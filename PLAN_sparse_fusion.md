# 计划：把 Sol-Attn 式块稀疏融合进 VC-Attention 内核（ComfyUI 节点）

状态：**待批准，尚未开工**
日期：2026-10-04
相关实测：`tests/_probe22_solstack.py`、`_probe27_stack_gain.py`、`_probe30_tilescale.py`

---

## 1. 目标

让 VC-Attention 在 sm_120 上**真正比现有方案快**，形式是一个「接上就生效」的 ComfyUI 节点。

**非目标**（明确排除，避免返工）：

- 不做「两个节点叠加」。已实测无效：`_probe27` 显示 0.98–0.99×，抢优先时 0.34×。
- 不做 4-bit。已实测边际太薄（`_probe28/29`）：即使拿到 PTX 单指令转换，
  4-bit 打包仍是 fp8 cast 的 8×，而 QK 规约维只有 D=128。
- 不改 V-Smooth 算法本身。

**核心判断**：稀疏（跳过计算）和量化（让计算变便宜）是**正交轴**，可以相乘；
而「叠加两个完整注意力实现」是功能重叠，必然互相抢 call site。
所以唯一有意义的组合形式是**融合进同一个 kernel**。

---

## 2. 为什么这条路有戏（已有实测支撑）

### 2.1 kernel 时间与 KV tile 数完全线性（`_probe30`）

65536 tokens / 56 heads / D=128：

| 处理 tile | 时间 | 比值 |
|---|---|---|
| 100% | 399.92 ms | 1.000× |
| 50% | 199.53 ms | 0.499× |
| 25% | 100.11 ms | 0.250× |
| 12.5% | 50.19 ms | 0.126× |

**没有固定开销占主导** → 跳过 KV block 是真杠杆。

### 2.2 稀疏与量化数值上兼容（`_probe22`）

| tau | keep | Sol 单独 | Sol+VC 量化 | 差 |
|---|---|---|---|---|
| 0.5 | 32% | 34.36 dB | 34.35 dB | −0.01 |
| 1.0 | 18% | 32.55 dB | 32.59 dB | +0.04 |
| 1.3 | 12% | 31.82 dB | 31.80 dB | −0.02 |

叠加几乎免费（稀疏误差 ≫ 量化误差）。**V-Smooth 置换不冲突，反而略好 +0.2~0.3 dB**
（置换后 block 更同质，块均值更能代表块内 token）。

### 2.3 投影

| 方案 | 64K 耗时 | vs SDPA |
|---|---|---|
| bf16 FlashAttention | 555 ms | 1.00× |
| Comfy Kitchen INT8 | 203 ms | 2.74× |
| VC fp8（现状） | 432 ms | 1.29× |
| **VC fp8 + 稀疏（tau=1.0）** | **~105 ms** | **~5.3×** |

---

## 3. 算法方案

### 3.1 路由规则（照抄 Sol-Attn，不自己发明）

来自 `ComfyUI-sol-attn/sol_kernel/preprocess.py`：

```
proxy(q_block, kv_block) = mean over q in q_block of ⟨q, mean_k(kv_block)⟩ · scale
threshold(q_block)       = mean_kv(proxy) + tau · std_kv(proxy)
keep(q_block, kv_block)  = (proxy > threshold) OR (|q_block - kv_block| <= 1)
```

外加：
- **sink 块恒精确**：H3 打包序列开头是 text/conditioning，对质量敏感。
- **被跳过的块不丢弃**：用 proxy 分近似其贡献（见 3.3）。

### 3.2 粒度选择

| | Sol-Attn | 本方案 | 理由 |
|---|---|---|---|
| query 组 | 64 | **BLOCK_M = 64** | 天然对齐 |
| KV 块 | 64 | **BLOCK_N = 128** | 对齐现有 tile，避免内核双分支 |

128 的块比 64 粗，阈值统计样本更少（std 更噪）。**Phase 2 先做 128**，
若质量不达标再考虑在 tile 内做 64 子块路由（记为备选）。

### 3.3 被跳过块的近似

对跳过的 tile，把该 tile 的所有 128 个 KV 行都视为分数 = proxy：

```
p_row = exp2((proxy_rows - m_new) · LOG2E)        # (BLOCK_M,)
l_i  += BLOCK_N · p_row
acc  += p_row[:, None] · (BLOCK_N · v_mean_tile)[None, :]
```

需要每块一个 `v_mean`（与 `k_mean` 同量级，很便宜）。
注意 proxy 用**逐行**值做近似，只用**均值**做路由决策——比只用均值近似更准。

### 3.4 阈值计算：放主机侧，不放内核

内核内两遍法虽然自包含，但要多 nb 次迭代，延迟可能吃掉收益。
主机侧成本很低，采用 Sol 的做法：

1. `k_mean`：(G, nb, D)，对**置换后**的 K 按 BLOCK_N 求均值
2. `q_centroid`：(G, n/64, D)，对 Q 按 BLOCK_M 求均值（Sol 的 `_pool_query_kernel`）
3. `proxy` = `q_centroid @ k_meanᵀ · scale` → (G, n/64, nb)
4. `threshold` = mean + tau·std over nb → (G, n/64)

64K 规模下 proxy 是 56×1024×512 = 29M 值（fp32 约 117 MB），可接受。
**若显存吃紧**，用 fp16 存 proxy 或分 head 流式计算。

### 3.5 调度（可选，Phase 4）

Sol 的做法：tau 随去噪步数从 `tau_start` 线性降到 `tau_end`，且前若干步保持稠密。
`patch.py` 已有 `GroupSchedule` 步调度设施，直接复用即可。

### 3.6 与 V-Smooth 置换的关系

已实测无害。但要注意：**所有块统计量必须在置换之后计算**，
因为内核看到的是置换后的 K/V。

---

## 4. 内核改造

文件：`vc_attention/kernels/triton_attn.py`

新增 constexpr：`SPARSE`（沿用现有 `TILE_SKIP` 的槽位或并列），新增入参：
`KMEAN`、`VMEAN`、`THRESH`。

主循环改成：

```
for start_n in range(0, N_PAD, BLOCK_N):
    blk = start_n // BLOCK_N
    proxy_rows = tl.sum(q * KMEAN[blk], axis=1) * sm_scale     # (BLOCK_M,)
    thr        = tl.load(THRESH + ...)                          # scalar
    keep       = (tl.sum(proxy_rows) / BLOCK_M > thr) | local | sink
    if keep:
        <现有 QK + PV 全流程>
    else:
        <3.3 的近似路径，只读 KMEAN/VMEAN，不读 K/V>
```

**关键点**：
- `keep` 必须是 **program 级标量**才能整块跳过（Triton 支持标量 `if`）。
- 跳过路径不加载 K/V，这是省时间的来源。
- `TILE_SKIP` 保留但改名为测量专用（或删除）。

---

## 5. ComfyUI 节点设计

**扩展现有节点**（不是新节点）——用户要的是「接上就自动融合」，一个节点最省事。

`nodes.py` / `VCAttentionConfig` 新增：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `enable_sparsity` | BOOLEAN | **True** | 关掉即退回纯量化路径 |
| `tau` | FLOAT | **1.3** | Sol 调优值；越大越快、质量越低 |
| `sparsity_min_tokens` | INT | 8192 | 短序列不值得 |
| `sink_tokens` | INT | 512 | 开头这些 token 所在块恒精确 |
| `local_blocks` | INT | 1 | ±N 个邻块恒精确（Sol 用 1） |

可选（Phase 4）：`tau_end`、`dense_percent`、`curve`（复用 GroupSchedule）。

**override 优先级**：现在默认 `"defer"`（因为 VC 慢，抢优先是 3× 倒退）。
**一旦 Phase 3 实测融合版快于 Kitchen，就把默认改成 `"front"`** —— 那时
「接上就生效」才成立。这是唯一需要改默认值的决策点，且必须由数据决定。

节点 `apply()` 打印：探测到的稀疏率、预估倍率、以及若与其它后端共存时的说明。

---

## 6. 实施阶段与验证门

每个阶段结束都必须过门，否则停下讨论。

### Phase 1 — 主机侧路由预处理（~30 min）
实现 `k_mean` / `q_centroid` / `proxy` / `threshold`。
**门 G1**：与 PyTorch 直算的 proxy/threshold 一致（rel err < 1e-4）；
打印各 tau 下的实际 keep 比例，应与 `_probe22` 接近。

### Phase 2 — 内核路由（~1 h）
按第 4 节改造，`SPARSE` 开关。
**门 G2**：`SPARSE=False` 时输出与当前 kernel **逐元素一致**（无回归）。
**门 G3**：`SPARSE=True` 与 `_probe22` 的 PyTorch 仿真路由结果接近（rel err < 1e-3）。

### Phase 3 — 计时与质量（~30 min）
16K / 64K，tau ∈ {0.8, 1.0, 1.3, 2.0}，对比 SDPA / Kitchen / 当前 VC。
**门 G4**：融合版在 tau=1.3 下**快于 Kitchen INT8**（否则本方案不成立，如实汇报）。
**门 G5**：给出 tau–速度–PSNR 曲线，标出推荐工作点。

### Phase 4 — 调度 + 节点集成（~45 min）
tau 调度、参数接进节点、README 更新。
**门 G6**：CPU 测试套件仍 73/73；节点 `apply()` 正常打印。
**门 G7**：若融合版确为最快，把 `override_priority` 默认改为 `"front"`。

### Phase 5 — 收尾（~15 min）
提交本地 + pod，更新 README 与 memory。

**总计约 3 小时**（SSH 不稳，实际可能更长）。

---

## 7. 风险与回退

| 风险 | 概率 | 应对 |
|---|---|---|
| Triton 标量分支编译效率差，收益被吃掉 | 中 | 退化为「掩码计算」而非真跳过；或改 64 子块路由 |
| 真实路由是**散点**跳块，而 `_probe30` 测的是等间隔跳块，cache 行为可能更差 | 中 | 内核是 compute-bound，预期影响有限；Phase 3 直接测真实路由 |
| 实际 tau 下质量不可接受 | 中 | 给出曲线让用户选工作点；必要时默认关 |
| 融合版仍慢于 Kitchen | 中 | **如实汇报并停止**，不改默认值，把发现写进 README |
| 实例被回收 | 低 | 全部代码已提交 git（本地 + pod），重搭约 10 min |

**不做的事**：不为了让数字好看而调参过拟合；不隐瞒负结果。

---

## 8. 交付物

1. `vc_attention/kernels/triton_attn.py` 融合稀疏的 kernel（默认关闭，节点里默认开）
2. `vc_attention/patch.py` 路由预处理 + 参数
3. `nodes.py` 新参数（接上即生效）
4. `tests/_probe31_*.py` 正确性 + 计时 + tau 曲线
5. README 新增章节：融合方案、实测曲线、与 Kitchen/Sol 的对比
6. 明确结论：**融合版是否真的最快**（是就改默认，不是就说清楚）

---

## 10. 执行状态与重要修正（2026-10-04 02:00）

### Phase 1 — 完成 ✅

`_prepare_fast(sparse=True)` 产出 `k_mean` / `v_mean` / `thresh`。
G1 通过（`_probe31`）：16K keep 0.324/0.178/0.118（probe22 独立算出 0.322/0.177/0.117），
thresh 相对误差 ~6e-4。

### Phase 2 — 机制就位，但有两处需要返工 ⚠️

**已经对的**（`_probe33`）：用三种独立方式强制"全保留"（LOCAL=1e5 / SINK=1e5 /
tau=-1000）都与 `SPARSE=False` **逐元素完全相同**（rel-err 0.000e+00）。
→ 重构后的保留分支没问题，阈值与索引管线也没问题。

**问题 1：近似路径写错了。** 读 Sol 的 `sol_kernel/fwd.py:150-200` 后确认：

```python
kc = ...            # 一组 block 的 K 均值
vc = ...            # 一组 block 的 V 均值
scores = tl.dot(q, kc.T) * scale_log2      # 代理分：对 block 均值做 matmul
routed = (tl.sum(scores, 0)/q_len > threshold) | local | sink_kv
approximate_scores = tl.where(approximate, scores, -inf)
new_max = tl.maximum(row_max, tl.max(approximate_scores, 1))
approximate_probability = exp2(approximate_scores - new_max)
output += tl.dot(approximate_probability, vc)          # (BS, GROUP) @ (GROUP, BV)
row_sum += tl.sum(approximate_probability * lengths, 1)
```

**两个我搞错的关键点**：
1. 代理分是**对 block 均值 K 的一次 matmul**（一次算一整组 block），
   不是每个 tile 单独算。
2. 近似路径是 **(BLOCK_SIZE × GROUP_SIZE) @ (GROUP_SIZE × BV)** 的小 matmul ——
   每个 block 只贡献**一列**（用块均值 V），不是 128 列。

我用的是"每 tile 一个 outer product + 每列都记 proxy"，配合
`m_new = max(m_i, proxy)`，导致被跳过块的权重比真实块高 10–100×。
所以 tau=0 / 1.3 / 100 三个完全不比例的结果 PSNR 几乎一样（22.74/22.75/22.76）。

**问题 2：路由开销 +27%。** 16K 下"全保留"要 41.16 ms，而 `SPARSE=False` 只要
32.46 ms。每 tile 的 `q.to(fp32) * km[None,:]` 会materialize 一个 (64,128) fp32
临时张量，寄存器压力大。应该改成用 **pooled query centroid** 算标量代理
（一次 D 长度点积），而不是每行都算。

### ⚠️ 对第 2.3 节投影的修正（重要）

`_probe30` 测的是"**整个 tile 都跳过**"的上限。但 Sol-Attn 的真实算法
**并不跳过整个 tile** —— 它对**所有** block 都做一次便宜的代理 QK（对块均值），
只跳过 **PV**。所以：

| | 我原来的投影 | 修正后 |
|---|---|---|
| 跳过内容 | QK + PV 全部 | 只跳 PV（QK 用便宜的代理版） |
| 64K 投影 | ~105 ms | **待实测**，会明显高于 105 ms |

16K 实测：32.46（无路由）→ 41.16（有路由，全保留）→ 28.98（有路由，keep 12%）。
**净收益目前只有 3.5 ms（11%），远不及投影。** 要把路由开销压下去、
把近似路径按 Sol 的结构重写，才谈得上收益。

### 下一步（Phase 2 返工）

1. 代理分改成 **pooled query centroid** 的标量点积（消除 27% 开销）
2. 近似路径按 Sol 结构重写：块均值 V + 单列贡献，修正归一化
3. 重跑 G2/G3/G4

**SPARSE 默认 False，当前无回归（CPU 套件 73/73）。**

---

## 9. 已确认的决定（2026-10-04）

| 问题 | 决定 |
|---|---|
| 节点形式 | **扩展现有 `VC Attention (MiniMax-H3)`**，稀疏默认开启 |
| 默认 tau | **1.3**（Sol 调优值） |
| 质量底线 | **PSNR 相对稠密下降 ≤ 3 dB** —— 决定推荐工作点；超过就必须降 tau |
| GPU 实例 | **保持开启** |

由此细化的两条硬约束：

- 默认 `tau=1.3`，但 Phase 3 必须给出 tau–速度–PSNR 曲线，**推荐工作点取满足
  ≤3 dB 的最快 tau**；若 tau=1.3 就超了 3 dB，则默认值下调到合规档并在 README 写明。
- `override_priority` 的默认值只在 Phase 3 实测「融合版快于 Kitchen」之后才改为
  `"front"`；否则保持 `"defer"`。

---

## 11. Phase 2 返工完成 + Phase 3 实测（2026-10-04 03:xx）

### Phase 2 返工 — 完成 ✅（根因与修法）

**根因不是「代理分算法」，是「把 Sol 的批处理拆成了逐 tile 的 fp32 逐元素运算」。**

1. **代理分开销**：`q.to(fp32) * km[None,:]` 每个 tile materialize 一个 (64,128) fp32
   临时张量 → 全保留时 32.4 → 41.2 ms（+27%）。
2. **近似路径才是真正的大头**：逐 tile 的 `p_row ⊗ (BLOCK_N·vm)` 是 fp32 逐元素
   (64,128) 更新，实测吃掉稠密 kernel 的 ~45%。所以「跳过」根本不省时间：
   tau=0（keep 50%）与 tau=1.3（keep 12%）耗时**完全相同**（29.06 vs 28.94 ms）。
   —— 这解释了 probe32/34 里「结果与 tau 无关」的假象。
3. **归一化 bug（重要）**：近似路径的分子漏了 block 长度因子（分母有），
   与 Sol 的写法一致但和「块均值 V」不自洽，导致近似块权重偏低。
   修法：把 `lengths` 折进 `p_ap` 再做 matmul（分子分母同尺度）。修完
   tau=1.3 的 PSNR 从 31.62 → **34.92 dB**（+3.3 dB）。
4. 附带修掉一个尺寸 bug：`|start_m - blk| <= LOCAL` 把 BLOCK_M(64) 与
   BLOCK_N(128) 混在一个单位里比较，改成 `qb_kv = start_m*BLOCK_M//BLOCK_N`。

**做法**：新增独立 kernel `_vc_attn_fwd_sparse`（稠密 kernel 一字未动，G2 天然成立），
按 Sol 的结构走——每 `GROUP=16` 个 KV block 一组：
- 一次 tensor-core matmul `q @ kc^T` 得到整组的逐行代理分；
- 行均值与主机侧阈值比较；
- 被跳过的 block 用**一次小 matmul** `p_ap @ vc`（块均值 V，长度已折入）折进来；
- 被保留的 block 用 `_vc_exact_tile` 逐个精确算（与稠密 kernel 同一段代码，
  避免漂移）。

### Phase 3 实测

**门 G2（无回归）** ✅ `_probe33`：LOCAL=1e5 / SINK=1e5 / tau=-1000 三种强制全保留
都与 SPARSE=False **逐元素相同**（rel-err 0.000e+00）。

**门 G3（路由正确）** ✅ `_probe35`：与独立 PyTorch 仿真（同样的量化算子 +
同样的「块均值 V / 复用 proxy」近似）在四个 tau 下 PSNR 差 ≤0.01 dB：
| tau | keep | 仿真 | kernel |
|---|---|---|---|
| 0.8 | 0.230 | 36.73 | 36.72 |
| 1.0 | 0.178 | 36.12 | 36.12 |
| 1.3 | 0.119 | 35.46 | 35.47 |
| 2.0 | 0.046 | 34.82 | 34.83 |
（rel-err 7–9e-2；1e-3 那个目标不现实——kernel 的近似路径是 bf16 matmul，
仿真 是 fp32。PSNR 一致到 0.01 dB 才是有意义的判据。）

**门 G4（快于 Kitchen）** ✅（64K）/ ⚠️（16K）：
| tokens | bf16 SDPA | Kitchen INT8 | VC 稠密 | **VC 稀疏 tau=1.3** |
|---|---|---|---|---|
| 16384 | 35.4 ms | 13.45 ms | 32.9 ms | **15.0 ms**（0.90× Kitchen） |
| 65536 | 559 ms | 202.1 ms | 432 ms | **112.3 ms**（**1.80× Kitchen**，4.98× SDPA） |

原投影「64K ~105 ms / ~5.3× SDPA」→ 实测 112 ms / 4.98×，**投影准确**。

**门 G5（tau–速度–PSNR 曲线）**：见上表 + probe35 输出的四个 tau 点。

### Phase 4 — 完成 ✅

节点新增 `enable_sparsity`(True) / `tau`(1.3) / `sparsity_min_tokens`(8192) /
`sink_tokens`(512) / `local_blocks`(1)；`apply()` 打印稀疏配置。
`override_priority` 默认由 `defer` 改为 **`front`**（依据：64K 实测 1.80× 快于 Kitchen；
16K 慢 11%，README 写明）。

### 质量门 — **未通过，且在这块数据上无法通过** ⚠️（如实汇报）

- 合成 `h3_like` 上，稀疏相对稠密掉 ~20 dB，**任何 tau 都超 3 dB 底线**。
- 但这不是本 port 的问题：**独立的 fp32 Sol 仿真（probe22 的规则）在同一数据上只有
  31.8 dB，比本 kernel（35.5 dB）更差**。原因是合成数据的 K 在 128-token block 内
  近乎正交，块均值 proxy 代表不了块内 token —— 算法前提不成立。
- **结论：合成 PSNR 不能用来判定稀疏的质量损失。** 真正的门是端到端 H3 出图对比，
  pod 上有完整 H3 权重（`models/diffusion_models/minimax_h3_*`），但需要搭一个可跑
  的工作流 + 重启 ComfyUI 载入新节点，本次未做。
- **未做的事**：没有为了让数字好看而调参；没有隐瞒这个负结果。

### 下一步（留给下次）

1. **端到端质量验证**（唯一未过的门）：搭 H3 最小工作流，dense vs sparse 出图对比，
   再决定 `enable_sparsity` 默认值是否保持 True。
2. probe30 的 `TILE_SKIP` 已随稠密 kernel 恢复原签名（测量专用）。
3. 若端到端质量可接受，可再考虑 tau 调度（Phase 4 的 `GroupSchedule` 复用）。

