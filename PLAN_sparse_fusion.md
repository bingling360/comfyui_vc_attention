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

## 9. 需要用户确认的点

1. **节点形式**：扩展 `VC Attention (MiniMax-H3)`（默认开稀疏），还是另开一个
   `VC Attention + Sparsity` 节点？（我倾向前者）
2. **默认 tau**：1.3（Sol 调优值，偏快）还是 1.0（论文默认，偏稳）？
3. **质量底线**：PSNR 掉多少以内可以接受？（决定 tau 的推荐工作点）
4. 实例保持开启？
