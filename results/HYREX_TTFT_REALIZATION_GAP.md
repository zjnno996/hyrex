# HyRex：更深 KV 命中为何可能增加 TTFT

## 1. 文档目的

本文记录 HyRex 当前最关键的 motivation 发现：对 Hybrid LLM 解除 Full-KV
与 recurrent state 的 528-token 共同边界后，系统能够命中更深的 KV prefix，
但 naive 实现没有降低 TTFT，部分配置反而更慢。

这个结果不说明独立索引没有价值，也不说明更深命中天然更慢。它说明：

> 发现可复用状态（reuse opportunity）与将其转化为端到端延迟收益
> （realized speedup）是两个不同问题。

本文只使用已经完成且口径明确的实验作主要证据。不同源码版本、异常 I/O
期间的结果以及未完成的 ablation 不用于定量归因。

---

## 2. 原生系统做了什么

当前 Qwen3.5-9B、vLLM prefix caching、`mamba_cache_mode=align` 配置中，
vLLM 为了让 Full-Attention page 的字节大小能够容纳一个 GDN/Mamba state page，
将 attention block 扩大到 528 token，并进一步将以下三种粒度绑定：

1. 物理 cache page 粒度；
2. prefix hash/index 粒度；
3. 可执行恢复边界。

运行日志明确记录：

```text
Setting attention block size to 528 tokens to ensure that attention page size
is >= mamba page size.
```

因此，对于第一轮长度 872、下一轮长度 903 的真实 ShareGPT session：

```text
上一轮： 0 ================================= 872
原生命中：0 ==================== 528
下一轮：                       528 -------------------- 903
                                             375 tokens
```

原生 vLLM+LMCache 并不是没有复用缓存。它已经从共同边界 528 恢复，显著优于
从 0 冷启动。它的局限是：由内存布局产生的 528 对齐约束同时成为了 prefix
索引和模型恢复约束，最后一个共同边界之后的已有状态不能被独立利用。

论文中应称其为 **conservative aligned recovery**，而不是“无效恢复”。

---

## 3. 三条恢复路径

固定例子：上一轮 872 token，下一轮 903 token。

### 3.1 原生对齐恢复

```text
Full KV hit = 528
State hit   = 528
Forward     = 903 - 528 = 375 tokens
```

### 3.2 Deep-KV + shallow-state replay

```text
Full KV hit = 864
State hit   = 528
Forward     = 375 tokens
```

528--864 区间已有 Full KV，因此 8 个 Full-Attention 层可以避免部分 K/V
projection；但为了构造后续层输入和推进 GDN state，整个模型仍需处理 375 token。

它节省的是 **layer-selective operators**，不是 336 个完整模型 token。

### 3.3 Tail-State recovery

额外维护 `S864`：

```text
Full KV hit = 864
State hit   = 864
Forward     = 903 - 864 = 39 tokens
```

这条路径减少 336 个完整模型 token（89.6%），但当前请求还要生成下一轮使用的
`S896`，并承担 state 查询、捕获、保存和传输成本。

---

## 4. 最新固定三臂结果

实验：Qwen3.5-9B BF16 eager，单请求，一条真实 ShareGPT session
`WRAImOg_0`，872 -> 903 token；每个正式请求前清 GPU prefix cache、保留配对内
CPU cache；每种方法 10 个无关 warmup，3 次 seed/resume 重复。

| Resume 指标 | 原生对齐 | Deep KV | Tail State |
|---|---:|---:|---:|
| Full KV hit | 528 | 864 | 864 |
| State hit | 528 | 528 | 864 |
| 完整模型 forward token | 375 | 375 | 39 |
| 每个 Full 层额外避免的 K/V projection 位置 | 0 | 336 | 336 |
| TTFT mean | 125.60 ms | 137.14 ms | 155.86 ms |
| TTFT median | 135.52 ms | 139.57 ms | 151.94 ms |

相对原生：

| 比较 | Mean 差值 | Median 差值 |
|---|---:|---:|
| Deep KV - Native | +11.54 ms | +4.05 ms |
| Tail State - Native | +30.26 ms | +16.42 ms |
| Tail State - Deep KV | +18.72 ms | +12.37 ms |

原生第二次恢复为 103.17 ms，明显低于另外两次，因此 mean 差值受样本波动放大；
三个重复不足以形成统计显著性结论。稳健地说，这个 case **没有观察到净加速**，
而不是证明所有 workload 下必然变慢。

原始结果：

- `/root/hyrex_results/deep_native_retest_20260929_v1/COMPARISON.md`
- `/root/hyrex_results/deep_native_retest_20260929_v1/1_optimized/online.jsonl`
- `/root/hyrex_results/deep_native_retest_20260929_v1/2_baseline/online.jsonl`
- `/root/hyrex_results/tail_verified_retest_20260929_v1/1_optimized/online.jsonl`

---

## 5. 根因一：更深 KV 覆盖不等于完整模型计算覆盖

这是 Deep-KV 路径收益小的首要原因。

Qwen3.5-9B 共 32 层，其中 8 层 Full Attention、24 层 GDN。更深 Full KV
只直接帮助 8 个 Full-Attention 层的历史 K/V 构造。以下工作仍然存在：

- 24 个 GDN 层的 state evolution；
- 所有层的 normalization；
- Q/gate projection；
- attention output projection；
- MLP；
- 层间 hidden-state propagation。

因此：

\[
\text{KV token coverage} \neq \text{whole-model computation coverage}.
\]

在一个已完成的 Nsight 诊断 case（831-token continuation）中，Native 与 Deep
都 forward 303 token。Deep 在 224 个位置复用 K/V，但 8 个 Full 层的 projection
kernel 总时间反而从 1.626 ms 变为 1.741 ms；全引擎 kernel sum 从 47.224 ms
变为 47.405 ms。原因是原生一个融合 QKV GEMM 被拆成两个更小、形状更差的 GEMM，
并增加 launch 和拼接开销。这个 case 说明“FLOPs 更少”不保证“GPU 时间更短”。

证据：`/root/hyrex_results/single_forward_profile_20260927_v1/DIAGNOSIS.md`。

---

## 6. 根因二：细粒度索引走了一条比原生更重的控制与传输路径

独立索引当前不是一次查询返回 `(L_KV, L_S)`，而是多个对象组、多个 namespace
和多次提交/检查。最新 Tail 路径仍先确认 Full-KV，再查询 tail-state；Deep 路径
虽然可以批量提交部分查询，但 retrieve 内部仍按 subgroup 顺序处理。

当前额外工作包括：

- Full-KV 与 state 分离 lookup；
- 多个 IPC future 的提交和状态检查；
- 多个 storage lock/lease 的生命周期管理；
- partial coarse object staging；
- 按层 `index_copy_`/scatter；
- 与原生 C++ transfer planner 不完全相同的 Python 组织路径。

四 session ABBA 诊断中，LMCache retrieve handler 的主机侧均值为：

```text
Native: 3.403 ms
Deep:   9.569 ms
```

这些是 host handler 墙钟时间，不是纯 PCIe DMA，也不能直接从 TTFT 相减；但它们
确认 Deep 当前使用了更重的提交/准备路径。对应 ABBA 端到端续轮均值：

```text
Native: 160.530 ms
Deep:   162.833 ms   (+2.303 ms, +1.43%)
```

证据：`/root/hyrex_results/deep_telemetry_abba_20260928_v1/DIAGNOSIS.md`。

历史微测还表明，16-token 索引本身不是性能问题：把每个 16-token page 逐页搬运
时，33 MiB Full KV 需要 18.753 ms；改为连续批量搬运后降为 1.853 ms，优于
原 528-token fallback 的 2.332 ms。问题在于 **细粒度索引被错误地实现成细粒度
数据移动和细粒度提交**。

因此索引技术仍是必要贡献，但必须满足：

> Fine-grained matching, coarse-grained transfer.

即索引按 16 token 精细匹配，传输仍对连续对象做少量大 DMA。

---

## 7. 根因三：Tail checkpoint 当前破坏了原生短前向 fast path

Tail-State 路径虽然把完整模型 forward 从 375 降到 39 token，但当前实现为了在
内部边界保存新 checkpoint，会将一个 GDN forward 分段。例如：

```text
39-token forward = 32-token recurrent call + 7-token recurrent call
```

当前源码仍包含：

- 每个 segment 单独调用 recurrent kernel；
- 对 `q/k/v/g/beta` 做切片；
- checkpoint state copy；
- conv-state copy；
- 多 segment 输出 `torch.cat`。

Qwen3.5-9B 有 24 个 GDN 层，因此一次内部 checkpoint 会把每层一次 recurrent
调用变成两次，并增加数十次 eager kernel launch 和 host/operator 工作。

历史 profiler（优化 direct-snapshot 之前的版本）中：

| 指标 | Deep | Tail |
|---|---:|---:|
| Forward tokens | 303 | 79 |
| GPU kernel sum | 47.405 ms | 26.662 ms |
| GPU kernel count | 1882 | 2096 |
| GDN recurrent calls | 24 | 48 |
| Model execution CPU range | 113.634 ms | 124.962 ms |

它证明 Tail 确实减少了 GPU arithmetic，但 checkpoint capture 同时增加了 kernel
数量和 CPU/operator 开销。当前实现已经去掉一部分 clone，但“分段调用 + copy +
cat”的结构仍在，因此旧 profiler 不能直接作为当前版本的毫秒归因，却仍能说明
瓶颈类型。

证据：`/root/hyrex_results/single_forward_profile_20260927_v1/DIAGNOSIS.md`。

---

## 8. 根因四：checkpoint 的成本发生在当前请求，收益发生在未来请求

恢复 `S864` 帮助当前请求少算 336 token；为了让下一轮继续受益，当前请求又需要
生成并保存 `S896`。因此 steady-state 每一轮都同时包含：

```text
使用上一轮 checkpoint 的收益
+ 维护下一轮 checkpoint 的成本
```

维护成本包括：

- checkpoint capture；
- 独立 GPU snapshot slot；
- STORE metadata/IPC submission；
- state D2H；
- CPU cache capacity；
- 对其他 KV/state 的 eviction opportunity cost。

当前 connector 的 STORE 完成是异步的，但请求会在模型 forward 之后、首 token
完成路径上提交 STORE；实际 D2H 还可能与采样或后续 GPU 工作争用 PCIe/显存带宽。

这部分存在两个层次：

1. **固有成本**：state 占用和最终 D2H 无法凭空消失；
2. **当前实现成本**：capture 分段、串行提交以及把维护放在 TTFT 关键路径上不是
   必然要求，可以通过自然 turn-boundary state、独立 snapshot slot 和异步 spill
   优化。

当前版本已经完成 no-new-checkpoint recovery-only ablation。它在命中 exact state
后跳过下一 checkpoint 的 capture 和 STORE；与 steady-state 版本相比，可分离“本轮
恢复”和“为下一轮维护”的成本。受控 512-token gap 上，recovery-only 与 steady
版本的 median TTFT 分别是 162.43 和 166.35 ms，即维护成本约 3.92 ms；五次样本
mean 差值为 10.00 ms。其他较小 gap 的差值受 host 波动影响、符号不稳定，因此不应
把单点维护成本外推成全局常数。

---

## 9. 根因五：单请求短前向存在较高固定成本

当前实验是单请求、BF16、eager。把 375 token 减到 39 token 并不会同比减少
所有算子时间：

- 每层仍需读取模型权重；
- kernel launch 数不会按 token 数同比下降；
- 小 GEMM 可能更难充分利用 GPU；
- eager 下新增的小 kernel 与 host dispatch 更明显；
- 39-token 路径还是自定义 checkpoint path，而不是原生 39-token fast path。

因此“少算 89.6% token”是逻辑工作量指标，不能直接写成“前向快 89.6%”。
必须使用 ideal oracle 测量同一模型真正的 39-token native forward 下界。

---

## 10. 总体成本模型

对于候选恢复路径 `P`：

\[
T(P)=T_{lookup}+T_{H2D}+T_{reconstruct}+T_{forward}
     +T_{maintenance}+T_{contention}.
\]

相对原生 aligned recovery 的变化可以写为：

\[
\Delta T(P)=
-S_{compute}
+C_{index}
+C_{transfer}
+C_{reconstruct}
+C_{maintenance}
+C_{contention}.
\]

- Deep-KV 的 `S_compute` 很窄，主要是 8 个 Full 层的部分 K/V projection；
- Tail-State 的 `S_compute` 更大，但当前 `C_reconstruct+C_maintenance` 也很大；
- 当额外成本大于被消除的 GPU 时间时，更深命中就会增加 TTFT。

当前结果的核心不是“load 永远比 recompute 慢”，而是：

> The current system pays a general-purpose heterogeneous-recovery tax to save
> a workload-dependent amount of computation.

---

## 11. 哪些成本是固有的，哪些是实现问题

| 成本 | 固有/实现 | 说明 |
|---|---|---|
| 更深 KV 的额外 H2D 字节 | 固有 | 除非 KV 已在 GPU 或与计算完全重叠 |
| recurrent checkpoint 的约 49.5 MiB 容量 | 固有 | 可通过选择性保存和多级放置控制 |
| checkpoint 最终 D2H | 固有 | 可移出 TTFT 路径，不能消除总带宽 |
| KV/state 两次串行 lookup | 实现问题 | 应由一次异构前沿查询返回 |
| 16-token page 逐页传输 | 实现问题 | 已证明可用批量 DMA 消除 |
| Full projection 拆成低效小 GEMM | 实现问题/算子依赖 | 需要融合或保持原生 GEMM |
| GDN 为 checkpoint 分成两次完整调用 | 实现问题 | 应使用自然终态或 kernel 内部输出 checkpoint |
| snapshot 与 running state 生命周期冲突 | 系统设计问题 | 需要独立 immutable slot |
| 高并发下 PCIe/GPU contention | 运行时固有 | 需要 cost-aware scheduling |

---

## 12. 对索引技术的直接要求

负 TTFT 结果不是删除独立索引的理由，反而给出了索引必须达到的设计要求。

### 12.1 一次查询返回异构前沿

不能先查 Full-KV、再查 tail state、再查 ordinary state。一次 prefix traversal
应返回：

\[
H=(L_{KV},\{L_{S_i}\},location_i,validity_i).
\]

### 12.2 精细匹配与批量传输分离

- hash/index：16-token KV page；
- state anchor：528 coarse checkpoint 或稀疏 semantic checkpoint；
- transfer：将连续 KV pages 合并成少量大对象；
- execution：由 planner 选择恢复边界，不由物理 page 大小决定。

### 12.3 State anchor 挂在已有 prefix node 上

不为每 16 token 保存 state。只在以下位置稀疏附着 checkpoint：

- turn end；
- tool-call 前后；
- branch root；
- rollback point；
- planner 判定高价值的位置。

这使索引同时支持 append-only chat 与 agent rollback/branch，而不会产生密集 state
容量开销。

---

## 13. 论文问题定义与核心挑战

不应将问题写成：

> 原生系统无法降低 TTFT，因此我们使用更细粒度 cache。

正确表述是：

> 原生 aligned recovery 已经减少了相对冷启动的 TTFT，但在 Hybrid 模型中，
> 物理 page 对齐限制了可见的恢复边界。独立索引暴露了更深的 layer-specific
> reuse opportunity，却不自动产生端到端收益。挑战是以低于被消除计算的成本，
> 将异构命中转化为可执行模型状态。

建议使用术语：

> **Heterogeneous Reuse Realization Gap**：异构 cache 中理论可复用计算与实际
> TTFT 收益之间的差距。

索引本身不是一个足够大的论文挑战。HyRex 应围绕以下两个核心挑战展开；
独立索引、Tail State 和 CPU Cache 管理分别作为解决它们的机制。

### Challenge 1：将异构命中转化为高效的可执行恢复前缀

Full-Attention KV 与 recurrent state 具有不同的匹配粒度、恢复语义和计算覆盖。
因此最深的物理 KV 命中不一定能直接执行，从浅 state 向深 KV replay 也不等于
跳过相同数量的完整模型 token。系统必须联合完成三件事：

1. 独立发现 KV 与各 recurrent state 的异构前沿；
2. 判断真正可以直接执行的恢复边界；
3. 在原生对齐恢复、Deep-KV replay 和 Exact-State 恢复之间构造并选择路径。

对应的 HyRex 机制是 **Heterogeneous Recovery Planner**。它通过一次联合查询返回

\[
(L_{KV},\{L_{S_i}\},L_{exec}),
\]

其中 `L_exec` 是 KV 和模型状态都足以继续执行的位置。Planner 使用自然请求结尾
已经生成的 recurrent state，避免为了捕获 checkpoint 拆分正常 forward，并以预测
TTFT 而非命中 token 数选择恢复路径。独立索引是该 Planner 的必要组件，而不是
单独的论文贡献。

### Challenge 2：在有限资源下只物化、保留和恢复有净收益的状态

Exact state 可以推进 `L_exec`，但不能为所有边界无条件保存。当前 Qwen3.5-9B
的一个 recurrent checkpoint 约为 49.5 MiB；仅为 16 个 session 各保存一个
checkpoint 就需要约 792 MiB。State 还会挤占 KV pages，并在保存和恢复时消耗
PCIe 带宽；高并发下，无约束的 checkpoint D2H/H2D 会进一步产生排队和批处理
碎片。因此系统必须在有限 CPU 容量、传输带宽和 GPU 执行资源下决定哪些边界
值得创建、保留和使用。

对应的 HyRex 机制是 **Recovery-aware Cache Orchestrator**。它联合管理 KV 与
state 的 admission、eviction、transfer 和 recovery scheduling，并最大化单位缓存
容量带来的预期净收益：

\[
U(o)=\frac{P_{reuse}(o)\left(T_{replay\ saved}(o)-T_{materialize}(o)
-T_{transfer}(o)-T_{restore}(o)\right)}{Size(o)}.
\]

当预测收益不足时，系统回退到原生 aligned recovery；当 checkpoint 能避免足够长
的 replay 时，才物化和加载 Exact State。并发恢复采用有界传输和 packed recurrent
execution，避免逐请求 kernel 调用破坏 batching。

两项挑战形成一条连续主线：

```text
异构 KV/state 命中
        -> 构造正确的 executable prefix（Challenge 1）
        -> 判断该 prefix 是否值得物化、缓存和恢复（Challenge 2）
        -> 实际 TTFT 收益
```

---

## 14. 受控 Gap Sweep：恢复何时真正获益

实验固定 coarse recurrent boundary 为 1056，上一轮结束在 `1056 + gap`，下一轮
再追加 128 token；`gap={0,64,128,256,384,512}`。每个点重复五次，每个请求后
验证 GPU prefix cache 已清空，CPU LMCache 保留。Native 使用 connector 强制的
528-token budget；Aligned-SF、Deep-KV 和两个 Exact 版本共享同一 single-forward
实现与 2112-token budget，用于做实现口径一致的因果对比。正式区间均无 JIT
warning，所有 30 个 continuation 的 first token 与 Native 一致。

| 避免的完整 replay | Aligned median | Deep-KV Δ | Exact recovery-only Δ | Exact steady Δ |
|---:|---:|---:|---:|---:|
| 0 | 140.73 ms | +0.40 ms | +20.98 ms | +16.56 ms |
| 64 | 144.64 ms | +1.13 ms | +13.40 ms | +13.09 ms |
| 128 | 144.31 ms | -0.58 ms | +6.12 ms | +13.17 ms |
| 256 | 143.84 ms | +4.39 ms | +15.95 ms | +12.78 ms |
| 384 | 143.61 ms | +3.37 ms | +11.14 ms | +11.59 ms |
| 512 | 199.57 ms | -2.72 ms | -37.14 ms | -33.22 ms |

这个实验给出三条直接证据：

1. Deep-KV 的报告命中从 1056 增到 1568，但完整模型仍从 state=1056 开始执行，
   TTFT 相对 Aligned 只在 -2.72 到 +4.39 ms 内波动，没有系统性收益；
2. Exact state 始终只执行新增的 128 token，但在小 gap 下固定恢复成本大于计算
   节省，不能无条件选择；
3. 当 Aligned 路径需要执行 640 token、跨过下一个 528 checkpoint 时出现明显
   latency cliff，Exact steady 仍只执行 128 token，median TTFT 下降 33.22 ms，
   mean 从 198.78 降到 166.34 ms（-16.3%）。

原始结果与分析见：

- `/root/hyrex_results/controlled_gap_sweep_20261001_v2/RESULTS.md`
- `/root/hyrex_results/controlled_gap_sweep_20261001_v2/paper_summary.json`
- `/root/hyrex_results/controlled_gap_sweep_20261001_v2/figure_gap_medians.csv`

由于宿主机 `/dev/shm` 只有 64 MiB，所有 arm 使用相同的 pickle IPC fallback。
这不影响 matched mechanism comparison 的方向，但正式发表数字仍应在充足 SHM
环境下复测。

---

## 15. 还必须完成的实验

### P0：Ideal oracle

预先将 `KV864` 和 `S864` 放入 GPU，不执行 lookup、H2D 和新 checkpoint 保存，
只运行原生 39-token forward：

\[
T_{oracle39}<T_{aligned375}
\]

是后续系统优化值得做的必要条件。如果 oracle 都不快，该 workload 应由 planner
选择 aligned recovery。

### P1：维护成本四阶段 ablation（已完成两端点）

在同一当前版本上依次测：

1. Tail restore，不生成下一 checkpoint（已完成）；
2. 加 checkpoint capture，不 STORE；
3. 加 STORE submission，不等待完成；
4. 完整 steady-state Tail（已完成）。

分别用 CUDA event 和 host monotonic clock 测量，不能混用日志时间相加。

### P2：索引/传输 ablation

保持完全相同的 payload 与 forward，仅替换：

- 串行双 lookup -> 一次联合 lookup；
- 当前 partial/scatter -> 原生批量 transfer planner；
- page-by-page -> coalesced object。

### P3：真实 session workload

必须保证下一轮 prefix 真正包含上一轮模型处理过的 token。当前“只生成 1 token，
下一轮使用数据集完整 assistant answer”的回放不能验证精确 turn checkpoint。
应使用完整生成、teacher forcing，或固定且严格连续的 token trace。

### P4：并发与容量

至少测 concurrency 1/4/8、不同 CPU/GPU checkpoint budget、不同 rollback/fan-out，
报告 p50/p95 TTFT、吞吐、传输字节、checkpoint hit rate 和单位容量收益。

---

## 16. 当前可以与不可以声称的结论

### 可以声称

- 原生 vLLM 将内存 page、prefix index 和 Hybrid 恢复边界耦合到 528 token；
- 独立索引能够发现更深的 Full-KV prefix；
- KV coverage 不等于 whole-model computation coverage；
- naive 更深恢复在当前 case 中没有降低 TTFT；
- Exact recovery 存在 request-dependent break-even；在受控 512-token gap 上，
  steady-state exact recovery 相对 matched aligned path 的 mean TTFT 降低 16.3%；
- 当前额外开销来自控制/传输路径、算子形状以及 checkpoint 维护，而不仅是 PCIe；
- 更深恢复需要联合优化 indexing、transfer、reconstruction 和 checkpoint placement。

### 不可以声称

- 更深 KV 命中天然比 aligned recovery 慢；
- 额外 30.26 ms 全部来自 H2D；
- 减少 89.6% token 就应减少 89.6% TTFT；
- 当前三次重复具有统计显著性；
- first-token 一致等价于完整输出正确；
- checkpoint 保存是全部根因（现有 ablation 只证明它是 512-gap 上约 3.92 ms
  median 的组成部分）。

---

## 17. 可直接用于 Introduction 的发现段落

> Existing aligned recovery already reduces latency relative to cold
> recomputation, but exposes only a conservative common boundary across
> full-attention KV entries and recurrent states. Decoupling the two reveals
> deeper layer-specific reuse. However, our measurements show that a deeper
> hit can increase TTFT: the saved computation may be narrower than the token
> coverage suggests, while additional lookup, transfer, state reconstruction,
> and checkpoint maintenance disrupt the original recovery fast path. We call
> this discrepancy the heterogeneous reuse realization gap.

> Closing this gap requires more than fine-grained matching. A practical system
> faces two challenges. First, it must convert heterogeneous KV and recurrent-
> state hits into an executable prefix without fragmenting the model's native
> recovery path. Second, under bounded CPU capacity and transfer bandwidth, it
> must materialize, retain, and restore only those prefixes whose expected saved
> recomputation exceeds their capture, transfer, restoration, and capacity costs.

> HyRex addresses these challenges with a heterogeneous recovery planner and a
> recovery-aware cache orchestrator. The planner jointly discovers KV and state
> frontiers, constructs feasible aligned, replay, and exact-state paths, and
> selects the path with the lowest predicted TTFT. The orchestrator jointly
> manages KV pages and recurrent checkpoints using expected latency benefit per
> byte, while bounding checkpoint traffic and preserving batched execution under
> concurrency.

---

## 18. 一句话结论

> HyRex 的关键不是“命中更多 token”，而是以低于所节省计算的成本，把更深的
> KV/state 命中转化为可执行状态；当前负结果正是这一技术问题存在的证据。
