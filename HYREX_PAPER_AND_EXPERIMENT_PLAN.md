# HyRex 论文与实验总计划

## 1. 论文要解决什么问题

HyRex 面向 Qwen3.5 一类同时包含 Full Attention 与 GDN/SSM 层的 Hybrid
LLM，重点研究多轮对话和 agent session 从 CPU cache 恢复到 GPU 时的 TTFT。
核心问题不是“能否命中更长前缀”，而是：

> 当 Full-KV 与 recurrent state 的可复用边界、大小和恢复代价不同，系统如何构造
> 正确的恢复路径，并在索引、H2D、replay、checkpoint 维护和并发排队之间选择真正
> 最快的路径？

原生 vLLM+LMCache 使用保守的共同边界，能够正确恢复并降低 TTFT，但会放弃共同
边界之后仍然存在的异构状态。简单追求更深 KV 命中也不够：它只减少部分 Attention
投影，并可能引入额外 lookup、传输和非融合算子；简单增加精确 state checkpoint
能够减少完整模型 replay，却需要支付 checkpoint 捕获、D2H、容量和并发争用成本。

因此论文主张应是 **cost-aware heterogeneous prefix recovery**，而不是“更长命中
必然更快”。

## 2. 两个集中挑战与对应技术

### 挑战一：异构命中不能直接转化为可执行、低延迟的恢复路径

Full-KV 可按较细 token page 命中，recurrent state 通常只在较稀的 checkpoint
存在。若强制共同边界，会浪费更深 KV；若仅把 KV 边界推进，完整模型仍需从较浅
state 处执行，而且细粒度传输和拆分投影可能抵消收益。

对应技术：

1. 独立索引 Full-KV 与 recurrent state，得到 `(L_KV, L_S)`，不因一类 miss
   抹掉另一类 hit；
2. 构造 aligned load、mixed load+replay、exact-state restore、recompute 等合法路径；
3. 使用 `lookup + H2D + reconstruct + forward` 的实测代价选择路径；
4. 保持“细粒度匹配、粗粒度传输”：token-page 级检索，连续页合并为少量大 DMA，
   同布局层共享 buffer pool、transfer descriptor 与批量提交。

### 挑战二：checkpoint 的未来收益与当前维护成本不在同一请求、同一资源队列

更密的 state checkpoint 可减少下一轮 replay，却会增加本轮捕获与 D2H、CPU
容量占用，并与其他请求争用 PCIe 和 GPU。单请求下有利的 all-load 策略，在高并发
下可能因 H2D 排队变差；GPU compute 饱和时选择又可能反转。

对应技术：

1. recovery-aware checkpoint admission：依据未来命中概率、预计节省时间和字节
   成本选择哪些 checkpoint 写入 CPU，而不是每个 KV page 都绑定一个 state；
2. 将自然 turn-boundary state 优先作为低成本 checkpoint，维护异步化并避免分段
   recurrent forward；
3. 同时建模 H2D 与 replay 队列，在 request/batch 级选择 load 或 recompute；
4. CPU/GPU cache 按“预计恢复时间收益/字节”进行 admission 与 eviction，并随并发
   和容量动态调整。

## 3. 需要区分的实验方案

| 方案 | Full-KV/state 边界 | 本质与用途 |
|---|---|---|
| Native vLLM+LMCache | 共同 528-token 边界 | 干净主基线，保守但路径成熟 |
| Aligned-SF | 共同边界 | 在同一原型代码路径上隔离实现开销 |
| Deep-KV | 更深 KV、较浅 state | 验证仅增加 KV 命中为何不一定降低 TTFT |
| Exact recovery-only | 更深 KV 与精确尾部 state，不维护下轮 checkpoint | 测量恢复收益上界 |
| Exact steady-state | 精确恢复并维护下一 checkpoint | 测量真实稳态净收益与维护税 |
| HyRex | 多候选路径、动态选择与容量管理 | 最终系统方案 |

Marconi、Sparse Prefix、CacheFlow/KVPR 可作为相关 cache/offload/scheduling
baseline，但论文必须明确哪些是原始 artifact，哪些是为 Hybrid recovery 加的保守
adapter，不能声称 adapter 完整复现原论文全部优化。

## 4. 已有 Motivation 证据

受控实验使用 Qwen3.5-9B、BF16、eager、单请求，每个正式请求后验证清空 GPU
prefix、配对内保留 CPU cache。共同 state 边界为 1056，额外 gap 为
`0/64/128/256/384/512`，下一轮追加 128 token，每点 5 次。

主要结果：

- Deep-KV 没有形成系统性 TTFT 收益：更深 Full-KV 只省 8 个 Full Attention
  层的部分 K/V projection，完整模型仍从较浅 state 边界 replay。
- gap=512 时，Exact steady-state 相对匹配的 Aligned-SF：mean 从 198.78 ms
  降至 166.34 ms（-16.3%），median 降低 33.22 ms。
- 同一点 Exact recovery-only 的 median 比 steady-state 再快 3.92 ms，显示维护
  下一 checkpoint 有可测成本；五样本 mean 维护税为 10.00 ms。
- 小 gap 上 Exact 往往更慢，证明存在 break-even point，正好支持动态路径选择，
  而不是固定使用最深命中。
- 五条路径与 Native 对比均为 0/30 首 token mismatch。

完整数据在 `results/controlled_gap_sweep_20261001_v2/RESULTS.md`，机制分析在
`results/HYREX_TTFT_REALIZATION_GAP.md`。

当前结果仍是 Motivation 证据，不是最终主表：容器 `/dev/shm` 只有 64 MiB，所有
方案一致使用 pickle IPC fallback。相同 fallback 保证机制对照仍有意义，但论文
数字必须在 `--shm-size=6g` 或更大配置下重跑。

## 5. 受控实验复现

准备 Qwen3.5-9B 权重和新 Python 环境后，从仓库根目录执行：

```bash
export HYREX_CONTROLLED_ROOT=/path/to/results/controlled_gap_sweep
export HYREX_AUDIT_RUNNER=$PWD/src/vllm-hyrex/benchmarks/motivation/audit_real_sharegpt_mp.py
export HYREX_PYTHON=/path/to/python
export HYREX_GPU=0
$HYREX_PYTHON results/run_controlled_gap_sweep_20261001.py
$HYREX_PYTHON results/analyze_controlled_gap_sweep_20261001.py
```

脚本固定执行：warmup、五方案、每请求 GPU prefix reset、CPU cache 保留、首 token
正确性检查，并记录 hit、forward token 与 TTFT。Native 使用其要求的 528 prefill
budget；Aligned/Deep/Exact 使用相同 2112 budget，避免把调度预算差异误认为恢复
机制收益。

## 6. 正式 HyRex 主实验

正式代码位于 `src/hyrex-vllm` 与 `src/hyrex-lmcache`，不是 Motivation 原型目录。
入口：

```bash
cd src/hyrex-vllm
python benchmarks/reproductions/run_hyrex_formal_campaign.py \
  --trace /path/to/trace.jsonl \
  --output-dir /path/to/output \
  --model-path /path/to/Qwen3.5-9B \
  --cuda-visible-devices 0 \
  --concurrencies 1,4,8,16,32 \
  --repetitions 3
```

正式矩阵需固定模型、请求到达、cache budget、输出长度和正确性 reference，交替运行
baseline/HyRex，报告 TTFT p50/p95/p99、SLO violation、吞吐、H2D/replay queue
time、传输字节、路径选择次数、admission/eviction 与 hit depth。

## 7. 接下来必须补的实验

按优先级执行：

1. **SHM 复测**：在共享内存不少于 6 GiB 的新容器复跑 gap sweep，确认 break-even
   与 checkpoint maintenance tax；这是投稿数据前置条件。
2. **真实多轮会话**：ShareGPT/WildChat 十轮与 16-session trace，逐请求 GPU reset、
   CPU cache 保留，报告每轮可复用 token、实际 replay token 和 TTFT。
3. **真实 agent trace**：BFCL multi-turn 与紧凑 SWE-bench session，覆盖追加、工具
   observation、回滚/分叉；分开报告追加式与非单调前缀。
4. **并发矩阵**：并发 1/4/8/16（资源允许再到 32），展示 H2D 与 compute queue
   如何改变最优恢复路径及 p99 TTFT。
5. **CPU cache 容量**：至少 1/2/4/8 GiB 或等比例 budget，对比统一 LRU、state-aware
   admission、HyRex value-per-byte eviction。
6. **关键消融**：独立索引；coalesced DMA；online cost model；batch-wide queue；
   checkpoint admission；异步 checkpoint spill；recovery-only 对 steady-state。
7. **正确性**：首 token/logprob、完整短输出、不同 hit-depth、跨 528 边界、eviction
   后恢复和并发 stream-order 测试。

主论文结果不能只展示 median。单请求 Motivation 可用 median 解释机制；主表必须以
p99 TTFT、SLO violation 和吞吐为中心，并同时给出路径选择与资源排队证据。

## 8. 论文叙事顺序

Introduction 建议按以下逻辑：Hybrid LLM 与多轮恢复的重要性；原生共同边界的保守
设计；“更深命中却不一定更快”的反直觉测量；成本来自异构计算覆盖、恢复路径实现
和跨请求 checkpoint/队列；由此提出 HyRex 的独立索引、合法路径构造、在线代价选择
和 recovery-aware cache management；最后用低并发、并发与有限 CPU budget 三类
实验闭环证明。

应避免写“原生系统不能复用”或“少算 token 必然等比例加速”。更准确的表述是：
原生 aligned recovery 有效但保守；naive deeper reuse 暴露了 realization gap；HyRex
的贡献是把潜在复用转换成可预测的端到端收益。
