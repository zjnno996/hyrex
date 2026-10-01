# HyRex：Hybrid Cache Recovery Scheduling 研究目标与实验计划

## 1. 一句话目标

面向高并发、多用户、多轮对话的 Hybrid LLM serving，构建 **HyRex**：在 CPU
offload 后，联合考虑 Full-Attention KV、recurrent state、cache readiness、H2D、
recomputation 和 queue contention 的恢复调度系统。论文主结论必须来自 HyRex 与
公开 baseline 在相同真实 vLLM 生命周期下的 **TTFT/TPOT/throughput** 对比。

当前主模型为 Qwen3.5-9B，BF16，单张 RTX 4090，
`gpu_memory_utilization=0.9`，LMCache local-CPU tier 为 24 GiB。

## 1.0 当前全局目标：统一外部 baseline 的端到端比较

当前唯一执行主线是：把每个公开 baseline 和 HyRex 放进**同一个**请求生命周期：

```text
single long-running vLLM service
→ multi-user turns arrive online and interleave randomly
→ normal prefill/decode produces and grows per-session Hybrid cache
→ GPU/CPU capacity pressure naturally causes promotion/offload/eviction
→ later turns naturally observe GPU hit / CPU hit / partial hit / cold miss
→ load/recompute/replay dispatch → completion fence → decode
```

主实验不采用“先把所有完整历史写入 CPU，再重启 worker 统一恢复”的两阶段构造。
也不得为某个方法额外制造公共 GPU prefix、local-prefix warmup、人工 residual 或指定
cache placement。所有方法在同一个持续运行的服务中消费相同 ShareGPT 多轮 trace：
同一 session 内保持 turn 顺序，不同 session 按固定随机种子交织到达。首轮、续轮、
cache admission、GPU→CPU offload、CPU eviction 和恢复都由正常 serving lifecycle
产生。GPU hit、CPU hit、partial hit 与 cold miss 是实验结果，不是预设条件。

固定主配置为：Qwen3.5-9B、BF16、单张 RTX 4090、
`gpu_memory_utilization=0.9`、LMCache local-CPU tier **24 GiB**、SLRU、允许 CPU
cache eviction；ShareGPT 多用户多轮 trace 的主压力集为 512 sessions、最多 3 turns，
并发为 `C1/C4/C8/C16`。到达过程至少包含固定种子的 uniform/Poisson 主配置，并用
bursty/Zipf 做压力敏感性。所有方法使用同一 tokenizer、session 内顺序、全局到达
序列、CPU capacity、eviction policy、输出长度和随机种子。

执行顺序固定为：

1. 定义并实现公共 cache-lifecycle/trace adapter，记录每个请求的 CPU admission、
   eviction、GPU prefix hit、CPU hit/miss、H2D/recompute/replay 和 completion。
2. **Marconi**：作为 Hybrid cache admission/eviction 对照；报告 cache 指标，且仅在
   接入真实 serving lifecycle 后报告 TTFT/TPOT。
3. **Tail-Replay**：原生 Hybrid 对照，在自然出现的 conversation resume 上执行
   Full-KV H2D 与论文规定的 recurrent-state replay；用于验证固定 replay 预算在不同
   cache 状态、上下文和并发下的局限。
4. **KVPR-H**：保留原 KVPR 同质 KV 逻辑，并明确标注 Hybrid state extension；接入
   同一 lifecycle 后再报告恢复延迟。
5. **HCache**：作为 Transformer activation-checkpoint restoration 的相关工作与
   实现参考保留；它不具备原生 Hybrid recurrent-state 恢复语义，因此不进入本文的
   Hybrid E2E 主表。
6. **CacheFlow-H**：保留 CacheFlow batch-aware dispatch，并为 Hybrid state
   readiness 增加适配；接入真实恢复 job、per-request observation 和 two-resource/3D
   dispatch，完成 fence 与输出一致性后进入端到端评测。
7. **HyRex**：在上述有效 baseline 上比较恢复合并、优先级、路径选择和队列调度。

硬性规则：分支隔离、惰性注册、相同资源配置；source/algorithm/runtime-level
reproduction 必须分别标注；unit test、CPU trace 或 planner-only 结果不能写成 E2E
性能结果。每一行 TTFT/TPOT 必须通过输出一致性和 cache-state gate；失败行仅保留为
diagnostic。GPU 不可用时只做 adapter/binding/trace 工作。

**不属于当前主线：** P1/P2/P3/P4/P5、TSM、Adaptive 与 EOSS 是已有的内部
load/replay 诊断原型，保留为历史材料，不能作为本阶段 baseline、主实验或论文结论，
除非后续以 HyRex 的一个明确组件重新定义并单独立项。

## 1.1 Baseline 复现目标与端到端准入条件

在 HyRex 的在线策略前，先用独立分支复现下列**不同问题层次**的公开
baseline。每个适配器只通过 `recovery_policy.py` 的惰性注册入口接入；不修改
其他 baseline 的逻辑，也不把近似实现标为原论文复现。

| 分支 | Baseline | 原问题 | 当前复现边界 | 进入 E2E 的条件 |
| --- | --- | --- | --- | --- |
| `hyrec/baseline-marconi` | Marconi | Hybrid cache admission/eviction | radix index、utility eviction、真实 trace replay | 同一 lifecycle 下 cache event 与 serving request 对齐 |
| `hyrec/baseline-tail-replay` | Tail-Replay | Full-KV load + fixed tail replay | 固定 replay budget planner；原生 Hybrid 语义 | 正常 H2D、Linear replay、输出与 quality/fence 核验 |
| `hyrec/baseline-kvpr-hybrid` | KVPR-H | KV partial-load + recompute | Hybrid-aware action planner；原 KVPR 与 extension 分开标注 | CPU hit 后的真实 load/recompute、输出与 fence 核验 |
| `hyrec/baseline-hcache` | HCache | activation checkpoint assisted restoration | checkpoint/storage/executor/vLLM binding | 不纳入 Hybrid E2E 比较；仅保留相关工作复现 |
| `hyrec/baseline-cacheflow` | CacheFlow | batch-aware 3D restoration | two-resource scheduler 与 dispatcher | 真实 observation/job、计划顺序、资源预算、输出与 fence 核验 |
| `hyrec/hyrex` | HyRex | Hybrid cache recovery scheduling | 本文方法 | 与所有有效 baseline 使用完全相同 lifecycle |

论文主表的核心比较只报告：**TTFT P50/P95/P99、TPOT P50/P95/P99、request/output-
token throughput**。H2D bytes/time、recompute/replay time、CPU hit/eviction、GPU
prefix hit 和资源使用量都只是解释指标：它们用于说明 TTFT/TPOT 为什么变化，不能
单独作为方法优劣结论。没有真实 E2E TTFT/TPOT 的 policy、trace 或 unit-test 行只能
标为 implementation/cache-management evidence，不能进入性能主表。最终 HyRex 只和
通过同一 correctness/cache-state gate 的 baseline 行比较；Marconi 是缓存管理对照，
Tail-Replay/KVPR-H/CacheFlow-H 是恢复/调度对照，不能混为同一种算法。HCache 不在主表中。

## 1.2 完成标准

- 五个方法都能消费同一份 24 GiB、允许 eviction 的 multi-turn trace，并记录统一
  request-level lifecycle 字段。
- 每个进入性能表的方法在 `C1/C4/C8/C16` 至少完成三次有效重复；每个 cell 都有
  TTFT/TPOT P50/P95/P99、throughput、输出一致性和 cache-state 记录。
- 任何 TTFT/TPOT 差异都同时附带 CPU/GPU cache hit、eviction、H2D bytes/time 与
  replay/recompute time；这些字段只用于归因，不替代核心延迟指标。
- CPU eviction 导致的 cold miss、GPU prefix miss、OOM/deferred 和 correctness failure
  单列报告，不并入恢复延迟均值。
- Marconi 若尚未完成真实 restore binding，只报告 cache 管理指标，不伪造 TTFT/TPOT。
- 所有外部 baseline 均达到 runtime-level 后，才运行并写入 HyRex 的统一比较结论。

## 1.3 当前执行计划与过程记录

### 固定执行阶段

1. **Online trace runner**：单个 vLLM 服务连续运行；回放真实 ShareGPT 多轮 session，
   支持固定种子的 uniform/Poisson/bursty 到达，并在请求级记录 cache lifecycle。
2. **统一观测**：记录 arrival、session/turn、prompt/output tokens、GPU/CPU/partial hit、
   admission/eviction、H2D、replay/recompute、queue/fence、TTFT、TPOT 和最终 action。
3. **Baseline E2E**：依次验证 All-Load、Marconi、Tail-Replay、KVPR-H、CacheFlow-H；
   HCache 只保留相关工作复现，不进入 Hybrid 主表。
4. **HyRex**：实现 request-level 路径选择、PCIe/GPU queue-aware cost、跨请求恢复合并、
   critical-path priority、admission/eviction 联动和 starvation protection。
5. **正式矩阵**：先用 32/64 sessions 做 correctness smoke，再运行 512 sessions、
   `C1/C4/C8/C16`、每 cell 三次，最后完成主表、归因、ablation 与 sensitivity。

### 过程记录

- **已完成**：明确论文主表只接受真实 vLLM lifecycle 的 TTFT/TPOT/throughput；
  planner、unit test 和微基准只作为实现或归因证据。
- **已完成**：Marconi/LMCache All-Load 已有真实 CPU hit 与 H2D 的初步 E2E 证据；
  旧结果仍需按新的 continuous-online runner 重跑，不能直接进入最终主表。
- **已完成**：Tail-Replay 已接入 Full-KV 恢复、严格 completion fence 和 recurrent-state
  replay 原型；partial-object IPC 的兼容模式已能完成 smoke，但同样需要在线 trace 重跑。
- **已否决**：人工 local GPU prefix、固定 residual、全量 seed-to-CPU 后重启恢复，以及
  P1/P2/P3/P4/P5 作为论文主实验。这些只保留为历史诊断。
- **2026-09-18 已完成**：在 `hyrec/hyrex` 实现单服务 continuous-online runner。
  runner 支持 saturated/uniform/Poisson/bursty 到达，同一 session 的后续 turn 严格等待
  前一 turn 完成，不同 session 可并发交织；输出请求级 arrival/session/turn、TTFT、TPOT
  与吞吐。旧的两阶段 runner 被显式标记为 `preconditioned`，不会混入在线主实验。
- **2026-09-18 已完成**：增加运行前资源门禁。它同时检查所选 GPU 的空闲显存和主机
  `MemAvailable`；主配置要求 `24 GiB LMCache + 20 GiB host reserve`。本次检查主机仅有
  约 36--37 GiB available，因此 24 GiB 主配置被正确拒绝，没有启动、没有 OOM。
- **2026-09-18 诊断运行**：在 GPU 空闲时用 8 GiB LMCache、C1、Poisson 到达启动
  Marconi smoke。LMCache 成功分配，Qwen3.5-9B 进入 safetensors 权重加载；运行在第
  0/4 shard 时随外层执行会话退出，未进入请求阶段、未生成 JSONL。该运行既不是算法
  失败也不是有效 E2E 数据，严禁写入论文表格。退出后 GPU 已恢复空闲，主机无遗留进程。
- **2026-09-18 已完成**：online runner 现在用确定性的 `X-Request-Id` 将请求 JSONL 与
  LMCache lookup 事件直接关联。每个请求记录 `gpu_hit_tokens`、`cpu_hit_tokens`、
  `h2d_tokens` 和 `cache_state={gpu,cpu,partial,cold}`；缺少事件时 cell 直接失败，禁止
  生成不完整结果。driver 与 join self-check、Python 编译和 diff check 均通过；实现提交为
  `hyrec/hyrex:2d8414434`。
- **2026-09-18 资源复查**：GPU 0 有 23.5 GiB 可用，但主机 `MemAvailable=37.0 GiB`，
  仍低于 24 GiB LMCache 主配置要求的 44.0 GiB；门禁再次正确拒绝启动，未产生 OOM。
- **2026-09-18 已完成**：worker connector 在真实 retrieve/store future 提交与完成点记录
  `retrieve_observed_ms`/`store_observed_ms`，并按同一 request ID 合并到请求 JSONL。该值包含
  LMCache 排队和传输，是恢复关键路径 observed latency，不标成纯 PCIe DMA 时间；实现与
  parser self-check 提交为 `hyrec/hyrex:2271f0114`。
- **2026-09-18 已完成**：online runner 启用 LMCache 原生 EventBus/Prometheus，并按 cell
  记录真实 `l1_write`（admission）与 `l1_evicted` chunk 增量；请求级记录实际
  `recovery_action`，所有 `h2d_tokens>0` 的结果必须观察到 retrieve completion fence，
  否则整个 cell 失败。16 MiB CPU-only LMCache smoke 已验证 `/metrics` 采集，未占用 GPU；
  实现提交为 `hyrec/hyrex:9cd0e7317`。
- **2026-09-18 架构审计**：统一 online runner 只存在于 `hyrec/hyrex` 评测分支，各
  baseline 分支保留隔离的算法源实现，通过注册表惰性加载到同一评测二进制。曾尝试向
  Tail-Replay 分支直接 cherry-pick runner，但其旧两阶段 API 与新 online API 冲突，已
  完整 abort，分支保持干净；不复制 runner 可避免不同 baseline 使用不同 serving 栈。
- **已确认缺口**：Marconi 与 Tail-Replay 已有 runtime policy；KVPR-H、CacheFlow-H 和
  HyRex 在统一注册表中仍为 `recovery_policy=None`。HyRex planner 虽存在，但在线
  LMCache lookup 尚未调用它，因此当前不能运行或宣称 HyRex E2E。
- **2026-09-18 baseline 源码审计**：KVPR-H 分支文档明确将 Hybrid split 标为
  analytical baseline；虽然原 KVPR activation callback 已有实现，但 Hybrid 版本仍缺
  layer-input checkpoint provenance、partial KV page ownership/writer 和对应 completion
  fence。CacheFlow-H 已有 native offload worker dispatcher/callback，但没有 LMCache
  lifecycle binding。二者都不能通过简单填充 `recovery_policy` 冒充 E2E baseline。
- **2026-09-18 已完成**：统一 runner 增加显式 cache-backend contract。Marconi 与
  Tail-Replay 使用 LMCacheMP；KVPR-H、CacheFlow-H、HyRex 使用 vLLM native
  `OffloadingConnector`，两种 backend 都由同一个 `--cpu-cache-gb`（主配置 24 GiB）
  和同一个 host-memory gate 约束。native 路径不再错误启动第二个 LMCache 进程；配置
  self-check 与 vLLM offloading config 13 项测试通过，提交为 `hyrec/hyrex:900943eb6`。
- **2026-09-18 已完成（algorithm-level）**：将公共 `recovery_policy.py` 惰性注册入口、
  KVPR-H policy、CacheFlow policy/executor 和 HyRex adapter 集成到统一评测分支。
  `kvpr_hybrid`、`cacheflow`、`hyrex` 均可按名称独立加载，联合 16 项测试通过。HyRex
  adapter 复用现有 planner 并传递 H2D/compute ready clocks，不复制算法。
- **2026-09-18 防误报 gate**：baseline 配置分离 `policy_plugin` 与 `runtime_bound`。
  三个算法当前均为 policy available、runtime unbound；online runner 会在启动任何
  CPU/GPU 进程前明确拒绝，而不是把 planner-only 结果写成 TTFT。相关提交为
  `hyrec/hyrex:e958efe5f`。
- **2026-09-18 已完成（native lifecycle）**：native `OffloadingConnector` 现按真实
  job/request 映射记录 lookup cache state、`gpu_hit_tokens/cpu_hit_tokens/h2d_tokens`、
  recovery action、transfer bytes、service/queue time、store admission 与 retrieve
  completion fence。online parser 同时消费 LMCache 与 native 结构化事件，并继续执行
  `h2d_tokens>0 => fence completed` gate。worker/metrics 与 schema 共 15 项测试通过；相关
  提交为 `hyrec/hyrex:021be424e`、`hyrec/hyrex:aaadde8c7`。
- **2026-09-18 工作树隔离**：`offloading/scheduler.py` 原有未提交 `all_replay` 两个 hunk
  未被本轮提交；仅逐 hunk 暂存本轮 native lifecycle 代码，用户改动仍原样保留。
- **当前下一项**：HyRex request-level binding 与 correctness gate 已完成；下一步按注册式
  接口完成 KVPR-H checkpoint/page writer 和 CacheFlow-H 的 pre-allocation chunk/page
  action binding，而不是仅重排已分配的 all-load job。然后做同配置 baseline/HyRex
  C1/C4/C8/C16 三次重复与 queue-aware 跨请求调度；每格启动前仍执行 44 GiB host 与单卡
  GPU 门禁，资源不足时只继续实现，不强行运行 24 GiB 主配置。
- **2026-09-19 已完成（HyRex request-level binding）**：在 native connector 完成真实
  CPU cache lookup、但尚未分配 GPU external blocks 的位置接入动态决策。决策使用实际
  external-hit token 数、每个 Full/Mamba group 的真实 `page_size_bytes`，以及显式提供的
  H2D、Full replay、recurrent replay 标定值，在 All-Load、Full-Load/Linear-Replay、
  Full-Replay/Linear-Load 三条当前可执行路径间选择；随后把 request-local policy 回写给
  core scheduler，并复用现有 suffix-only mixed replay 与 completion fence。缺少标定值时
  runner 会在启动前拒绝，避免默认零成本导致伪动态选择。定向测试 13 项、runner 两项
  self-check 与 diff check 均通过，实现提交为 `hyrec/hyrex:c72e36b00`。All-Replay 暂不
  进入候选集，因为当前 native store 会随 `loaded_group_indices=()` 跳过所有 group 写回，
  破坏后续多轮 cache 生命周期；必须先补独立的 store-group contract 才能安全开放。
- **2026-09-19 资源复查**：主机 `MemAvailable` 约 69 GiB，已高于 24 GiB CPU cache +
  20 GiB reserve；GPU 1/2 各约 23.5 GiB 空闲，GPU 0 正在满载。后续 smoke 只能选空闲卡，
  并仍需在启动瞬间重新执行门禁。
- **2026-09-19 HyRex native diagnostic smoke**：在 GPU 1、24 GiB native CPU tier、
  Qwen3.5-9B BF16、单服务 continuous 32-session/66-request trace 上完成 C1 smoke；运行前
  `MemAvailable=91.1 GiB`、GPU free=23.5 GiB，运行中未 OOM。有效结果为 cold=32、
  GPU-hit=26、CPU-hit=8，8/8 CPU H2D 均观察到 completion fence；TTFT P50/P95/P99=
  294.5/477.4/644.4 ms，TPOT P50/P95/P99=0.98/1.20/1.49 ms。该行只证明 lifecycle，
  未经输出 reference gate，不能进入论文主表。
- **2026-09-19 已修复（cost bytes）**：首次 smoke 暴露 scheduler 将 per-layer
  `page_size_bytes` 错当成 whole-group bytes。native worker 实际传输 Qwen3.5-9B 的
  Full `2,162,688 B × 8 layers` 与 recurrent `2,146,304 B × 24 layers`，一页约
  65.625 MiB。修复后 40-request diagnostic 中两个自然 CPU hit 均真实选择
  Full-Replay/Linear-Load，H2D 降为约 49.1 MiB，retrieve=2.84/3.02 ms，2/2 fence
  完成；实现提交为 `hyrec/hyrex:efae5ffda`，决策输入日志为 `33394c8bf`。这证明动态
  policy 已改变 native 执行路径，但样本太少且无 baseline/correctness 对照，不能作为
  性能结论。
- **2026-09-19 已完成（correctness gate plumbing）**：online trace runner 逐请求保存
  完整输出的 SHA-256，并支持 `--correctness-reference`；arrival index 缺失或任一 digest
  不一致会使 cell 失败。native engine 自动追加 request-id 后缀与 retrieve/store event
  单调 fence 合并也已修复，提交为 `54a515311`、`a61ef105a`、`49ad4e77e`。
- **2026-09-19 已完成（mixed recovery store contract）**：mixed recovery 只对选中的
  object group 做 H2D，但 replay 后把所有已完整物化的 group 写回 CPU tier，避免下一轮
  会话恢复读到不完整缓存。40-request lifecycle smoke 中 CPU request 33 实际 retrieve
  `51,511,296 B`、随后 store `68,812,800 B`；request 36 复用已有完整对象而无需重复写入。
  实现提交为 `hyrec/hyrex:12a44ec36`。
- **2026-09-19 已完成（独立输出一致性）**：用同一 32-session ShareGPT 多轮 trace、
  Qwen3.5-9B BF16、24 GiB CPU tier、C1、40 requests，先运行 Marconi/LMCache All-Load
  生成独立 reference，再运行 HyRex。HyRex 40/40 output SHA-256 与 reference 完全一致，
  `correctness_checked=true`；自然状态为 cold=26、GPU=12、CPU=2，两个 CPU request 均选择
  `replay_full_load_linear`，实际 H2D 从完整 `68,812,800 B` 降至 `51,511,296 B`（约
  25.1%），且 2/2 completion fence 成立。该格 TTFT P50/P95/P99=
  460.3/841.6/1356.6 ms，TPOT P50/P95/P99=1.00/1.13/1.17 ms；它证明 mixed path
  correctness 与 byte reduction，不构成性能优于 baseline 的结论，因为尚无重复且
  Marconi 首次运行在请求完成后的 attribution 汇总阶段失败。
- **2026-09-19 已修复（Marconi attribution）**：失败根因不是推理或 H2D，而是统一
  runner 强制等待 HyRex 专用结构化 marker。现已直接解析 LMCache 原生日志中的
  `lookup complete`、`prefix match`、`H2D retrieve`，在保存原 baseline 实现的同时生成
  统一 request-level cache state/action/fence；旧日志离线回放得到 cold=26、GPU=12、
  CPU=2，自检通过。实现提交为 `hyrec/hyrex:124447c08`，后续需用修复后的 runner
  正式重跑 Marconi cell 才能进入性能表。
- **2026-09-19 已完成（KVPR-H conservative runtime binding）**：在真实 native CPU
  lookup 后、external-token allocation 前，用实际 Full-KV bytes/token、recurrent
  checkpoint bytes、显式 H2D 带宽与 replay/token 代价选择 528-token 对齐切分。vLLM
  实际只恢复所选 Hybrid prefix，剩余 suffix 由普通 prefill recompute；request event 同时
  保存 available/load/replay tokens，zero-load 仍记为 CPU cache available，因而不能把
  all-load 或 cold miss 伪报成 KVPR-H split。runtime planner 会计入切分点 recurrent
  checkpoint H2D，14 项定向测试与 runner self-check 通过，实现提交为
  `hyrec/hyrex:d6c704187`。该 conservative binding 在 retrieve fence 后串行执行 suffix
  prefill；原 KVPR 的 layer-wise H2D/compute overlap 仍需要 layer-input provenance 和
  partial KV-page writer，因此不把 overlap 标为已复现。
- **2026-09-19 资源门禁**：本轮 host `MemAvailable≈82.3 GiB`，满足 24+20 GiB 门禁；但
  GPU 0/1 各被占约 21.5 GiB、GPU 2 被占约 18.2 GiB，GPU 3 虽有显存却持续存在
  16--35% 外部 SM 活动。为避免污染 TTFT/TPOT，未强跑修复后的 Marconi/KVPR-H 性能格，
  只完成无 GPU 的实现和验证。
- **2026-09-19 已完成（CacheFlow-H conservative runtime binding）**：真实 CPU lookup
  后按 recurrent checkpoint 粒度运行两指针 planner，并在 allocation 前裁剪实际 external
  tokens；同一 scheduler step 内的真实 H2D jobs 按剩余 replay work 跨请求排序。日志记录
  available/load/replay tokens，20 项定向测试、runner self-check 与 check-only 资源契约通过，
  实现提交为 `hyrec/hyrex:274e2fa9f`。当前 native vLLM 只能表达连续恢复前缀，因此把
  CacheFlow tail-load chunk 的数量映射为等量 prefix load，且 H2D fence 后才执行 suffix
  prefill；这是可运行的单 GPU CacheFlow-H adaptation，不是 arbitrary tail-page 与完整
  token/layer/GPU overlap executor。
- **2026-09-19 Marconi/HyRex 首个同机相邻 C1 pair**：GPU 2、40-request 同一 trace 与
  24 GiB CPU tier 下，Marconi cold/GPU/CPU=26/12/2，TTFT P50/P95/P99=
  186.6/371.5/940.1 ms，TPOT P50/P95/P99=1.08/2.36/9.48 ms，LMCache write=41、
  eviction=0；紧随其后的 HyRex 40/40 digest 一致、状态同为 26/12/2，TTFT=
  182.5/293.0/472.9 ms，TPOT=1.03/1.16/1.35 ms。整体 P50/P95/P99 分别约改善
  2.2%/21.1%/49.7%，但不能全部归因于动态恢复：仅有两个 CPU hit，request 33 改善约
  28.6%，request 36 反而慢约 24.0%，且 cold/GPU-hit 也存在 backend 差异。该 pair 是
  motivation signal，不是最终 superiority claim。
- **2026-09-19 第二重复状态**：HyRex r2 再次 40/40 correctness 通过，TTFT
  P50/P95/P99=267.4/405.8/554.7 ms；反向顺序的 Marconi r2 在初始门禁后被外部进程
  抢占 GPU 2，EngineCore 仅见 5.02 GiB free、低于 0.9 utilization 所需 21.17 GiB，安全
  退出且无结果行。该次只记为 resource-race failure，不与 HyRex r2 配对、不进入统计。
- **2026-09-19 已完成（scheduler-step queue-aware path selection）**：native scheduler
  在每个 step 内维护预测 H2D/compute ready clocks；每个真实 CPU hit 在 pre-allocation
  决策后按其 All-Load 或 mixed path 分别占用 I/O/compute timeline，后续请求使用累计 queue
  delay 重新选择路径，connector metadata 构建完成后重置 clocks。decision event 同时记录
  queue-before 与 ready-after，便于验证拥塞是否真正改变选择。单测证明同一 528-token
  Hybrid hit 在空闲时选择 All-Load，而已有 25 ms H2D queue 时切换为
  Full-Load/Linear-Replay，并更新 ready clocks 到 H2D=35 ms、compute=30 ms；17 项定向测试
  通过，实现提交为 `hyrec/hyrex:cf8fc54ce`。当前 clocks 是基于标定成本的 step-local
  prediction，尚需用 completed transfer/replay telemetry 做在线校正，并补跨 step/高并发
  验证。
- **2026-09-19 已完成（measured H2D feedback）**：worker metadata 现随 completed job
  回送真实 transfer bytes、service time 与 queue time；多 worker 聚合保留最慢 completion
  作为 critical-path observation。scheduler 仅对真实 load completion 更新持久 EWMA H2D
  bandwidth，后续 HyRex path decision 优先使用测量值，并在 decision event 标记
  `configured/measured` 来源。24 项 metadata/lifecycle/policy 测试通过，实现提交为
  `hyrec/hyrex:d673937ed`。
- **2026-09-19 已完成（measured replay feedback）**：GPU model runner 仅在整个
  scheduler step 都属于 mixed replay 时用 CUDA events 测量真实 model-forward 时间，
  避免把普通 prefill/decode 混入 replay 样本；step 时间按各请求 replay token 数分摊，
  经 `ModelRunnerOutput -> core scheduler -> native connector` 回传。HyRex scheduler 分别
  维护 Full 与 recurrent replay 的 per-token EWMA，后续 path decision 优先使用测量值并
  标记 `configured/measured` 来源。混合普通/replay batch 不产生有偏样本。25 项定向测试、
  `py_compile` 与全仓 `ModelRunnerOutput` positional-call 审计通过，实现提交为
  `hyrec/hyrex:dbdf64018`。代码闭环已完成；仍需 GPU lifecycle smoke 证明首个 replay
  completion 后的后续请求实际切换到 measured source。
- **2026-09-19 replay GPU smoke 进展**：首次 24 GiB CPU tier、9B、GPU 2 启动通过
  Host/GPU 门禁并完成模型与 native offloading 初始化，随后在首批请求暴露
  `sample_tokens()` 未携带局部 replay observation 的真实 lifecycle bug；runner 清理后
  GPU/Host 内存均恢复。修复将 observation 纳入 `ExecuteModelState`，并增加字段/构造顺序
  AST gate，提交为 `hyrec/hyrex:02dd36890`。同时 check-only 新增 vLLM executable gate，
  避免错误 Python 环境直到启动阶段才失败，提交为 `hyrec/hyrex:608455adf`。修复后的重跑
  被资源门禁安全拒绝：外部任务将 GPU 2 free memory 从 23.5 GiB 降至 5.8 GiB；GPU 3
  虽有 23.5 GiB free memory，但持续约 37% SM load，不作为性能卡。该次无结果行、不进入
  性能统计；待获得真正空闲卡后重跑相同 smoke。
- **2026-09-19 已修复（E2E dispatch activation）**：审计确认 native HyRex load-job
  priority scheduler 虽已有 runtime 实现和单测，但全局默认关闭，统一 online runner 此前
  没有显式启用。现在仅 `hyrex` baseline 设置 `VLLM_HYREX_SCHEDULE_LOADS=1`，其他 baseline
  保持关闭以维持隔离，提交为 `hyrec/hyrex:6374f20e8`。这使下一次 ShareGPT E2E 会真实
  执行同 step 跨请求 H2D 排序；它不等于 physical transfer merge。当前不同请求拥有不同
  GPU destination blocks，即使 CPU source 相同也仍需分别物化；真正减少重复 H2D 需要扩展
  transfer spec 为一次 H2D 加 GPU D2D fan-out，不能用 planner 的 coalesced 标记替代。
- **2026-09-19 measured-feedback GPU gate 已通过**：修复后在 GPU 2 运行相同 9B、
  24 GiB CPU tier、40-request C1 ShareGPT smoke，40/40 请求完成，cache state 为
  cold/GPU/CPU=`26/12/2`，退出后 GPU/Host 内存完整释放。第一个 CPU hit 使用 configured
  cost，选择 Full-Replay/Linear-Load，实际 retrieve `51,511,296 B / 3.083 ms`；该次真实
  model forward 形成 Full replay `0.121375 ms/token` 样本。第二个 CPU hit 同时使用 measured
  replay cost 与 measured H2D `16.708 GB/s`，决策切换为 Full-Load/Linear-Replay，实际
  retrieve `17,301,504 B / 0.972 ms`。因此“completion telemetry 能改变后续执行路径”的
  lifecycle 证据成立。该 smoke 未提供 correctness reference，且 TTFT P50/P95/P99=
  `307.0/3000.9/4878.9 ms` 明显受当次系统扰动，不进入性能主表，也不据此声称加速；正式
  paired run 必须带 reference 并重复。
- **2026-09-19 已完成（decision/dispatch evidence persistence）**：统一 runner 现在把
  `HYREX_NATIVE_DECISION` 合并到请求 JSONL，并在 cell 结果汇总 policy、H2D cost source
  与 replay cost source；真实 smoke log 回放得到两个决策、两种 mixed policy，以及
  configured/measured 各一次，提交为 `hyrec/hyrex:674d3ecc2`。native scheduler 还会为
  实际多-job step 输出逐请求 `dispatch_rank` 与 `dispatch_batch_size`，parser 同步持久化，
  两项 scheduler 测试与 runner self-check 通过，提交为 `hyrec/hyrex:a274692ee`。因此后续
  C4/C8/C16 能验证是否真的形成跨请求 recovery batch，而不只检查环境开关。
- **2026-09-19 已完成（cross-step contention state）**：原实现会在每次
  `build_connector_meta()` 后把预测 H2D/compute ready clocks 清零，只能感知同一 scheduler
  step 内的竞争。现在 queue debt 在 step 间保留，并从 metadata 提交（真实 job 可开始执行）
  的墙钟时刻按 elapsed time 衰减；每个新 step 只 advance 一次，多个 request 的决策仍共享
  同一累计队列。这样后续 step 能看到前序尚未消化的 H2D/replay contention。8 项 HyRex
  定向测试通过，实现提交为 `hyrec/hyrex:860ae52e8`；随后 native offloading scheduler 与
  HyRex policy 的联合完整回归 66/66 通过，覆盖 pending jobs、flush 与 completion 生命周期。
  GPU 运行验证仍需空闲单卡；本轮四卡
  均被同一外部任务占用约 12.6 GiB，门禁在模型启动前安全拒绝，无实验结果行。
- **2026-09-19 已完成（arrival-aware recovery dispatch）**：native batch planner 过去在
  policy/deadline 相同的情况下最终按 request ID 字典序排序，会让 `request-10` 错误先于
  更早到达的 `request-2`，并且直接使用 vLLM priority 会反转其“小值优先”语义。现在把
  request arrival time 传入 planner，并将原生 priority 正确映射；显式 `hyrex_priority`
  仍保留为高值优先覆盖项。9 项 HyRex 定向测试通过，实现提交为
  `hyrec/hyrex:875ac554e`。底层审计同时确认当前 transfer handler 对多个 destination 仍
  逐项执行 H2D；physical merge 需要 H2D leader + GPU D2D follower completion contract，
  现有 source-coalesced planner 标记不能作为“减少 PCIe bytes”的证据。
- **2026-09-19 已完成（TTFT-slack-aware dispatch）**：统一 runner 新增可选
  `--hyrex-ttft-slo-ms`（默认不设置，不改变公平 baseline）；native scheduler 根据
  `SLO - request 已等待时间` 计算当前剩余 slack，deadline 更紧的 recovery job 优先。
  dispatch event 同时持久化 estimated finish、remaining deadline 与 predicted SLO miss，
  从而可在 C4/C8/C16 直接审计尾延迟优先级是否执行。10 项 HyRex 定向测试与 runner
  self-check 通过，实现提交为 `hyrec/hyrex:fec59a619`。
- **2026-09-19 已完成（strict paired-result gate）**：新增统一 online pair analyzer，只有
  trace、请求数、并发、arrival 配置、生成长度、CPU/GPU cache 配置及 cache-state 分布均
  完全一致，candidate 明确执行 correctness reference，且每个 arrival index 的
  session/turn/prompt/output digest 全部匹配时才计算 TTFT/TPOT/throughput 改善。self-check
  通过；对已有 Marconi/HyRex C1 pair 回放验证 40/40 digest 与 `26/12/2` cache-state 一致，
  得到 TTFT P50/P95/P99 改善 `2.22%/21.12%/49.70%`，但旧 row 无新增 decision telemetry，
  因而不会被当作 dispatch 证据。实现提交为 `hyrec/hyrex:86d4e3f3b`。
- **2026-09-19 已完成（repetition/statistics gate）**：新增 paired-result 聚合器；默认要求
  至少 3 个互不重复的结果文件 pair，每个 pair 先经过上述严格 correctness/fairness gate，
  并再次检查所有重复的 workload/cache 配置一致，才报告 improvement mean、sample stddev
  与 Student-t 95% CI。单次结果默认被拒绝；只有显式 diagnostic `min_repetitions=1` 时输出，
  且 stddev/CI 为 null，不能进入主表。三重复 synthetic self-check、现有 C1 单次 diagnostic
  和默认拒绝 gate 均通过，实现提交为 `hyrec/hyrex:5ea0dbd77`。
- **2026-09-19 聚合 gate 加固**：pair gate 进一步要求 prompt field、execution mode 与
  workload 一致；重复聚合固定 reference/candidate baseline 名称，拒绝把不同方法混入同一
  CI。现有 C1 pair 与 synthetic 三重复复测通过，提交为 `hyrec/hyrex:dba03ee81`。
- **2026-09-19 已完成（resumable formal-matrix orchestrator）**：新增统一矩阵 runner，
  默认执行 `C1/C4/C8/C16 × 3 repetitions × {Marconi, HyRex}`；重复间采用
  `M→H, H→M, M→H` 交替顺序降低运行顺序偏差。每个 cell 继续调用原 runner 的 Host/GPU
  门禁；HyRex-first repetition 使用同并发上一重复的 Marconi digest reference，当前 pair
  完成后仍逐请求重新验证。中断恢复只复用请求数完整的 cell，每个 pair 立即经过 strict
  gate，每个并发达到全部重复后才生成 mean/stddev/95% CI。C1/C4 三重复 plan self-check
  生成并验证 12 个 cell，未启动 GPU，实现提交为 `hyrec/hyrex:98fa8b43f`。
- **2026-09-19 矩阵恢复 gate 加固**：已完成 cell 只有在 baseline、concurrency、trace、
  请求总数与请求文件行数全部匹配时才允许复用，避免旧目录中的异配置 row 被误当作当前
  repetition；plan self-check 复测通过，提交为 `hyrec/hyrex:589ded348`。
- **2026-09-19 已完成（source provenance gate）**：每个 online cell 现在持久化实际 Git
  HEAD、完整 dirty path 列表与 tracked binary diff SHA-256；strict pair、跨重复聚合与矩阵
  断点恢复都要求 fingerprint 一致。这样运行中修改代码或用旧目录续跑不会把不同实现混入
  同一主表。provenance smoke、旧 C1 diagnostic 兼容回放、pair/matrix self-check 均通过，
  实现提交为 `hyrec/hyrex:4ffc027c8`。
- **2026-09-19 已完成（formal profile + 512-session trace）**：矩阵 runner 现在显式区分
  smoke/formal；formal 默认要求至少 512 个独立 session、使用 trace 全部请求并生成 16 个
  output token，避免原 2-token smoke 只有一个 TPOT interval。已从本地 ShareGPT/Qwen
  tokenizer 生成 `/root/hyrex_sharegpt_multiturn_512_l528_seed0.jsonl`：512 sessions、1018
  events、每会话 1–3 turns、13 MiB，SHA-256=
  `13509fc4291b002d11292d7602890e7430664487a0a0487812bdab6760b03ffe`；arrival/turn/token
  bounds 全部通过。formal plan 生成 `4 concurrencies × 3 reps × 2 methods = 24` cells，
  每 cell 1018 requests、max_tokens=16；当前 32-session trace 会在 GPU 启动前被 formal gate
  拒绝。每个结果同时保存 trace SHA-256，pair/重复/断点恢复要求 hash 一致。实现提交为
  `hyrec/hyrex:d6157fadd`；本轮 GPU 仍被外部四卡任务占满，未启动正式 cell。
- **2026-09-19 已完成（matrix execution journal）**：正式矩阵现在用 fsync 的 append-only
  JSONL journal 记录每个 cell 的 start/reuse/completed/failed、完整命令、返回码、源码与
  trace fingerprint，以及 pair validation；失败仍立即抛出并停止，不会自动进入结果表。
  journal self-check 与 24-cell formal plan 复测通过，实现提交为
  `hyrec/hyrex:3ab952e74`。
- **2026-09-19 已完成（matrix/GPU mutual exclusion）**：非 plan 矩阵运行现在同时持有
  output-directory 与 numeric CUDA device 的 non-blocking `flock`；重复恢复同一目录或本项目
  另一矩阵抢同一卡会在启动 cell 前拒绝，进程退出由内核自动释放，不产生 stale lock。
  lock contention self-check 与 24-cell formal plan 复测通过，实现提交为
  `hyrec/hyrex:745675855`。外部不遵守该锁的作业仍由每 cell 显存/Host 门禁处理。
- **2026-09-19 已完成（launch-time resource recheck）**：cell runner 除初始 preflight 外，
  在 vLLM `Popen` 前再次检查 Host/GPU；结果同时保存 preflight 与 launch snapshot。该 gate
  缩小外部任务在 LMCache/config 准备期间抢卡导致模型 OOM 的窗口，失败仍由 finally 清理
  已启动的辅助进程。runner self-check 通过，实现提交为 `hyrec/hyrex:0100caec4`。
- **2026-09-19 已完成（bounded formal execution）**：online client 的 per-request timeout
  现可配置并写入结果；cell runner 新增总 execution timeout，timeout 在父 runner 内触发，
  因而仍会进入 finally 清理 vLLM/LMCache。formal/smoke 默认总时限分别为 7200/1800 秒，
  request 默认 600 秒；timeout 配置纳入 strict pair gate。client/cell/matrix self-check、
  24-cell formal timeout plan 与旧 C1 diagnostic 回放通过，实现提交为
  `hyrec/hyrex:b919d5675`。
- **2026-09-19 已完成（matrix failure attribution）**：matrix runner 捕获 cell runner 的
  短 stdout/stderr（vLLM 大日志仍独立落盘），失败 journal 保存各 4 KiB 尾部并分类为
  `resource_gate/timeout/correctness/runtime`，成功输出继续转发；因此资源竞争不会被统计成
  算法失败。四类 self-check 与 24-cell formal plan 通过，实现提交为
  `hyrec/hyrex:366836423`。
- **尚未完成**：CacheFlow-H 的 arbitrary-page/完整 3D overlap、KVPR-H 的 layer-wise
  overlap（若作为 exact baseline 要求）、至少两个额外完整 Marconi/HyRex paired
  repetitions、HyRex replay telemetry GPU gate 与跨请求恢复合并/优先级调度、
  修复后 Marconi 与其他有效 baseline 的同配置 C1/C4/C8/C16 三次
  重复、正式 512-session 矩阵、最终论文主表与结论。

## 历史内部诊断（非当前执行目标）

以下章节保存早期 P1/P2/P5/Adaptive/EOSS 的实现、诊断和原始实验审计，便于追溯；
它们不定义当前待办、不触发新实验，也不构成论文主结果。

## 2. 研究问题与核心 insight

一次 agent continuation 可抽象为：

```text
GPU 命中的共享 prefix + CPU 命中的后续 prefix/state + 新增 suffix
```

它需要回答的不是“请求要不要 load”，而是：

```text
state/segment × 实际缺失长度 × 当前 queue 压力 -> Load 或 Replay
```

Qwen3.5-9B 的 Hybrid cache 有两类语义不同的对象：

| State | 语义 | 恢复时真正需要什么 |
| --- | --- | --- |
| FullAttention KV | token-indexed 历史 | 缺失区间的全部 KV 页 |
| Mamba/GDN recurrent state | prefix 末端的 sufficient state | 能正确开始 suffix 的末端 state |

当前 align 模式把两类对象都按共同的 528-token page 存取。一个完整 Hybrid
page 的 GPU/CPU对象大小为约 65.625 MiB：Full KV 16.5 MiB，加上三个
Mamba/GDN group 共 49.125 MiB。

由此得到两个应分别验证的 insight：

1. **Load vs Replay 不是固定规则。** 短、未对齐或低压区间可能 Replay 更便宜；
   多页缺失或高计算代价区间通常更适合 Load。实际 `need_h2d_tokens`，而非名义
   suffix 长度，才是决策变量。
2. **recurrent state 不应被当作 sequence history 恢复。** 对多页 CPU-only
   区间，Full KV 仍需要每一页；Mamba/GDN 理论上只需最后一页的 endpoint state。
   这形成 P5 Terminal-State Materialization (TSM) 的可验证压缩机会。

注意：这些是系统恢复 insight，不等价于“FullAttention 总应 Load”或
“LinearAttention 总应 Replay”。也不意味着只恢复 KV/state 就可以跳过中间
decoder layer；跳层需要 hidden/residual activation checkpoint，属于独立扩展。

## 3. 已验证事实、诊断观察与待验证假设

### 已验证事实

- Qwen3.5-9B 当前布局为三个 `MambaSpec` group 加一个 `FullAttentionSpec`
  group；每个 align page 为 528 tokens。
- Native pinned H2D 与 LMCache local-CPU 都是实际 CPU -> GPU 路径；LMCache
  能记录 lookup、object-group H2D bytes 和严格等待时间。
- P1/P2 可以构成真实 All Load / All Replay 对照；P1 有 CPU hit、GPU miss 和
  H2D trace，P2 不应有 H2D。
- 当前单卡本地 pinned-memory H2D 很便宜。已有有效 C4/C8 长区间结果中，P1
  明显快于 P2；仅凭出现 H2D queue 不能推导 Replay 应胜出。
- 一页所有 state 的恢复约为 65.625 MiB；长 context × 并发会耗尽 GPU page
  pool，导致共享 GPU prefix 被驱逐。这是容量实验，不是有效的 partial-recovery
  对比。
- P5 原型已经实现为两次 LMCache retrieve：Full group 取全缺失区间，Mamba
  groups 只取最终 chunk。它目前只允许 Qwen 的 verified 528-token 对齐布局。

### 仅作诊断、不能写成论文结论的观察

- P3/P4 在部分运行中较慢，说明当前路径会叠加 H2D 和 Replay；这不能证明某类
  attention 天生不适合 Load。
- 部分 C8/C16 的跨服务 first-token signature 不一致，或 GPU prefix gate 失败；
  这些行不能参加 speedup 比较。
- 当前 Adaptive 的 queue-only 分支可工作，但在本地 H2D 上导致回退，不能宣称
  Adaptive 已优于 P1。

### 需要验证的假设

- 在同一硬件上，P1/P2 是否随 **实际缺失页数** 出现稳定 crossover。
- P5 是否在两页及以上缺失时保持输出正确，并按预期减少 H2D bytes 与 TTFT。
- 当真实的 `T_load > T_replay` 时，基于成本的 Adaptive 是否能优于静态策略。
- EOSS 是否能把 copy/compute barrier 变为正确的 execution-order overlap；它与
  TSM 是可组合但独立的机制。

## 4. 研究范围与策略定义

### 主线：KV/state materialization

| Policy | 含义 | 在论文中的位置 |
| --- | --- | --- |
| P1 | All Load：所有 CPU-resident state 都恢复 | 主基线 |
| P2 | All Replay：不 lookup/H2D，完整计算 | 主基线 |
| P3 | Full Load + Linear Replay | 静态类型策略诊断 |
| P4 | Linear Load + Full Replay | 静态类型策略诊断 |
| P5 / TSM | Full KV 全页 + Mamba 仅 terminal page | 主创新候选 |
| Adaptive | 用测量成本选择 P1/P2/P5 类动作 | 最终系统策略 |

P3/P4 必须在 suffix-only continuation 下验证输出正确，且明确记录实际 H2D group
和 replay 范围。它们不能被解释为“某种 attention 层被跳过”。

### 独立扩展：Activation-assisted recovery

若目标是跳过完整 decoder tile，必须额外恢复 layer-boundary hidden state 与
residual。该路径要单独比较 D2H store、CPU capacity、H2D restore、checkpoint
hit rate 和 skip 的计算收益；不得与纯 KV/state 主线混合成一个数字。

### 独立扩展：EOSS

EOSS 按 decoder execution tile 分块、在该 tile 执行前等待其 state，同时传输后续
tile。它验证的是 H2D 与 suffix compute overlap，不是 state 压缩。比较对象为
Barrier P1 与 EOSS window=1/2/4/8；P5/TSM 的字节缩减是另一条维度。

## 5. 每条结果必须满足的测量契约

### Cache-state precondition

每个记录到结果表的 Load cell 都必须有：

```text
local_gpu_tokens = shared prefix 长度
cpu_tokens >= local_gpu_tokens + 实际恢复区间
need_h2d_tokens > 0
CPU_to_GPU bytes > 0
```

P2 则必须有：

```text
no LMCache retrieve / no CPU_to_GPU H2D
```

CPU tier 不清空、不为了测试驱逐；GPU 侧若为了构造 CPU-only suffix 使用 filler
pressure，必须保留共享 prefix，并在日志中验证上述 precondition。若
`local_gpu_tokens=0`，该行归入 GPU-capacity 结果，不能与 partial recovery 比较。

### Correctness contract

- 同一输入、同一采样设置下，P1/P2/P5 必须具有相同 first-token signature；
  对长 output 再记录前若干 token 或 logits tolerance。
- 每个 policy 至少检查：请求数、CPU hit、GPU prefix hit、H2D group、H2D bytes、
  suffix-only replay marker（如策略包含 replay）。
- 一旦 signature 或 cache-state gate 失败，保留日志但标为 `diagnostic`，不写入
  主比较图表。

### 每请求/每组必须记录的字段

| 类别 | 字段 |
| --- | --- |
| 命中 | request id、local GPU hit tokens、CPU hit tokens、`need_h2d_tokens` |
| state | group/state type、Full bytes、Linear bytes、object group ids |
| 传输 | lookup time、H2D submit/queue/service/wait、H2D bytes、copy stream queue |
| 计算 | replay prefill time、scheduler queue、batch shape、prefill tokens |
| E2E | TTFT p50/p90/p99、TPOT、first-token signature、token/logit check |
| 资源 | GPU KV blocks、GPU prefix eviction、LMCache CPU used/capacity、OOM/deferred |

## 6. [历史诊断] P1/P2 完整矩阵

这是形成可靠 motivating result 的第一优先级。统一使用：Qwen3.5-9B BF16、单卡、
0.9 GPU utilization、LMCache local CPU 24 GiB、SLRU、输出 256 tokens、真实
ShareGPT prompt token ids。

长度用 **实际 `need_h2d_tokens`** 分桶，而不是只按输入参数标注：

| 名义 suffix | 实际 `need_h2d_tokens` | 实际恢复页数 | 用途 |
| ---: | ---: | ---: | --- |
| 784 | 528 | 1 | 最短真实 materialization |
| 1568 | 1056 | 2 | P1/P2 与 P5 的首个关键点 |
| 2112 | 1584 | 3 | 当前 corpus 可用的长缺失段 |
| 2640 | 2112 | 4 | 目标点；当前 corpus 不可用 |

并发矩阵：`C1, C4, C8, C10`。`C12, C16` 单独作为 GPU capacity sweep，除非能保证
所有请求的 shared GPU prefix 仍命中。

每个有效 cell 至少三次重复，采用相同 prompt 集、相同并发、相同 output length；
优先在同一健康环境下成对运行 P1/P2，减少独立服务的 JIT/启动方差。

预期输出：

1. `TTFT vs actual missing pages`：C1 下的 P1/P2 crossover 或其不存在。
2. `TTFT vs concurrency`：每个长度下 P1/P2 的排队、Replay 计算和容量边界。
3. `H2D breakdown`：lookup、queue、service、strict wait 与 replay prefill 的对比。
4. `capacity boundary`：context × concurrency 导致的 GPU prefix eviction，而非把它
   误报为 H2D/Recompute crossover。

运行模板：

```bash
python benchmarks/reproductions/run_native_independent_cells.py \
  --backend lmcache --policies P1,P2 \
  --suffix-only --retain-shared-prefix \
  --shared-prefix-tokens 528 --suffix-tokens <target> \
  --requests <C> --concurrencies <C> --output-tokens 256 \
  --gpu-memory-utilization 0.9 --lmcache-kv-gb 24 \
  --lmcache-chunk-size 528 --lmcache-eviction-policy SLRU
```

其中 `<target>` 只用于构造工作负载；最终分桶依据输出 JSONL 中的
`need_h2d_tokens`。

## 7. [历史诊断] P5 / Terminal-State Materialization 实验

P5 的假设只针对当前 recurrent Mamba/GDN layout：对于 `K` 个 CPU-only page，

| Policy | 理论 H2D volume |
| --- | --- |
| P1 | `65.625 MiB × K` |
| P5 | `16.5 MiB × K + 49.125 MiB` |

因此一页时 P5 与 P1 相同；两页时为 `82.125 MiB vs 131.25 MiB`；四页时为
`115.125 MiB vs 262.5 MiB`。P5 减少的是 **CPU -> GPU materialization**，不减少
当前 all-page CPU cache 的存储空间。

执行顺序：

1. C1，`need_h2d=528`：P1/P5/P2 输出一致；P5 不应声称速度收益。
2. C1，`need_h2d=1056` 与 `1584`：验证 P5 的 Full group 传 K 页、三个 Mamba
   group 各传 1 页，以及 signature 与 P1/P2 相同。
3. C4、C8：比较 P1/P5 的 TTFT、H2D queue/service 和 tail latency。
4. 只有 P5 correctness 和 bytes 都通过后，才加入成本型 Adaptive。

P5 的必须通过条件：

```text
Full group H2D objects = K
each Mamba group H2D objects = 1
P5 first-token / token check == P1 == P2
CPU hit + GPU prefix hit + nonzero H2D all observed
```

运行模板：

```bash
python benchmarks/reproductions/run_native_independent_cells.py \
  --backend lmcache --policies P1,P2,P5 \
  --suffix-only --retain-shared-prefix \
  --shared-prefix-tokens 528 --suffix-tokens <target> \
  --requests <C> --concurrencies <C> --output-tokens 256 \
  --gpu-memory-utilization 0.9 --lmcache-kv-gb 24 \
  --lmcache-chunk-size 528 --lmcache-eviction-policy SLRU
```

## 8. [历史诊断] Adaptive：先成本模型，后策略收益

Adaptive 不能使用“看到 H2D queue 就 Replay”的单阈值规则。已有实验已经说明，在
本地 pinned CPU 路径中，即使 C8 有排队，Replay 长区间仍远慢于 Load。

最小成本模型：

```text
T_load(state/segment) = lookup + H2D_queue + bytes / observed_bandwidth + materialization
T_replay(segment)     = observed_prefill(segment, missing_tokens, batch_shape)
decision              = argmin(T_load, T_replay)
```

实施步骤：

1. 用 P1/P2/P5 主矩阵收集实际计时；不先训练复杂 ML。
2. 对 Full history 与 Linear endpoint 分别拟合简单分段/阈值模型。
3. 在相同 trace 上构造 offline oracle，比较静态策略的 regret。
4. 再实现在线 threshold policy，并验证每个请求决策与实际成本一致。
5. 若本地 H2D 没有 crossover，不伪造结论；改用受控带宽限制、真实高并发或远端
   tier 来制造可测的传输压力，并将该环境清楚报告。

Adaptive 的最终比较对象为 P1、P2、P5 与 offline oracle；指标为 TTFT、p99、H2D
bytes、Replay FLOPs 和 correctness。

## 9. [历史诊断] GPU/CPU 容量实验（独立报告）

短 context 不会更容易占满内存；占用近似随：

```text
active requests × cacheable context pages × 65.625 MiB/page
```

例如三页 context：C8 约 1.54 GiB，C16 约 3.08 GiB 的 Hybrid state，尚未包含模型
BF16 weights、workspace、activation 和 scheduler buffers。9B BF16 模型在 24 GiB
GPU、0.9 utilization 下留给 KV/page pool 的空间有限，因此 C12/C16 出现
`local_gpu_tokens=0` 是可预期容量现象。

容量 sweep 应单列：固定 context（2/3/5 pages），提高并发直到发生 GPU prefix
eviction，记录最大 valid concurrency、KV block usage、deferred/oom。它的结论是
“partial GPU+CPU recovery 的可维持工作集”，不是 P1/P2 性能优劣。

LMCache 24 GiB CPU tier 与 GPU KV pool 独立。CPU tier 是否有足够内容由 CPU hit
和 eviction trace 判断；GPU shared prefix 是否还在由 `local_gpu_tokens` 判断，二者
不能互相替代。

## 10. [历史诊断] 完成标准与论文结论边界

### 主线完成标准

- P1/P2 在至少 3 个非零实际缺失长度、C1/C4/C8 的有效配对矩阵。
- 每个主表行都通过 cache-state gate 与输出一致性检查。
- P5 至少在 2-page、4-page C1 与一个并发点完成 correctness + H2D-byte 验证。
- Adaptive 与静态策略、offline oracle 的对比包含完整成本分解。
- capacity failure、异步 correctness failure、启动/JIT 异常均单独保留，不混入均值。

### 可支持的结论（满足上述标准后）

> Hybrid LLM 的恢复决策应以 state/segment 为单位，并同时考虑 state 表示、实际
> 缺失页数和实时传输成本；统一 all-load 或 all-replay 会在不同工作点留下空间。

若 P5 通过：

> 对具有 endpoint sufficient recurrent state 的 Hybrid 模型，Linear state 的恢复
> 不应按所有缺失页 materialize；Full KV history 与 terminal recurrent state 应使用
> 不同的数据单位和成本模型。

### 当前不能声称的结论

- FullAttention 永远 Load、LinearAttention 永远 Replay。
- H2D queue 单独足以使 Replay 更优。
- P3/P4 已经代表“只计算一种 attention”。
- KV restore 可以跳过完整 decoder layer。
- Adaptive 已经在当前本地 H2D 硬件上稳定优于 P1。

## 11. [历史诊断] 结果与实现入口

- 实验记录：`benchmarks/reproductions/QWEN35_HYBRID_RECOVERY_RESULTS.md`
- P1--P5 runner：`benchmarks/reproductions/run_native_independent_cells.py`
- Offline gate/oracle：`benchmarks/reproductions/analyze_hybrid_recovery.py`
- EOSS 设计与状态：`benchmarks/reproductions/HYBRID_STATE_STREAMING_IDEA.md`
- P5 LMCache connector：当前 uv 环境的
  `/root/.cache/uv/archive-v0/ZtnCK5kBImFAagQN/lmcache/integration/vllm/lmcache_mp_connector.py`

每次新增实验后，先更新“有效/diagnostic/容量失败”的状态，再根据有效结果更新本文件
的主矩阵，不以单次 TTFT 或未验证 cache hit 形成结论。

离线审计必须用与实验同一协议的 JSONL 行运行，例如：

```bash
.venv/bin/python benchmarks/reproductions/analyze_hybrid_recovery.py \
  --self-check /root/qwen35_goal_*.jsonl
```

该脚本只在 P1/P2 都通过 cache-state 和 first-token gate 时生成 baseline
oracle；P5、P3/P4 与 Adaptive 各自单独校验签名，失败者保留为 diagnostic，不能
使有效 P1/P2 对照被丢弃或被纳入速度结论。

## 12. [历史诊断] Latest execution audit

A fresh two-page K=2/C1 P1/P2/P5 attempt on GPU3 was started with the same
suffix-only protocol and `start_index=8`. GPU3 had sufficient free memory, but
vLLM remained in weight-file I/O for the 300-second startup deadline and the
runner exited without writing a JSONL row. This is recorded as a startup/I/O
diagnostic and is excluded from all performance and correctness comparisons.

The first architecture-level value check is now reproducible through the
offline analyzer's `--architecture` mode. Across valid rows, transferred
bytes per request are:

```text
P1: K=1/2/3 -> Mamba 49.5/99.0/148.5 MiB, Full 16.5/33.0/49.5 MiB
P5: K=1/2/3 -> Mamba 49.5/49.5/49.5 MiB, Full 16.5/33.0/49.5 MiB
```

This validates the architectural motivation for terminal-state materialization:
the three recurrent groups transfer one endpoint page under P5, while the
FullAttention group remains proportional to the missing history. It is a
state-size/bandwidth result, not yet a TTFT result. The next execution gate is
therefore to use the same valid cells to measure whether this byte reduction
survives concurrency and whether the P1/P2 oracle headroom is large enough to
justify online Adaptive.

The current execution has three correctness-passing repetitions for the main
P1/P2 K=2 matrix at C1/C4/C8, and three valid P1/P2 repetitions for K=3 at
C1/C4. A separate deterministic K=1 prompt set (`start_index=8`) now passes
P1/P2 at C4/C8, while the original start=0 K=1 rows remain diagnostic because
of cross-policy first-token mismatches. A real
ShareGPT suffix long enough to produce K=4 is unavailable in the current
corpus: suffix=2640 has zero eligible prompts. The suffix=2112 run is valid
K=3, not K=4, because the retained 528-token shared prefix is not part of the
missing interval.

P5 has passed output and object-byte checks at K=2/C1,C4,C8 and K=3/C1. At
K=3, P1 transfers 792 MiB for four requests while P5 transfers 396 MiB; this
is a 50% materialization reduction. The normal K=3/C4 and C8 P5 paths have
cross-policy correctness mismatches, while the serialized-terminal fallback
passes those P2/P5 checks. The missing K=4 P5 point remains an explicit
workload-availability gap. The raw rows and classifications are maintained in
`benchmarks/reproductions/QWEN35_HYBRID_RECOVERY_RESULTS.md`; these results do
not yet justify marking the overall goal complete until repeated cells,
additional page lengths, and the remaining required Adaptive/oracle work are
performed.

The recent C8 trace clarified the transfer metric boundary. LMCache reported
retrieve-future waits of 84--167 ms for 132 MiB/request, but the final CUDA
synchronization was only 0.07--0.28 ms. A direct pinned BF16 CPU-to-GPU copy
microbenchmark on the same GPU measured about 20.0 GB/s. Thus the large value
is not a slow bare H2D link: it includes LMCache worker/prefetch scheduling,
many object-group transfers, and the strict correctness barrier before model
forward. TTFT comparisons should retain this end-to-end recovery wait, while
raw copy time must be instrumented and reported separately.

The length-aware Adaptive branch has now been exercised validly at K=2/C1,
C4,C8: all eight requests in each cell selected Replay at the 1056-token
threshold, had zero H2D, and produced the expected deterministic signature.
This is routing and replay-tracing evidence, not an Adaptive speedup claim;
the exact same protocol still needs paired P1/P2 references and an offline
oracle comparison. The existing correctness-safe P1/P2 oracle shows that
Replay is the winner only at K=3/C4; it selects Load at K=2/C1,C4,C8 and
K=3/C1. Therefore the current fixed K=2 threshold branch is explicitly not
an optimization result. The K=3 Adaptive attempt was excluded because its
CPU suffix precondition was not met.

A strict K=1/C1 serial-All-Load probe failed the output signature gate (P1
`a0d2400bba2b816d` vs P2 `7968cf1b1c1b72b8`) despite uniform one-page CPU
recovery and complete four-group H2D traces. A later explicit suffix-only
rerun showed the same issue (`9038d882fa0d5bb2` vs `7968cf1b1c1b72b8`) for
start=0, while the independent start=8 C4/C8 pair passed. The discrepancy is
therefore workload/cache-state sensitive, and the start=0 rows remain
diagnostic. The analyzer gate was corrected: `suffix_only` is required for
P3/P4 mixed replay, while P1/P2/P5 are judged by the actual shared-GPU-prefix
plus CPU-suffix cache-state gate.

The earlier K=3/C4 P1/P2 mismatch was not reproduced after the settle=3
protocol and repeated runs. The three valid C4 pairs have P1/P2 p50 medians
of 17,411.515/16,608.363 ms; P2 is about 4.8% lower in this high-queue
condition, so this cell does not support a Load speedup claim but is valid
evidence that the winner can depend on queue pressure. The normal concurrent
K=3/C4 P5 path still has a one-request signature mismatch, while the
serialized-terminal fallback passes P2/P5 correctness. K=3/C8 All Load still
fails the cross-policy signature check even with per-request All Load
serialization, so it remains diagnostic and needs a correctness fix or a
bounded-safe execution mode before entering the main P1 table.

The third K=2 repetition completed valid P1/P2 pairs at C1/C4/C8. P1 p50 was
169.864/710.982/823.014 ms and P2 was 265.888/17,404.484/37,279.698 ms;
all pair signatures matched and P1 had 528/528/1,056 MiB H2D respectively.
Together with the two earlier valid K=2 rows, this gives three repetitions per
concurrency. The endpoint-backed vLLM timing fields were also validated:
P2's measured prefill is a replay-compute proxy, while its queue time is
reported separately from prefill and E2E TTFT.

Across the three valid K=2 repetitions, the paired TTFT p50 medians are
175.223/265.888 ms (P1/P2) at C1, 694.133/16,794.896 ms at C4, and
823.014/34,064.130 ms at C8. This corresponds to P1 reductions of 34.1%,
95.9%, and 97.6%, respectively. These are motivating results for this local
CPU/H2D configuration, not a universal rule: K=1 remains diagnostic because
of cross-policy signature mismatches; K=3/C4 is valid for P1/P2 but not for
normal concurrent P5; and K=3/C8 remains diagnostic for P1/P2. A separate
filler=16 attempt was correctly rejected when the
shared GPU prefix was evicted, and remains a capacity/control failure.

A second K=3/C1 P1/P2/P5 repetition also passed all gates. P1/P2/P5 p50 was
213.284/301.854/182.881 ms, with common signature
`fd9af8c6805d5e37`. P1 transferred 792 MiB (all four groups over three
missing pages), while P5 transferred 396 MiB: three Full pages plus one
terminal page per Mamba group, a 50% reduction. This confirms the TSM object
selection and byte reduction at output length 256 for a second K=3/C1 run;
The subsequent settle=3 P1/P2 probe also passed with p50 206.429/318.893 ms,
uniform actual K=3 and common signature `fd9af8c6805d5e37`; it is the third
valid K=3/C1 P1/P2 repetition. The intervening one-request mismatch
(`ea62b817fb56a82a` vs `fd9af8c6805d5e37`) remains diagnostic and is retained
separately. A settle=3 P5 probe then passed at 191.963 ms with the same
signature and 396 MiB H2D, completing three K=3/C1 P5/P2 correctness points.

The settle=3 K=3/C4 run had uniform K=3 and P1/P2 signatures both
`fd9af8c6805d5e37` (P1 17,411.515 ms; P2 16,375.518 ms), but P5 produced
`d3cffb256dd38b0d` for one request despite correct K=3 terminal object
selection and 396 MiB H2D. Therefore K=3/C4 is currently valid for the P1/P2
pair but diagnostic for P5; the discrepancy is a concurrency-sensitive
terminal-state correctness issue, not a speedup result.

At K=3/C8 with the same settle and serialized P5 fallback, P1 remained a
correctness diagnostic (`0cd0fb7df15756c3`) while P2 and P5 matched
(`d0dcd3115c7de5dc`) across all eight uniform-K=3 requests. P5's p50 was
37,206.112 ms and its H2D volume was 792 MiB (Full 396 + terminal Mamba
groups 132 each); this is a correctness-safe P5/P2 point, but not a valid P1
comparison. The result reinforces that All Load is not currently safe at
high-concurrency multi-page Hybrid recovery, whereas serialized terminal
materialization is.

An opt-in `--p5-serialize-terminal` fallback was then tested at K=3/C4. It
serialized each request's Full and terminal H2D copies and passed P2/P5
correctness (`fd9af8c6805d5e37`) for all four requests, with the same 396 MiB
P5 volume and TTFT p50 16,754.925 ms. This is a correctness-safe fallback,
not yet a proof that the normal concurrent P5 path is safe; its extra
serialization cost must be included in any final P5 latency comparison.

The output=256 P5 concurrency sweep at K=2 now has valid P2/P5 pairs at C1,
C4, and C8. P5 p50 was 159.184/461.480/680.482 ms, versus P2
246.842/18,378.942/41,698.756 ms; all signatures matched. P5 transferred
330/330/660 MiB for 4/4/8 requests, versus the corresponding P1 volume
528/528/1,056 MiB, a stable 37.5% reduction. The P1 row in the C4 run was
mixed (one request had only one missing page), so that row is excluded from
the P1 comparison; the P5/P2 rows had uniform actual K=2 and remain valid.

The runner now writes `lmcache_need_h2d_tokens`, `lmcache_missing_pages`, and
`lmcache_recovery_length_uniform` for every LMCache row, making actual-page
bucket validation explicit rather than inferred from the nominal suffix. It
also exposes `--warmup-settle-seconds` (default 3 s) so asynchronous CPU-store
completion is part of the recorded protocol.

A suffix-only K=2/C10 P1/P2 pair then passed the same gates with common
signature `247a563bac96db57`: P1 p50 876.727 ms versus P2 p50 43,070.703 ms.
P1 transferred 1.289 GiB for ten requests, while P2 had 438.255 s aggregate
scheduler queue time. This adds a valid high-contention point; it is still
reported as a pressure-aware observation rather than a universal Load rule.
A final execution update: the explicit suffix-only K=1/C1 rerun passed its
cache-state and one-page gates but retained the P1/P2 signature mismatch
(`9038d882fa0d5bb2` versus `7968cf1b1c1b72b8`), so it remains diagnostic. This
confirms that the analyzer metadata correction does not hide the existing All
Load correctness issue.

The suffix-only K=2/C10 P1/P2 pair passed all gates with common signature
`247a563bac96db57`: P1 p50 876.727 ms versus P2 p50 43,070.703 ms. P1
transferred 1.289 GiB for ten requests, while P2 had 438.255 s aggregate
scheduler queue time. The matching K=2/C10 P5 run passed as well: P5 p50
885.384 ms and 0.805 GiB H2D, a 37.5% reduction versus P1, but no material
TTFT improvement. These are pressure-aware and materialization-byte results,
not universal latency rules.
A different deterministic ShareGPT suffix set (`start_index=8`) resolved the
K=1 high-concurrency coverage gap: P1/P2 passed correctness at C4/C8 with
common signature `3163847c9065406e`, P1 p50 392.373/581.270 ms and P2 p50
30,213.724/38,048.725 ms. These rows remain separate from the start=0 K=1
diagnostic rows, since prompt/cache state is part of the reproducibility
protocol.

The suffix-only K=2/C10 pair passed with common signature
`247a563bac96db57`: P1 p50 876.727 ms versus P2 43,070.703 ms, and P1
transferred 1.289 GiB for ten requests. The matching P5 C10 run also passed:
P5 p50 885.384 ms and 0.805 GiB H2D, a 37.5% materialization-byte reduction
versus P1 without a material TTFT improvement at this point.
The K=2/C1 P3/P4 probe verified state-type H2D routing but both mixed policies
failed the output signature gate: P1/P2 were `80f1de9cc34d7d21`, while P3/P4
were `67f4fc18605c4e61`. They remain routing diagnostics, not speed results.

A second prompt set at K=3/C8 with feasible filler=16 reproduced the
multi-page high-concurrency All Load correctness mismatch: P1
`9f8fb4d476a9e45d` versus P2 `a45ee06d699a318d`, despite uniform K=3 and
complete H2D traces. The 64-filler attempt was rejected before measurement
because the corpus has only 237 qualifying candidates for the required 512.

The first same-command K=2/C10 P1/P2/Adaptive attempt omitted GPU eviction;
P1 therefore had signature `873865596b548b4b` while P2 and Adaptive had
`247a563bac96db57`, so it was correctly classified as diagnostic. A corrected
Adaptive-only K=2/C10 run with GPU eviction passed the cache-state and output
gates, selected Replay for all ten requests at the 1056-token threshold, and
measured TTFT p50 42,463.860 ms with zero H2D. Its warmup count was 72 while
the established P1/P2 reference uses 10, so it is not yet merged into a
formal analyzer cell; a same-warmup Adaptive rerun remains required before
claiming regret or speedup.

The runner was hardened after an interrupted Adaptive C8 trial: the E2E client
now accepts `--request-timeout` (default 180 s), and every warmup, eviction,
retain, and measured-load request uses it. A stalled streaming request now
fails the cell and triggers the existing vLLM/LMCache cleanup instead of
waiting indefinitely. The interrupted breakdown trial produced no JSONL row
and is excluded from all conclusions.

For the formal multi-user ShareGPT matrix, cache pressure is now an explicit
validity condition rather than an assumed property of a small CPU cache. The
native offloading manager emits per-request admission, eviction, and rejection
events; the HyRex cell runner aggregates them, while the Marconi path uses its
LMCache write/eviction counters. Formal cells require nonzero admission and
nonzero eviction for each method before a TTFT/TPOT pair may be reported. The
two counters retain their own units (native blocks vs. LMCache chunks), so they
establish pressure within each implementation and are not compared as byte-for-
byte cache-size measurements. The runner and matrix self-checks passed, as did
the relevant native-offloading regression suite (67 passed). No new GPU result
was recorded because all four suitable GPUs remained occupied by external jobs;
the first formal C1 pair will run only after the existing 9B resource gate
confirms at least 22.59 GiB free GPU memory and 44 GiB free host memory.

The online runner now also supports an explicit `--auto-cpu-cache-gb` mode for
resource-constrained diagnostic runs. At matrix start it computes one effective
CPU-cache capacity from `MemAvailable - host_reserve`, enforces a configurable
minimum (4 GiB by default), and passes that same fixed capacity to both
Marconi and HyRex. The selected budget is journaled and embedded in every cell
result; resume logic treats a different cache capacity as a different cell, so
results cannot accidentally mix cache configurations. Formal reporting remains
on the fixed 24 GiB setting unless an experiment is deliberately labelled as a
different cache-capacity condition. The runner self-checks, matrix self-check,
and formal C1 plan construction passed after this change.

The resumable online matrix is no longer hard-coded to Marconi. Its
`--reference-baseline` selector now produces independently named, paired
baseline-versus-HyRex cells for Marconi, Tail-Replay, KVPR-H, or CacheFlow-H,
while preserving the old Marconi filenames for existing diagnostics. It passes
the required calibrated policy parameters to KVPR-H and CacheFlow-H, validates
cache pressure according to each backend, and keeps each baseline's three
repetitions isolated for aggregation. Matrix self-check plus formal C1
plan-only checks passed for Tail-Replay, KVPR-H, and CacheFlow-H. These are
execution-plan and runtime-binding checks; their GPU E2E measurements remain
pending the same host-memory gate.

Formal execution now rejects untraceable recovery-cost constants. A schema-v1
calibration record must contain provenance plus H2D bandwidth, Full replay, and
recurrent replay rates; its SHA-256 is written into every cell and is part of
both pair validation and resumable-cell identity. The matrix maps this one
measured record consistently to HyRex and to the comparable KVPR-H/CacheFlow-H
Full-replay estimates. Plan-only remains available without an artifact, so
resource-free adapter validation is not confused with a formal run. Cell,
matrix, pair, and aggregate self-checks passed; an attempted formal run without
the record was correctly rejected before any GPU process was started. A real
calibration artifact is still required before formal GPU measurements can be
claimed.

An artifact generator now derives the formal calibration record only from a
completed native server log's `HYREX_NATIVE_DECISION` events that explicitly
carry measured H2D and measured Full/recurrent replay feedback. It uses robust
medians, embeds SHA-256 provenance for the server log (and optional diagnostic
result), and rejects configured-only cold-start decisions. This diagnostic
calibration pass is deliberately separate from the formal matrix, so it does
not contribute TTFT/TPOT rows. Its own self-check and the matrix validation
self-check passed. When host memory returns, the sequence is: run a bounded
native feedback pass, derive its artifact, then start the paired formal matrix.

The bounded feedback pass is now executable as a separate runner. Its default
uses the first 512 natural events from the formal trace, which contains 113
distinct sessions with an actual later turn (rather than fabricated CPU hits),
requires at least 100 such resumed sessions, and writes a diagnostic result
plus the server-log path before deriving `recovery_calibration.json`. Bootstrap
rates are only used to make the first dynamic decisions possible; the artifact
generator still requires later measured scheduler feedback. The pass defaults
to C4 and two output tokens and is explicitly excluded from the formal
TTFT/TPOT matrix. Its runner self-check and the underlying cell self-check
passed.

The online workload now includes a deterministic bounded Zipf inter-arrival
sensitivity model in addition to uniform, Poisson, and bursty traffic. It
samples Zipf ranks for gaps but normalizes by their expected rank, preserving
the requested mean arrival rate while changing only the tail/burst structure.
The exponent and maximum rank are recorded request-side, propagated through
cell/matrix/calibration commands, and included in strict pair identity, so a
Zipf result can never be paired with a different offered-load distribution.
Trace, cell, matrix, pair, calibration self-checks and a formal Zipf plan-only
construction passed. Zipf is a sensitivity condition, not a replacement for
the fixed Poisson/uniform main configuration.

The final paper-table builder is now also gated rather than manually assembled.
For every selected reference baseline, C1/C4/C8/C16, and each of three repeats,
it requires the exact paired result files, re-runs request-level output/config
validation, and rejects a row without nonzero cache admission and eviction for
both methods. Only then does it emit Markdown plus machine-readable evidence
containing median baseline/HyRex TTFT P50/P99 and TPOT P50 values and deltas.
It supports the preserved Marconi naming and the isolated Tail-Replay, KVPR-H,
and CacheFlow-H matrix directories. The builder self-check passed. Until these
real E2E cells exist, it cannot and does not create a paper-performance table.

Each online cell now persists a recovery-attribution summary alongside TTFT and
TPOT: actual native retrieve bytes, retrieve service/queue time, LMCache H2D
token coverage, and the separately labelled modeled load bytes/replay time
selected by the Hybrid policy. The main-table builder requires this telemetry
from every repeat and writes median baseline/HyRex values into the evidence
artifact; it deliberately does not treat modeled bytes or replay time as a
latency result. Runner and table-builder self-checks passed. The native
scheduler audit also confirms that its present cross-request "coalescing" is a
cost-model annotation, not physical H2D deduplication, so it is not claimed as
a completed runtime merging contribution.

KVPR-H and CacheFlow-H native bindings now emit the same structured recovery
telemetry contract consumed by the unified runner: selected policy, calibrated
H2D/replay cost source, calibrated bandwidth, and explicitly modeled selected
load/replay work. This changes neither policy's binding nor its documented
approximation boundary; it only makes their runtime choice attributable beside
HyRex in the final evidence artifact. The runner parser and self-check passed.
The interrupted policy-test process was subsequently terminated and no vLLM,
LMCache, or pytest process from this work remained.

The unified cell parser now has direct self-check coverage for both
`KVPR_H_NATIVE_DECISION` and `CACHEFLOW_H_NATIVE_DECISION` markers, including
their policy/source accounting and modeled load/replay aggregation. This closes
the parser-to-evidence link for the two Hybrid extensions without changing the
baseline algorithms. The cell self-check passed.

HyRex now supports an opt-in `hyrex_starvation_ms` threshold. A native recovery
job whose observed request wait reaches that threshold receives a bounded-batch
priority promotion before H2D dispatch; the emitted dispatch event records both
waited time and whether promotion occurred. The option is propagated from the
cell and matrix runners, allowing an ablation without changing baseline
configuration. Static compilation and cell/matrix self-checks passed. The
targeted scheduler pytest entered uninterruptible I/O in this host environment
and was terminated cleanly rather than counted as a pass; no GPU or serving
process was started by it.

The complete formal campaign is now executable through one orchestrator. It
performs the same 24 GiB CPU / GPU headroom gate before any launch, runs the
bounded measured-feedback calibration pass, then executes separate resumable
three-repeat C1/C4/C8/C16 matrices for Marconi, Tail-Replay, KVPR-H, and
CacheFlow-H against HyRex before invoking the gated main-table builder. The
`--plan-only` mode emitted all six stages (calibration, four comparisons, and
table) and the campaign self-check passed. This runner does not weaken any
per-cell resource, calibration, correctness, cache-pressure, or resume gate.

A resource-free regression audit compiled all seven current benchmark/campaign
entrypoints and ran every built-in self-check successfully (cell, matrix,
trace, calibration derivation, bounded calibration pass, table builder, and
campaign orchestrator). A non-plan campaign invocation was then deliberately
attempted at the current resource state: it rejected before spawning any vLLM
or cache service because host memory was 15.0 GiB versus the required 44.0
GiB, while GPU free memory was 23.5 GiB versus the 22.6 GiB GPU threshold.
This confirms the combined gate enforces both resources rather than treating a
free GPU as sufficient.

The campaign and bounded calibration runner now take one common arrival
configuration: uniform/Poisson/bursty/Zipf mode, offered request rate, burst
size, and Zipf exponent/rank. The campaign propagates the exact values to both
calibration and every formal baseline matrix, validating positive parameters
before planning. Calibration and campaign self-checks passed, and a Zipf
campaign plan confirmed the parameters appear in all generated commands. This
makes the required sensitivity studies reproducible through the same gates as
the primary Poisson/uniform matrix.

**2026-09-21 runtime environment correction:** the previously used vLLM
environment is `/root/hybrid-model-offloading/.venv`. A newly created
`/root/hybrid-model-offloading-hyrex/.venv` contained PyTorch but not vLLM,
which incorrectly triggered a source editable-install attempt. The redundant
build was terminated before completion. Future experiments must reuse and
verify the existing environment first, then bind the HyRex worktree source as
needed; they must not rebuild the complete vLLM stack merely because the
worktree-local venv is absent.
