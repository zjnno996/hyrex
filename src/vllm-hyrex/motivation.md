# Hybrid 前缀恢复的 Motivation 实验

## 实验目的

在真实多轮对话的 CPU offload/restore 中，比较原生 vLLM + LMCache、
原生 SGLang + HiCache 与最小增量方案的实际历史重算量、TTFT，
以及额外 SSM checkpoint 的生成、保存和加载成本。
先分别测出原生系统的真实恢复边界；不能预设 CPU 中的 KV 比 SSM 保存得深，
也不能把不同 SGLang Mamba 策略合并为一个结果。

## 实验设置

- 模型：Qwen3.5-9B，单卡。
- 系统：干净 vLLM + 官方 LMCache，先确认所选官方 connector 支持 Hybrid。
  之前 `LMCacheConnectorV1` 的启动失败只代表该路径，不能推断所有 connector。
- 数据：固定 4 条 WildChat 真实长对话 session，每条至少 5 轮，
  Qwen3.5-9B tokenize 后控制在 4096-token 上下文以内；预先固定 session ID，
  覆盖不同的 528-token 边界余数，不按实验结果筛选。
  使用同一 tokenizer/chat template，逐轮计算请求之间的真实 token 公共前缀。
- 请求并发为 1；不同 session 可交错，但每条 session 内的轮次保持原序。
- CPU cache：从 4 GiB 开始，检查实际 KV/SSM 字节数和淘汰日志；
  若出现淘汰，增加到 6–8 GiB 后重跑，不把容量 miss 当作恢复边界损失。
- 原生对照：vLLM + 官方 LMCache MP connector；SGLang + HiCache，
  `no_buffer` 与指定参数的 `extra_buffer` 分开报告。
  增量方案在干净基线之上单独实现：① KV 独立保留、缺 state 时补算；
  ② 在选择的末端 KV 边界增加 SSM checkpoint，缺失时退化为①。
  官方 MP connector 不读取 `discard_partial_chunks` 或 `save_unfull_chunk`，
  不能把这两个开关当作有效的基线消融。

## 实验怎么做

1. **跑完上一轮，检查 CPU 保存结果。**
   输入真实对话，生成回答，等待异步 offload 完成。分别检查 Full-KV 的实际
   保存范围、各 SSM/GDN checkpoint 的 token 位置。记录输入与生成 token 数；
   若回答部分未保存，检查 `save_decode_cache`，再单独开启该配置做对照。

2. **清空 GPU prefix cache，保留 CPU cache。**
   4 条 session 不保证自然冲掉 GPU 缓存，因此每次续聊都要用经过验证的
   local-only reset 或 GPU 淘汰方式，并从传输日志确认 CPU 命中。
   vLLM 与 SGLang 的清理方式须分别验证，不能重启后丢失 HiCache CPU 池。

3. **发送下一轮，测实际恢复边界。**
   分别记录 CPU Full-KV 可以连续命中到的位置 `L_KV`、同一前缀上各 recurrent
   group 的有效 checkpoint，以及调度器最终采用的边界 `L_resume`。
   同时记录实际加载字节数和重新计算的 token 数。
   尾块“已保存”不等于下一轮“能检索到”，这两项要分别检查。

4. **对同一批对话比较原生系统及两级增量方案。**
   判断缺失发生在保存、检索还是最终恢复选择阶段。
   记录①与②的实际历史重算量差异；如果原生 SGLang 已有接近完整的
   state 命中，允许其重算量优于或等于增量方案，不预设胜负。

5. **配对测量 TTFT 和 checkpoint 成本。**
   同一模型、输入、CPU 容量和 GPU 清理条件下，每个配置重复至少 5 次，
   记录下一轮 TTFT、实际重算的历史 token、加载字节、checkpoint 制作/写入/加载耗时，
   并报告上一轮 checkpoint 开销 + 下一轮 TTFT 的两轮合计成本。
   同一个模型内配对比较；不同引擎的绝对 TTFT 不直接归因于恢复方法。

## 记录的数据与判定

每次恢复记录：session/轮次、实际公共前缀长度、CPU KV 保存深度、
CPU recurrent checkpoint 位置、`L_KV`、`L_resume`、重算历史 token 数、
首 token 延迟、KV/SSM CPU 占用、CPU→GPU 字节数及 checkpoint 开销。

| 实测情况 | 对应解释 |
|---|---|
| Full KV 已保存且可命中更深，但恢复退到较浅的 state checkpoint | 支持“共同恢复边界浪费已有 KV”的核心 insight |
| KV 和 state 都只保存到共同对齐边界 | 损失发生在保存阶段，需检查尾块与 checkpoint 保存规则 |
| KV 尾块已经在 CPU，但下一轮无法命中 | 问题在 chunk/key 的跨轮检索规则 |
| 两类状态都完整保存且正常恢复 | 此场景没有证明边界不一致问题，应如实报告 |

例如：只有实测得到 CPU KV 可命中 2064、可用 state checkpoint 为 1584，
并且调度器实际恢复到 1584，才能说这次恢复未利用 480 token 的已有 KV。
单纯计算 `floor(P / 528) * 528` 不能作为该结论的实验证据。

## 最终产出

一张真实长对话逐轮的 KV/state/恢复边界示意图，一张 4-session 的原生/增量
配对表：历史重算 token、TTFT、checkpoint 开销、KV/SSM CPU 占用及原因分布。
这是机制性 motivation case study；4 条 session 不用于声明总体性能显著性，
只做 motivation 所需测量，不扩展为完整端到端性能主表。

已有历史结果的汇总传输量和共同 lookup 长度不能证明各组独立存储深度；
本实验需要重新采集这些字段。当前本文是实验方案，不代表实验已完成。

## 2026-09-25 实施边界与进度

预热后的干净 vLLM + 官方 LMCache MP 对照已经完成 4 session × 10 轮：
`/root/hyrex_results/motivation_lmcache_9b_warm_4s10t/online.jsonl`（40 条）。
每轮只清 GPU、保留 2 GiB LMCache L1；36 个续聊请求的 CPU 预取均成功。
这仍是**共同边界 baseline**，不是 HyRex 增量方案。

真正的“更深 KV”实验必须在**写入 CPU 时**实现组级独立性：

1. Full-Attention KV 以自身的 token-prefix key 和块索引持续写入，形成 `L_KV`；
   recurrent/GDN checkpoint 单独决定保存位置和 `L_S`。不能先把两者都写到同一深度，
   再人为删除较深的 SSM 对象并称为独立存储。
   **Full KV 按实际 16-token kernel page 独立生成 CPU key**，查出最深连续
   `L_KV`；recurrent/GDN 按 checkpoint 粒度独立查出 `L_S`。vLLM 的 528-token
   逻辑调度块不是 Full-KV 的 CPU 索引粒度，不能用共同命中截断 KV 查找。
2. LMCache MP 的对象组需要按 Hybrid engine group 分开。官方
   `--separate-object-groups` 仅按 sliding-window 大小分组；Qwen3.5-9B 的
   Full 与 GDN 组有相同窗口属性，原生会落在同一个 CPU 对象组；实验分支
   已增加按 engine group 拆分。
3. STORE/LOOKUP/RETRIEVE 都按对象组使用各自粒度，分别返回 `L_KV` 与
   `L_S`。`L=max(L_KV,L_S)` 只是**目标边界**，不是已经完整恢复的前缀。
   `L_KV>L_S` 时加载深 KV 和浅 checkpoint，并只补算缺失的 recurrent
   state；补算过程中保护已加载的 Full-KV 槽位。若 `L_S>L_KV`，还需补齐
   缺失 KV，不能直接把 `L_S` 宣告为完整命中。
4. 16-token page 只能存完整有效页；不足 16 token 的尾部要明确重算，或另设
   带有效长度的 partial-page 对象。实验先采用重算尾部，避免伪造命中。
5. 必须对照同一 prompt 的首 token/输出一致性、逐轮 TTFT、实际 D2H/H2D
   字节和 CPU 占用。输出不一致的 TTFT 不可作为收益。

LMCache 0.5.3 的实验分支位于 `/root/lmcache-hyrex`。已完成按 engine group
拆分 CPU 对象组、选择性 STORE/LOOKUP/RETRIEVE、双边界匹配，以及实验性
replay 路径；但 Full-KV CPU key、对象布局和传输仍按全局 528-token chunk，
**不是上述 16-token 独立索引**。现有深命中实验第 8、9 轮首 token 与官方
基线不一致（实验分支 `!`，基线分别为 ` B`、` C`），不能比较 TTFT 收益。
仅写入、不进行深层加载的诊断结果与基线首 token 一致；问题尚在深层
H2D/replay 路径，必须先修复正确性，再测试 16-token 索引的收益。

写入侧 9B smoke test：
`/root/hyrex_results/motivation_lmcache_9b_independent_store_1s10t/`。
真实 ShareGPT session `Ud3L2sd_124` 共 10 轮，528-token chunk、2 GiB CPU L1、
每轮清 GPU 前缀缓存。日志中的 `group=3` 为 Full Attention；例如长轮次实际提交
Full-KV `[528,1056)` 和 `[1056,1584)` 的 CPU STORE，`state_boundary=528`；
LMCache 端对应 STORE 均成功返回。这证明了独立写入路径被执行，不等于证明
下一轮能独立检索这些 KV。原生共同 LOOKUP 仍只报告 528-token 命中；第 7～9
轮 TTFT 分别是 291.43、293.00、297.04 ms，而同一输入的预热官方基线是
224.48、173.39、154.09 ms。**当前实验分支是写入侧 smoke test，不是有效
HyRex 性能结果**；额外写入开销和多次重算仍在，必须完成独立恢复后重测。

`LMCACHE_HYREX_STATE_STORE_CAP_TOKENS=528` 是用于制造可控边界差异的实验配置，
不是最终自适应 checkpoint 策略。当前原型只覆盖完整 528-token chunk，不能声称
已经保留或命中 16-token 粒度的尾部 KV。
此前 `/root/hyrex_results/motivation_lmcache_9b_deeper_kv_4s10t/` 是被中止的
“删 SSM 对象”错误方向实验，其 33 条临时请求**不用于任何结论**。

## 2026-09-26：9B 逐请求清 GPU 的机制对照

同一 GPU、4 条真实 ShareGPT session × 10 轮、每轮生成 1 token；每次请求后
`reset_prefix_cache?reset_external=false` 成功（39/39），CPU L1 容量 2 GiB。
首轮不计入续聊 TTFT 中位数。三种配置均先做不相关请求的 warmup，再按相同顺序
执行 40 个请求。原始数据分别在以下目录的 `online.jsonl`、`resets.jsonl` 和
`lmcache.log`：

| 配置 | 续聊 TTFT 中位数 | CPU 命中 token 总数 | 与官方基线首 token 差异（36 轮） |
|---|---:|---:|---:|
| 干净 vLLM + 官方 LMCache，共同 528-token 边界 | 157.93 ms | 32,736 | 0 |
| 独立 16-token Full-KV 索引 + 528-token state；原 QKV 投影 | 214.24 ms | 42,368 | 11 |
| 同上，命中区域 Full-Attention 改为 Q-only 投影 | 208.72 ms | 42,368 | 9 |

独立索引确实让更多 KV 命中，Q-only 对同一独立索引路径约省 5.5 ms，但
**目前没有获得相对官方基线的净 TTFT 收益，也未通过输出一致性检查**。
不能把这组数据写成 HyRex 的性能提升。LMCache 日志显示官方单路恢复耗时
中位数约 26 ms；实验路径分离 state 与 Full-KV 后分别约 16.5 和 57 ms。
后续源码审计纠正：state 对象组不含 Full KV，并没有在两路中重复传输 Full KV。
问题主要是细页逐对象搬运/散写开销，以及 state 路径加载了多个历史 checkpoint。
这里的恢复耗时是 CPU handler 墙钟时间，包含准备和派发，不是纯 PCIe DMA 时间。

尝试让 528-token 公共对象同时携带 Full KV，仅用 16-token 页补尾部；但当前
LMCache 对同一个 object group 只允许一个 chunk size，公共操作涉及 528 与 16
两种粒度时 warmup 直接报 `Mixed chunk sizes require separate object-group operations`。
该不兼容尝试已撤回，**没有作为有效实验结果**。后续改用细粒度索引、批量搬运，
不需要双粒度 Full-KV 别名；实现和复测见下节。

## 2026-09-26：批量 Full KV 搬运 + 仅恢复末端 state，正式复测

已完成同 GPU1 的两组 40 请求：4 session × 10 轮，round-robin，始终单请求。
两组均使用 LMCache CPU L1 2 GiB、不相关请求 warmup、正式请求之间 39/39 次
GPU prefix reset 成功，保留 CPU cache；不是设备重启，也不重新加载权重。
每轮回放真实对话历史而非把此次生成的 1 token 作为下一轮历史。TTFT 从客户端
发起 HTTP 到收到第一个非空流式文本，包含服务/传输/前向开销；轮间等待和 reset
在计时之外。这里只比较 36 个续聊请求，四个首轮另存原始记录。

| 配置 | 续聊 TTFT 中位数 | 平均数 | 首文本与本次原生基线不一致 |
|---|---:|---:|---:|
| 原生共同 528 边界，本次同卡重跑 | 160.24 ms | 168.77 ms | — |
| 独立索引，旧细页搬运 + Q-only | 208.72 ms | 215.89 ms | 9/36 |
| 独立索引，批量 Full KV + 末端 state + Q-only | 162.31 ms | 172.87 ms | 8/36 |

修复版比旧实现的中位数下降 22.24%，但仍比本次原生基线高 1.29%，
仅 11/36 个续聊请求更快；一次配对运行不能证明统计显著性。两次原生基线
40/40 首文本完全一致。实验分支的 8 个差异未解决，不能声称精确恢复已通过，
也不能挑出更快的请求当作有效加速结果。

原生 CPU 命中累计 32,736 token，独立 Full KV 命中累计 42,368 token，
增加 9,632 token（29.42%）。后者是 Full KV 可复用长度，不是整个模型免算长度：
从 state 边界到 Full KV 边界仍需要执行 recurrent、Q、attention 和 MLP 等计算。

本次最小修改：Full KV 仍按 16-token key 独立索引，但连续 CPU 页批量上传，
按层批量 index_copy，避免逐页 Python/torch fallback 调用；state 仍按 528
checkpoint 检索，仅加载选中的末端一份，同时不发布未加载的历史 GPU state hash。
正式恢复日志排除 warmup 后：旧实现 state/Full handler 中位数 16/57 ms，
修复版 14/7 ms，原生共同恢复 25 ms。不同 handler 的中位数不能相加当成
精确的请求分解；这些数值也不是 CUDA event 测量的纯 H2D 时间。

独立等字节搬运微测（GPU2，1056 token、8 Full 层、33 MiB，10 次测量，
计时含 CUDA synchronize，三种输出逐位一致）：

| 搬运实现 | 中位时间 |
|---|---:|
| 原 528-token 粗块 fallback | 2.332 ms |
| 16-token 逐页 fallback | 18.753 ms |
| 16-token key + 新批量搬运 | 1.853 ms |

这证明此前明显的搬运劣化主要来自实现，而不是细粒度索引必然需要更多传输时间。
不过更少投影 FLOPs 不必然更快：单层 BF16 投影微测（434 token，其中 264 replay、
170 新 token，50 次）中，融合 QKV 为 0.307 ms，当前拆分 Q-only 为 0.313 ms。
该合成微测不是模型 TTFT，只表明拆分 GEMM、拼接等开销可能抵消投影节省。

结果目录：

- 原生重跑：`/root/hyrex_results/motivation_9b_clean_repeat_20260926_4s10t_gpu1/`
- 修复版：`/root/hyrex_results/motivation_9b_bulk_full_last_state_4s10t_strict_reset_gpu1/`
- 配对逐轮表：`/root/hyrex_results/motivation_9b_comparison_20260926.csv`
- 搬运微测：`/root/hyrex_results/motivation_transfer_micro_20260926.json`

当前结论：独立索引能命中更深的 KV，批量搬运能消除主要实现劣化，但这组短对话
尚未证明净 TTFT 收益。下一项必要工作是定位输出差异（同 prompt 的无缓存参考、
逐层中间值/首 token logits 对照），再在正确性通过后做配对重复测量。
不能把尚未验证的“更深命中必然更快”作为论文结论。

## 2026-09-26：按层恢复的实际分配缺陷与修复

新增隔离诊断：4 session 各前两轮，逐请求清 GPU，不 warmup，仅用于正确性，
不比较 TTFT。关闭 Q-only 后，保留深 KV 有两个续聊首文本与基线不同；允许
replay 覆盖深 KV 后仍有两个不同；关闭 CPU 恢复、从头计算则全部与基线一致。
目录分别为 `motivation_9b_replay_diagnosis_masked_20260926`、
`motivation_9b_replay_diagnosis_overwrite_20260926`、
`motivation_9b_replay_diagnosis_shallow_20260926`（最后这个历史命名实际指无 CPU
恢复，不是浅边界恢复），均位于 `/root/hyrex_results/`。

发现确定的分配缺陷：connector 向 scheduler 报告 max(state, Full KV) 作为
异步加载目标，但 KVCacheCoordinator 把它也传给 MambaManager 作为 computed
边界。例：state=528、Full KV=752，MambaManager 的跳过规则得到
floor((752-1)/528)=1，把 checkpoint 528 的目的槽位设成 null block 0。
这不是合法的独立 state 恢复。不同组的层还可以通过 shared_by 共用底层缓冲区，
不能向共享 null block 写请求的真实状态。

修复：异步加载阶段把 recurrent load boundary 显式传入 coordinator，块数预算、
外部块分配、运行块分配和回收都按组处理：Full 保持深边界；Mamba 保留实际
checkpoint，后续 replay 才分配运行状态块。原生未设置独立边界时行为不变。
不能仅修改 lookup 或最终 num_computed_tokens，分配/回收边界也必须解耦。

回归测试覆盖零 state、跨多个块、1/3 个 state group，确认 checkpoint 是独立
非 null 物理块，后续 replay 有合法运行块；也以未解耦的原路径复现 null 槽位。
完整 prefix-cache 测试先通过 80 项，扩展多 group 后新增 4 个参数用例通过。

修复后 Q-only 同 GPU1 的 40 请求已完成，39/39 次 reset 成功：原先 8 个首文本
差异全部消失，但 Ud3L2sd_124 的第二轮新增空格/双换行差异（1/36）。
续聊 TTFT 中位数 164.765 ms，均值 173.251 ms，仍不能宣称净加速。
目录：`/root/hyrex_results/motivation_9b_state_allocation_fixed_4s10t_20260926/`。
正在另测保留融合 QKV 的分支和首 token top-5 logprobs；这些数值诊断不能
替代逐层 tensor 对照或更长生成的正确性验证。

### 同一搬运实现的浅/深加载消融

新增 `LMCACHE_HYREX_FULL_LOAD_TO_STATE=1`：在存在可用 state checkpoint 时，
仍执行相同的 16-token Full KV lookup，但只加载到 state 边界；释放未加载尾部
的 lookup 锁，保留真实 CPU 存储深度，避免把尾部误当未存储而重复 STORE。
开关关闭时加载到独立 Full KV 命中边界。无 state 命中时暂不截断，两组行为相同。

两组必须固定末端 state-only、Full 页批量搬运、同卡、warmup、逐请求 GPU reset
及请求顺序。该开关不是 `LMCACHE_HYREX_DEEP_LOOKUP=0`（后者会禁用 CPU 恢复）。
先用融合 QKV 验证加载边界切换的正确性，再测 Q-only 的增量效果；融合 QKV 组
没有跳过 K/V 投影，不能单独据此否定深 KV 的计算收益。
当前仅完成消融开关与单元验证，尚未运行此配置的 40 请求 TTFT 对照。

### 同搬运消融实测更新：基础正确性尚未稳定

已在 GPU1 顺序完成浅、深各 40 请求，warmup、2 GiB CPU L1、末端 state、
细粒度索引和批量搬运相同，各 39/39 次 GPU reset 成功。浅组没有 replay spans，
因此不进入 Q-only；深组启用 Q-only。

| 配置 | 36 续聊 TTFT 中位数 | 平均数 | 对原生首文本差异 |
|---|---:|---:|---:|
| 同实现，KV 截至 state | 158.11 ms | 168.093 ms | 1/36 |
| 同实现，KV 加载到深命中 | 159.11 ms | 168.164 ms | 4/36 |

目录：`/root/hyrex_results/motivation_9b_matched_transfer_shallow_20260926/`
和 `/root/hyrex_results/motivation_9b_matched_transfer_deep_20260926/`。
浅组差异为 Pr8nMeM_0 的 turn_index=1；深组为 WRAImOg_0 的 1、3、7，
以及 Ud3L2sd_124 的 9。差异位置与此前运行不同，不能宣称基础正确性完整通过，
也不能将差异直接归因于浮点误差。此组仅作诊断，不作为有效净加速结论。

下一步先验证数据：新增 `LMCACHE_HYREX_VERIFY_FULL_H2D=1`，在批量 NHD
Full KV 加载路径逐层把目的页读回 CPU，与原 CPU 对象逐字节比较，出错时报
group/layer/page 区间。该开关强制同步并增加 D2H，禁止用于 TTFT 性能数据。
它只验证当前批量 Full H2D 路径，不覆盖 STORE、state 恢复或 replay 数值正确性。
校验单元测试两项通过（正常复制通过、故意破坏复制必须报错）；GPU1 合成
1056-token/8-layer/33-MiB 搬运亦通过逐字节校验。校验模式中的搬运计时包含
逐层 D2H 和同步，不能与关闭校验的搬运性能比较；真实模型恢复尚未用此开关复测。

## 2026-09-27：额外尾部 checkpoint 的独立机制验证（进行中）

候选切入点：更深 KV 命中不是完整的可跳过前缀；同位置的完整 recurrent
checkpoint 才能消除整段前向。但 checkpoint 的捕获、保存和容量不是免费的。
本节验证这项交换，不预设收益，不宣称新增 checkpoint 或流水线本身为首次提出。

最小原型保持正常 528-token checkpoints 和 Full KV 16-token 索引，额外在指定
1040-token 边界截断一次前向，把所有 recurrent 层完整状态通过 LMCache MP
STORE 保存。下轮经 CPU LOOKUP/RETRIEVE 恢复该状态；没有命中则走原路径。
它是单边界实验，不是已完成的动态多 session 尾部管理器。
尾部用单独命名空间，完整真实前缀 SHA256 纳入键；传输仍使用既有 opaque
state 对象布局。为了不被后续前向原位覆盖，实验显式等待尾部 STORE 完成。
日志 SAVE_BARRIER_MS 包含同一步的其他 STORE 和同步，不能称为纯 state DMA。

选取真实 ShareGPT `Ud3L2sd_124` 原始 turn 5→6：历史 1054 token，下轮1191，
精确共同前缀1054。基线恢复528，尾部目标1040，整段前向理论减少512 token。
这是明确挑选的大尾差机制例子，不代表总体平均。两路固定同卡GPU1、2GiB CPU
缓存、Full批量搬运和单末端state加载；每请求间验证GPU reset成功，CPU保留；
重复间清空两级缓存。先warmup，再重复6对，第一对仅用于该形状预热，后5对
计算平均值。生成32token检查文本，记录首token top5 logprobs；相同输出仍不等于
所有输入下的数值等价证明。baseline是同搬运栈的浅边界消融，不冒充原生未改基线。

入口：`benchmarks/motivation/run_tail_probe_pair.py --arm {shallow,tail}`。
同时报告上一轮TTFT/总耗时、下一轮TTFT、两轮合计和额外49.5MiB state容量。
不能只展示下一轮收益而隐藏捕获阶段代价。

初次冒烟 `tail_probe_1040_20260927_v1` 未通过边界门槛：尾部已STORE但检索miss，
恢复仍528；这组数据无效，不计加速。原因是原型曾用虚拟零token派生传输键，
与LMCache同请求内已经memoize的真实前缀hash不一致。已改为真实前缀token配合
完整前缀salt，正在复测。现有prefix-cache回归84项、connector及tail测试14项通过。

`v2`发现另一项实验干扰：seed也生成32token，生成内容与下轮历史部分一致，
使原始checkpoint跨到1056，实际Full命中1072。这没有验证1040尾部恢复，不纳入
该对照。修正为seed固定1token（与原主表一致）、resume生成32token。故本实验
仍是teacher-forced真实对话历史的受控恢复，不是自由生成闭环对话。`v3`启动前
端口探测受TIME_WAIT影响退出，无测量；已改进探测，`v4`正在执行修正协议。

### 修正协议的实测结果：单加尾部 checkpoint 尚无 TTFT 收益

两路均已完成6对请求，各11/11次GPU reset成功。尾部组6次恢复均有独立
`TAIL_PROBE HIT state=1040 full=1040 base=528` 日志。排除每路第一对后：

| 指标（后5对平均） | 浅恢复消融 | 额外1040尾部state |
|---|---:|---:|
| 下一轮可直接恢复前缀 | 528 | 1040 |
| 下一轮剩余前向token | 663 | 151 |
| seed TTFT | 217.362 ms | 335.874 ms |
| resume TTFT | 242.862 ms | 260.704 ms |
| 两请求总耗时之和（含resume 32token生成，不含固定reset等待） | 1558.546 ms | 1706.242 ms |
| 相对浅恢复额外CPU checkpoint容量（布局推导） | 0 | 49.5 MiB |

该例少算512token（剩余前向token减少77.2%），但恢复TTFT增加17.842ms（约7.35%），
seed TTFT增加118.512ms。尾部组后5次SAVE_BARRIER_MS平均68.966ms，包含同一步
Full KV STORE和同步等待，不能全部归为state D2H，也不能与请求TTFT相加做分解。
同条件复测中，5对seed首文本和resume32token文本全部相同；首token logprobs有
差异，两路各自重复间也有变化。因此只通过抽样生成文本检查，未证明逐层数值一致。
这是顺序运行的单样本机制实验，不是随机交错或多session总体性能结论。

结果目录：`/root/hyrex_results/shallow_probe_1040_20260927_v1`、
`/root/hyrex_results/tail_probe_1040_20260927_v4`。
逐次数据为各目录`online.jsonl`；汇总为
`/root/hyrex_results/tail_probe_pair_comparison_20260927.json`。

当前能支持的切入点：**增加完整可恢复前缀、甚至真正减少整段前向token，仍不足以
保证TTFT改善；恢复和checkpoint维护的实际关键路径必须被测量和优化。**
尚不能支持“尾部checkpoint必然更快”或“现有系统根本无法做高效恢复”。

下一项待验证的具体假设：恢复到1040后，为继续保存1056常规checkpoint，scheduler
仍把剩余151token拆成16+135两次前向；浅恢复则是528+135，两者并未减少前向次数。
这一分段可从scheduler逻辑推出，尚需实际执行计数/时间剖面验证其TTFT占比。
还需分离新增KV传输、第三路lookup和保存屏障开销。任何跳过1056 checkpoint的
消融必须同时禁止将S1040或S1191错误发布为S1056，不能只关闭scheduler对齐。

### 尾部 checkpoint 与维护消融：完成一轮验证（2026-09-27）

复核以下各组原始 online.jsonl/resets.jsonl：均12请求、11次成功GPU reset。
恢复性能排除第一对，报告后5对平均；浅恢复为同搬运栈消融，不是原生未改基线。

| 配置 | 恢复边界 | 剩余前向token | resume TTFT | seed TTFT |
|---|---:|---:|---:|---:|
| 浅恢复 | 528 | 663 | 242.862 ms | 217.362 ms |
| 额外尾部state | 1040 | 151 | 260.704 ms | 335.874 ms |
| 尾部state，且恢复请求不再生成/发布新state checkpoint | 1040 | 151 | 193.066 ms | 343.222 ms |

第三组实际STEP日志确认1040→1191为一次151-token前向；同时禁用后续state CPU
STORE和GPU哈希发布，Full KV STORE仍保留。这是维护开销消融，不是完整多轮策略。
相对浅恢复，resume平均减少49.796 ms（20.5%），但seed平均增加125.860 ms，
两请求总耗时之和仍从1558.546 ms增至1629.246 ms。额外尾部state为49.5 MiB。

**正确性门槛未通过，以上仅为诊断测量，不能作为有效加速结果。** 第三组后5次中
一次32-token生成文本不同。浅恢复、普通尾部恢复和该消融各自重复时首token
logprob也漂移。新增同分段无缓存参考（528+512+151）已完成6次，其首token
logprob全部为-0.19320979714393616；第三组首次恢复与它一致，后续不一致。
这不能直接归因为普通BF16分段误差，需要排查恢复或物理块复用等路径。

独立state字节审计组的opaque对象D2H/H2D检查通过，但这不证明模型实际结构化
状态视图、语义块映射以及Full KV恢复都正确；该次审计未启用Full KV字节检查。
字节审计会同步GPU，不计入性能测量。下一步优先定位数值漂移，再重复性能对照。

结果：`/root/hyrex_results/tail_probe_no_new_states_comparison_20260927.json`；
原始目录：`tail_probe_no_new_states_20260927_v1`、`tail_probe_state_bytes_20260927_v1`、
`tail_probe_cold_fused_reference_20260927_v1`（均位于`/root/hyrex_results/`）。

### 恢复漂移定位：Python fallback 的 CUDA stream 顺序错误

模型可见状态审计（`tail_probe_model_state_bytes_20260927_v1`）的3次恢复中，
24层共48份conv/recurrent状态与seed在1040边界的状态哈希一致，状态复制字节
检查也通过。进一步审计模型Full KV（`tail_probe_model_full_bytes_20260927_v1`）：
第一次恢复所有页一致；第二次8层的前13页不一致，第三次前17页不一致。
坏页中第n页多为原第n-1页，呈现暂存缓冲区读取落后一批的特征。

在同环境的独立CUDA复现中，非默认stream先排队sleep、写入17，再调用
`torch_ops.lmcache_memcpy_async`做D2H；等待该stream后CPU仍读到旧值0。
源码原因：fallback调用同步`cudaMemcpy`，却没有遵守生产者所在的非阻塞stream；
“同步拷贝”不意味着等待其他非阻塞stream的生产者。

修复在`/root/lmcache-hyrex/lmcache/v1/platform/torch_ops.py`：改用当前CUDA
stream的`cudaMemcpyAsync`，按原生实现切分host注册边界；不增加全设备同步。
回归`tests/v1/test_torch_ops_stream_order.py`验证非默认stream上的H2D/D2H顺序，
2项通过；相关tail和字节审计测试11项通过。9B端到端恢复审计与TTFT重测另行记录。

**这属于本实验环境的fallback实现问题，不应当包装为原生vLLM/LMCache的
固有限制或论文motivation。** 修复后须同时重跑浅恢复与尾部恢复，不能拿修复前
的浅恢复数据比较修复后的方案。此前性能数字仅保留为排错记录。

修复后的9B审计`tail_probe_stream_fixed_audit_20260927_v1`已完成3对、5次成功reset。
每次恢复48份state和8层×65页Full KV全部与seed边界字节一致；32-token文本和
首token top-5 logprobs均与同分段冷参考一致。校验结果为该目录`verified.json`。
`benchmarks/motivation/verify_tail_model_bytes.py`可重复验证；对修复前错页目录运行
会以`model Full KV mismatch`拒绝，而非把抽样文本相同当成恢复正确。
额外84项prefix-cache回归全部通过。该审计带同步、哈希开销，不能用于TTFT比较。

### 修复后正式TTFT对照（单真实pair，三组均已完成）

同一GPU1、Qwen3.5-9B、2GiB LMCache CPU缓存；每组无关请求warmup后重复6对，
排除第0对，报告后5对均值。每组11/11次GPU reset成功，组间顺序运行。
三组都使用修复后的同一Python fallback搬运栈，关闭所有字节/哈希诊断；
浅恢复是受控消融，不是未改动原生vLLM+LMCache性能基准。

| 指标 | 浅恢复 | 加1040尾部state | 尾部state＋不维护后续state checkpoint |
|---|---:|---:|---:|
| 可直接恢复边界 | 528 | 1040 | 1040 |
| 剩余前向token | 663 | 151 | 151 |
| 恢复后的prefill分段 | 528+135 | 16+135（实测STEP日志） | 151（实测6次STEP日志） |
| resume平均TTFT | 243.952 ms | 255.706 ms | 190.076 ms |
| seed平均TTFT | 222.222 ms | 331.244 ms | 327.176 ms |
| seed+resume TTFT之和 | 466.174 ms | 586.950 ms | 517.252 ms |
| 两请求总耗时之和（含32token生成，不含reset等待） | 1593.152 ms | 1731.652 ms | 1631.204 ms |
| 相对浅恢复额外CPU state容量 | 0 | 49.5 MiB | 49.5 MiB |

第三组resume样本为178.57、190.35、195.96、191.98、193.52 ms；相对浅恢复
平均减少53.876 ms（22.08%），相对普通尾部组减少65.630 ms（25.67%）。
但seed平均增加104.954 ms；两请求总耗时之和仍增加38.052 ms（约2.39%）。
Full KV H2D从528增到1040 token，多16 MiB；单次加载的完整state仍为49.5 MiB。
浅恢复组本身也独立存储较深Full KV，故不能把这16 MiB额外加载量说成额外CPU存储量。

正确性：三组全部6次resume的32-token文本一致，并与冷参考一致；每组内部首token
top-5 logprobs完全稳定。第三组还与同分段冷参考的top-5 logprobs逐项一致。
不同分段之间存在小幅数值差异，不要求跨不同分段算法bitwise相同。
这些检查支持当前选定例子，不是对任意请求/并发/模型的全面正确性证明。

**目前可用于motivation的证据：更深的完整checkpoint能减少整段前向token，
但token减少不必然转化为TTFT下降；恢复边界之后的checkpoint维护与前向分段
会影响收益能否实现。** 同样1040恢复边界、同样151个剩余token，两种维护策略
实测TTFT不同。该消融同时改变保存工作和前向分段，不能把65.630 ms全部归因于
纯D2H传输或纯GPU计算，若要细分仍需时间剖面。

第三组牺牲了本请求后续checkpoint的可用性，不能称为完整多轮管理方案；创建
尾部state还会强制seed多分段。因此下一步技术目标应是低开销捕获与按收益选择
checkpoint，并避免维护强制拆分完整模型前向，而不是单纯“增加尾部state”。
本轮只验证一个选定大尾差pair，尚未证明多session平均收益、原生栈收益或新颖性。

三组目录：`/root/hyrex_results/tail_probe_stream_fixed_{shallow,tail,tail_no_new_states}_20260927_v1`。
逐轮数据见各目录`online.jsonl`、reset记录见`resets.jsonl`。
汇总：`/root/hyrex_results/tail_probe_stream_fixed_tail_comparison_20260927.json`与
`/root/hyrex_results/tail_probe_stream_fixed_no_new_states_comparison_20260927.json`。

### 深Full KV＋浅state replay：修复后重新验证

入口`benchmarks/motivation/run_deep_kv_ablation.py`，三组为`shallow`、`deep_qkv`、
`deep_qonly`。均为4个真实session×10轮，交错执行，每请求清GPU cache，CPU 2GiB，
相同Full16索引、批量搬运、末端state-only策略和无关warmup。没有额外尾部state。
同一修复后源码栈只改变Full加载边界与是否跳过命中token的K/V投影；因此是受控
消融，不能冒充干净原生baseline。统计36续聊的平均TTFT，保留所有长尾样本。

需要纠正计数：深KV组API `cached_tokens`报告较深Full命中，不等于实际可以跳过
的完整模型前向token。state仍按528命中，实际replay必须从state边界开始。
为了产生后续GDN层的输入，replay仍执行Q、attention输出、输出投影、GDN和MLP。
Q-only分支仅跳过命中区间的K/V投影；当前代码还保留K norm/RoPE（作用于占位K）、
分配零KV缓冲与cat操作。未启用Q-only时仍执行融合QKV，不能算作省掉了K/V投影。

本地9B配置：32层、8层Full，hidden=4096，MLP intermediate=12288，KV heads=4，
head_dim=256。每replay token可省的K/V投影权重乘加项为
`8×4096×(2×4×256)=67,108,864`；仅32层MLP已有
`32×3×4096×12288=4,831,838,208`项，前者为后者的1.3889%。
这只是FLOPs比例，不是TTFT加速上限；它说明多命中512 token的KV不等于少算512
token的完整模型。对于512-token replay，省去的K/V投影约68.72 GFLOPs，额外
加载Full KV为16 MiB，仍需测实际kernel效率与恢复开销。

正式TTFT与profiling分开。`run_tail_probe_pair.py --arm deep_qkv/deep_qonly
--profile-resume --repetitions 2`只对形状预热后的恢复请求启用原生torch profiler，
输出实际forward起点/调度token/Q-only区间；这些运行不得计入TTFT主表。
`bench_replay_projection.py`单独测528-token前向中不同replay长度的融合投影与
当前拆分Q-only实现，包括zero/cat开销；该microbenchmark不代表模型TTFT。

修复后主表三组已各完成40请求、39次成功reset，首token文本三组全部一致。
保留所有样本，36续聊结果如下（不是重复试验的显著性结论）：

| 配置 | 平均TTFT | 续聊样本标准差 | 最大TTFT | 相对浅组更快的请求数 |
|---|---:|---:|---:|---:|
| 浅KV、融合QKV | 174.719 ms | 29.335 ms | 236.72 ms | — |
| 深KV、融合QKV | 182.411 ms | 49.267 ms | 412.95 ms | 21/36 |
| 深KV、Q-only | 173.305 ms | 29.244 ms | 253.01 ms | 25/36 |

Q-only相对浅组仅减少1.414 ms（0.810%），不能据单次顺序试验声称稳定显著收益。
两种深KV组的API命中深度逐请求一致，平均比浅组多267.556 token，即多加载
8.361 MiB Full KV；未因此少做对应token的全模型前向。
融合深组最后一次WnjND3T_0/turn9的412.95 ms中，日志定位Full恢复服务路径约
260 ms（典型相邻请求约10 ms）；该计时含CPU管理/发起等，不是纯PCIe DMA时间。
保留该样本，不能删除它来制造收益，也不能把融合深组与Q-only的9.106 ms均值
差全部归因于投影优化。首token一致只是一项抽样检查，不是全输出正确性证明。

形状匹配projection微测（GPU1 RTX4090，BF16，每次528 token，交替测量）：

| replay token | 融合QKV，单层ms | 当前拆分Q-only，单层ms |
|---:|---:|---:|
| 112 | 0.3410 | 0.3613 |
| 264 | 0.3301 | 0.3426 |
| 416 | 0.3321 | 0.3292 |
| 512 | 0.3307 | 0.3294 |

即使replay命中512 token，拆分后的投影块也几乎没有变快：额外GEMM、zero和cat
抵消节省。这是当前实现的机会点，但不能因此把可省计算扩大成整个Full层。

第一份GPU剖面`deep_kv_profile_deep_qkv_20260927_v1`确认：Full命中1040，实际
计算仍为state528→1056的528 token、再1056→1191的135 token。第一段32层
MLP gate/up与down各32次矩阵乘法，GPU kernel合计36.665 ms；8层融合QKV合计
2.830 ms。这里只是profiling下的kernel和，不是无侵入TTFT。诊断中的32-token
生成文本也与同输入参考一致。第二份Q-only剖面结果另行补充。

第二份Q-only剖面已完成：第一段8次Q/gate矩阵乘法合计2.364 ms，8次
新token K/V矩阵乘法合计0.245 ms，总2.609 ms，对照融合投影2.830 ms。
这只减少约0.221 ms矩阵乘法kernel时间；32层MLP仍36.513 ms。不同profiler
运行间存在扰动，该差值用于定位量级，不能直接视为TTFT因果差值。

进一步最小优化：Q/gate与KV保留独立buffer，移除cat后立即split的整块复制；
保留KV补零及原有norm/RoPE、屏蔽KV写入逻辑。无replay时仍融合QKV。
`VLLM_HYREX_Q_ONLY_NO_CAT=0`保留旧布局作为同版本对照（默认1）。
微测输出`deep_kv_projection_no_cat_20260927.json`：

| replay token | 融合QKV ms | 旧Q-only ms | 无cat Q-only ms |
|---:|---:|---:|---:|
| 112 | 0.33765 | 0.35848 | 0.34481 |
| 264 | 0.33148 | 0.34290 | 0.32853 |
| 416 | 0.33128 | 0.32942 | 0.31482 |
| 512 | 0.33322 | 0.32920 | 0.31508 |

这是单层投影micro，独立输出与旧输出逐元素相等；不是全模型推理正确性证明。
512 token时去cat约省0.0141 ms/层，8层约0.113 ms，不能宣传成大幅TTFT提升。
真实9B seed/resume、逐请求GPU reset、CPU缓存保留的32-token输出测试另行记录。

上述真实检查现已完成，两组各4对请求（8请求、7次GPU reset），预先执行不相关
warmup，无profiler。旧cat通过环境开关恢复，其他恢复配置一致；先无cat后旧cat，
不同服务进程，尚非交错重复统计试验。保留全部4次恢复TTFT：

| 布局 | 4次恢复TTFT ms | 全部恢复平均ms |
|---|---|---:|
| 旧cat | 291.03 / 280.58 / 247.70 / 256.23 | 268.8850 |
| 无cat | 275.13 / 234.87 / 226.86 / 243.53 | 245.0975 |

4次32-token生成文本和首token top-logprobs跨组完全一致。观察到均值差23.7875 ms，
但远大于投影微测所解释的约0.113 ms，不能将差值全部归因为去cat，也不能据此
宣称稳定8.85%提升。此机制探针与此前40请求/输出1token的主表不同，不横向拼表。
无cat组恢复阶段日志中state服务耗时8–13 ms，Full服务耗时2–6 ms；这些是CPU
服务路径计时，并非纯DMA，不能简单与GPU kernel时间相加构造TTFT分解。
目前已定位的净计算收益是K/V投影及cat复制；未减少的是GDN、MLP、Q/Attention
输出，额外负担仍包括更深Full KV传输及拆分GEMM/placeholder处理。

真实结果目录：`/root/hyrex_results/deep_kv_no_cat_probe_20260927_v1`、
`/root/hyrex_results/deep_kv_cat_control_20260927_v1`。

主表目录`/root/hyrex_results/deep_kv_stream_fixed_{shallow,deep_qkv,deep_qonly}_4s10t_20260927_v1`；
汇总`deep_kv_stream_fixed_4s10t_comparison_20260927.json`；投影微测
`deep_kv_projection_micro_20260927.json`（均在`/root/hyrex_results/`）。

## 2026-09-27：固定分支四组复测（完整结束）

结果目录：`/root/hyrex_results/frozen_4arm_4s10t_20260927_v3/`。
该目录README记录六个worktree及commit；各组design.json记录准确源码版本、
命令、环境开关和trace哈希。运行前后检查源码干净且commit未变，使用
PYTHONSAFEPATH=1排除cwd导入遮蔽，并在imports.log验证实际导入路径。
原始工作目录的未提交修改和索引保留不动。v1/v2为启动/隔离诊断，不计入结果。

Qwen3.5-9B BF16 eager，GPU1，CPU缓存2 GiB，四个真实ShareGPT session各10轮，
round-robin单请求执行，先无关warmup，每请求结束后清GPU prefix cache，保留CPU。
各组40请求、39次reset全部成功，共160请求。均值包含全部36续轮，不剔除异常值。
基线仅保留reset成功返回及公共CUDA fallback流顺序正确性补丁，因此应称
“原始恢复策略＋公共正确性修复”，不能称完全零修改的原生二进制。

| 配置 | 续轮平均TTFT ms | 相对基线延迟变化 | 比基线快的续轮数 |
|---|---:|---:|---:|
| 原始对齐恢复 | 167.0922 | — | — |
| 独立存储/传输，但Full恢复截到SSM边界 | 171.4272 | +2.59% | 16/36 |
| 深KV、仍融合QKV投影 | 174.6614 | +4.53% | 12/36 |
| 深KV、Q-only＋去cat | 172.4136 | +3.18% | 13/36 |

优化组相对未优化深KV组减少2.2478 ms（1.29%），相对浅恢复仍增加0.9864 ms，
相对基线增加5.3214 ms。本轮未测到相对原始恢复的净加速。只是各组一次顺序运行，
不能证明统计显著性或所有负载均无收益，也不能将配置均值差当作独立阶段计时。

三段消融差值：改变恢复组织但不加深命中 +4.3350 ms；再加深命中但不省投影
+3.2342 ms；再启用投影优化 -2.2478 ms。这是配置对比，不是纯H2D/纯GEMM因果分解。

四组40个首token文本全部一致。基线与浅组逐请求命中长度相同，两种深KV组命中
长度也逐请求相同；平均多267.556 token，相当于多8.361 MiB Full KV。首token
一致是有限正确性检查，不代表多token输出或逐层状态的完整正确性证明。

预先使用过的512-token gap例子（Ud3L2sd_124，turn_index=6，输入1191）：

| 配置 | API cached_tokens | TTFT ms |
|---|---:|---:|
| 基线 | 528 | 218.52 |
| 浅恢复 | 528 | 216.07 |
| 深KV融合投影 | 1040 | 243.00 |
| 深KV优化投影 | 1040 | 219.31 |

这里API缓存1040不等于跳过1040 token的完整前向；state仍从528重放，GDN/MLP
等计算没有省掉。这个既有例子也未显示优化深KV比基线更快。

本轮不启用profiler。TTFT为HTTP发起到首非空SSE文本，启动/warmup/reset不计入；
恢复服务日志不是完整DMA时间。完整阶段归因仍需同请求跨进程时间线和CUDA事件，
不能拿此前不同运行的kernel时间与本轮TTFT相减填充“其他开销”。
