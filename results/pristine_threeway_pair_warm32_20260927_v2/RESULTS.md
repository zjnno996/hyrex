# 9B三组恢复实验：补齐32-token预热后的复测

## 范围与口径

- Qwen3.5-9B，GPU 1，BF16，eager，CPU缓存2 GiB，单请求串行。
- 一个真实ShareGPT会话的1054→1191 token片段，每组独立启动，重复5次。不是4个session×10轮的连续工作负载。
- 每次启动后用无关token预热：651、1191、1708、1922长度各两次，再做1054→1191的无关恢复请求，共10个请求，全部生成32 tokens，启用与测量相同的首token logprobs。
- 每次预热后清GPU缓存；预热结束清GPU和CPU缓存。正式seed→resume之间只清GPU，保留CPU；不同重复之间清CPU，防止上一对污染下一对。
- 正式seed生成1 token，resume生成32 tokens。TTFT从客户端HTTP请求开始，到第一个非空SSE文本片段；不是引擎内核执行时间。
- 所有5次样本均保留，不删除慢样本。三组顺序为baseline、deep、replacement，尚未做交错顺序或多次独立服务启动的统计确认。

## 结果（毫秒）

| 方案 | 保存轮平均TTFT | 恢复轮平均TTFT | 恢复轮相对baseline | 两轮TTFT均值之和 |
|---|---:|---:|---:|---:|
| 未修改vLLM＋LMCache，对齐恢复 | 197.398 | 238.502 | 基准 | 435.900 |
| Full KV独立16-token索引＋粗粒度state replay | 214.898 | 246.326 | 慢3.280% | 461.224 |
| 将最后coarse state替换到Full KV尾部 | 263.864 | 256.032 | 慢7.350% | 519.896 |

两轮TTFT之和不是完整对话耗时，不包含32-token生成完成时间。当前实现没有测得净TTFT加速；小样本、有波动，不能据此断言方法本身没有优化空间。

| 方案 | 保存轮5次TTFT | 恢复轮5次TTFT | 恢复轮样本标准差 |
|---|---|---|---:|
| baseline | 216.24, 203.07, 171.80, 207.36, 188.52 | 228.81, 253.11, 239.87, 261.69, 209.03 | 20.709 |
| deep | 174.94, 230.98, 245.56, 212.67, 210.34 | 242.69, 274.84, 252.42, 242.04, 219.64 | 19.962 |
| replacement | 283.16, 227.08, 272.06, 235.39, 301.63 | 228.78, 242.33, 256.60, 240.42, 312.03 | 32.826 |

## 校验与机制

- 每组10条正式记录、9条请求间GPU reset成功记录；所有正式seed的cached_tokens为0。
- baseline恢复命中528；deep/replacement的Full KV命中1040。
- 以API请求日志分隔10个预热请求与10个正式请求，正式阶段没有发现`JIT compilation during inference`警告。此检查不是完整的编译事件追踪。
- 两个实验组的所有生成文本与baseline对应记录相同，包括5次32-token续写。这是样例级正确性验证，不是完整模型正确性证明。
- replacement日志5次确认`state=1040 full=1040 base=0`，保存边界依次为1040、1184，各5次。没有跳过新checkpoint维护。

| 方案 | 恢复Full KV | 恢复state | 剩余前向token范围 |
|---|---:|---:|---|
| baseline | 528 | 528 | 528:1191，共663 tokens |
| deep | 1040 | 528 | 仍需从528前向，复用其中512 tokens的Full KV；不是跳过全部层 |
| replacement | 1040 | 1040 | 1040:1191，共151 tokens，按144＋7分段保存新tail |

replacement的10次`SAVE_BARRIER_MS`均值为11.156 ms，范围9.444–14.064 ms。它只计保存同步等待这一段，不是完整checkpoint维护成本，也不能单独解释全部TTFT差额。额外前向分段、传输、查找、同步等仍需独立归因；本轮不做因果占比推断。

replacement仍是隔离seed/resume实验：旧tail保留到这对请求后的CPU clear，未实现连续多轮下的旧tail回收。因此不能把本结果称为连续多轮、恒定state容量的完整缓存策略。

## 复现与环境

命令、源码commit及trace哈希见`design.json`和每组`command.json`。测量阶段未修改恢复源码；此次按ponytail最小修改原则，仅扩展已有预热流程和记录预热生成长度。

三组使用同一份由官方未修改LMCache源码编译的CUDA12.8扩展，导入检查确认绑定native kernel。扩展SHA256：`808afab59217242426918b5b6024f16c6a9ec686d5f56b4599ee0289561a320c`。

原始记录：各组`online.jsonl`、`warmup.jsonl`、`resets.jsonl`、`vllm.log`、`lmcache.log`；均值见`summary.json`。

上一轮`pristine_threeway_pair_20260927_v1`包含首次decode JIT峰值，不用其87%左右的表面加速比作为性能结论。
