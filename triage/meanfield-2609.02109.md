# MeanField 调度代理 — 评估后决定不复现

**论文**：[MeanField Surrogate Modeling for Scalable Runtime Scheduling of Concurrent Heterogeneous AI Inference on Shared GPUs](https://arxiv.org/abs/2609.02109)
（arXiv 2609.02109，Youssef Ennouri、Soonhoi Ha / 首尔大学，2026-09-01）

**结论：问题选得好、建模简化合理，但对 LLM 的建模粒度太粗——不区分 prefill/decode、
不建模 KV cache。加上无代码、单卡小规模验证，转化价值有限。**

## 论文做了什么

共享 GPU 上多模型并发推理的运行时调度。核心是一个建模简化：预测每个模型的性能时，
不建模模型间所有两两交互，只用「本模型的本地配置 + GPU 聚合状态（显存占用、利用率、
负载水平）」。profiling 复杂度从 O(k^N) 降到 O(N·k)，每个模型训一个轻量 MLP（64-32），
再接遗传算法做配置搜索。

RTX 3090 单卡，Qwen2.5（FP16 / AWQ）+ YOLO11，N=2–6。
R² ≈ 0.96，决策延迟 26 ms 中位数（实测），穷举 131 ms，0% SLA 违规。

## 定位澄清

它做的不是请求调度（那是 vLLM scheduler 的职责），而是**部署配置搜索**——
N 个模型共享一张卡时，每个模型该用哪个量化版本、并发度设多少。
`request_concurrency ∈ {1,2,4}`、YOLO 的 `skip_rate` / `imgsz` 都是运行时可调参数，
不需要重载模型，所以在线调整是合理的。这是真实的运维问题。

## 不做的理由

1. **LLM 建模粒度太粗。** 把 LLM 当黑盒，不区分 prefill/decode 阶段，不建模 KV cache
   占用。作者自陈：*"Future work will extend the LLM workload model to capture
   prefill/decode phases and KV-cache occupancy."* 但现代 LLM serving 的性能正是被
   这两件事主导（continuous batching、chunked prefill、PagedAttention）。
   R² 0.96 只说明它在那个特定窄负载上拟合得好，换负载形态未必成立。

2. **假设 GA 搜索期间 GPU 聚合状态固定**，不预测状态演化。真实负载动态波动时该假设失效。

3. **无开源代码。**

4. **无 Limitations 章节。** 唯一的前瞻性约束只出现在结论里。

5. **验证规模小。** 单张 RTX 3090，N ≤ 6，未涉及多卡。

## 可以带走的知识点

**多模型共享 GPU 时，用「聚合状态」代替「两两交互建模」，可把 profiling 成本
从指数降到线性。** 论文测得达到 R² ≥ 0.95 所需样本量约 n* ≈ 20N（N∈{2..6}
对应约 40/60/80/100/120 个样本）。

如果以后要做多模型混布的容量规划，这个近似值得一试——但 LLM 那部分必须自己补上
prefill/decode 与 KV cache 两个维度，否则模型在真实 serving 负载下不可信。
