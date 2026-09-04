# CRISP — 评估后决定不复现

**论文**：[CRISP: Cliff-awaRe Input-adaptive Sparse Prefilling with Structural-Mass-Motivated Routing](https://arxiv.org/abs/2609.01925)
（arXiv 2609.01925，Adobe Research + University of Oregon，2026-09-01，EMNLP 2026 Main，16 页）

**结论：不复现。技术方向已在生产引擎中，本方法未落地，且评测口径离生产太远。**

## 论文做了什么

稀疏 prefill attention，对 FlexPrefill 的两处修补：

- 用结构代理 `C_struct` 替掉 JSD 路由（去掉 pooling 和 KL 计算，路由一致率 94.0%–88.1%）
- 用 sink-aware 噪声底阈值替掉累积 γ-阈值

报告 512K 上下文 5.30× 加速（attention-only），retrieval 任务最高 +28.0pp。

## 引擎现状

| 引擎 | 稀疏 attention 支持 |
|---|---|
| vLLM | dual-chunk flash attention + MInference 稀疏（PR #11844）；Qwen2.5-1M 依赖此路径 |
| SGLang | HiSparse backend（v0.5.10, PR #20343）；DSA（DeepSeek Sparse Attention），`--attention-backend dsa`，DeepSeek V3.2 / GLM-5 原生 |
| CRISP | 无开源代码，全文不提任何推理引擎，仅单卡 H100 上的 attention-only 延迟 |

它的两个基线（MInference、FlexPrefill）都已进入生产引擎，CRISP 自己没有。

## 不做的理由

1. **没有代码，没有引擎集成。** 转化路径的第一步就缺失。
2. **评测口径离生产太远。** attention-only、单序列、单卡。生产 prefill 与 continuous
   batching、chunked prefill 混合执行，收益会被稀释多少，论文未讨论。
3. **加速比不是它的强项。** CRISP 报 512K 时 5.30×，而 SGLang 集成的 MInference
   报 8×（口径不同不可直接比，但足以说明 CRISP 的卖点是精度而非速度）。
4. **价值窗口在收窄。** CRISP 属于「给 dense 模型做 training-free 稀疏化」。
   产业界最新路线（DeepSeek V3.2 的 DSA、GLM-5）是在预训练阶段就用稀疏注意力——
   没有动态路由开销，效果更稳。架构层面的改动天花板高于推理时的补丁。
5. **论文自陈的限制。** 只覆盖 prefill 不覆盖 decode；仅验证至 8B；主要改进 VS 路径；
   依赖 attention sink 存在，gated attention 架构下失效。

## 可以直接带走的知识点

**累积覆盖率阈值在长上下文下会累积 O(n) 噪声。** 这是 CRISP 最有价值的部分——它给出了
渐近分析，并解释了 FlexPrefill 为何在 retrieval 任务上崩溃（CRISP 修回 +17.8pp / +28.0pp）。

如果要自己实现稀疏 prefill 的块选择，不要用「选到累积注意力质量达 γ 为止」这种规则。
post-softmax 质量存在断崖，越过断崖后继续选块只是在收集噪声，且噪声量随序列长度线性增长。

这条信息不需要跑实验就能用。

## 该跟的方向

原生稀疏注意力（DSA），而非推理时的 training-free 稀疏化。DeepSeek V3.2 已在 SGLang 主干运行。

## 参考

- [SGLang Attention Backend 文档](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/attention_backend.md)
- [SGLang v0.5.10 Release（HiSparse）](https://github.com/sgl-project/sglang/releases/tag/v0.5.10)
- [vLLM PR #11844](https://github.com/vllm-project/vllm/pull/11844)
- [microsoft/MInference](https://github.com/microsoft/MInference)
- [ByteDance-Seed/FlexPrefill](https://github.com/ByteDance-Seed/FlexPrefill)
