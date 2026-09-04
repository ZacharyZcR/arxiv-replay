# arxiv-replay

论文复现与研究。

## 说明

对感兴趣的论文做最小可运行的复现，并记录复现过程中的观察与偏差。

每个复现独立成目录，包含：

- 论文出处与核心思路
- 复现代码
- 结果与原文的对照

## 目录

| 目录 | 论文 | 状态 | 一句话结论 |
|---|---|---|---|
| [`hllm/`](hllm/) | [hLLM: Single Pass Decoding for Generative Reranking](https://arxiv.org/abs/2609.01807) (Meta, 2026) | 已归档 | 有效的部分（prefill-only 解码）是 vLLM 早有的成熟能力，论文冠名的匈牙利指派实测无效 |

## 评估后未复现

关注点是学术成果能否转化到工业推理栈，所以先查技术是否已在 vLLM / SGLang 中落地。
判定不值得投入的，留一份记录，省得以后重复评估。

| 记录 | 论文 | 不做的原因 |
|---|---|---|
| [`triage/crisp-2609.01925.md`](triage/crisp-2609.01925.md) | [CRISP](https://arxiv.org/abs/2609.01925) (EMNLP 2026) | 无代码无引擎集成；attention-only 单序列口径；产业界已转向原生稀疏注意力 |

## License

MIT
