# arxiv-replay

论文复现与研究。

立场是**学术成果能否转化到工业推理栈**，所以每篇先问三件事：有没有开源代码、
有没有进 vLLM / SGLang、评测口径离生产有多远。够格的才动手复现。

## 优化手段索引

这是本仓库的主表。一篇论文往往捆绑多个优化，需要逐条拆开判定——
论文整体的成色，和其中某一项技术是否可用，是两回事。

**判定口径**：`实测` 表示本仓库跑过实验；`评估` 表示仅做文献与工程现状分析，未跑代码。

| 优化手段 | 出处 | 判定 | 依据 | 局限性 | 验证 |
|---|---|---|---|---|---|
| prefill-only 解码（不逐 token 生成，一次前向读出全部候选分数） | hLLM | **有效，但非原创** | 实测 19.4× 加速（1568.5 → 80.8 ms），质量不降反升 | 该机制在 vLLM v0.7.0 `/rerank` 中已是标准能力，早于论文；论文未引用亦未对比 | 实测 2026-09-05 |
| 排序输出的合法性保证 | hLLM | **有效，但非匈牙利独家** | 自回归解码合法率仅 2.6%，一次前向恒为 100% | `argsort` 输出天然是排列，同样 by construction，不需要求解器 | 实测 2026-09-05 |
| 匈牙利指派解码（最优二部图匹配） | hLLM | **无效** | 同一矩阵换 `argsort`：R@10 高 46%、NDCG@10 高 31%，延迟相同 | 结论成立于**单正例真值监督**下（矩阵仅第 0 列有信号）。完整排列监督下可能翻转，但论文未做此对照 | 实测 2026-09-05 |
| Sinkhorn 松弛作为训练信号 | hLLM | **无效** | 比朴素 CE 更差（R@10 0.3927 vs 0.4413） | 同上，未在教师蒸馏条件下验证 | 实测 2026-09-05 |
| Self-Attention 打分头（L=2，item 间交互） | hLLM | **无效** | 4.73 M 参数探针峰值 R@1 0.150，1 K 参数线性探针 0.148；随后严重过拟合 | 单正例监督、4000 训练样本下的结论 | 实测 2026-09-05 |
| N×K 物品-位置矩阵表示 | hLLM | **无效** | 仅第 0 列携带信号，其余 K−1 列是噪声负担 | 同上 | 实测 2026-09-05 |
| 秩 1 打分矩阵下的指派求解 | hLLM | **等价于排序** | `M_ij = s_i·w_j` 且 w 单调时，重排不等式保证最优指派 ≡ `argsort(s)`；300 次试验 Kendall τ = 1.0000 | 数学结论，无条件成立 | 实测 2026-09-04 |
| 逐物品独立打分头 + 指派求解 | hLLM | **结构性失效** | 矩阵无列变化时全部 N! 个排列目标值相同（实测差距 0.00e+00），求解器输出零排序信息 | 解释了论文 Linear Probe 变体表现差的真实原因（论文归因于 capacity gap） | 实测 2026-09-04 |
| 矩形指派做 top-K 部分排序 | hLLM | **可行且更快** | O(N·K·min(N,K))；N=1000 时 top-10 为 0.023 ms，全排列 29.6 ms，快 1295× | 论文未讨论此场景，属文档缺失而非方法缺陷 | 实测 2026-09-04 |
| 累积覆盖率阈值做稀疏块选择（γ-cumsum） | CRISP 指出其缺陷 | **已知有害** | post-softmax 质量存在断崖，越过后继续选块只收集噪声，噪声量随序列长度线性增长 O(n) | 论文给出渐近分析并解释 FlexPrefill 在 retrieval 上崩溃的原因（修回 +17.8pp / +28.0pp）；本仓库未独立验证 | 评估 2026-09-05 |
| sink-aware 噪声底阈值 | CRISP | **未验证，有理论价值** | 声称 512K 时 5.30× attention-only 加速 | 无开源代码；attention-only 单序列口径；仅覆盖 prefill 不覆盖 decode；仅验证至 8B；依赖 attention sink 存在，gated attention 架构下失效 | 评估 2026-09-05 |
| `C_struct` 结构代理替代 JSD 路由 | CRISP | **未验证，增量优化** | 去掉 pooling 与 KL 计算，路由一致率 94.0%–88.1% | 同上 | 评估 2026-09-05 |
| 模型自声明注意力范围 + block table 重写 | DA | **未验证，数字存疑** | 声称 attended tokens 减少 31–52%，精度降 1.27–2.75pp | wall-clock 为 roofline 估算而非实测，且排除 prefill；global 模式占 80%+ attended tokens 且随上下文增长；多跑约 33% decode step；不兼容 thinking mode；小模型完全失效；无 batching 实验；无代码 | 评估 2026-09-05 |
| hook attention metadata builder 改写 block table（作为集成路径） | DA | **有效的集成方式** | 不改 kernel、不改 scheduler；vLLM V1 有 `supports_update_block_table` 原生路径 | 这是对「集成成本」的判定，与该方法本身的收益无关 | 评估 2026-09-05 |

## 论文清单

| 论文 | 状态 | 记录 | 一句话结论 |
|---|---|---|---|
| [hLLM: Single Pass Decoding for Generative Reranking](https://arxiv.org/abs/2609.01807)<br/>Meta, 2026 | 已复现，已归档 | [`hllm/`](hllm/) | 有效的部分（prefill-only 解码）是 vLLM 早有的成熟能力，论文冠名的匈牙利指派实测无效 |
| [CRISP: Cliff-awaRe Input-adaptive Sparse Prefilling](https://arxiv.org/abs/2609.01925)<br/>Adobe Research, EMNLP 2026 | 仅评估 | [`triage/crisp-2609.01925.md`](triage/crisp-2609.01925.md) | 无代码无引擎集成；attention-only 单序列口径；产业界已转向原生稀疏注意力（DSA） |
| [Language Models Can Control Their Own Attention](https://arxiv.org/abs/2609.02737)<br/>KAIST AI + Google DeepMind, 2026 | 仅评估 | [`triage/da-2609.02737.md`](triage/da-2609.02737.md) | 集成路径是看过最干净的，但收益全是 roofline 估算，且上下文越长节省越少 |

## 说明

每个复现独立成目录，包含论文出处与核心思路、复现代码、结果与原文的对照。
评估后判定不值得复现的，在 `triage/` 留一份记录，省得以后重复评估。

## License

MIT
