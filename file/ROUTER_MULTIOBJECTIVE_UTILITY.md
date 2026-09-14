# 多目标效用路由实验

## 1. 结论

多目标效用路由在同一 24 题冻结测试集上，同时超过当前保留的 120 题固定降权路由：

| 方法 | Answer Correctness | Evidence Recall | 路由 V/L/P | 平均延时 |
|---|---:|---:|---:|---:|
| 120题固定降权路由 | 0.6042 | 0.6049 | 8/5/11 | 3531.2 ms |
| 300题固定降权路由 | 0.5733 | 0.6257 | 8/6/10 | 3236.4 ms |
| 多目标效用路由 | **0.6762** | **0.6424** | 7/7/10 | **3296.4 ms** |

相对 120 题固定降权版本，Answer Correctness 提高 0.0721，Evidence Recall 提高
0.0375，同时平均检索与生成总延时降低约 234.8 ms。相对三个固定方法，它取得最高的
Answer Correctness；Evidence Recall 仅比固定 Vector 的 0.6521 低 0.0097。

## 2. 方法

分类器不再把每道题压缩成一个硬银标，而是在 276 道开发题上使用六个连续监督目标：三种
方法各自的 Answer Correctness 和 Evidence Recall。输入仍只有问题的 BGE-M3 CLS 向量，
不得使用答案、证据、题型或检索结果作为推理特征。

模型为多输出 Ridge 回归。对问题预测六个分数后，按照下式选择方法：

```text
Utility(method) = predicted Answer Correctness
                + λ × predicted Evidence Recall
```

固定候选为 Ridge `alpha ∈ {0.001, 0.01, 0.1, 1, 10, 100}`，
`λ ∈ {0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.50, 1.0}`。使用按题型分层的固定 5 折 OOF
预测。候选必须在 OOF Answer Correctness 和 Evidence Recall 上同时不低于当前 300 题
固定降权路由，再最大化两个绝对增益中的较小值，其次最大化增益之和。测试集不参与模型
拟合或候选选择。

最终由开发集选择：

```text
Ridge alpha = 0.001
Evidence Recall weight λ = 0.15
```

## 3. 开发集 OOF

| 方法 | OOF Answer Correctness | OOF Evidence Recall | 联合调和均值 | 路由 V/L/P |
|---|---:|---:|---:|---:|
| 300题固定降权硬标签路由 | 0.6203 | 0.5930 | 0.6063 | 109/59/108 |
| 多目标效用路由 | **0.6306** | **0.6053** | **0.6177** | 98/81/97 |
| 变化 | +0.0103 | +0.0123 | +0.0113 | -11/+22/-11 |

若仍用 `threshold=0.60, margin=0.10` 的硬银标衡量分类，多目标路由的 OOF Accuracy 为
0.4022、Macro-F1 为 0.3728。该值不是本模型的选择目标；相比之下，端到端 OOF 的两个
实际效用指标都得到提升。

## 4. 冻结测试集

| 方法 | Answer Correctness | Evidence Recall | 联合调和均值 |
|---|---:|---:|---:|
| Vector | 0.5845 | **0.6521** | 0.6164 |
| LightRAG | 0.5337 | 0.5313 | 0.5325 |
| PathRAG | 0.5815 | 0.5215 | 0.5499 |
| 120题固定降权路由 | 0.6042 | 0.6049 | 0.6045 |
| 300题固定降权路由 | 0.5733 | 0.6257 | 0.5984 |
| 多目标效用路由 | **0.6762** | 0.6424 | **0.6589** |

相对 120 题固定降权路由的逐题配对 bootstrap（20,000 次，种子 42）：

- Answer Correctness 平均增益 0.0721，95% 区间 `[0.0033, 0.1579]`；6 题改善、15 题
  不变、3 题下降。
- Evidence Recall 平均增益 0.0375，95% 区间 `[0.0000, 0.0958]`；2 题改善、22 题
  不变、0 题下降。

对应硬银标的测试分类 Accuracy 为 0.5417，Macro-F1 为 0.4327，也高于 120 题固定降权
版本的 0.4583/0.3485。但核心结论仍以直接优化的 Correctness/Recall 为准。

## 5. 限制

- 测试集只有 24 题，置信区间仍宽；而且此前实验已经多次查看该测试集，不能继续根据这些
  测试数字调整参数。
- 新方法使用开发集上的生成评测分数作为回归监督，仍会受到 LLM Judge 噪声影响。
- 当前效用没有成本惩罚；平均 token 数与旧路由接近，但将来如果需要严格控制成本，应在
  新的开发集上预先加入成本项，不能根据现有测试集回调参数。
- 正式最终结论应增加一批从未用于设计的新测试题再验证。

## 6. 产物

```text
configs/router_multiobjective_utility.json
src/router/run_multiobjective_utility_router.py
result/shared_expanded/p1/multiobjective_utility_router/
test/test_multiobjective_utility_router.py
```

本实验完全离线，未调用检索器、生成模型、评测模型或外部 API。
