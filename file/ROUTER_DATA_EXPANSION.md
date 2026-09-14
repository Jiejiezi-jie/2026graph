# 路由分类器改进步骤二：扩充问题数量

## 1. 结论

本步骤将开发集从 96 道唯一问题扩充到 276 道，增幅为 187.5%，同时保持原 24 道测试题
及其 ID 完全不变。主分类指标没有提升：当前开发集最优特征仍为 BGE-M3，但 5 折准确率
从 0.5000 降至 0.4022，Macro-F1 从 0.4164 降至 0.3650。因此不能宣称“只增加问题
数量”解决了路由分类问题。

端到端 RAG 指标出现了不同趋势：扩充后的 BGE-M3 路由器在冻结测试集上的 Answer
Correctness 从 0.5946 升至 0.6115，Evidence Recall 从 0.5382 升至 0.6049。不过这两个
数值来自仅 24 题的最终测试集，只能作为消融观察，不能用于继续选模型。

## 2. 固定实验设计

- 数据池：GraphRAG-Bench Medical，共 2,062 道问题；本实验仍只使用 Fact Retrieval 和
  Complex Reasoning。
- 原实验：120 题，其中开发集 96 题、测试集 24 题。
- 新增数据：180 题，Fact Retrieval 90 题、Complex Reasoning 90 题；其中 144 题标为
  train、36 题标为 validation，不新增测试题。
- 扩充后：总计 300 题，开发集 276 题，冻结测试集仍为原 24 题。
- 抽样种子：4242；分类与 5 折种子：42。
- 标签规则：`Answer Correctness >= 0.60` 且不低于本题最高正确率减 0.10，再从合格
  方法中选择 token 成本最低者；无合格方法时保留最高质量回退标签。
- 特征候选：TF-IDF 与 BGE-M3；模型仍为逻辑回归；按题型分层 5 折，超参数只按开发集
  OOF Macro-F1、再按 OOF Accuracy 选择。
- 生成与评分：当前 main 版本、DeepSeek `deepseek-chat`、温度 0；官方评估指纹与旧实验
  完全相同，为 `209096a999b2a761cfd08eb9ddadd88a804f51ecd79acf8a61a8b592ed9fd851`。
- 图输入：LightRAG 和 PathRAG 复用字节一致的共享 LightRAG 图，SHA-256 为
  `4c2f2c35c00b94aa8eb4a728572ffbebc66ec31c2ad5caa6c58546613fdc1a21`；Vector 不使用图。

新增样本清单在任何 API 调用前已冻结。清单 SHA-256 为
`796430b2a18ab1490c5fbfdd33f9a90191afbe46ecf6b5a9cceccb3b7170f927`，与原 120 题零重叠。

## 3. 分类结果

| 特征 | 开发题数 | 5折准确率 | 5折 Macro-F1 | 测试准确率 | 测试 Macro-F1 | 测试路由 V/L/P |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF，扩充前 | 96 | 0.4479 | 0.3751 | 0.4167 | 0.3278 | 12/8/4 |
| TF-IDF，扩充后 | 276 | **0.4964** | 0.3317 | **0.5000** | 0.2984 | 19/0/5 |
| BGE-M3，扩充前 | 96 | **0.5000** | **0.4164** | **0.4583** | **0.3485** | 10/5/9 |
| BGE-M3，扩充后 | 276 | 0.4022 | 0.3650 | 0.4167 | 0.3175 | 9/5/10 |

这里的 `V/L/P` 是预测路由分布。扩充后的 TF-IDF 虽然 Accuracy 上升 4.85 个百分点，
却完全不预测 LightRAG，导致 Macro-F1 下降 4.34 个百分点，因此不算有效提升。按开发集
Macro-F1 选择，扩充后仍应选 BGE-M3；其相对扩充前的变化为：

| 指标 | 扩充前 | 扩充后 | 变化 |
|---|---:|---:|---:|
| 5折 Accuracy | 0.5000 | 0.4022 | -0.0978 |
| 5折 Macro-F1 | 0.4164 | 0.3650 | -0.0514 |
| 测试 Accuracy | 0.4583 | 0.4167 | -0.0417 |
| 测试 Macro-F1 | 0.3485 | 0.3175 | -0.0310 |

BGE-M3 扩充后的 5 折混淆矩阵如下，行是真实银标、列是预测，顺序为
Vector/LightRAG/PathRAG：

```text
[[56, 38, 41],
 [18, 10, 14],
 [34, 20, 45]]
```

逐类 OOF F1 为 Vector 0.4609、LightRAG 0.1818、PathRAG 0.4523。LightRAG F1 比扩充前
的 0.1290 有所提高，但 Vector 和 PathRAG 同时下降，所以总体 Macro-F1 仍降低。

## 4. 银标分布与原因分析

| 开发集 | Vector | LightRAG | PathRAG | all_failed |
|---|---:|---:|---:|---:|
| 扩充前 96 题 | 48 | 19 | 29 | 20 |
| 新增 180 题 | 87 | 23 | 70 | 43 |
| 扩充后 276 题 | 135 | 42 | 99 | 63 |

问题类型数量已经平衡，但方法银标仍不平衡：扩充后占比约为 Vector 48.9%、LightRAG
15.2%、PathRAG 35.9%。逻辑回归的最优 BGE-M3 候选已经使用 `class_weight=balanced`，
所以本轮下降不能简单归因于忘记做类别权重。

更主要的原因是：银标由某次检索、答案生成、官方裁判分数与成本共同决定，不完全等价于
问题文本本身的“结构复杂度”。扩充后 `all_failed` 仍有 63/276（22.8%），这些回退标签
噪声较高；同一种表述的问题也可能因为知识图覆盖与检索命中不同而对应不同方法。BGE-M3
CLS 向量加线性边界只能看到问题文本，无法直接观察索引覆盖和检索置信度。旧 96 题上的
0.5000 也可能是小样本下偏乐观的估计，276 题结果更稳定地暴露了这一限制。

因此，本轮证据表明“数量不足”确实是原问题的一部分，但不是主要瓶颈；继续无目标地随机
增加问题，成本较高且不保证提高 5 折指标。下一步更合理的是在不接触测试集的前提下，先
提高标签可学习性：例如做联合分层以稳定每折标签比例、将检索可用性作为规则特征，或把
硬三分类改成预测三种方法效用再按成本决策。是否进入下一步应另行决定。

## 5. 冻结测试集上的端到端观察

| 路由器 | Answer Correctness | Evidence Recall | 路由 V/L/P |
|---|---:|---:|---:|
| 扩充前 BGE-M3 | 0.5946 | 0.5382 | 10/5/9 |
| 扩充后 BGE-M3 | **0.6115** | **0.6049** | 9/5/10 |
| 变化 | +0.0169 | +0.0667 | -1/0/+1 |

分类银标准确率下降、端到端质量上升并不矛盾：某些“错标签”路由到了质量相近甚至更好的
方法。它再次说明硬银标 Accuracy 不是唯一目标，但本步骤的主要验收指标仍是开发集 5 折，
所以总体结论仍为“没有分类提升”。

## 6. 沿用上一步固定样本权重

为避免把“扩充数据”和“退回未加权训练”混在一起，又使用上一步端到端指标曾改善的固定
权重做了离线复核：唯一合格方法的 `stable=1.0`，多个合格方法的 `ambiguous=0.5`，没有
方法合格的 `all_failed=0.3`。银标签仍是三分类硬标签，阈值仍为 0.60、容差仍为 0.10；
这里是样本权重，不是软标签。本次只复用已缓存的 300 题官方评分和 BGE-M3 向量，新增
API 调用为 0。

| 加权 BGE-M3 | 开发题数 | 5折准确率 | 5折 Macro-F1 | 测试准确率 | 测试 Macro-F1 | 测试路由 V/L/P |
|---|---:|---:|---:|---:|---:|---:|
| 扩充前 | 96 | 0.4167 | 0.3470 | 0.4583 | 0.3485 | 8/5/11 |
| 扩充后 | 276 | 0.4167 | **0.3720** | 0.3333 | 0.2587 | 8/6/10 |
| 变化 | +180 | 0.0000 | **+0.0250** | -0.1250 | -0.0898 | 0/+1/-1 |

与扩充后的未加权 BGE-M3 相比，加权使 5 折准确率从 0.4022 升到 0.4167，Macro-F1
从 0.3650 升到 0.3720，分别提高 0.0145 和 0.0070。因此，沿用原固定权重确实对开发集
交叉验证有小幅帮助。但它仍低于扩充前未加权基线的 Macro-F1 0.4164，也没有改善冻结
测试集分类结果。

端到端指标同样需要保守解释：扩充前加权路由的 Answer Correctness/Evidence Recall 为
0.6042/0.6049，扩充后为 0.5733/0.6257，即正确性下降 0.0308、证据召回上升 0.0208。
所以本轮结论是“固定降权缓解了一部分开发集标签噪声”，而不是“扩充加权方案整体已经
超过旧方案”。扩充后最优参数为 `C=0.5, class_weight=balanced`；开发集权重类别为
stable 108、ambiguous 105、all_failed 63。

为保证可复现，以上数字使用与原加权实验一致的 scikit-learn 1.7.2 生成。环境中的
scikit-learn 1.9.0 会因 LogisticRegression 实现版本变化产生不同结果，因此正式结果
记录并固定 1.7.2。

## 7. 调用量、耗时与产物

新增 180 题的生成调用为 Vector 180 次、LightRAG 360 次、PathRAG 360 次；三种方法的
官方裁判各 720 次。生成阶段平均单题总耗时分别为 1.102 秒、2.280 秒和 5.672 秒。
从冻结样本清单到最终分类汇总约 35 分 28 秒，其中包括一次评估指纹审计后的安全重启；
三路生成和三路评分均使用断点续跑。

主要产物：

```text
configs/router_data_expansion.json
src/backend/common/prepare_router_data_expansion.py
src/backend/common/run_official_expansion.py
src/router/analyze_router_data_expansion.py
configs/router_weighted_data_expansion.json
src/router/analyze_weighted_router_data_expansion.py
result/shared_expanded/splits/expansion_questions.jsonl
result/shared_expanded/splits/expansion_manifest.json
result/shared_expanded/p1/{vector,lightrag,pathrag}.jsonl
result/shared_expanded/p1/{vector,lightrag,pathrag}_evaluated.jsonl
result/shared_expanded/p1/router_data_expansion/
result/shared_expanded/p1/weighted_router_data_expansion/
tests/test_router_data_expansion.py
```

API 密钥未写入配置、结果或报告。两条裁判瞬时异常最终均已通过单条重试，正式结果中无
失败行。
