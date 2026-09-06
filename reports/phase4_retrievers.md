# 阶段 4：共享数据基础上的三个检索器

## 实现边界

本阶段在完全相同的 62 个 phase2 Chunk 和同一份 phase3 知识图谱上实现
`VectorRetriever`、`NeighborhoodRetriever` 和 `PathRetriever`。后两者分别是
LightRAG Local 风格的一跳邻域检索与 PathRAG 风格的关系路径检索课程简化版，
**不是** LightRAG 或 PathRAG 的完整复现。

本阶段未调用生成式 LLM、未生成答案、未评价答案正确率，未读取测试集，也没有
使用训练题中的 `answer`、`evidence`、`evidence_triple`。问题加载器只投影
`id`、`question`、`question_type`，其中题型仅用于本报告分组。

## 配置与索引

- 随机种子：42
- 共享 Chunk 预算：5
- 最多种子实体：4
- 实体语义阈值：0.55
- 邻域跳数：1
- 最大路径跳数：3；每实体对最多路径：3
- Path 排除关系：['AUTHORED_BY', 'PORTRAYED_BY']
- 嵌入模型：`BAAI/bge-small-en-v1.5`
- 模型 revision：`5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`
- 权重 SHA-256：`3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad`
- 向量维度：384；pooling：cls；L2 normalize：True
- 查询前缀：`Represent this sentence for searching relevant passages: `；文档不加前缀并保持原文。
- 编码环境：Transformers 5.13.0.dev0，PyTorch 2.6.0+cu124，NumPy 2.4.4，NetworkX 3.6.1。

Vector 只对归一化 Chunk 向量进行余弦排序，不接触图。Neighborhood 先做规范名/
别名匹配，再用实体名称、别名和描述向量补充实体链接，展开一跳后仅从节点来源、
边来源与 `chunk_entity_map` 收集 Chunk。Path 在排除元数据边后的无向投影上搜索
不超过三跳的简单路径；失败时返回明确状态，不进行向量回退。

## 62 道训练问题覆盖率

“同一分量”按至少两个链接实体中存在一对在排除元数据边后的图中连通计算；这里
只测可检索覆盖，不使用答案或标准证据。

| 分组 | 问题数 | ≥1 实体 | ≥2 实体 | 同一分量 | Vector 非空 | 邻域非空 | 找到路径 |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 62 | 100.00% | 96.77% | 83.87% | 100.00% | 100.00% | 83.87% |
| Fact Retrieval | 34 | 100.00% | 94.12% | 79.41% | 100.00% | 100.00% | 79.41% |
| Complex Reasoning | 28 | 100.00% | 100.00% | 89.29% | 100.00% | 100.00% | 89.29% |

## 状态、返回量与耗时

| 检索器 | 状态计数 | 平均 Chunk 数 | 均值 ms | P50 ms | P95 ms |
|---|---|---:|---:|---:|---:|
| vector | {'success': 62} | 5.0000 | 0.047 | 0.046 | 0.054 |
| neighborhood | {'success': 62} | 4.9516 | 2.370 | 2.406 | 2.840 |
| path | {'disconnected': 8, 'insufficient_entities': 2, 'success': 52} | 4.0806 | 3.628 | 3.267 | 6.104 |

- Path 失败原因：`{'disconnected': 8, 'insufficient_entities': 2}`
- 至少两个实体的问题中，同一分量比例：86.67%
- 所有图检索结果均记录 `vector_fallback_used=false`。

## 确定性与风险

- 结果结构哈希：`4e4d66be4b4bd3cbf18b4519e50bf3137d77fa6c7bb68269f78e0392ed50bc5e`
- 诊断结构哈希：`334ab031ff88b1c8fdefe1ed51d9c4f4d01cb11015d43b69e90d0e676072be93`
- 除实测耗时字段外，重复运行结果一致。
- phase3 图有较多孤立节点且关系过滤较保守，因此 Path 覆盖率可能显著低于向量
  覆盖率；本阶段如实保留失败状态，没有反向修改图谱。
- 实体链接仍可能受作品名/动物名 `Dandy Dick` 等同形别名影响；语义阈值和最多
  四个种子实体均固定在配置中，未根据答案或测试集调参。
- 下一步可以在这三个共享检索器之上评估检索证据，但本阶段不开始答案生成、
  伪标签或分类器训练。
