# 阶段 5：训练集答案生成、评估与伪标签

## 协议边界

本阶段只使用冻结的 186 条 phase4 训练检索结果、62 个 phase2 Chunk 和 62 条
phase1 训练问题。生成阶段只投影问题 ID 与问题正文，生成全部结束后评估脚本才
加载标准答案和标准证据。未读取或运行测试集，未重跑检索器，未训练分类器。

三种方法统一使用 `deepseek-v4-pro`、temperature=0、thinking
关闭、同一系统 Prompt、最多 5 个 Chunk。Path 的 10 条非 success 结果未调用
LLM，直接记为证据不足。Neighborhood/Path 只附加 phase4 结果中已有的图上下文。

Answer Correctness 直接复用固定 GraphRAG-Bench 官方实现：LLM 将答案与参考答案
拆为陈述并计算事实性 F1（权重 0.75），BGE 余弦相似度映射至 [0,1]（权重
0.25）；多参考答案取最高分。证据覆盖按归一化包含匹配，失败时以 token recall
≥0.8 判为覆盖。

## 方法结果

| 方法 | 平均正确性 | 平均证据覆盖 | 可接受率 | 上下文字符 | Prompt tok | Completion tok | 总 tok | 检索 ms | 生成 ms | 无效引用 | 胜出 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vector | 0.5090 | 0.5766 | 20.97% | 11416.5 | 3417.6 | 70.4 | 3488.1 | 0.047 | 2372.7 | 0 | 25 |
| neighborhood | 0.6430 | 0.6277 | 32.26% | 24055.7 | 8065.3 | 71.7 | 8137.0 | 2.370 | 2564.6 | 0 | 23 |
| path | 0.4701 | 0.5228 | 20.97% | 27457.8 | 9773.4 | 63.5 | 9837.0 | 3.628 | 2195.7 | 0 | 14 |

阈值在运行前固定为 Answer Correctness ≥
0.75 且 Evidence Coverage ≥
0.5。可接受时按 vector、neighborhood、
path 的难度顺序选最简单方法；三者均不可接受时依次比较正确性、证据覆盖、总
tokens 和难度，并保留为 `best_available`。

## 标签与审计

- 标签分布：`{'vector': 25, 'neighborhood': 23, 'path': 14}`
- 选择模式：`{'acceptable': 23, 'best_available': 39}`
- 类别失衡风险：无。
- 生成有效缓存：176；直接跳过：10
- 最后一次离线生成重建 API 调用：0
- 官方评审有效缓存：302；已解决的结构失败审计缓存：2；最后一次离线评估 API 调用：0
- 官方评审有效缓存 token：`{'completion_tokens': 38159, 'prompt_tokens': 149604, 'total_tokens': 187763}`
- 输入 SHA-256：`{'train_questions': {'path': 'data/processed/phase1/train_questions.jsonl', 'sha256': '84b25f68b6658c11c92af381f684e14efb09444d28e34475671bcf468c697332'}, 'chunks': {'path': 'data/processed/phase2/chunks.jsonl', 'sha256': 'a081ba0f863a37e71fae17a8c7b42f2aa46019e8d62ca1d98e818e4988811427'}, 'retrieval_results': {'path': 'data/processed/phase4/train_retrieval_results.jsonl', 'sha256': '71b69cb61a302129d0891431646f93e40620b8c33337cd2f2ccda0845be231f1'}}`

非耗时产物已通过重复运行一致性验证。阶段 1～4 产物未修改。下一步可训练路由
分类器，但本阶段按要求在此停止。
