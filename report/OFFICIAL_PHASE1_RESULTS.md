# Official Phase-one Results

完成时间：2026-09-05（Asia/Shanghai）。本报告只统计 `results_official/`
中的 Vector RAG、HKUDS LightRAG 和 BUPT-GAMMA PathRAG；原 `results/`
中的 `light_proxy`、`path_proxy` 和 `answer_proxy` 仍保留，但不进入下表。

## 实验状态

- P0：两类各 5 题，三个后端均为 10/10 有效答案、非空检索证据。
- P1：Fact Retrieval 60 题、Complex Reasoning 60 题；三个后端均为
  120/120 有效答案和有效官方指标。
- 固定划分：训练 72、验证 24、测试 24，seed=42；三个后端的问题 ID
  完全一致。
- 生成/抽取：Qwen2.5-VL-7B-Instruct 纯文本模式，temperature=0，答案最多
  256 token。
- Embedding：BAAI/bge-m3，三个后端共用。
- Judge：GraphRAG-Bench 官方指标代码加同一个 Qwen2.5-VL-7B 本地模型；
  事实分类最多 2048 token。生成模型与 judge 同族，存在自评偏差。

## 服务器与模型摘要

| 用途 | 实际选择 | 状态 |
|---|---|---|
| GPU | 6 × RTX 4090（每卡约 23.5 GiB） | P1 三路按 0/1、2/3、4/5 并行 |
| 实体关系抽取 | Qwen2.5-VL-7B-Instruct | 5/5 JSON 探针通过 |
| 答案生成 | Qwen2.5-VL-7B-Instruct | 纯文本模式，5/5 QA 探针通过 |
| Embedding | BAAI/bge-m3 | 1024 维归一化向量 |
| 轻量路由 | TF-IDF + Logistic Regression | 验证集 Macro-F1 选参 |
| T5 路由 | 无本地 T5-Small/Base 权重 | 未运行，不以其他结果代替 |
| 独立 judge | 没有更强的独立本地文本模型 | 使用同族 Qwen，并明确降级说明 |

完整环境审计见 `MODEL_AUDIT.md`，集成边界见
`SERVER_INTEGRATION_NOTES.md`。

## 索引结果与成本

| 后端 | Chunk/节点/边 | 索引时间 | LLM 调用 | 输入 token | 输出 token |
|---|---:|---:|---:|---:|---:|
| Vector RAG | 199 chunks | 18.85 s | 0 | 0 | 0 |
| LightRAG | 2,002 nodes / 163 edges | 137.05 min | 431 | 1,434,011 | 159,369 |
| PathRAG | 1,850 nodes / 675 edges | 112.71 min | 402 | 1,513,294 | 255,094 |

LightRAG GraphML 为 1,092,180 bytes，SHA-256 为
`f58864e06ea669259fd70288193dfe7f2b8edbfe961d3aa571d1776dbef4765d`；
PathRAG GraphML 为 1,045,525 bytes。

## P0 冒烟结果（10 题均值）

| 后端 | Answer Correctness | ROUGE-L | Evidence Recall | 在线 token | 延迟 |
|---|---:|---:|---:|---:|---:|
| Vector RAG | 0.6229 | 0.3616 | 0.8190 | 5,222.2 | 3.240 s |
| LightRAG | 0.4158 | 0.1522 | 0.8500 | 6,727.9 | 3.685 s |
| PathRAG | 0.4985 | 0.1638 | 0.6721 | 5,730.6 | 4.015 s |

## P1 测试集结果（24 题）

在线 token 是每题生成阶段实际输入与输出 token 之和；索引成本未混入在线成本。

| 策略 | Answer Correctness | ROUGE-L | Evidence Recall | 在线 token | 延迟 | 路由分布 |
|---|---:|---:|---:|---:|---:|---|
| 固定 Vector | 0.7135 | 0.3130 | 0.8472 | 5,206.3 | 1.966 s | V=24 |
| 固定 LightRAG | 0.3524 | 0.1182 | 0.7674 | 6,815.4 | 2.427 s | L=24 |
| 固定 PathRAG | 0.4908 | 0.1833 | 0.7639 | 5,695.1 | 3.022 s | P=24 |
| Adaptive | 0.7135 | 0.3130 | 0.8472 | 5,206.3 | 1.966 s | V=24 |
| Oracle | 0.7266 | 0.3189 | 0.8681 | 5,272.7 | 2.176 s | V=21, P=3 |

Adaptive 相比固定 PathRAG 的在线 token 降低 8.58%，但相比固定 Vector
没有质量提升。原因不是路由器学到了有效的三路边界，而是它在测试集上把全部
24 题都预测成 Vector。

## 银标与路由器

- 全部 120 题银标：Vector 111、LightRAG 4、PathRAG 5；其中 7 题
  `all_failed=true`。
- 验证集 24 个银标全部为 Vector，因此所有候选模型的验证 Macro-F1 上限
  在三分类口径下只有 0.3333。
- 选中的配置：排除 `all_failed`、word TF-IDF、`C=0.5`、无 class weight。
- 测试 Accuracy：0.8750；Macro-F1：0.3111。
- Majority Route 的 Accuracy/Macro-F1 同样为 0.8750/0.3111，说明当前
  Adaptive 路由没有超过多数类基线。

混淆矩阵（行是真实银标，列是预测；顺序 Vector/LightRAG/PathRAG）：

```text
[[21, 0, 0],
 [ 0, 0, 0],
 [ 3, 0, 0]]
```

## 代表案例

成功路由：

1. `Medical-83e047ca`：regional prostate cancer 分期；Vector
   Correctness=0.9937，路由正确。
2. `Medical-819d9c6c`：肝内胆管癌的解剖来源；Vector=0.9883，
   PathRAG=0.8308，按质量与成本选择 Vector。
3. `Medical-0be2176d`：FRα 阳性铂耐药卵巢癌；LightRAG=0.9924、
   Vector=0.9857、PathRAG=0.9349。三者都达到正确阈值时，银标按在线成本
   选择 Vector，符合“能正确时选最便宜策略”。

失败路由：

1. `Medical-f3af6f24`：原发 CNS 淋巴瘤脑脊液受累；银标 PathRAG
   0.6532，路由器选择 Vector 0.4673，Correctness 差 0.1859。
2. `Medical-dbfb3406`：口腔癌 TNM 中 T2 定义；银标 PathRAG 0.6013，
   路由器选择 Vector 0.4725，差 0.1288。

机器生成的完整 3 个成功与 2 个失败案例见
`results_official/p1/analysis/representative_cases.json`。

## Judge JSON 兼容修复

GraphRAG-Bench 的事实分类函数直接对模型响应调用 `json.loads`。本地 Qwen
最初返回 Markdown-fenced JSON，导致 factuality 被错误记为 0，Correctness
只剩 25% 权重的语义相似度。最终处理如下：

1. 保存所有原始 judge prompt/response；
2. 适配层只规范化响应中可验证的完整 JSON，不改变分类内容；
3. 对完整旧 trace 用官方 F1 公式确定性重算；
4. 对截断或结构错误的行单独补评，并加入仅约束输出格式的 JSON system prompt；
5. 最终 P0/P1 共 390 个后端-问题结果的 factuality 分类全部通过结构校验。

所有修复前文件保留在 `results_official/debug/eval_*`，未覆盖或冒充有效结果。

## 实际命令

```bash
# P0
bash scripts/run_official_local.sh --stage p0 --backend pathrag --skip-index
.venv_official/bin/python -m scripts.validate_p0

# P1 三路并行时分别使用以下参数
bash scripts/run_official_local.sh --stage p1 --backend vector --skip-index \
  --llm-device cuda:0 --embedding-device cuda:1
bash scripts/run_official_local.sh --stage p1 --backend lightrag --skip-index \
  --llm-device cuda:2 --embedding-device cuda:3
bash scripts/run_official_local.sh --stage p1 --backend pathrag --skip-index \
  --llm-device cuda:4 --embedding-device cuda:5

.venv_official/bin/python -m scripts.validate_p0 \
  --dir results_official/p1 --expected-count 120

# 三路评测同样使用 --llm-device/--embedding-device 分配 GPU
.venv_official/bin/python -m scripts.evaluate_official \
  --stage p1 --backend vector --llm-device cuda:0 --embedding-device cuda:1
.venv_official/bin/python -m scripts.analyze_official --stage p1
.venv_official/bin/python -m unittest discover -s tests -v
```

所有运行均支持按 question ID 续跑；PathRAG 抽取调用另有逐次 fsync 的
`adapter_llm_cache.jsonl`，意外退出无需从零重做两小时建图。

## 主要产物（绝对路径）

- 项目：`/home/user/wangyuhan/知识工程综合实践/adaptive_graphrag_official`
- P1 逐题结果：`/home/user/wangyuhan/知识工程综合实践/adaptive_graphrag_official/results_official/p1/{vector,lightrag,pathrag}.jsonl`
- P1 逐题评分：`/home/user/wangyuhan/知识工程综合实践/adaptive_graphrag_official/results_official/p1/{vector,lightrag,pathrag}_evaluated.jsonl`
- 汇总：`/home/user/wangyuhan/知识工程综合实践/adaptive_graphrag_official/results_official/p1/analysis/summary.json`
- 银标：`/home/user/wangyuhan/知识工程综合实践/adaptive_graphrag_official/results_official/p1/analysis/silver_labels.jsonl`
- 路由模型：`/home/user/wangyuhan/知识工程综合实践/adaptive_graphrag_official/results_official/p1/analysis/router.joblib`
- 质量—成本图：`/home/user/wangyuhan/知识工程综合实践/adaptive_graphrag_official/results_official/p1/analysis/quality_cost.png`
- 混淆矩阵：`/home/user/wangyuhan/知识工程综合实践/adaptive_graphrag_official/results_official/p1/analysis/router_confusion_matrix.png`
- 案例：`/home/user/wangyuhan/知识工程综合实践/adaptive_graphrag_official/results_official/p1/analysis/representative_cases.json`
- 人工复核模板：`/home/user/wangyuhan/知识工程综合实践/adaptive_graphrag_official/results_official/p1/analysis/manual_review_sample.jsonl`
- LightRAG GraphML：`/home/user/wangyuhan/知识工程综合实践/adaptive_graphrag_official/artifacts/lightrag/graph_chunk_entity_relation.graphml`
- PathRAG GraphML：`/home/user/wangyuhan/知识工程综合实践/adaptive_graphrag_official/results_official/indexes/pathrag/graph_chunk_entity_relation.graphml`

## 限制与下一步

1. 20 题人工复核模板已随机生成，但真实人工 verdict 尚未填写；不能写成已完成人审。
2. judge 与生成模型同族，正式论文结论应换用更强的独立文本 judge 再复评。
3. 没有本地 T5-Small/Base 权重，因此只完成了必需的 TF-IDF 路由基线。
4. 银标高度不平衡，且验证集全是 Vector，当前 120 题不足以证明 Adaptive
   路由有效；应扩大样本或优化图抽取/检索后再训练路由器。
5. LightRAG 只有 163 条合并边，PathRAG 关系更多但在线质量仍低于 Vector；
   后续优先检查实体合并质量、关系噪声与上下文裁剪，而不是直接宣称图方法无效。
