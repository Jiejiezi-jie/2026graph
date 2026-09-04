# Adaptive GraphRAG：阶段一可复现实验

本项目把 Adaptive-RAG 的核心协议迁移到三个异构检索后端：

1. `vector`：直接文本向量检索；
2. `light_proxy`：种子文本块加一跳图扩散；
3. `path_proxy`：查询分解、种子定位与关系路径检索。

三个后端先在 GraphRAG-Bench Medical 上逐题运行；随后根据“证据覆盖近似最优时优先选择便宜路线”的规则自动构造银标，并用问题文本重新训练一个轻量三分类器。推理时，分类器只读取问题并选择一个后端，不使用题型标签、参考答案或证据。

## 重要边界

当前环境没有 LLM API、Ollama、GPU 和 Transformers，因此仓库中的首轮数字属于**检索层离线代理实验**：

- 使用真实 GraphRAG-Bench Medical 语料、问题、答案和证据；
- 使用确定性的TF-IDF、文本块图扩散和路径搜索；
- 真实训练并测试了轻量路由器；
- 没有冒充官方LightRAG或PathRAG的端到端生成结果；
- `answer_proxy`仅用于验证“检索到回答”的最小流程，不能作为医疗回答质量结论。

这套协议的作用是先验证数据、日志、银标、路由训练和评测链路。获得模型条件后，只需要将三个代理替换成官方后端，银标和路由训练部分可以保持不变。

## 数据

使用 [GraphRAG-Bench](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark) 的 Medical 子集。阶段一保留：

- Fact Retrieval：1098题；
- Complex Reasoning：509题；
- 合计1607题；
- 训练/验证/测试：963/322/322。

语料在切词后按180词窗口、40词重叠切分为1255个文本块。

## 一键复现

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/fetch_benchmark.sh
bash scripts/run_phase1.sh
python -m unittest discover -s tests -v
```

如果已经有官方benchmark仓库：

```bash
bash scripts/run_phase1.sh /absolute/path/to/GraphRAG-Benchmark
```

运行一条自动路由查询：

```bash
python -m src.demo \
  "Why are fair skin and organ transplantation both risk factors for BCC?" \
  --benchmark-dir data/vendor/GraphRAG-Benchmark
```

## 银标协议

对于问题 `q`，分别运行三个检索器。令证据覆盖率为 `Q(r,q)`，若一个方法满足：

```text
Q(r,q) >= max_r Q(r,q) - 0.02
```

则视为质量近似最优；从这些方法中按 `vector -> light_proxy -> path_proxy` 选择最低成本路线。这个规则对应 Adaptive-RAG 的“能够完成任务时优先选择更简单策略”。

## 产物

- `results/summary.json`：完整结果与限制；
- `results/backend_results.csv`：三种后端逐题指标；
- `results/processed_questions.jsonl`：数据划分与银标；
- `results/router_test_predictions.csv`：测试集路由概率与预测；
- `results/representative_cases.json`：三条正确路由与两条错误路由案例；
- `results/*.png`：质量—成本、标签分布和混淆矩阵；
- `artifacts/router.joblib`：训练后的轻量路由器。

## 接入正式后端时的实验约束

正式复现实验必须统一：

- 原始语料和数据划分；
- 生成LLM与温度；
- embedding模型；
- 最大生成长度；
- 检索上下文预算；
- 逐题token、延迟、调用次数日志。

推荐将官方LightRAG的`naive`模式作为Vector RAG，将`hybrid`作为LightRAG，并使用PathRAG官方实现作为第三条路线。正式结果应使用GraphRAG-Bench官方LLM评测脚本，不能把本项目的词项覆盖率当作官方指标。

## 正式本地后端（新增）

原 `src/backends.py`、`results/` 和 `artifacts/router.joblib` 仍是 phase-one
proxy 基线，未被覆盖。正式实现位于 `src/official_backends/`，正式输出只写入
`results_official/`。

固定上游版本：

- GraphRAG-Bench `fdbab5959b18c96532580877ffe27d112bccc0ec`；
- HKUDS LightRAG `v1.5.7` / `28ff1b05f2ac3f3e6fa14dd2cd33656579bd0c9c`；
- BUPT-GAMMA PathRAG `32567bfc93605b8393996d5fa9ccdc0edbb865b2`。

环境安装命令：

```bash
bash scripts/fetch_benchmark.sh
git clone --depth 1 --branch v1.5.7 \
  https://github.com/HKUDS/LightRAG.git third_party/LightRAG
git clone --depth 1 \
  https://github.com/BUPT-GAMMA/PathRAG.git third_party/PathRAG

uv venv --python /home/user/.miniforge3/envs/qwen_saliency/bin/python \
  --system-site-packages .venv_official
uv pip install --python .venv_official/bin/python \
  -e third_party/LightRAG \
  scikit-learn==1.7.2 matplotlib==3.10.6 joblib==1.5.2 \
  rouge-score==0.1.2 json5 'langchain-core>=0.3,<2'
```

三个后端统一使用 `BAAI/bge-m3`（本机路径
`/home/user/wangyuhan/models/bge-m3`）。Hugging Face 大文件通道在服务器
代理下不稳定，因此从 BAAI ModelScope 官方镜像拉取同一权重：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
  https://www.modelscope.cn/BAAI/bge-m3.git \
  /home/user/wangyuhan/models/bge-m3
git -C /home/user/wangyuhan/models/bge-m3 lfs pull \
  --include='pytorch_model.bin,tokenizer.json,sentencepiece.bpe.model' \
  --exclude='onnx/**,colbert_linear.pt,sparse_linear.pt'
```

本次镜像 commit 为 `a46a13810c9d7f876fccd8d7017512ebc265d2c1`，
`pytorch_model.bin` SHA-256 为
`b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38`。

Qwen2.5-VL-7B 的纯文本 5 问答 + 5 JSON 探针：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. conda run --no-capture-output \
  -n qwen_saliency .venv_official/bin/python scripts/probe_local_model.py \
  --benchmark-dir data/vendor/GraphRAG-Benchmark \
  --model-path /home/user/wangyuhan/models/Qwen2.5-VL-7B-Instruct \
  --device cuda:0 \
  --output results_official/model_probe_qwen25vl7b.json
```

P0 必须先完整通过（每类 5 题、三个后端）：

```bash
bash scripts/run_official_local.sh --stage p0 --backend all
PYTHONPATH=. .venv_official/bin/python -m scripts.validate_p0
PYTHONPATH=. conda run --no-capture-output -n qwen_saliency \
  .venv_official/bin/python -m scripts.evaluate_official --stage p0
```

LightRAG 完成全部 Chunk 的实体/关系抽取和图合并后，可校验并导出 GraphML：

```bash
PYTHONPATH=. .venv_official/bin/python -m scripts.export_lightrag_graph
```

脚本只接受同时存在完成态索引清单、非空节点和非空边的图，输出到
`artifacts/lightrag/graph_chunk_entity_relation.graphml`，并生成包含节点数、
边数和 SHA-256 的 `manifest.json`。未完成的中间索引不会被当作成果导出。

P1 固定为每类 60 题，60%/20%/20% 划分在模型运行前冻结；runner 支持按
逐题 JSONL 自动续跑：

```bash
bash scripts/run_official_local.sh --stage p1 --backend all
PYTHONPATH=. conda run --no-capture-output -n qwen_saliency \
  .venv_official/bin/python -m scripts.evaluate_official --stage p1
PYTHONPATH=. .venv_official/bin/python -m scripts.analyze_official --stage p1
```

测试命令：

```bash
.venv_official/bin/python -m unittest discover -s tests -v
```

模型、GPU 和上游兼容性细节见 `MODEL_AUDIT.md` 和
`SERVER_INTEGRATION_NOTES.md`。GraphRAG-Bench 的 Answer Correctness、
ROUGE-L、Evidence Recall 代码被直接复用；当前唯一可用 judge 与生成模型同属
Qwen2.5-VL-7B，报告会明确标注潜在自评偏差。
