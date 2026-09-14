# Adaptive GraphRAG

> 仓库整合说明(2026-09):本仓库已并入三条历史分支的全部成果——**Medical 官方后端评测主线**(vector / LightRAG / PathRAG 的 P0/P1、索引与 router)、**知识图谱 Adaptive 路由**(`src/router/`)与 **知图 · Medical 检索工作台**(FastAPI + React/Cytoscape 图谱问答演示)。2026-09 完成目录重组,代码按 `src / data / test / result / file / deps` 分区。正文前半保留最初阶段一协议实验文档作历史参考。

## 目录结构

```text
src/
├── frontend/          React 19 + Vite + Cytoscape 工作台前端
├── backend/
│   ├── app/           FastAPI 应用(Medical 检索工作台)
│   ├── scripts/       serve_web.py / prepare_medical.py / install_medical_router.py
│   ├── common/        共享设施:后端接口、模型客户端、官方数据与评测、跨方法 runner
│   ├── vector/        Vector 基线后端与 runner
│   ├── lightRAG/      LightRAG 后端、runner、图谱导出与检索探针
│   └── pathRAG/       PathRAG 后端、runner、索引恢复/调参与图谱导出
└── router/            Adaptive 分类器特征、训练、网格与多目标实验、分析脚本
data/                  数据集(GraphRAG-Bench 经 data/vendor/ 拉取,不入库)
test/                  全部测试(test/ 为官方流水线,test/workbench/ 为工作台)
result/                全部结果:
                        api/                    官方 P0/P1 索引与评测产物
                        official/                正式本地后端输出
                        shared_{current_main,expanded,unified_latency}/  共享图谱实验批次
                        phase1_proxy/{code,outputs}/  阶段一离线代理实验(冻结)
file/                  汇报稿、PPT、运行手册与技术文档
deps/{LightRAG,PathRAG}/  固定提交的精简上游依赖
configs/               全部实验 JSON 配置
```

关键文档:[file/P0_P1_RUNBOOK.md](file/P0_P1_RUNBOOK.md)、[file/MEDICAL_WORKBENCH.md](file/MEDICAL_WORKBENCH.md)、[file/WORKBENCH_README.md](file/WORKBENCH_README.md)、[file/MODEL_AUDIT.md](file/MODEL_AUDIT.md)、[file/SERVER_INTEGRATION_NOTES.md](file/SERVER_INTEGRATION_NOTES.md)。

## 阶段一：可复现离线代理实验(历史)

本项目把 Adaptive-RAG 的核心协议迁移到三个异构检索后端：

1. `vector`：直接文本向量检索；
2. `light_proxy`：种子文本块加一跳图扩散；
3. `path_proxy`：查询分解、种子定位与关系路径检索。

三个后端先在 GraphRAG-Bench Medical 上逐题运行；随后根据“证据覆盖近似最优时优先选择便宜路线”的规则自动构造银标，并用问题文本重新训练一个轻量三分类器。推理时，分类器只读取问题并选择一个后端，不使用题型标签、参考答案或证据。

### 重要边界

当前环境没有 LLM API、Ollama、GPU 和 Transformers，因此仓库中的首轮数字属于**检索层离线代理实验**：

- 使用真实 GraphRAG-Bench Medical 语料、问题、答案和证据；
- 使用确定性的TF-IDF、文本块图扩散和路径搜索；
- 真实训练并测试了轻量路由器；
- 没有冒充官方LightRAG或PathRAG的端到端生成结果；
- `answer_proxy`仅用于验证“检索到回答”的最小流程，不能作为医疗回答质量结论。

这套协议的作用是先验证数据、日志、银标、路由训练和评测链路。获得模型条件后，只需要将三个代理替换成官方后端，银标和路由训练部分可以保持不变。

### 数据

使用 [GraphRAG-Bench](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark) 的 Medical 子集。阶段一保留：

- Fact Retrieval：1098题；
- Complex Reasoning：509题；
- 合计1607题；
- 训练/验证/测试：963/322/322。

语料在切词后按180词窗口、40词重叠切分为1255个文本块。

### 一键复现

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash src/backend/common/fetch_benchmark.sh
bash result/phase1_proxy/code/run_phase1.sh
python -m unittest discover -s test -v
```

如果已经有官方benchmark仓库：

```bash
bash result/phase1_proxy/code/run_phase1.sh /absolute/path/to/GraphRAG-Benchmark
```

运行一条自动路由查询：

```bash
python result/phase1_proxy/code/demo.py \
  "Why are fair skin and organ transplantation both risk factors for BCC?" \
  --benchmark-dir data/vendor/GraphRAG-Benchmark
```

### 银标协议

对于问题 `q`，分别运行三个检索器。令证据覆盖率为 `Q(r,q)`，若一个方法满足：

```text
Q(r,q) >= max_r Q(r,q) - 0.02
```

则视为质量近似最优；从这些方法中按 `vector -> light_proxy -> path_proxy` 选择最低成本路线。这个规则对应 Adaptive-RAG 的“能够完成任务时优先选择更简单策略”。

### 产物

阶段一代码与输出整体冻结在 `result/phase1_proxy/`：

- `result/phase1_proxy/outputs/results/summary.json`：完整结果与限制；
- `result/phase1_proxy/outputs/results/backend_results.csv`：三种后端逐题指标；
- `result/phase1_proxy/outputs/results/processed_questions.jsonl`：数据划分与银标；
- `result/phase1_proxy/outputs/results/router_test_predictions.csv`：测试集路由概率与预测；
- `result/phase1_proxy/outputs/results/representative_cases.json`：三条正确路由与两条错误路由案例；
- `result/phase1_proxy/outputs/results/*.png`：质量—成本、标签分布和混淆矩阵；
- `artifacts/router.joblib`：训练后的轻量路由器。

### 接入正式后端时的实验约束

正式复现实验必须统一：

- 原始语料和数据划分；
- 生成LLM与温度；
- embedding模型；
- 最大生成长度；
- 检索上下文预算；
- 逐题token、延迟、调用次数日志。

推荐将官方LightRAG的`naive`模式作为Vector RAG，将`hybrid`作为LightRAG，并使用PathRAG官方实现作为第三条路线。正式结果应使用GraphRAG-Bench官方LLM评测脚本，不能把本项目的词项覆盖率当作官方指标。

## 正式本地后端

`result/phase1_proxy/` 中的代理代码与输出仍是 phase-one 基线，未被覆盖。正式实现位于 `src/backend/`，正式输出只写入 `result/official/`。

固定上游版本：

- GraphRAG-Bench `fdbab5959b18c96532580877ffe27d112bccc0ec`；
- HKUDS LightRAG `v1.5.7` / `28ff1b05f2ac3f3e6fa14dd2cd33656579bd0c9c`；
- BUPT-GAMMA PathRAG `32567bfc93605b8393996d5fa9ccdc0edbb865b2`。

上游依赖已随仓库提供精简副本(`deps/LightRAG`、`deps/PathRAG`)；如需重新拉取：

```bash
bash src/backend/common/fetch_benchmark.sh
git clone --depth 1 --branch v1.5.7 \
  https://github.com/HKUDS/LightRAG.git deps/LightRAG
git clone --depth 1 \
  https://github.com/BUPT-GAMMA/PathRAG.git deps/PathRAG
```

环境安装命令：

```bash
uv venv --python /home/user/.miniforge3/envs/qwen_saliency/bin/python \
  --system-site-packages .venv_official
uv pip install --python .venv_official/bin/python \
  -e deps/LightRAG \
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
  -n qwen_saliency .venv_official/bin/python -m src.backend.common.probe_local_model \
  --benchmark-dir data/vendor/GraphRAG-Benchmark \
  --model-path /home/user/wangyuhan/models/Qwen2.5-VL-7B-Instruct \
  --device cuda:0 \
  --output result/official/model_probe_qwen25vl7b.json
```

P0 必须先完整通过（每类 5 题、三个后端）：

```bash
bash src/backend/common/run_official_local.sh --stage p0 --backend all
PYTHONPATH=. .venv_official/bin/python -m src.backend.common.validate_p0
PYTHONPATH=. conda run --no-capture-output -n qwen_saliency \
  .venv_official/bin/python -m src.backend.common.evaluate_official --stage p0
```

LightRAG 完成全部 Chunk 的实体/关系抽取和图合并后，可校验并导出 GraphML：

```bash
PYTHONPATH=. .venv_official/bin/python -m src.backend.lightRAG.export_lightrag_graph
```

脚本只接受同时存在完成态索引清单、非空节点和非空边的图，输出到
`artifacts/lightrag/graph_chunk_entity_relation.graphml`，并生成包含节点数、
边数和 SHA-256 的 `manifest.json`。未完成的中间索引不会被当作成果导出。

P1 固定为每类 60 题，60%/20%/20% 划分在模型运行前冻结；runner 支持按
逐题 JSONL 自动续跑：

```bash
bash src/backend/common/run_official_local.sh --stage p1 --backend all
PYTHONPATH=. conda run --no-capture-output -n qwen_saliency \
  .venv_official/bin/python -m src.backend.common.evaluate_official --stage p1
PYTHONPATH=. .venv_official/bin/python -m src.router.analyze_official --stage p1
```

模型、GPU 和上游兼容性细节见 [file/MODEL_AUDIT.md](file/MODEL_AUDIT.md) 与
[file/SERVER_INTEGRATION_NOTES.md](file/SERVER_INTEGRATION_NOTES.md)。GraphRAG-Bench 的 Answer Correctness、
ROUGE-L、Evidence Recall 代码被直接复用；当前唯一可用 judge 与生成模型同属
Qwen2.5-VL-7B，报告会明确标注潜在自评偏差。

## Adaptive 路由

`src/router/` 保存分类器的全部链路：特征构建(`router.py`)、CV 与硬权重网格
(`run_router_hard_weight_grid_300.py` 等)、多目标/加权效用实验、共享图谱规则网格，
以及 `official_analysis.py` 等分析脚本。`from src.router import build_router`
保持可用。

## Medical 检索工作台

FastAPI + React/Cytoscape 工作台,在 LightRAG / PathRAG 共享的 Medical 知识图谱上做
检索问答演示,展示 vector / LightRAG / PathRAG / Adaptive 四种方法。

```bash
# 后端(仓库根目录)
python src/backend/scripts/serve_web.py --port 8000 \
  --medical-bundle src/backend/data/medical \
  --medical-embedding-path <BGE-M3 权重目录> \
  --medical-pathrag-root deps/PathRAG
# 前端
cd src/frontend && npm ci && npm run dev
```

导入索引与模型、preflight、四方法验证、常见问题见
[file/WORKBENCH_README.md](file/WORKBENCH_README.md) 与
[file/MEDICAL_WORKBENCH.md](file/MEDICAL_WORKBENCH.md)。

## 测试

```bash
python -m unittest discover -s test -v     # 官方流水线测试
python -m pytest test/workbench -q        # 工作台测试(需 fastapi / pytest-asyncio / lightrag)
cd src/frontend && npm run build           # 前端类型检查 + 生产构建
```

根目录 `pytest.ini` 已设置 `pythonpath = . src/backend`,使 `src.*` 与工作台的 `app.*` 均可导入。
