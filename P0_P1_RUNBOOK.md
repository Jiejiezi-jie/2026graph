# lightRAG API 线补齐：P0 → P1 操作手册（2026-09-07 实测版）

> 环境：`.wt-lightrag` worktree（HEAD `60ebc48`），Python venv `.venv_official`。
> 目标：补齐 PathRAG（索引未完成 → 重建），对齐已有 vector/lightrag 结果，跑通 P0→P1→评估→路由分析。

## 0. 前置（一次性）

```bash
cd /home/szj/2026graph/.wt-lightrag
nvidia-smi                                  # 确认 GPU 可见
# 装 GPU torch（cu126，torch>=2.6 才能加载 bge-m3；cu121 会被 transformers 5.6.2 拒载）
.venv_official/bin/pip install --index-url https://download.pytorch.org/whl/cu126 \
  "torch==2.14.0+cu126" "torchvision==0.29.0+cu126"
.venv_official/bin/python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
export DEEPSEEK_API_KEY='你的key'           # 只放环境变量
export PYTHONPATH=/home/szj/2026graph/.wt-lightrag
```

已就绪（勿删）：`data/vendor/GraphRAG-Benchmark`(fdbab59)、`third_party/LightRAG`(28ff1b05)、`third_party/PathRAG`(32567bfc)、`models/bge-m3`(2.2GB)。

## 1. PathRAG 索引：现状与决策

`results_api/indexes/pathrag/` 只有 **graphml(4568 节点/9432 边) + identity + vdb_chunks(仅向量)**；
缺 `official_index_manifest.json`，`vdb_entities/relationships` 为空，kv 存储为 `{}`。

**结论：该目录不能触发 cached 复用，runner 会全量重建 `ainsert()`。**

选项：
- **A（推荐，干净）**：先备份/移走残缺目录，让重建从零开始：
  ```bash
  mv results_api/indexes/pathrag results_api/indexes/pathrag.INCOMPLETE.bak
  ```
- B：保留现状直接跑（残留 graphml 可能造成状态污染/重复，不推荐）。

## 2. PathRAG P0（40 题，与现有 vector/lightrag 严格对齐）

> ⚠️ 必须 `--p0-per-type 20`：config 默认 `p0_per_type=5`（10 题），会用 10 题覆盖 splits，
> 破坏与现有 40 行 vector/lightrag 的对齐。实测 `--p0-per-type 20` 重写的 splits 与已提交的
> `results_api/splits/*.jsonl` **逐字节顺序一致**（已验证）。
> ⚠️ 不要用 `scripts/run_official_local.sh`：它内部 `conda run -n qwen_saliency`，本机无此 env。

```bash
.venv_official/bin/python -m scripts.run_official_experiment \
  --config configs/official_api.json --stage p0 --backend pathrag --p0-per-type 20
```
- 首次将全量重建 PathRAG 索引（数百次 LLM 调用，1-2 小时级 + bge-m3 向量化）；
- 断点续跑：已完成 question_id 自动跳过，失败行记 `error`（P0 失败会 raise）；
- 产物：`results_api/p0/pathrag.jsonl`（40 行）。

## 3. LightRAG P0（确认/补齐）

```bash
.venv_official/bin/python -m scripts.run_official_experiment \
  --config configs/official_api.json --stage p0 --backend lightrag --p0-per-type 20
```
- 现有 `results_api/p0/lightrag.jsonl` 40 行已存在（cached 索引复用）；重复跑会跳过已完成行，幂等。

## 4. 验证 P0（注意与 40 题规模的差异）

⚠️ `scripts/validate_p0.py` **硬编码期望 10 行、默认目录 `results_official/p0`**，仅适配本地 10 题线，
**不适用于 API 40 题线**。40 题规模的等价校验改为：

```bash
.venv_official/bin/python - <<'EOF'
import json
from pathlib import Path
from src.official_evaluation import load_jsonl
from src.official_backends.base import METHODS
d = Path("results_api/p0")
ids = {}
for m in METHODS:
    rows = load_jsonl(d / f"{m}.jsonl")
    assert len(rows) == 40, f"{m}: {len(rows)} rows"
    for r in rows:
        assert not r.get("error"), f"{m}/{r['question_id']}: error"
        assert r.get("answer","").strip() and r.get("contexts"), f"{m}: empty"
    ids[m] = [r["question_id"] for r in rows]
    print(m, "OK", len(rows))
base = ids["vector"]
for m in METHODS[1:]:
    assert ids[m] == base, f"{m} 与 vector 题序不一致"
print("P0 alignment OK: 40 x", len(METHODS))
EOF
```

## 5. P0 官方评估（三指标）

```bash
.venv_official/bin/python -m scripts.evaluate_official \
  --config configs/official_api.json --stage p0
```
- 生成 `results_api/p0/{vector,lightrag,pathrag}_evaluated.jsonl`（version-2 协议，指纹复用 + `.bak` 归档）；
- `--check-only` 可离线查看可复用/待评行；`--force` 全量重评。

## 6. P1 全部三后端（120 题）

```bash
.venv_official/bin/python -m scripts.run_official_experiment \
  --config configs/official_api.json --stage p1 --backend all
```
- 120 题 = FR 60 / CR 60（train 36 + val 12 + test 12 每类）；失败行自动重试、断点续跑；
- 三后端将先各自 index（vector cached；lightrag cached；**pathrag 需重建或复用步骤 1 决策**）。

## 7. P1 评估 + 路由分析

```bash
.venv_official/bin/python -m scripts.evaluate_official --config configs/official_api.json --stage p1
.venv_official/bin/python -m scripts.analyze_official  --config configs/official_api.json --stage p1
```
- `analyze_official` 产出 `results_api/p1/analysis/`：summary.json、silver_labels.jsonl、
  router_tuning.json、router_test_predictions.jsonl、router.joblib、quality_cost.png、router_confusion_matrix.png。

## 8. 结果检查

```bash
wc -l results_api/p1/{vector,lightrag,pathrag}.jsonl          # 应各 120
python3 -m json.tool results_api/p1/analysis/summary.json      # 三策略 + 路由指标
ls results_api/p1/analysis/
```

## 备注

- **PathRAG 索引完整性判定**（勿只信 graphml）：完成 = 存在 `official_index_manifest.json`
  且 `vdb_entities/vdb_relationships` 非空 + kv 存储非空。当前分支该目录为不完整态。
- 若后续改走"PathRAG 复用 LightRAG 图"：需重新生成辅助索引 → 重跑 → 重评 → 重训路由器，
  现有 P0/P1 结果不可沿用。
- API 结果为 engineering validation（同模型 judge、非本地官方基准）；官方结论以本地 GPU 线为准。
