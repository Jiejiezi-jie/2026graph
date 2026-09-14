# API backend testing

## Evaluation integrity (version 2)

The judge now has its own 4096-token output budget and temperature zero.
Complete fenced JSON is accepted; truncated JSON is never repaired into scores.
Statement lists, TP/FP/FN fields, and complete evidence coverage are validated
before calling the upstream scoring logic. The official weights and formulas
are unchanged. This is a guarded adapter, not byte-for-byte unmodified upstream
evaluation behavior.

Invalid responses get one retry. Exhausted failures and non-finite scores are
recorded with `evaluation.status=failed`, null scores, and diagnostic traces.
Analysis refuses failed, legacy, or mixed-protocol evaluations rather than
silently counting them as zero or excluding difficult questions.

Re-run the evaluation commands below against existing generated answers.
Old evaluations are automatically archived as timestamped `.bak` files before
replacement. Successful version-2 rows are reused only when source data and the
evaluation configuration/upstream metric fingerprints match. Failed rows are
retried on the next run. `--force` also archives successful scores and rescores
everything. Generation and graph indexing do not need to run again.

Use `--check-only` with either evaluation command to report reusable/pending
rows without an API key, requests, or writes. Each judge attempt and completed
question prints progress. Network retries are bounded independently (one retry
by default for evaluation); a socket timeout is not a total per-question deadline.

These checks remove identifiable evaluator failures, not LLM sampling variance
or same-model judging bias. Compare aligned question sets under the same protocol;
use held-out questions and repeated or independent judging for stronger claims.

The repository supports two independent model configurations:

- `configs/official_local.json`: the original local Transformers/Qwen path.
- `configs/official_api.json`: DeepSeek-compatible chat API plus local BGE-M3 embeddings.

API keys are read only from environment variables. They are never stored in the
configuration file or result files.

## Setup on Windows PowerShell

```powershell
python scripts/setup_official_sources.py
.venv_api\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv_api\Scripts\python.exe -m pip install -r requirements-official-api.txt
$env:DEEPSEEK_API_KEY = "your-deepseek-key"
```

The API configuration uses the downloaded local model at
`models/bge-m3`. Only the DeepSeek key is required; the embedding model does
not make network requests.

The source setup script pins GraphRAG-Bench, LightRAG, and PathRAG to the
commits used by this project. It is safe to run again.

## Check the API and local embedding

```powershell
.venv_api\Scripts\python.exe -m scripts.probe_api --config configs/official_api.json
```

The probe must return a non-empty chat response and a BGE-M3 embedding shape
of `[2, 1024]` with unit-length vectors.

## Run the real upstream backends on a tiny corpus

This is the cheapest integration check and writes only to
`results_api/smoke/`:

```powershell
.venv_api\Scripts\python.exe -m scripts.smoke_api_backends --backend all
```

If a previous smoke run stopped during indexing, rebuild its generated cache:

```powershell
.venv_api\Scripts\python.exe -m scripts.smoke_api_backends --backend all --reset
```

The command requires the DeepSeek key. It validates that official LightRAG and
PathRAG each build a non-empty graph, retrieve contexts, and return entities,
relationships, and an answer. Embeddings are computed locally by BGE-M3.

## Run the fixed P0 sample

```powershell
.venv_api\Scripts\python.exe -m scripts.run_official_experiment --config configs/official_api.json --stage p0 --backend lightrag
.venv_api\Scripts\python.exe -m scripts.run_official_experiment --config configs/official_api.json --stage p0 --backend pathrag
```

To compare Vector and LightRAG on 40 fixed questions (20 Fact Retrieval and 20
Complex Reasoning), keep the same configuration and override the P0 size:

```powershell
.venv_api\Scripts\python.exe -m scripts.run_official_experiment --config configs/official_api.json --stage p0 --backend vector --p0-per-type 20
.venv_api\Scripts\python.exe -m scripts.run_official_experiment --config configs/official_api.json --stage p0 --backend lightrag --p0-per-type 20
.venv_api\Scripts\python.exe -m scripts.evaluate_official --config configs/official_api.json --stage p0 --backend vector
.venv_api\Scripts\python.exe -m scripts.evaluate_official --config configs/official_api.json --stage p0 --backend lightrag
```

The runner resumes by `question_id`, so the existing ten rows are retained and
only the additional questions are generated. The official answer correctness,
ROUGE-L, and evidence recall scores should be compared over all 40 successful
rows, with the same model and temperature for both backends.

Run Vector separately if the API configuration should be compared with the
existing local Vector result. The API results live under `results_api/` and do
not overwrite `results_official/`.

After all three API result files contain ten successful rows, validate them:

```powershell
.venv_api\Scripts\python.exe -m scripts.validate_p0 --dir results_api/p0
```

API-backed results are engineering validation, not the original local-Qwen
benchmark run. The original author can keep using `configs/official_local.json`
on the GPU server and generate the official P1 results independently.
