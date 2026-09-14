#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$project_dir"
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
python_bin="${OFFICIAL_PYTHON:-$project_dir/.venv_official/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  python_bin="${PYTHON:-python}"
fi
export OFFICIAL_PYTHON="$python_bin"

active_lightrag_pid="${1:?usage: continue_official_pipeline.sh ACTIVE_LIGHTRAG_PID}"
while kill -0 "$active_lightrag_pid" 2>/dev/null; do
  sleep 30
done

"$python_bin" - <<'PY'
import json
from pathlib import Path

root = Path("result/official")
manifest = root / "indexes" / "lightrag" / "official_index_manifest.json"
rows_path = root / "p0" / "lightrag.jsonl"
if not manifest.exists() or not rows_path.exists():
    raise SystemExit("LightRAG P0 did not finish; refusing to continue the pipeline")
rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line]
if len(rows) != 10 or any(row.get("error") for row in rows):
    raise SystemExit("LightRAG P0 is incomplete or contains errors")
PY

bash src/backend/common/run_official_local.sh --stage p0 --backend pathrag
"$python_bin" -m src.backend.common.validate_p0
"$python_bin" -m src.backend.common.evaluate_official --stage p0

bash src/backend/common/run_official_local.sh --stage p1 --backend all
"$python_bin" -m src.backend.common.evaluate_official --stage p1
"$python_bin" -m src.router.analyze_official --stage p1
