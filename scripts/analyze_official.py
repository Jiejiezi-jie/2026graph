from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.official_analysis import analyze_official
from src.official_evaluation import fingerprint, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/official_local.json"))
    parser.add_argument("--stage", choices=["p0", "p1"], default="p1")
    args = parser.parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else project_dir / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    stage_dir = project_dir / config["results_dir"] / args.stage
    rows = {
        method: load_jsonl(stage_dir / f"{method}_evaluated.jsonl")
        for method in ("vector", "lightrag", "pathrag")
    }
    for method, evaluated in rows.items():
        source_rows = load_jsonl(stage_dir / f"{method}.jsonl")
        sources = {row["question_id"]: fingerprint(row) for row in source_rows}
        if len(sources) != len(source_rows) or set(sources) != {row["question_id"] for row in evaluated}:
            raise ValueError(f"{method}: evaluation does not cover the current source question set")
        if any(row.get("evaluation", {}).get("source") != sources[row["question_id"]] for row in evaluated):
            raise ValueError(f"{method}: stale evaluations; re-evaluate current answers first")
    summary = analyze_official(rows, stage_dir / "analysis", seed=config["seed"])
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
