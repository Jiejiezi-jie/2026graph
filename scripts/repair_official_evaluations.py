from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.official_evaluation import extract_json_payload, load_jsonl


def _classification(trace: list[dict[str, str]]) -> dict[str, list[Any]] | None:
    matches = [
        item
        for item in trace
        if "Given a ground truth and an answer statements" in item.get("prompt", "")
    ]
    if len(matches) != 1:
        return None
    raw = matches[0].get("response", "")
    normalized = extract_json_payload(raw)
    try:
        value = json.loads(normalized)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(value, dict) or any(
        not isinstance(value.get(key), list) for key in ("TP", "FP", "FN")
    ):
        return None
    if normalized != raw:
        matches[0]["normalized_response"] = normalized
    return value


def repair_row(row: dict[str, Any]) -> dict[str, Any] | None:
    classification = _classification(row.get("judge", {}).get("trace", []))
    if classification is None:
        return None
    benchmark_dir = Path(__file__).resolve().parents[1] / "data/vendor/GraphRAG-Benchmark"
    benchmark_text = str(benchmark_dir.resolve())
    if benchmark_text not in sys.path:
        sys.path.insert(0, benchmark_text)
    from Evaluation.metrics.answer_accuracy import fbeta_score

    factuality = fbeta_score(
        len(classification["TP"]),
        len(classification["FP"]),
        len(classification["FN"]),
    )
    # Every pre-fix classification response was Markdown-fenced, so the
    # official json.loads path returned factuality=0.  Its saved correctness
    # value is therefore exactly the still-valid 0.25 semantic component.
    old_correctness = float(row["answer_correctness"])
    repaired = {**row, "answer_correctness": old_correctness + 0.75 * factuality}
    repaired["judge"] = {
        **row["judge"],
        "compatibility_repair": {
            "reason": "Normalized saved Markdown-fenced JSON for official factuality F1",
            "old_answer_correctness": old_correctness,
            "factuality": factuality,
            "formula": "old_semantic_component + 0.75 * factuality",
        },
    }
    return repaired


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=["repair-old", "filter-valid"], default="repair-old"
    )
    args = parser.parse_args()
    repaired: list[dict[str, Any]] = []
    skipped: list[str] = []
    for row in load_jsonl(args.input):
        value = (
            row
            if args.mode == "filter-valid"
            and _classification(row.get("judge", {}).get("trace", [])) is not None
            else None
        )
        if args.mode == "repair-old":
            value = repair_row(row)
        if value is None:
            skipped.append(row["question_id"])
        else:
            repaired.append(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".repair.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in repaired:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "repaired": len(repaired),
                "skipped_for_reevaluation": len(skipped),
                "mode": args.mode,
                "skipped_ids": skipped,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
