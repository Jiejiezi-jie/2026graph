from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.official_backends.base import METHODS
from src.official_evaluation import load_jsonl


REQUIRED_FIELDS = {
    "question_id",
    "method",
    "answer",
    "contexts",
    "entities",
    "relations",
    "paths",
    "input_tokens",
    "output_tokens",
    "llm_calls",
    "retrieval_time_ms",
    "generation_time_ms",
    "total_time_ms",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("results_official/p0"))
    parser.add_argument("--expected-count", type=int, default=10)
    args = parser.parse_args()
    ids_by_method = {}
    for method in METHODS:
        rows = load_jsonl(args.dir / f"{method}.jsonl")
        if len(rows) != args.expected_count:
            raise AssertionError(
                f"{method}: expected {args.expected_count} rows, found {len(rows)}"
            )
        ids_by_method[method] = [row["question_id"] for row in rows]
        for row in rows:
            missing = REQUIRED_FIELDS - row.keys()
            if missing:
                raise AssertionError(f"{method}/{row.get('question_id')}: missing {missing}")
            if row.get("error"):
                raise AssertionError(f"{method}/{row['question_id']}: {row['error']}")
            if not row["answer"].strip():
                raise AssertionError(f"{method}/{row['question_id']}: empty answer")
            if not row["contexts"]:
                raise AssertionError(f"{method}/{row['question_id']}: empty contexts")
    first = ids_by_method[METHODS[0]]
    for method in METHODS[1:]:
        if ids_by_method[method] != first:
            raise AssertionError(f"P0 question alignment differs for {method}")
    print(
        json.dumps(
            {
                "status": "passed",
                "questions": args.expected_count,
                "backends": list(METHODS),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
