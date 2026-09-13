from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_medical_benchmark
from src.official_backends.base import METHODS
from src.official_evaluation import valid_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze an expansion set without changing the existing test set."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/router_data_expansion.json")
    )
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline_paths = {
        method: resolve(config["baseline_inputs"][method]) for method in METHODS
    }
    baseline_rows = {method: load_jsonl(path) for method, path in baseline_paths.items()}
    if any(len(rows) != 120 for rows in baseline_rows.values()):
        raise ValueError(
            "Expected exactly 120 baseline rows per method; got "
            + repr({method: len(rows) for method, rows in baseline_rows.items()})
        )
    if any(not valid_evaluation(row) for rows in baseline_rows.values() for row in rows):
        raise ValueError("All baseline inputs must contain valid official evaluations")
    ids_by_method = {
        method: {row["question_id"] for row in rows}
        for method, rows in baseline_rows.items()
    }
    if len({frozenset(ids) for ids in ids_by_method.values()}) != 1:
        raise ValueError("Baseline question IDs are not aligned across methods")

    baseline_by_id = {
        row["question_id"]: row for row in baseline_rows["vector"]
    }
    split_counts = Counter(row["split"] for row in baseline_by_id.values())
    if split_counts != {"train": 72, "validation": 24, "test": 24}:
        raise ValueError(f"Baseline split changed: {dict(split_counts)}")

    benchmark_dir = resolve(config["benchmark_dir"])
    _, all_questions = load_medical_benchmark(benchmark_dir)
    question_types = list(config["question_types"])
    baseline_ids = set(baseline_by_id)
    selected: list[dict[str, Any]] = []
    train_count = int(config["additional_train_per_type"])
    validation_count = int(config["additional_validation_per_type"])
    per_type = int(config["additional_per_type"])
    if train_count + validation_count != per_type:
        raise ValueError("Additional train and validation counts must sum to per-type total")

    for type_index, question_type in enumerate(question_types):
        candidates = sorted(
            (
                dict(row)
                for row in all_questions
                if row.get("question_type") == question_type
                and row.get("id") not in baseline_ids
            ),
            key=lambda row: row["id"],
        )
        if len(candidates) < per_type:
            raise ValueError(f"Only {len(candidates)} unused rows for {question_type}")
        rng = random.Random(int(config["expansion_seed"]) + type_index)
        sampled = rng.sample(candidates, per_type)
        rng.shuffle(sampled)
        for position, row in enumerate(sampled):
            row["split"] = "train" if position < train_count else "validation"
            row["selection_source"] = "router_data_expansion"
            selected.append(row)

    if len({row["id"] for row in selected}) != len(selected):
        raise AssertionError("Expansion contains duplicate question IDs")
    if baseline_ids.intersection(row["id"] for row in selected):
        raise AssertionError("Expansion overlaps the existing 120-question experiment")

    expected_counts = {
        (kind, "train"): train_count for kind in question_types
    } | {(kind, "validation"): validation_count for kind in question_types}
    actual_counts = Counter((row["question_type"], row["split"]) for row in selected)
    if dict(actual_counts) != expected_counts:
        raise AssertionError(f"Unexpected expansion counts: {dict(actual_counts)}")

    question_path = resolve(config["question_file"])
    write_jsonl(question_path, selected)
    benchmark_questions = benchmark_dir / "Datasets" / "Questions" / "medical_questions.json"
    manifest = {
        "selection_policy": {
            "seed": int(config["expansion_seed"]),
            "question_types": question_types,
            "additional_per_type": per_type,
            "additional_train_per_type": train_count,
            "additional_validation_per_type": validation_count,
            "test_additions": 0,
        },
        "baseline": {
            "count": len(baseline_ids),
            "split_counts": dict(split_counts),
            "test_ids": sorted(
                qid for qid, row in baseline_by_id.items() if row["split"] == "test"
            ),
            "input_sha256": {
                method: sha256(path) for method, path in baseline_paths.items()
            },
        },
        "expansion": {
            "count": len(selected),
            "split_counts": dict(Counter(row["split"] for row in selected)),
            "question_type_counts": dict(Counter(row["question_type"] for row in selected)),
            "question_ids": [row["id"] for row in selected],
        },
        "combined": {
            "count": len(baseline_ids) + len(selected),
            "development_count": 96 + len(selected),
            "test_count": 24,
        },
        "files": {
            "benchmark_questions": display_path(benchmark_questions),
            "benchmark_questions_sha256": sha256(benchmark_questions),
            "question_file": display_path(question_path),
            "question_file_sha256": sha256(question_path),
        },
    }
    return selected, manifest


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows, manifest = prepare(config)
    manifest_path = resolve(config["selection_manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "questions": len(rows),
                "split_counts": manifest["expansion"]["split_counts"],
                "type_counts": manifest["expansion"]["question_type_counts"],
                "frozen_test_count": manifest["combined"]["test_count"],
                "question_sha256": manifest["files"]["question_file_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
