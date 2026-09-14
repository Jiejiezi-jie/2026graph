from __future__ import annotations

import random
from collections import Counter
from typing import Iterable


def stratified_sample_and_split(
    questions: Iterable[dict],
    question_types: list[str],
    per_type: int,
    seed: int = 42,
) -> list[dict]:
    """Sample each type and freeze a 60/20/20 split before any model runs."""

    grouped = {kind: [] for kind in question_types}
    for item in questions:
        kind = item.get("question_type")
        if kind in grouped:
            grouped[kind].append(dict(item))
    selected: list[dict] = []
    for type_index, kind in enumerate(question_types):
        rows = sorted(grouped[kind], key=lambda row: row["id"])
        if len(rows) < per_type:
            raise ValueError(f"Only {len(rows)} questions available for {kind}")
        rng = random.Random(seed + type_index)
        sampled = rng.sample(rows, per_type)
        rng.shuffle(sampled)
        train_end = int(per_type * 0.60)
        validation_end = train_end + int(per_type * 0.20)
        for position, row in enumerate(sampled):
            if position < train_end:
                split = "train"
            elif position < validation_end:
                split = "validation"
            else:
                split = "test"
            row["split"] = split
            selected.append(row)
    counts = Counter((row["question_type"], row["split"]) for row in selected)
    expected = {
        (kind, "train"): int(per_type * 0.60) for kind in question_types
    }
    expected.update(
        {(kind, "validation"): int(per_type * 0.20) for kind in question_types}
    )
    expected.update(
        {
            (kind, "test"): per_type - int(per_type * 0.60) - int(per_type * 0.20)
            for kind in question_types
        }
    )
    if dict(counts) != expected:
        raise AssertionError(f"Unexpected split counts: {counts}")
    return selected


def p0_subset(p1_rows: list[dict], question_types: list[str], per_type: int) -> list[dict]:
    rows: list[dict] = []
    for kind in question_types:
        candidates = [row for row in p1_rows if row["question_type"] == kind]
        rows.extend(candidates[:per_type])
    return rows

