#!/usr/bin/env python3
"""Evaluate phase-five train answers and build one auditable label per question."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase5_pipeline import (
    Phase5Error,
    canonical_json,
    evaluate_and_build_labels,
    load_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "phase5_generation_evaluation.json",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require official-judge caches and make zero network calls.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config, paths = load_config(ROOT, args.config.resolve())
        _scores, _labels, summary, _manifest = evaluate_and_build_labels(
            ROOT, config, paths, offline=args.offline
        )
    except (Phase5Error, json.JSONDecodeError, OSError) as exc:
        print(f"阶段5评估与标签失败：{exc}", file=sys.stderr)
        return 1
    print(canonical_json(summary, pretty=True), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
