#!/usr/bin/env python3
"""Generate grounded answers for the frozen 62 x 3 training retrieval matrix."""

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
    generate_train_answers,
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
        help="Require validated caches and make zero network calls.",
    )
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="Retry cached generation failures during an online run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config, paths = load_config(ROOT, args.config.resolve())
        _rows, audit = generate_train_answers(
            ROOT,
            config,
            paths,
            offline=args.offline,
            retry_failures=args.retry_failures,
        )
    except (Phase5Error, json.JSONDecodeError, OSError) as exc:
        print(f"阶段5答案生成失败：{exc}", file=sys.stderr)
        return 1
    print(canonical_json(audit, pretty=True), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
