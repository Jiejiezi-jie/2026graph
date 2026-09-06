#!/usr/bin/env python3
"""Build normalized BGE chunk/entity indices for phase four."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase4_retrieval import Phase4Error, build_indices, canonical_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "retrieval_config.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        metadata = build_indices(ROOT, args.config.resolve())
    except Phase4Error as exc:
        print(f"阶段4索引失败：{exc}", file=sys.stderr)
        return 1
    print(canonical_json(metadata, pretty=True), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
