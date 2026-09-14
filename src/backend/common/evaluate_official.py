from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from src.backend.common.model_client import (
    build_chat_client,
    build_embedding_client,
)
from src.backend.common.official_evaluation import (
    EVALUATION_VERSION, _load_official_metrics, evaluate_official_row,
    fingerprint, load_jsonl, valid_evaluation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/official_local.json"))
    parser.add_argument("--stage", choices=["p0", "p1"], default="p0")
    parser.add_argument(
        "--backend", choices=["all", "vector", "lightrag", "pathrag"], default="all"
    )
    parser.add_argument("--force", action="store_true", help="Back up and re-evaluate completed rows")
    parser.add_argument("--check-only", action="store_true", help="Validate inputs and report pending rows without API calls or writes")
    return parser.parse_args()


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {row["question_id"] for row in load_jsonl(path) if valid_evaluation(row)}


def reusable_rows(target: Path, rows: list[dict], protocol: str) -> list[dict]:
    sources = {row["question_id"]: fingerprint(row) for row in rows}
    if len(sources) != len(rows):
        raise ValueError("Duplicate question IDs in source results")
    if not target.exists():
        return []
    kept = {}
    for row in load_jsonl(target):
        meta = row.get("evaluation", {})
        if (valid_evaluation(row) and meta.get("protocol") == protocol
                and meta.get("source") == sources.get(row["question_id"])):
            kept[row["question_id"]] = row
    return list(kept.values())


def prepare_target(target: Path, kept: list[dict]) -> None:
    if target.exists() and load_jsonl(target) != kept:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = target.with_name(target.name + f".{stamp}.bak")
        # Create the replacement before archiving, so a failed write retains the original.
        temporary = target.with_name(target.name + ".tmp")
        with temporary.open("x", encoding="utf-8") as handle:
            for row in kept:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        target.rename(backup)
        temporary.replace(target)
        print(f"Backed up previous evaluation: {backup}", flush=True)


async def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parents[3]
    config_path = args.config if args.config.is_absolute() else project_dir / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result_dir = project_dir / config["results_dir"] / args.stage
    benchmark_dir = project_dir / config["benchmark_dir"]
    llm_config = config["llm"]
    embedding_config = config["embedding"]
    evaluation_config = config.get("evaluation", {})
    max_tokens = int(evaluation_config.get("max_new_tokens", 4096))
    retries = int(evaluation_config.get("response_retries", 1))
    if max_tokens <= 0 or retries < 0:
        raise ValueError("Invalid evaluation token budget or retry count")
    judge_config = {**llm_config, "temperature": 0, "max_new_tokens": max_tokens,
                    "max_callback_new_tokens": max_tokens,
                    "max_retries": int(evaluation_config.get("request_retries", 1))}
    _load_official_metrics(benchmark_dir)
    protocol = fingerprint({"version": EVALUATION_VERSION, "judge": judge_config,
                            "embedding": embedding_config, "retries": retries,
                            "metrics": {p.name: p.read_text(encoding="utf-8") for p in
                                        sorted((benchmark_dir / "Evaluation" / "metrics").glob("*.py"))}})
    methods = ["vector", "lightrag", "pathrag"] if args.backend == "all" else [args.backend]
    work = []
    for method in methods:
        rows = load_jsonl(result_dir / f"{method}.jsonl")
        target = result_dir / f"{method}_evaluated.jsonl"
        kept = reusable_rows(target, rows, protocol)
        if args.force:
            kept = []
        if any(row.get("error") for row in rows):
            raise ValueError(f"{method}: source contains failed generations; retry generation first")
        print(f"[{method}] total={len(rows)} reusable={len(kept)} pending={len(rows) - len(kept)}", flush=True)
        work.append((method, rows, target, kept))
    if args.check_only:
        return
    judge = build_chat_client(judge_config)
    embedding = None
    failures = 0
    try:
        embedding = build_embedding_client(embedding_config)
        for method, rows, target, kept in work:
            prepare_target(target, kept)
            done = {row["question_id"] for row in kept}
            with target.open("a", encoding="utf-8") as handle:
                for position, row in enumerate(rows, start=1):
                    if row["question_id"] in done:
                        continue
                    if row.get("error"):
                        raise RuntimeError(
                            f"Cannot evaluate failed result {row['question_id']}: {row['error']}"
                        )
                    print(f"[{method}] evaluate {position}/{len(rows)}", flush=True)
                    started = time.monotonic()
                    evaluated = await evaluate_official_row(
                        row, judge, embedding, benchmark_dir,
                        max_tokens=max_tokens, retries=retries, protocol=protocol,
                    )
                    handle.write(json.dumps(evaluated, ensure_ascii=False, allow_nan=False) + "\n")
                    handle.flush()
                    status = evaluated["evaluation"]["status"]
                    failures += status != "ok"
                    print(f"[{method}] {row['question_id']} {status} ({time.monotonic() - started:.1f}s)", flush=True)
                    if status != "ok":
                        print("    " + "; ".join(evaluated["evaluation"]["errors"]), flush=True)
    finally:
        judge.unload()
        if embedding is not None:
            embedding.unload()
    if failures:
        raise SystemExit(f"{failures} evaluations failed; scores are null. Rerun to retry failed rows.")


if __name__ == "__main__":
    asyncio.run(main())

