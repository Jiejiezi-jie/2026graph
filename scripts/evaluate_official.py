from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from src.official_backends.model_client import (
    TransformersChatClient,
    TransformersEmbeddingClient,
)
from src.official_evaluation import evaluate_official_row, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/official_local.json"))
    parser.add_argument("--stage", choices=["p0", "p1"], default="p0")
    parser.add_argument(
        "--backend", choices=["all", "vector", "lightrag", "pathrag"], default="all"
    )
    return parser.parse_args()


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {row["question_id"] for row in load_jsonl(path)}


async def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else project_dir / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result_dir = project_dir / config["results_dir"] / args.stage
    benchmark_dir = project_dir / config["benchmark_dir"]
    llm_config = config["llm"]
    embedding_config = config["embedding"]
    judge = TransformersChatClient(
        model_path=llm_config["model_path"],
        device=llm_config["device"],
        dtype=llm_config["dtype"],
        temperature=0,
        max_new_tokens=llm_config["max_new_tokens"],
        max_callback_new_tokens=llm_config["max_callback_new_tokens"],
    )
    embedding = TransformersEmbeddingClient(
        model_path=embedding_config["model_path"],
        device=embedding_config["device"],
        batch_size=embedding_config["batch_size"],
        max_length=embedding_config["max_length"],
        normalize=embedding_config["normalize"],
    )
    methods = ["vector", "lightrag", "pathrag"] if args.backend == "all" else [args.backend]
    try:
        for method in methods:
            source = result_dir / f"{method}.jsonl"
            target = result_dir / f"{method}_evaluated.jsonl"
            done = completed_ids(target)
            rows = load_jsonl(source)
            with target.open("a", encoding="utf-8") as handle:
                for position, row in enumerate(rows, start=1):
                    if row["question_id"] in done:
                        continue
                    if row.get("error"):
                        raise RuntimeError(
                            f"Cannot evaluate failed result {row['question_id']}: {row['error']}"
                        )
                    print(f"[{method}] evaluate {position}/{len(rows)}", flush=True)
                    evaluated = await evaluate_official_row(
                        row, judge, embedding, benchmark_dir
                    )
                    handle.write(json.dumps(evaluated, ensure_ascii=False) + "\n")
                    handle.flush()
    finally:
        judge.unload()
        embedding.unload()


if __name__ == "__main__":
    asyncio.run(main())

