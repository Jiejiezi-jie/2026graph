"""Run real LightRAG/PathRAG indexing and retrieval on a tiny medical corpus."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from pathlib import Path

from src.backend.common.run_official_experiment import _build_backends
from src.backend.common.model_client import build_chat_client, build_embedding_client


SMOKE_CORPUS = """
Basal cell carcinoma (BCC) is a common skin cancer. Ultraviolet radiation can
damage DNA in skin cells and is a major risk factor for BCC. Organ transplant
recipients take immunosuppressive medicines to prevent graft rejection.
Immunosuppression weakens immune surveillance and increases the risk of BCC.
Mohs surgery removes thin layers of cancer-containing skin and is commonly used
for high-risk BCC in cosmetically sensitive areas.
""".strip()

SMOKE_QUESTION = (
    "Why can ultraviolet exposure and immunosuppression after organ transplantation "
    "both increase the risk of basal cell carcinoma?"
)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/official_api.json"))
    parser.add_argument(
        "--backend", choices=["all", "lightrag", "pathrag"], default="all"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Remove the generated smoke index/results before rebuilding",
    )
    args = parser.parse_args()
    project_dir = Path(__file__).resolve().parents[3]
    config_path = args.config if args.config.is_absolute() else project_dir / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config = {
        **config,
        "results_dir": str(Path(config["results_dir"]) / "smoke"),
        "chunk_tokens": 300,
        "chunk_overlap_tokens": 30,
        "lightrag_top_k": 10,
        "lightrag_chunk_top_k": 3,
        "pathrag_top_k": 10,
        "max_context_tokens": 5000,
    }
    output_dir = project_dir / config["results_dir"]
    if args.reset and output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"[smoke] removed previous output: {output_dir}", flush=True)
    llm = build_chat_client(config["llm"])
    embedding = build_embedding_client(config["embedding"])
    names = ["lightrag", "pathrag"] if args.backend == "all" else [args.backend]
    backends = _build_backends(names, project_dir, config, llm, embedding)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for name, backend in backends.items():
            print(f"[{name}] indexing tiny medical corpus", flush=True)
            index_stats = await backend.index(SMOKE_CORPUS)
            print(f"[{name}] querying", flush=True)
            row = await backend.query(SMOKE_QUESTION, "api-smoke-1")
            if not row["answer"].strip() or not row["contexts"]:
                raise RuntimeError(f"{name} returned an empty answer or context")
            if not row["entities"] or not row["relations"]:
                raise RuntimeError(f"{name} returned no entities or relations")
            result = {
                "status": "passed",
                "backend": name,
                "llm_model": llm.model_name,
                "embedding_model": embedding.model_name,
                "index": index_stats,
                "result": row,
            }
            output_path = output_dir / f"{name}.json"
            output_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"[{name}] passed: {output_path}", flush=True)
    finally:
        for backend in backends.values():
            await backend.close()
        llm.unload()
        embedding.unload()


if __name__ == "__main__":
    asyncio.run(main())
