"""Make one inexpensive call to each configured API client."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from src.official_backends.model_client import build_chat_client, build_embedding_client


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/official_api.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    llm = build_chat_client(config["llm"])
    embedding = build_embedding_client(config["embedding"])
    try:
        generation = await llm.generate(
            "Reply with exactly: API_OK",
            system_prompt="You are a connectivity probe. Do not add any explanation.",
            max_new_tokens=16,
        )
        vectors = await embedding.embed(
            [
                "Basal cell carcinoma is a common skin cancer.",
                "What is basal cell carcinoma?",
            ]
        )
        print(
            json.dumps(
                {
                    "llm_model": llm.model_name,
                    "llm_response": generation.text,
                    "llm_usage": llm.snapshot().__dict__,
                    "embedding_model": embedding.model_name,
                    "embedding_shape": list(vectors.shape),
                    "embedding_norms": [float((vector**2).sum() ** 0.5) for vector in vectors],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        llm.unload()
        embedding.unload()


if __name__ == "__main__":
    asyncio.run(main())
