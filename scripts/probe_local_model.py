from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

import torch
from src.data import chunk_by_word_window, load_medical_benchmark
from src.official_backends.model_client import TransformersChatClient


EXTRACTION_SYSTEM = (
    "Extract medical entities and explicit relationships from the text. Return one JSON "
    "object only, with keys entities and relations. entities is a list of objects with "
    "name, type, description. relations is a list of objects with source, target, "
    "relation. Return exactly one compact line with at most 3 entities and 3 relations. "
    "Do not use markdown or code fences."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--output", type=Path, default=Path("results_official/model_probe.json")
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    device_index = int(args.device.split(":", 1)[1])
    free, total = torch.cuda.mem_get_info(device_index)
    if free < 10 * 2**30:
        raise RuntimeError(
            f"Refusing model probe: {args.device} has only {free / 2**30:.2f} GiB free"
        )
    corpus, questions = load_medical_benchmark(args.benchmark_dir)
    chunks = chunk_by_word_window(corpus, size=180, overlap=40)[:5]
    client = TransformersChatClient(
        model_path=args.model_path,
        device=args.device,
        dtype="bfloat16",
        temperature=0,
        max_new_tokens=256,
        max_callback_new_tokens=512,
    )
    qa_rows = []
    for question in questions[:5]:
        response = await client.generate(
            question["question"],
            system_prompt="Answer the English medical question directly and concisely.",
        )
        qa_rows.append(
            {
                "id": question["id"],
                "question": question["question"],
                "answer": response.text,
                "nonempty": bool(response.text.strip()),
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            }
        )
    extraction_rows = []
    for index, chunk in enumerate(chunks):
        response = await client.generate(
            chunk, system_prompt=EXTRACTION_SYSTEM, max_new_tokens=512
        )
        parsed = None
        error = None
        try:
            cleaned = response.text.strip()
            fenced = re.fullmatch(
                r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.I
            )
            if fenced:
                cleaned = fenced.group(1)
            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict):
                raise TypeError("top-level output is not an object")
            if not isinstance(parsed.get("entities"), list):
                raise TypeError("entities is not a list")
            if not isinstance(parsed.get("relations"), list):
                raise TypeError("relations is not a list")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            parsed = None
        extraction_rows.append(
            {
                "chunk_index": index,
                "chunk": chunk,
                "raw_output": response.text,
                "parsed": parsed,
                "json_valid": parsed is not None,
                "error": error,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            }
        )
    result = {
        "model": str(args.model_path.resolve()),
        "device": args.device,
        "temperature": 0,
        "max_new_tokens": 256,
        "gpu_free_before_gib": free / 2**30,
        "gpu_total_gib": total / 2**30,
        "qa_success": sum(row["nonempty"] for row in qa_rows),
        "qa_total": len(qa_rows),
        "json_success": sum(row["json_valid"] for row in extraction_rows),
        "json_total": len(extraction_rows),
        "usage": client.snapshot().__dict__,
        "qa": qa_rows,
        "extractions": extraction_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({key: result[key] for key in result if key not in {"qa", "extractions"}}, indent=2))
    client.unload()


if __name__ == "__main__":
    asyncio.run(main())
