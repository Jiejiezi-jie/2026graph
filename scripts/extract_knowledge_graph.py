#!/usr/bin/env python3
"""Extract and build the phase-three Dandy Dick knowledge graph.

Only ``data/processed/phase2/chunks.jsonl`` is accepted as document input.
LLM credentials are read from LLM_API_KEY, LLM_BASE_URL and LLM_MODEL and are
never persisted. Valid per-chunk responses are cached for deterministic,
offline post-processing.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Sequence

import networkx as nx


EXPECTED_CORPUS = "Novel-40700"
EXPECTED_TITLE = "Dandy Dick"
CACHE_VERSION = "phase3-cache-v1"
FORBIDDEN_SUPERVISION_KEYS = {"question", "answer", "evidence", "evidence_triple"}
SYMMETRIC_REVIEW_CATEGORIES = {
    "family": {"PARENT_OF", "SIBLING_OF", "MARRIED_TO"},
    "emotion": {"ROMANTIC_WITH", "FRIEND_OF"},
    "production": {
        "AUTHORED_BY",
        "PORTRAYED_BY",
        "PERFORMED_AT",
        "MANAGED_BY",
        "PRECEDED_BY",
    },
}


class Phase3Error(RuntimeError):
    """Raised when phase-three protocol or artifact validation fails."""


def canonical_json(value: Any, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def jsonl_dump(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(canonical_json(row) + "\n" for row in rows)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def normalize_ws(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def normalized_name_key(name: str) -> str:
    value = normalize_ws(name).strip(" \t\r\n.,;:!?\"'`_()[]{}")
    value = value.replace("’", "'").replace("‘", "'")
    value = re.sub(r"\s+", " ", value)
    return value.casefold()


def clean_name(name: str) -> str:
    return normalize_ws(name).strip(" \t\r\n,;:\"'`_")


def normalized_contains(text: str, candidate: str) -> bool:
    haystack = normalized_name_key(text)
    needle = normalized_name_key(candidate)
    if not needle:
        return False
    return bool(
        re.search(r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])", haystack)
    )


def evidence_matches(quote: str, chunk_text: str) -> bool:
    quote_norm = normalize_ws(quote)
    return bool(quote_norm) and quote_norm in normalize_ws(chunk_text)


def stable_id(prefix: str, *parts: str, width: int = 12) -> str:
    digest = sha256_text("\x1f".join(parts))[:width]
    slug = re.sub(r"[^a-z0-9]+", "_", parts[0].casefold()).strip("_")[:28]
    return f"{prefix}_{slug}_{digest}" if slug else f"{prefix}_{digest}"


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise Phase3Error(f"文件不存在：{path}") from exc
    except json.JSONDecodeError as exc:
        raise Phase3Error(f"JSON 无效：{path}: {exc}") from exc


def load_chunks(path: Path, expected_count: int) -> list[dict[str, Any]]:
    if path.name != "chunks.jsonl" or path.parent.name != "phase2":
        raise Phase3Error("阶段3唯一正文输入必须是 phase2/chunks.jsonl")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise Phase3Error(f"阶段2切块不存在：{path}") from exc
    chunks: list[dict[str, Any]] = []
    required = {
        "chunk_id",
        "corpus_name",
        "title",
        "section_id",
        "section_type",
        "act",
        "scene",
        "speakers",
        "text",
    }
    for line_no, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Phase3Error(f"chunks.jsonl 第 {line_no} 行无效：{exc}") from exc
        if not isinstance(row, dict) or not required <= set(row):
            raise Phase3Error(f"chunks.jsonl 第 {line_no} 行缺少必要字段")
        leaked = FORBIDDEN_SUPERVISION_KEYS & set(row)
        if leaked:
            raise Phase3Error(f"输入 chunk 含禁止监督字段：{sorted(leaked)}")
        if row["corpus_name"] != EXPECTED_CORPUS or row["title"] != EXPECTED_TITLE:
            raise Phase3Error(f"输入 chunk 语料不匹配：{row.get('chunk_id')}")
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise Phase3Error(f"输入 chunk 正文为空：{row.get('chunk_id')}")
        chunks.append(row)
    if len(chunks) != expected_count:
        raise Phase3Error(f"预期 {expected_count} 个 chunk，实际 {len(chunks)}")
    ids = [row["chunk_id"] for row in chunks]
    if len(ids) != len(set(ids)):
        raise Phase3Error("输入 chunk_id 不唯一")
    return chunks


def load_config(path: Path) -> dict[str, Any]:
    config = load_json(path)
    if not isinstance(config, dict):
        raise Phase3Error("kg_config.json 必须为 JSON 对象")
    fixed = {"temperature": 0, "max_retries": 2, "expected_chunk_count": 62}
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in fixed.items()
        if config.get(key) != value
    }
    if mismatches:
        raise Phase3Error(f"KG 固定配置不匹配：{mismatches}")
    for key in ("entity_types", "relation_types", "symmetric_relations"):
        if not isinstance(config.get(key), list) or not config[key]:
            raise Phase3Error(f"配置缺少非空数组：{key}")
    if not set(config["symmetric_relations"]) <= set(config["relation_types"]):
        raise Phase3Error("symmetric_relations 含未允许关系")
    return config


def runtime_from_env(*, required: bool = True) -> dict[str, str]:
    names = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")
    values = {name: os.environ.get(name, "").strip() for name in names}
    missing = [name for name, value in values.items() if not value]
    if required and missing:
        raise Phase3Error(f"缺少 LLM 环境变量：{', '.join(missing)}")
    return values


def extraction_fingerprint(prompt: str, config: dict[str, Any], model: str) -> str:
    extraction_config = {
        key: config[key]
        for key in (
            "schema_version",
            "temperature",
            "max_retries",
            "max_tokens",
            "thinking",
            "entity_types",
            "relation_types",
            "minimum_confidence",
        )
    }
    return sha256_text(
        canonical_json(
            {"prompt": prompt, "config": extraction_config, "model": model}
        )
    )


def output_name_lookup(entities: Sequence[dict[str, Any]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for entity in entities:
        for value in [entity["name"], *entity["aliases"]]:
            key = normalized_name_key(value)
            if key and key not in lookup:
                lookup[key] = entity["name"]
    return lookup


def validate_extraction(
    payload: Any, chunk: dict[str, Any], config: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(payload, dict):
        raise Phase3Error("模型输出不是 JSON 对象")
    expected_top = {"entities", "relations", "no_fact_reason"}
    if set(payload) != expected_top:
        raise Phase3Error(
            f"模型输出顶层字段必须严格为 {sorted(expected_top)}，实际 {sorted(payload)}"
        )
    if not isinstance(payload["entities"], list) or not isinstance(
        payload["relations"], list
    ):
        raise Phase3Error("entities/relations 必须为数组")
    if payload["no_fact_reason"] is not None and not isinstance(
        payload["no_fact_reason"], str
    ):
        raise Phase3Error("no_fact_reason 必须为字符串或 null")

    allowed_entities = set(config["entity_types"])
    allowed_relations = set(config["relation_types"])
    forbidden = {
        normalized_name_key(value) for value in config["forbidden_entity_names"]
    }
    warnings: list[str] = []
    entities: list[dict[str, Any]] = []
    seen_entities: set[tuple[str, str]] = set()
    entity_fields = {"name", "type", "description", "aliases"}
    for index, raw in enumerate(payload["entities"]):
        if not isinstance(raw, dict) or set(raw) != entity_fields:
            warnings.append(f"entity[{index}]_invalid_schema")
            continue
        name = clean_name(raw["name"]) if isinstance(raw["name"], str) else ""
        entity_type = raw["type"]
        description = normalize_ws(raw["description"]) if isinstance(raw["description"], str) else ""
        aliases = raw["aliases"]
        if (
            not name
            or entity_type not in allowed_entities
            or not description
            or not isinstance(aliases, list)
            or not all(isinstance(value, str) and clean_name(value) for value in aliases)
        ):
            warnings.append(f"entity[{index}]_invalid_value")
            continue
        if normalized_name_key(name) in forbidden:
            warnings.append(f"entity[{index}]_forbidden_name")
            continue
        aliases_clean = []
        for alias in aliases:
            value = clean_name(alias)
            if normalized_name_key(value) not in forbidden and value not in aliases_clean:
                aliases_clean.append(value)
        key = (normalized_name_key(name), entity_type)
        if key in seen_entities:
            warnings.append(f"entity[{index}]_duplicate")
            continue
        seen_entities.add(key)
        entities.append(
            {
                "name": name,
                "type": entity_type,
                "description": description,
                "aliases": aliases_clean,
            }
        )

    lookup = output_name_lookup(entities)
    relation_fields = {
        "source",
        "target",
        "relation",
        "description",
        "evidence_quote",
        "source_chunk_id",
        "confidence",
    }
    relations: list[dict[str, Any]] = []
    for index, raw in enumerate(payload["relations"]):
        if not isinstance(raw, dict) or set(raw) != relation_fields:
            warnings.append(f"relation[{index}]_invalid_schema")
            continue
        source = clean_name(raw["source"]) if isinstance(raw["source"], str) else ""
        target = clean_name(raw["target"]) if isinstance(raw["target"], str) else ""
        relation = raw["relation"]
        description = normalize_ws(raw["description"]) if isinstance(raw["description"], str) else ""
        quote = normalize_ws(raw["evidence_quote"]) if isinstance(raw["evidence_quote"], str) else ""
        confidence = raw["confidence"]
        if (
            not source
            or not target
            or relation not in allowed_relations
            or not description
            or raw["source_chunk_id"] != chunk["chunk_id"]
            or not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not 0 <= float(confidence) <= 1
        ):
            warnings.append(f"relation[{index}]_invalid_value")
            continue
        if not evidence_matches(quote, chunk["text"]):
            warnings.append(f"relation[{index}]_evidence_mismatch")
            continue
        source_key = normalized_name_key(source)
        target_key = normalized_name_key(target)
        if source_key not in lookup or target_key not in lookup:
            warnings.append(f"relation[{index}]_endpoint_missing")
            continue
        if float(confidence) < float(config["minimum_confidence"]):
            warnings.append(f"relation[{index}]_below_confidence")
            continue
        relations.append(
            {
                "source": lookup[source_key],
                "target": lookup[target_key],
                "relation": relation,
                "description": description,
                "evidence_quote": quote,
                "source_chunk_id": chunk["chunk_id"],
                "confidence": float(confidence),
            }
        )

    no_fact = normalize_ws(payload["no_fact_reason"] or "") or None
    if no_fact and (entities or relations):
        warnings.append("no_fact_reason_ignored_because_facts_exist")
        no_fact = None
    if not entities and not relations and not no_fact:
        raise Phase3Error("模型既未返回合法事实，也未给出 no_fact_reason")
    return {"entities": entities, "relations": relations, "no_fact_reason": no_fact}, warnings


def parse_api_response(
    response: dict[str, Any], chunk: dict[str, Any], config: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise Phase3Error("API 响应缺少 choices[0].message.content") from exc
    if not isinstance(content, str) or not content.strip():
        raise Phase3Error("API 返回空 content")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise Phase3Error(f"模型 content 不是合法 JSON：{exc}") from exc
    return validate_extraction(payload, chunk, config)


def usage_from_response(response: dict[str, Any]) -> dict[str, int]:
    usage = response.get("usage") or {}
    keys = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
    )
    return {
        key: int(usage.get(key, 0) or 0)
        for key in keys
        if isinstance(usage.get(key, 0), (int, float))
    }


class OpenAICompatibleClient:
    def __init__(self, runtime: dict[str, str], config: dict[str, Any], prompt: str):
        self.api_key = runtime["LLM_API_KEY"]
        self.base_url = runtime["LLM_BASE_URL"].rstrip("/")
        self.model = runtime["LLM_MODEL"]
        self.config = config
        self.prompt = prompt

    def call_once(self, chunk: dict[str, Any]) -> tuple[dict[str, Any], float]:
        metadata = {
            key: chunk[key]
            for key in (
                "chunk_id",
                "corpus_name",
                "title",
                "section_id",
                "section_type",
                "act",
                "scene",
                "speakers",
            )
        }
        user_content = (
            "CHUNK METADATA JSON:\n"
            + canonical_json(metadata, pretty=True)
            + "SOURCE CHUNK:\n"
            + chunk["text"]
        )
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.config["temperature"],
            "max_tokens": self.config["max_tokens"],
            "response_format": {"type": "json_object"},
            "thinking": {"type": self.config["thinking"]},
            "stream": False,
        }
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=canonical_json(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "2026graph-phase3/1.0",
            },
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(
                request, timeout=self.config["timeout_seconds"]
            ) as reply:
                response_bytes = reply.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:800]
            raise Phase3Error(f"HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise Phase3Error(f"网络错误：{exc.reason}") from exc
        elapsed = time.monotonic() - started
        try:
            response = json.loads(response_bytes)
        except json.JSONDecodeError as exc:
            raise Phase3Error("API 响应体不是 JSON") from exc
        return response, elapsed


def cache_path(cache_dir: Path, chunk_id: str) -> Path:
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", chunk_id):
        raise Phase3Error(f"不安全的 chunk_id：{chunk_id}")
    return cache_dir / f"{chunk_id}.json"


def validate_cache(
    cache: Any,
    chunk: dict[str, Any],
    config: dict[str, Any],
    fingerprint: str,
    model: str,
) -> dict[str, Any]:
    if not isinstance(cache, dict):
        raise Phase3Error("缓存不是 JSON 对象")
    required = {
        "cache_version",
        "chunk_id",
        "input_sha256",
        "extraction_fingerprint",
        "model",
        "outcome",
        "request_audit",
    }
    if not required <= set(cache):
        raise Phase3Error("缓存字段不完整")
    if (
        cache["cache_version"] != CACHE_VERSION
        or cache["chunk_id"] != chunk["chunk_id"]
        or cache["input_sha256"] != sha256_text(chunk["text"])
        or cache["extraction_fingerprint"] != fingerprint
        or cache["model"] != model
    ):
        raise Phase3Error("缓存指纹与当前输入/配置不匹配")
    if cache["outcome"] == "failure":
        return cache
    if cache["outcome"] not in {"success", "no_fact"}:
        raise Phase3Error("缓存 outcome 无效")
    validated, warnings = validate_extraction(cache.get("validated_output"), chunk, config)
    expected_outcome = "no_fact" if validated["no_fact_reason"] else "success"
    if expected_outcome != cache["outcome"]:
        raise Phase3Error("缓存 outcome 与内容不一致")
    result = dict(cache)
    result["validated_output"] = validated
    result["validation_warnings"] = warnings
    return result


def extract_chunks(
    chunks: Sequence[dict[str, Any]],
    config: dict[str, Any],
    prompt: str,
    runtime: dict[str, str],
    cache_dir: Path,
    *,
    offline: bool,
    retry_failures: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    model = runtime["LLM_MODEL"]
    fingerprint = extraction_fingerprint(prompt, config, model)
    client = None if offline else OpenAICompatibleClient(runtime, config, prompt)
    outcomes: list[dict[str, Any]] = []
    cache_hits = 0
    api_calls = 0
    for position, chunk in enumerate(chunks, 1):
        path = cache_path(cache_dir, chunk["chunk_id"])
        cached: dict[str, Any] | None = None
        cache_error: str | None = None
        if path.exists():
            try:
                cached = validate_cache(
                    load_json(path), chunk, config, fingerprint, model
                )
            except Phase3Error as exc:
                cache_error = str(exc)
        if cached is not None and (cached["outcome"] != "failure" or not retry_failures):
            cache_hits += 1
            outcomes.append(cached)
            print(
                f"[{position:02d}/{len(chunks)}] {chunk['chunk_id']}: cache {cached['outcome']}",
                flush=True,
            )
            continue
        if offline:
            reason = cache_error or "缓存不存在"
            raise Phase3Error(f"离线重建缺少有效缓存 {chunk['chunk_id']}：{reason}")
        assert client is not None
        attempts: list[dict[str, Any]] = []
        raw_responses: list[dict[str, Any]] = []
        success_cache: dict[str, Any] | None = None
        for attempt in range(config["max_retries"] + 1):
            api_calls += 1
            attempt_started = dt.datetime.now(dt.timezone.utc).isoformat()
            try:
                response, elapsed = client.call_once(chunk)
                raw_responses.append(response)
                validated, warnings = parse_api_response(response, chunk, config)
                usage = usage_from_response(response)
                attempts.append(
                    {
                        "attempt": attempt + 1,
                        "status": "success",
                        "started_at_utc": attempt_started,
                        "elapsed_seconds": round(elapsed, 6),
                        "token_usage": usage,
                        "finish_reason": (
                            response.get("choices", [{}])[0].get("finish_reason")
                        ),
                    }
                )
                outcome = "no_fact" if validated["no_fact_reason"] else "success"
                success_cache = {
                    "cache_version": CACHE_VERSION,
                    "chunk_id": chunk["chunk_id"],
                    "input_sha256": sha256_text(chunk["text"]),
                    "extraction_fingerprint": fingerprint,
                    "model": model,
                    "outcome": outcome,
                    "validated_output": validated,
                    "validation_warnings": warnings,
                    "request_audit": {
                        "attempt_count": len(attempts),
                        "attempts": attempts,
                        "total_elapsed_seconds": round(
                            sum(item["elapsed_seconds"] for item in attempts), 6
                        ),
                        "token_usage": usage,
                    },
                    "raw_response": response,
                }
                break
            except Exception as exc:  # retry all parse/network failures with audit
                elapsed = max(
                    0.0,
                    (
                        dt.datetime.now(dt.timezone.utc)
                        - dt.datetime.fromisoformat(attempt_started)
                    ).total_seconds(),
                )
                attempts.append(
                    {
                        "attempt": attempt + 1,
                        "status": "failed",
                        "started_at_utc": attempt_started,
                        "elapsed_seconds": round(elapsed, 6),
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:800],
                    }
                )
                if attempt < config["max_retries"]:
                    time.sleep(config["retry_backoff_seconds"] * (attempt + 1))
        if success_cache is None:
            success_cache = {
                "cache_version": CACHE_VERSION,
                "chunk_id": chunk["chunk_id"],
                "input_sha256": sha256_text(chunk["text"]),
                "extraction_fingerprint": fingerprint,
                "model": model,
                "outcome": "failure",
                "request_audit": {
                    "attempt_count": len(attempts),
                    "attempts": attempts,
                    "total_elapsed_seconds": round(
                        sum(item["elapsed_seconds"] for item in attempts), 6
                    ),
                    "token_usage": {},
                },
                "raw_responses": raw_responses,
                "failure_reason": attempts[-1].get("error", "unknown failure"),
            }
        write_text_atomic(path, canonical_json(success_cache, pretty=True))
        outcomes.append(success_cache)
        print(
            f"[{position:02d}/{len(chunks)}] {chunk['chunk_id']}: API {success_cache['outcome']}",
            flush=True,
        )

    summary = {
        "api_calls_this_run": api_calls,
        "cache_hits_this_run": cache_hits,
        "extraction_fingerprint": fingerprint,
    }
    return outcomes, summary


class AliasResolver:
    def __init__(self, value: dict[str, Any]):
        if not isinstance(value, dict):
            raise Phase3Error("alias_overrides.json 必须为 JSON 对象")
        self.value = value
        self.explicit: dict[tuple[str, str], str] = {}
        self.explicit_targets: dict[str, set[tuple[str, str]]] = collections.defaultdict(set)
        self.canonical_types: dict[str, str] = {}
        self.aliases_by_canonical: dict[tuple[str, str], list[str]] = {}
        for item in value.get("canonical_entities", []):
            canonical = clean_name(item["canonical"])
            entity_type = item["type"]
            self.canonical_types[canonical] = entity_type
            configured_aliases = [canonical, *item.get("aliases", [])]
            self.aliases_by_canonical[(canonical, entity_type)] = configured_aliases
            for alias in configured_aliases:
                key = (normalized_name_key(alias), entity_type)
                old = self.explicit.get(key)
                if old is not None and old != canonical:
                    raise Phase3Error(f"别名覆盖冲突：{alias}")
                self.explicit[key] = canonical
                self.explicit_targets[key[0]].add((normalized_name_key(canonical), entity_type))
        self.blocked: dict[str, dict[str, Any]] = {
            normalized_name_key(item["alias"]): item
            for item in value.get("blocked_aliases", [])
        }

    def resolve(self, name: str, entity_type: str) -> tuple[str | None, str]:
        key = normalized_name_key(name)
        if key in self.blocked:
            return None, "blocked_ambiguous"
        explicit = self.explicit.get((key, entity_type))
        if explicit:
            return explicit, "explicit_override"
        for (alias_key, expected_type), canonical in self.explicit.items():
            if alias_key == key and expected_type != entity_type:
                return None, "type_conflict_omitted"
        return clean_name(name), "normalized_exact"

    def aliases_for(self, canonical: str, entity_type: str) -> list[str]:
        return self.aliases_by_canonical.get((canonical, entity_type), [canonical])

    def unique_known_target(self, name: str) -> tuple[str, str] | None:
        targets = self.explicit_targets.get(normalized_name_key(name), set())
        return next(iter(targets)) if len(targets) == 1 else None


def relation_support_reason(
    relation: dict[str, Any],
    source_mentions: Sequence[str],
    target_mentions: Sequence[str],
    config: dict[str, Any],
    source_node: dict[str, Any],
    target_node: dict[str, Any],
) -> str | None:
    """Conservatively reject weak or semantically unsupported model edges."""
    if relation["relation"] in set(config.get("weak_relation_types", [])):
        return "weak_relation_excluded"
    quote = relation["evidence_quote"]
    relation_type = relation["relation"]
    source_name = source_node["name"]
    target_name = target_node["name"]
    if relation_type == "PARENT_OF" and (source_name, target_name) not in {
        ("Augustin Jedd", "Salome Jedd"),
        ("Augustin Jedd", "Sheba Jedd"),
    }:
        return "unsupported_parent_direction"
    if relation_type == "ARRESTS" and source_name != "Noah Topping":
        return "unsupported_arrest_direction"
    if relation_type == "OWNS":
        if target_node["type"] in {"CHARACTER", "PERSON", "EVENT", "ORGANIZATION"}:
            return "ownership_target_type_invalid"
        if target_name == "Dandy Dick" and source_name not in {
            "Georgiana Tidman",
            "Sir Tristram Mardon",
        }:
            return "dandy_owner_not_explicit"
    if relation_type == "AUTHORED_BY" and not (
        source_node["type"] == "CREATIVE_WORK" and target_node["type"] == "PERSON"
    ):
        return "authorship_endpoint_types_invalid"
    if relation_type == "PORTRAYED_BY" and not (
        source_node["type"] == "CHARACTER" and target_node["type"] == "PERSON"
    ):
        return "portrayal_endpoint_types_invalid"
    if (
        relation_type == "PORTRAYED_BY"
        and not relation["source_chunk_id"].startswith("dd_cast_")
    ):
        return "portrayal_not_cast_evidence"
    if relation_type == "PERFORMED_AT" and not (
        source_node["type"] == "CREATIVE_WORK" and target_node["type"] == "LOCATION"
    ):
        return "performance_endpoint_types_invalid"
    if relation_type == "LOCATED_AT":
        if target_node["type"] != "LOCATION":
            return "location_target_type_invalid"
        rejected_pairs = {
            ("Augustin Jedd", "Durnstone"),
            ("Augustin Jedd", "Deanery Stables"),
            ("Blore", "Tattersall's Ring"),
            ("Georgiana Tidman", "Pear Tree Lane"),
            ("Hatcham", "The Swan Inn"),
            ("Major Tarver", "London"),
            ("Major Tarver", "St. Marvells"),
            ("Noah Topping", "Durnstone"),
            ("Nugent Darbey", "St. Marvells"),
        }
        if (source_name, target_name) in rejected_pairs:
            return "location_not_asserted_for_source"
        if any(term in quote.casefold() for term in ("shan't be took", "won't be taken")):
            return "negated_or_future_location"
    if (
        relation_type == "PARTICIPATES_IN"
        and source_name == "Augustin Jedd"
        and target_name == "St. Marvells Spring Meeting"
    ):
        return "event_mentioned_not_participated"
    if (
        relation_type == "PARTICIPATES_IN"
        and source_name == "Blore"
        and target_name == "Durnstone Handicap"
    ):
        return "racing_tip_not_participation"
    if len(quote.split()) < 4:
        return "evidence_too_short"
    if not any(normalized_contains(quote, value) for value in source_mentions):
        return "source_not_named_in_evidence"
    if not any(normalized_contains(quote, value) for value in target_mentions):
        return "target_not_named_in_evidence"
    keyword_groups = {
        "PARENT_OF": ("father", "mother", "daughter", "son", "child", "papa", "papsey"),
        "SIBLING_OF": ("sister", "brother", "sibling"),
        "MARRIED_TO": ("married", "wife", "husband"),
        "ROMANTIC_WITH": ("love", "marry", "proposal", "propose", "engaged", "jilted", "passion", "sweetheart", "paying attention"),
        "FRIEND_OF": ("friend",),
        "EMPLOYED_BY": ("butler", "servant", "groom", "employ", "worked for", "cook", "manager", "lessee"),
        "OWNS": ("owner", "owns", "own", "belong", "bought", "buy", "purchased", "property"),
        "PARTICIPATES_IN": (
            "race",
            "races",
            "meeting",
            "handicap",
            "ball",
            "auction",
            "dinner",
            "concert",
            "attend",
            "perform",
            "back",
            "subscribed",
            "offered",
            "entered",
            "runs",
            "won",
        ),
        "HELPS": ("help", "save", "rescue", "assist", "aid", "release", "unlock", "free"),
        "ARRESTS": ("arrest", "collared", "lock-up", "prisoner", "handcuff", "handcuffs", "took my man"),
        "OPPOSES": ("oppose", "against", "denounce", "refuse", "won't", "will not", "forbid", "dislike"),
        "CAUSES": ("cause", "because", "result", "poison", "killed", "died from", "made "),
        "BEFORE": ("before", "earlier", "then"),
        "AFTER": ("after", "later", "then"),
        "AUTHORED_BY": ("author", "by", "wrote", "written"),
        "PERFORMED_AT": ("performed", "production", "played at", "theatre"),
        "MANAGED_BY": ("manager", "lessee", "management", "lease"),
        "PRECEDED_BY": ("preceded", "before", "following", "succeeded"),
    }
    keywords = keyword_groups.get(relation_type)
    if keywords and not any(normalized_contains(quote, term) for term in keywords):
        return "predicate_not_explicit_in_evidence"
    return None


def expand_evidence_quote(
    quote: str,
    chunk_text: str,
    source_mentions: Sequence[str],
    target_mentions: Sequence[str],
    *,
    context_chars: int = 140,
    max_chars: int = 360,
) -> str:
    """Expand a model quote with nearby named speaker context, still verbatim."""
    start = chunk_text.find(quote)
    if start < 0:
        return quote
    end = start + len(quote)
    lowered = chunk_text.casefold()
    for mention_index, mentions in enumerate((source_mentions, target_mentions)):
        if any(normalized_contains(chunk_text[start:end], value) for value in mentions):
            continue
        previous_candidates: list[tuple[int, int]] = []
        following_candidates: list[tuple[int, int]] = []
        window_start = max(0, start - context_chars)
        window_end = min(len(chunk_text), end + context_chars)
        for mention in mentions:
            needle = clean_name(mention).casefold()
            if len(needle) < 3:
                continue
            position = lowered.rfind(needle, window_start, start)
            if position >= 0:
                previous_candidates.append((position, position + len(needle)))
            position = lowered.find(needle, end, window_end)
            if position >= 0:
                following_candidates.append((position, position + len(needle)))
        candidates = (
            previous_candidates
            if mention_index == 0 and previous_candidates
            else previous_candidates + following_candidates
        )
        if candidates:
            position, position_end = min(
                candidates,
                key=lambda item: min(abs(start - item[1]), abs(item[0] - end)),
            )
            proposed_start = min(start, position)
            proposed_end = max(end, position_end)
            if proposed_end - proposed_start <= max_chars:
                start, end = proposed_start, proposed_end
    return normalize_ws(chunk_text[start:end])


def correct_relation_direction(
    relation_type: str,
    source_node: dict[str, Any],
    target_node: dict[str, Any],
) -> bool:
    """Return true when a high-confidence schema rule requires swapping endpoints."""
    if relation_type == "AUTHORED_BY":
        return source_node["type"] == "PERSON" and target_node["type"] == "CREATIVE_WORK"
    if relation_type == "PORTRAYED_BY":
        return source_node["type"] == "PERSON" and target_node["type"] == "CHARACTER"
    if relation_type == "PERFORMED_AT":
        return source_node["type"] == "LOCATION" and target_node["type"] == "CREATIVE_WORK"
    if relation_type == "MANAGED_BY":
        return source_node["type"] == "PERSON" and target_node["type"] in {"LOCATION", "ORGANIZATION"}
    if relation_type == "PARENT_OF":
        return source_node["name"] in {"Salome Jedd", "Sheba Jedd"} and target_node["name"] == "Augustin Jedd"
    if relation_type == "EMPLOYED_BY":
        known_employee_employer = {
            ("Augustin Jedd", "Blore"),
            ("Augustin Jedd", "Hannah Topping"),
            ("Sir Tristram Mardon", "Hatcham"),
        }
        return (source_node["name"], target_node["name"]) in known_employee_employer
    return False


def evidence_quality_score(relation_type: str, quote: str) -> int:
    strong_terms = {
        "PARENT_OF": ("parent", "daughter", "daughters", "father", "mother", "children"),
        "SIBLING_OF": ("sister", "brother", "sibling"),
        "MARRIED_TO": ("wife", "husband", "married", "wed", "bride"),
        "ROMANTIC_WITH": ("proposed", "proposal", "marry", "love", "paying attention"),
        "FRIEND_OF": ("friend", "acquaintance"),
        "OWNS": ("owner", "belongs", "half share", "purchased", "bought"),
        "ARRESTS": ("handcuffs", "collared", "arrest", "lock-up", "prisoner"),
        "AUTHORED_BY": ("author", "wrote", "written", " by "),
        "PORTRAYED_BY": ("mr.", "mrs.", "miss"),
        "PERFORMED_AT": ("performed", "produced", "transferred", "ran"),
        "PARTICIPATES_IN": (
            "subscribed",
            "offered",
            "entered",
            "runs",
            "races",
            "won",
            "back",
        ),
    }
    padded = f" {quote.casefold()} "
    return sum(1 for term in strong_terms.get(relation_type, ()) if term in padded)


def merge_graph_data(
    chunks: Sequence[dict[str, Any]],
    outcomes: Sequence[dict[str, Any]],
    config: dict[str, Any],
    alias_config: dict[str, Any],
) -> dict[str, Any]:
    resolver = AliasResolver(alias_config)
    chunk_by_id = {row["chunk_id"]: row for row in chunks}
    nodes_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    local_maps: dict[str, dict[str, tuple[str, str]]] = {}
    alias_records: dict[tuple[str, str, str], dict[str, Any]] = {}
    unresolved: list[dict[str, Any]] = []
    postprocess_warnings: list[dict[str, Any]] = []
    pending_type_conflicts: list[dict[str, Any]] = []

    for blocked in alias_config.get("blocked_aliases", []):
        matches = [
            chunk["chunk_id"]
            for chunk in chunks
            if normalized_name_key(blocked["alias"])
            in normalized_name_key(chunk["text"])
        ]
        unresolved.append(
            {
                "alias": blocked["alias"],
                "reason": blocked["reason"],
                "candidates": blocked.get("candidates", []),
                "source_chunk_ids": matches,
            }
        )

    for outcome in outcomes:
        chunk_id = outcome["chunk_id"]
        local: dict[str, tuple[str, str]] = {}
        local_maps[chunk_id] = local
        if outcome["outcome"] != "success":
            continue
        for entity in outcome["validated_output"]["entities"]:
            canonical, method = resolver.resolve(entity["name"], entity["type"])
            if canonical is None:
                if method == "type_conflict_omitted":
                    known_target = resolver.unique_known_target(entity["name"])
                    if known_target is not None and known_target in nodes_by_key:
                        local[normalized_name_key(entity["name"])] = known_target
                        postprocess_warnings.append(
                            {
                                "chunk_id": chunk_id,
                                "kind": "type_conflict_relinked_to_known_entity",
                                "name": entity["name"],
                                "canonical": nodes_by_key[known_target]["name"],
                            }
                        )
                        continue
                    if known_target is not None:
                        pending_type_conflicts.append(
                            {
                                "chunk_id": chunk_id,
                                "name": entity["name"],
                                "known_target": known_target,
                            }
                        )
                        continue
                    unresolved.append(
                        {
                            "alias": entity["name"],
                            "reason": "Model assigned a known entity alias a conflicting type; the conflicting node was omitted and no unique in-chunk endpoint could be linked.",
                            "candidates": [],
                            "source_chunk_ids": [chunk_id],
                        }
                    )
                postprocess_warnings.append(
                    {
                        "chunk_id": chunk_id,
                        "kind": method,
                        "name": entity["name"],
                    }
                )
                continue
            key = (normalized_name_key(canonical), entity["type"])
            if key not in nodes_by_key:
                node_id = stable_id("n", canonical, entity["type"])
                nodes_by_key[key] = {
                    "node_id": node_id,
                    "name": canonical,
                    "type": entity["type"],
                    "description": entity["description"],
                    "descriptions": [],
                    "aliases": [],
                    "source_chunk_ids": [],
                }
            node = nodes_by_key[key]
            if entity["description"] not in node["descriptions"]:
                node["descriptions"].append(entity["description"])
            if chunk_id not in node["source_chunk_ids"]:
                node["source_chunk_ids"].append(chunk_id)
            local[normalized_name_key(entity["name"])] = key
            local[normalized_name_key(canonical)] = key
            for raw_name in [entity["name"], *entity["aliases"]]:
                alias = clean_name(raw_name)
                alias_key = normalized_name_key(alias)
                alias_canonical, alias_method = resolver.resolve(alias, entity["type"])
                if alias_canonical != canonical:
                    if raw_name != entity["name"]:
                        postprocess_warnings.append(
                            {
                                "chunk_id": chunk_id,
                                "kind": "model_alias_not_auto_merged",
                                "name": alias,
                                "candidate": canonical,
                            }
                        )
                    continue
                if alias != canonical and alias not in node["aliases"]:
                    node["aliases"].append(alias)
                record_key = (alias_key, entity["type"], canonical)
                alias_records[record_key] = {
                    "alias": alias,
                    "canonical": canonical,
                    "type": entity["type"],
                    "method": method if raw_name == entity["name"] else "model_alias_then_" + alias_method,
                }

    for pending in pending_type_conflicts:
        chunk_id = pending["chunk_id"]
        known_target = pending["known_target"]
        if known_target in nodes_by_key:
            local_maps[chunk_id][normalized_name_key(pending["name"])] = known_target
            postprocess_warnings.append(
                {
                    "chunk_id": chunk_id,
                    "kind": "type_conflict_relinked_to_known_entity",
                    "name": pending["name"],
                    "canonical": nodes_by_key[known_target]["name"],
                }
            )
        else:
            unresolved.append(
                {
                    "alias": pending["name"],
                    "reason": "Model assigned a known entity alias a conflicting type; the conflicting node was omitted and no canonical node was extracted.",
                    "candidates": [],
                    "source_chunk_ids": [chunk_id],
                }
            )
            postprocess_warnings.append(
                {
                    "chunk_id": chunk_id,
                    "kind": "type_conflict_omitted",
                    "name": pending["name"],
                }
            )

    edges_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    symmetric = set(config["symmetric_relations"])
    for outcome in outcomes:
        chunk_id = outcome["chunk_id"]
        if outcome["outcome"] != "success":
            continue
        local = local_maps[chunk_id]
        raw_entities = {
            normalized_name_key(entity["name"]): entity
            for entity in outcome["validated_output"]["entities"]
        }
        for relation in outcome["validated_output"]["relations"]:
            source_key = local.get(normalized_name_key(relation["source"]))
            target_key = local.get(normalized_name_key(relation["target"]))
            if source_key is None or target_key is None:
                postprocess_warnings.append(
                    {
                        "chunk_id": chunk_id,
                        "kind": "relation_endpoint_unresolved",
                        "relation": relation["relation"],
                    }
                )
                continue
            source_node = nodes_by_key[source_key]
            target_node = nodes_by_key[target_key]
            source_raw = raw_entities.get(normalized_name_key(relation["source"]), {})
            target_raw = raw_entities.get(normalized_name_key(relation["target"]), {})
            source_mentions = list(
                dict.fromkeys(
                    [
                        relation["source"],
                        *source_raw.get("aliases", []),
                        *resolver.aliases_for(source_node["name"], source_node["type"]),
                    ]
                )
            )
            target_mentions = list(
                dict.fromkeys(
                    [
                        relation["target"],
                        *target_raw.get("aliases", []),
                        *resolver.aliases_for(target_node["name"], target_node["type"]),
                    ]
                )
            )
            relation = dict(relation)
            if correct_relation_direction(relation["relation"], source_node, target_node):
                source_key, target_key = target_key, source_key
                source_node, target_node = target_node, source_node
                source_mentions, target_mentions = target_mentions, source_mentions
                postprocess_warnings.append(
                    {
                        "chunk_id": chunk_id,
                        "kind": "relation_direction_corrected",
                        "relation": relation["relation"],
                        "source": source_node["name"],
                        "target": target_node["name"],
                    }
                )
            relation["evidence_quote"] = expand_evidence_quote(
                relation["evidence_quote"],
                chunk_by_id[chunk_id]["text"],
                source_mentions,
                target_mentions,
            )
            rejection = relation_support_reason(
                relation,
                source_mentions,
                target_mentions,
                config,
                source_node,
                target_node,
            )
            if rejection:
                postprocess_warnings.append(
                    {
                        "chunk_id": chunk_id,
                        "kind": "relation_rejected_" + rejection,
                        "relation": relation["relation"],
                        "source": source_node["name"],
                        "target": target_node["name"],
                    }
                )
                continue
            if source_node["node_id"] == target_node["node_id"]:
                postprocess_warnings.append(
                    {
                        "chunk_id": chunk_id,
                        "kind": "alias_induced_self_loop_omitted",
                        "relation": relation["relation"],
                        "entity": source_node["name"],
                    }
                )
                continue
            directed = relation["relation"] not in symmetric
            source_id, target_id = source_node["node_id"], target_node["node_id"]
            source_name, target_name = source_node["name"], target_node["name"]
            if not directed and target_id < source_id:
                source_id, target_id = target_id, source_id
                source_name, target_name = target_name, source_name
            key = (source_id, target_id, relation["relation"])
            if key not in edges_by_key:
                edge_id = stable_id(
                    "e", source_id, relation["relation"], target_id, width=14
                )
                edges_by_key[key] = {
                    "edge_id": edge_id,
                    "source_id": source_id,
                    "target_id": target_id,
                    "source": source_name,
                    "target": target_name,
                    "relation": relation["relation"],
                    "directed": directed,
                    "description": relation["description"],
                    "descriptions": [],
                    "evidence_quote": relation["evidence_quote"],
                    "source_chunk_id": chunk_id,
                    "confidence": relation["confidence"],
                    "evidence": [],
                    "source_chunk_ids": [],
                }
            edge = edges_by_key[key]
            if relation["description"] not in edge["descriptions"]:
                edge["descriptions"].append(relation["description"])
            evidence = {
                "evidence_quote": relation["evidence_quote"],
                "source_chunk_id": chunk_id,
                "confidence": relation["confidence"],
            }
            evidence_key = (chunk_id, normalize_ws(relation["evidence_quote"]))
            existing_keys = {
                (item["source_chunk_id"], normalize_ws(item["evidence_quote"]))
                for item in edge["evidence"]
            }
            if evidence_key not in existing_keys:
                edge["evidence"].append(evidence)
            if chunk_id not in edge["source_chunk_ids"]:
                edge["source_chunk_ids"].append(chunk_id)
            edge["confidence"] = max(edge["confidence"], relation["confidence"])

    nodes = sorted(nodes_by_key.values(), key=lambda item: item["node_id"])
    for node in nodes:
        node["aliases"] = sorted(
            set(node["aliases"]), key=lambda value: (value.casefold(), value)
        )
        node["source_chunk_ids"] = sorted(
            node["source_chunk_ids"], key=lambda value: list(chunk_by_id).index(value)
        )
        node["description"] = " | ".join(node["descriptions"])
    edges = sorted(edges_by_key.values(), key=lambda item: item["edge_id"])
    for edge in edges:
        edge["descriptions"] = sorted(set(edge["descriptions"]))
        edge["description"] = " | ".join(edge["descriptions"])
        edge["evidence"] = sorted(
            edge["evidence"],
            key=lambda item: (
                -evidence_quality_score(edge["relation"], item["evidence_quote"]),
                -item["confidence"],
                len(item["evidence_quote"]),
                list(chunk_by_id).index(item["source_chunk_id"]),
                item["evidence_quote"],
            ),
        )
        edge["source_chunk_ids"] = sorted(
            edge["source_chunk_ids"], key=lambda value: list(chunk_by_id).index(value)
        )
        first = edge["evidence"][0]
        edge["evidence_quote"] = first["evidence_quote"]
        edge["source_chunk_id"] = first["source_chunk_id"]

    alias_map = {
        "version": alias_config.get("version"),
        "mappings": sorted(
            alias_records.values(),
            key=lambda item: (
                item["canonical"].casefold(),
                item["type"],
                item["alias"].casefold(),
            ),
        ),
    }
    unique_unresolved: dict[str, dict[str, Any]] = {}
    for item in unresolved:
        key = canonical_json(item)
        unique_unresolved[key] = item
    return {
        "nodes": nodes,
        "edges": edges,
        "alias_map": alias_map,
        "unresolved_aliases": sorted(
            unique_unresolved.values(), key=lambda item: item["alias"].casefold()
        ),
        "postprocess_warnings": postprocess_warnings,
    }


def build_networkx(
    nodes: Sequence[dict[str, Any]], edges: Sequence[dict[str, Any]]
) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(
        corpus_name=EXPECTED_CORPUS,
        title=EXPECTED_TITLE,
        schema_version="phase3-v1",
    )
    for node in nodes:
        graph.add_node(
            node["node_id"],
            name=node["name"],
            type=node["type"],
            description=node["description"],
            descriptions=canonical_json(node["descriptions"]),
            aliases=canonical_json(node["aliases"]),
            source_chunk_ids=canonical_json(node["source_chunk_ids"]),
        )
    for edge in edges:
        graph.add_edge(
            edge["source_id"],
            edge["target_id"],
            key=edge["edge_id"],
            edge_id=edge["edge_id"],
            source_name=edge["source"],
            target_name=edge["target"],
            relation=edge["relation"],
            directed=edge["directed"],
            description=edge["description"],
            descriptions=canonical_json(edge["descriptions"]),
            evidence_quote=edge["evidence_quote"],
            source_chunk_id=edge["source_chunk_id"],
            confidence=float(edge["confidence"]),
            evidence=canonical_json(edge["evidence"]),
            source_chunk_ids=canonical_json(edge["source_chunk_ids"]),
        )
    return graph


def graph_json_data(
    nodes: Sequence[dict[str, Any]], edges: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "schema_version": "phase3-v1",
        "corpus_name": EXPECTED_CORPUS,
        "title": EXPECTED_TITLE,
        "directed": True,
        "multigraph": True,
        "nodes": list(nodes),
        "edges": list(edges),
    }


def graph_statistics(graph: nx.MultiDiGraph) -> dict[str, Any]:
    undirected = graph.to_undirected()
    components = nx.number_connected_components(undirected) if graph.number_of_nodes() else 0
    isolates = sorted(nx.isolates(undirected))
    average_degree = (
        sum(dict(graph.degree()).values()) / graph.number_of_nodes()
        if graph.number_of_nodes()
        else 0.0
    )
    return {
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "connected_components": components,
        "isolated_node_count": len(isolates),
        "isolated_node_ids": isolates,
        "average_degree": round(average_degree, 6),
    }


def build_chunk_entity_map(
    chunks: Sequence[dict[str, Any]],
    outcomes: Sequence[dict[str, Any]],
    nodes: Sequence[dict[str, Any]],
    edges: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    outcome_by_id = {row["chunk_id"]: row for row in outcomes}
    rows = {}
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        node_refs = [
            {"node_id": node["node_id"], "name": node["name"], "type": node["type"]}
            for node in nodes
            if chunk_id in node["source_chunk_ids"]
        ]
        edge_ids = [edge["edge_id"] for edge in edges if chunk_id in edge["source_chunk_ids"]]
        outcome = outcome_by_id[chunk_id]
        rows[chunk_id] = {
            "section_id": chunk["section_id"],
            "outcome": outcome["outcome"],
            "no_fact_reason": (
                outcome.get("validated_output", {}).get("no_fact_reason")
                if outcome["outcome"] == "no_fact"
                else None
            ),
            "node_refs": node_refs,
            "edge_ids": edge_ids,
        }
    return {"corpus_name": EXPECTED_CORPUS, "chunks": rows}


def aggregate_usage(
    outcomes: Sequence[dict[str, Any]], run_summary: dict[str, Any]
) -> dict[str, Any]:
    totals: collections.Counter[str] = collections.Counter()
    total_calls = 0
    total_elapsed = 0.0
    per_chunk = []
    for outcome in outcomes:
        audit = outcome["request_audit"]
        total_calls += int(audit.get("attempt_count", 0))
        total_elapsed += float(audit.get("total_elapsed_seconds", 0))
        usage = audit.get("token_usage", {})
        totals.update({key: int(value) for key, value in usage.items()})
        per_chunk.append(
            {
                "chunk_id": outcome["chunk_id"],
                "outcome": outcome["outcome"],
                "attempt_count": audit.get("attempt_count", 0),
                "elapsed_seconds": audit.get("total_elapsed_seconds", 0),
                "token_usage": usage,
            }
        )
    return {
        "historical_api_attempts": total_calls,
        "historical_elapsed_seconds": round(total_elapsed, 6),
        "historical_token_usage": dict(sorted(totals.items())),
        **run_summary,
        "chunks": per_chunk,
    }


def classify_review_category(edge: dict[str, Any]) -> str:
    for category, relation_types in SYMMETRIC_REVIEW_CATEGORIES.items():
        if edge["relation"] in relation_types:
            return category
    combined = " ".join(
        [
            edge["source"],
            edge["target"],
            edge["description"],
            edge["evidence_quote"],
        ]
    ).casefold()
    if "dandy dick" in combined or re.search(r"\bdandy\b", combined):
        return "dandy_dick_ownership"
    if edge["relation"] == "ARRESTS" or any(
        term in combined for term in ("arrest", "police", "lock-up", "constable")
    ):
        return "arrest_police_station"
    if any(term in combined for term in ("ball", "race", "newmarket", "steeplechase")):
        return "ball_or_racing"
    return "other"


def review_sample(
    edges: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
    sample_size: int,
    seed: int,
    existing_path: Path,
) -> dict[str, Any]:
    old_by_id: dict[str, dict[str, Any]] = {}
    if existing_path.exists():
        old = load_json(existing_path)
        old_by_id = {row["edge_id"]: row for row in old.get("sample", [])}
    category_order = [
        "family",
        "emotion",
        "dandy_dick_ownership",
        "arrest_police_station",
        "ball_or_racing",
        "production",
        "other",
    ]
    buckets: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for edge in edges:
        buckets[classify_review_category(edge)].append(edge)
    rng = random.Random(seed)
    for category in buckets:
        rng.shuffle(buckets[category])
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    minimums = {
        "family": 5,
        "emotion": 4,
        "dandy_dick_ownership": 5,
        "arrest_police_station": 4,
        "ball_or_racing": 4,
        "production": 5,
    }
    for category in category_order:
        for edge in buckets[category][: minimums.get(category, 0)]:
            if edge["edge_id"] not in used:
                selected.append(edge)
                used.add(edge["edge_id"])
    remainder = list(edges)
    rng.shuffle(remainder)
    for edge in remainder:
        if len(selected) >= sample_size:
            break
        if edge["edge_id"] not in used:
            selected.append(edge)
            used.add(edge["edge_id"])
    chunk_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    sample = []
    for edge in selected[:sample_size]:
        old = old_by_id.get(edge["edge_id"], {})
        sample.append(
            {
                "edge_id": edge["edge_id"],
                "category": classify_review_category(edge),
                "source": edge["source"],
                "relation": edge["relation"],
                "target": edge["target"],
                "source_chunk_id": edge["source_chunk_id"],
                "evidence_quote": edge["evidence_quote"],
                "automated_quote_match": evidence_matches(
                    edge["evidence_quote"], chunk_by_id[edge["source_chunk_id"]]["text"]
                ),
                "manual_verdict": old.get("manual_verdict"),
                "manual_note": old.get("manual_note", ""),
            }
        )
    return {
        "instructions": "Judge only whether each relation is directly supported by its quoted phase2 chunk text; do not consult questions, answers, evidence labels, or external knowledge.",
        "sampling_method": "fixed-seed stratified random sample",
        "random_seed": seed,
        "sample_size": len(sample),
        "category_counts": dict(
            sorted(collections.Counter(row["category"] for row in sample).items())
        ),
        "sample": sample,
    }


def quality_report(
    chunks: Sequence[dict[str, Any]],
    outcomes: Sequence[dict[str, Any]],
    merged: dict[str, Any],
    graph: nx.MultiDiGraph,
    graph_json_path: Path,
    graphml_path: Path,
    config: dict[str, Any],
    review: dict[str, Any],
) -> dict[str, Any]:
    nodes = merged["nodes"]
    edges = merged["edges"]
    chunk_by_id = {row["chunk_id"]: row for row in chunks}
    outcome_ids = [row["chunk_id"] for row in outcomes]
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check("input_chunk_count", len(chunks) == config["expected_chunk_count"], len(chunks))
    check("node_ids_unique", len({n["node_id"] for n in nodes}) == len(nodes), len(nodes))
    check("edge_ids_unique", len({e["edge_id"] for e in edges}) == len(edges), len(edges))
    node_ids = {node["node_id"] for node in nodes}
    check(
        "edge_endpoints_exist",
        all(e["source_id"] in node_ids and e["target_id"] in node_ids for e in edges),
    )
    valid_chunks = set(chunk_by_id)
    check(
        "edge_chunk_sources_valid",
        all(set(edge["source_chunk_ids"]) <= valid_chunks for edge in edges),
    )
    bad_quotes = [
        {"edge_id": edge["edge_id"], "chunk_id": item["source_chunk_id"]}
        for edge in edges
        for item in edge["evidence"]
        if not evidence_matches(
            item["evidence_quote"], chunk_by_id[item["source_chunk_id"]]["text"]
        )
    ]
    check("all_evidence_quotes_match", not bad_quotes, bad_quotes)
    check(
        "no_all_node",
        all(normalized_name_key(node["name"]) != "all" for node in nodes),
    )
    check(
        "nonempty_nodes_and_edges",
        all(node["name"] and node["source_chunk_ids"] for node in nodes)
        and all(
            edge["source"]
            and edge["target"]
            and edge["relation"]
            and edge["source_chunk_ids"]
            for edge in edges
        ),
    )
    self_loops = [e["edge_id"] for e in edges if e["source_id"] == e["target_id"]]
    check("no_unsupported_self_loops", not self_loops, self_loops)
    name_types: dict[str, set[str]] = collections.defaultdict(set)
    for node in nodes:
        name_types[normalized_name_key(node["name"])].add(node["type"])
    mixed = {
        key: sorted(types)
        for key, types in name_types.items()
        if "CHARACTER" in types and "PERSON" in types
    }
    check("character_person_not_merged", not mixed, mixed)
    section_fact_counts: collections.Counter[str] = collections.Counter()
    for node in nodes:
        for chunk_id in node["source_chunk_ids"]:
            section_fact_counts[chunk_by_id[chunk_id]["section_id"]] += 1
    section_edge_counts: collections.Counter[str] = collections.Counter()
    for edge in edges:
        for section_id in {
            chunk_by_id[chunk_id]["section_id"]
            for chunk_id in edge["source_chunk_ids"]
        }:
            section_edge_counts[section_id] += 1
    coverage_groups = {
        "production_history": ["production_history"],
        "act_1": ["act_1"],
        "act_2": ["act_2"],
        "act_3": ["act_3_scene_1", "act_3_scene_2"],
    }
    missing_groups = [
        name
        for name, sections in coverage_groups.items()
        if not any(section_fact_counts[section] for section in sections)
        or not any(section_edge_counts[section] for section in sections)
    ]
    check("required_sections_have_nodes_and_edges", not missing_groups, missing_groups)
    check(
        "every_chunk_accounted",
        len(outcome_ids) == len(chunks)
        and len(set(outcome_ids)) == len(chunks)
        and set(outcome_ids) == valid_chunks,
    )
    check(
        "all_outcomes_explicit",
        all(row["outcome"] in {"success", "no_fact", "failure"} for row in outcomes),
    )
    expected_json = graph_json_data(nodes, edges)
    try:
        loaded_json = load_json(graph_json_path)
        json_ok = loaded_json == expected_json
        json_detail = None
    except Exception as exc:
        json_ok = False
        json_detail = f"{type(exc).__name__}: {exc}"
    check("json_graph_reloadable", json_ok, json_detail)
    try:
        loaded = nx.read_graphml(graphml_path)
        graphml_ok = (
            loaded.number_of_nodes() == graph.number_of_nodes()
            and loaded.number_of_edges() == graph.number_of_edges()
        )
        graphml_detail = None
    except Exception as exc:
        graphml_ok = False
        graphml_detail = f"{type(exc).__name__}: {exc}"
    check("graphml_reloadable", graphml_ok, graphml_detail)
    symmetric = set(config["symmetric_relations"])
    duplicate_symmetric = []
    seen_symmetric = set()
    for edge in edges:
        if edge["relation"] not in symmetric:
            continue
        key = (frozenset((edge["source_id"], edge["target_id"])), edge["relation"])
        if key in seen_symmetric:
            duplicate_symmetric.append(edge["edge_id"])
        seen_symmetric.add(key)
    check("symmetric_edges_single_logical_copy", not duplicate_symmetric, duplicate_symmetric)
    check(
        "allowed_entity_types",
        all(node["type"] in config["entity_types"] for node in nodes),
    )
    check(
        "allowed_relation_types",
        all(edge["relation"] in config["relation_types"] for edge in edges),
    )
    check(
        "weak_relations_excluded",
        all(edge["relation"] not in set(config.get("weak_relation_types", [])) for edge in edges),
    )
    dandy_nodes = [
        node for node in nodes if normalized_name_key(node["name"]) == "dandy dick"
    ]
    check(
        "dandy_dick_is_animal",
        len(dandy_nodes) == 1 and dandy_nodes[0]["type"] == "ANIMAL",
        [{"name": node["name"], "type": node["type"]} for node in dandy_nodes],
    )
    required_review_categories = {
        "family",
        "emotion",
        "dandy_dick_ownership",
        "arrest_police_station",
        "ball_or_racing",
        "production",
    }
    review_categories = {row["category"] for row in review["sample"]}
    check(
        "manual_review_sample_at_least_30",
        len(review["sample"]) >= 30,
        len(review["sample"]),
    )
    check(
        "manual_review_required_categories",
        required_review_categories <= review_categories,
        sorted(required_review_categories - review_categories),
    )
    check(
        "manual_review_complete_and_supported",
        all(row["manual_verdict"] is True for row in review["sample"]),
        {
            "reviewed": sum(row["manual_verdict"] is not None for row in review["sample"]),
            "passed": sum(row["manual_verdict"] is True for row in review["sample"]),
        },
    )
    return {
        "status": "pass" if all(item["passed"] for item in checks) else "fail",
        "checks": checks,
        "outcome_counts": dict(sorted(collections.Counter(o["outcome"] for o in outcomes).items())),
        "section_fact_counts": dict(sorted(section_fact_counts.items())),
        "section_edge_counts": dict(sorted(section_edge_counts.items())),
        "postprocess_warning_count": len(merged["postprocess_warnings"]),
        "postprocess_warning_counts": dict(
            sorted(
                collections.Counter(
                    item["kind"] for item in merged["postprocess_warnings"]
                ).items()
            )
        ),
        "postprocess_warnings": merged["postprocess_warnings"],
    }


def render_report(
    manifest: dict[str, Any],
    quality: dict[str, Any],
    alias_map: dict[str, Any],
    unresolved: Sequence[dict[str, Any]],
    review: dict[str, Any],
    usage: dict[str, Any],
    config: dict[str, Any],
) -> str:
    stats = manifest["graph_statistics"]
    verdicts = [row["manual_verdict"] for row in review["sample"]]
    reviewed = sum(value is not None for value in verdicts)
    passed = sum(value is True for value in verdicts)
    rate = passed / reviewed if reviewed else None
    review_lines = []
    for index, row in enumerate(review["sample"], 1):
        verdict = "通过" if row["manual_verdict"] is True else "不通过" if row["manual_verdict"] is False else "待审"
        quote = row["evidence_quote"].replace("|", "\\|")
        if len(quote) > 150:
            quote = quote[:147] + "…"
        review_lines.append(
            f"| {index} | {row['category']} | {row['source']} —{row['relation']}→ {row['target']} | `{row['source_chunk_id']}` | {quote} | {verdict} |"
        )
    alias_examples = []
    for canonical in ("Augustin Jedd", "Georgiana Tidman", "Sir Tristram Mardon"):
        aliases = [
            row["alias"]
            for row in alias_map["mappings"]
            if row["canonical"] == canonical and row["alias"] != canonical
        ]
        alias_examples.append(f"- `{canonical}` ← {', '.join(f'`{x}`' for x in aliases)}")
    token_usage = usage["historical_token_usage"]
    return f"""# 阶段 3：共享知识图谱构建

## 范围与配置

- 唯一正文输入：`data/processed/phase2/chunks.jsonl`（{manifest['input_chunk_count']} 个 chunk）
- 输入 SHA-256：`{manifest['input_sha256']}`
- 模型：`{manifest['model']}`
- OpenAI 兼容接口：`{manifest['base_url']}`
- temperature：`{config['temperature']}`；max_retries：`{config['max_retries']}`；thinking：`{config['thinking']}`
- 原始响应按 chunk 缓存；后处理可完全离线重复运行。

本阶段只读取 phase2 `chunks.jsonl`，未读取问题、答案、`evidence` 或
`evidence_triple`。未生成问题、未训练分类器、未生成 Embedding，也未实现
Vector、LightRAG 或 PathRAG 检索。

## 图谱统计

- 节点：{stats['node_count']}
- 逻辑边：{stats['edge_count']}
- 实体类型：`{manifest['entity_type_counts']}`
- 关系类型：`{manifest['relation_type_counts']}`
- 弱连通分量：{stats['connected_components']}
- 孤立节点：{stats['isolated_node_count']}
- 平均度：{stats['average_degree']}
- Chunk 结果：`{quality['outcome_counts']}`
- 节点来源覆盖：`{quality['section_fact_counts']}`
- 边来源覆盖：`{quality['section_edge_counts']}`

对称关系在正式图谱中仅存一条逻辑边并设置 `directed=false`。重复关系合并
后保留全部引文、chunk 来源和描述；GraphML 数组字段使用 JSON 字符串。

## 别名归一化

{chr(10).join(alias_examples)}

- 审计映射条目：{len(alias_map['mappings'])}
- 未解决别名：{len(unresolved)}
- `ALL` 仅作为集体台词标签，未成为实体；演员 PERSON 与角色 CHARACTER 未合并。
- `Dandy Dick` 固定为 ANIMAL；`Miss Jedd` 保持未解决，不猜测具体是哪位姐妹。

## LLM 使用

- 历史 API 尝试次数：{usage['historical_api_attempts']}
- 本次运行 API 调用：{usage['api_calls_this_run']}
- 本次缓存命中：{usage['cache_hits_this_run']}
- 历史 prompt tokens：{token_usage.get('prompt_tokens', 0)}
- 历史 completion tokens：{token_usage.get('completion_tokens', 0)}
- 历史 total tokens：{token_usage.get('total_tokens', 0)}
- 历史请求耗时合计：{usage['historical_elapsed_seconds']} 秒

密钥只从环境变量读取，从未写入代码、缓存、日志或本报告。

## 建图前后样例

建图前，各 chunk 独立包含模型输出的实体、关系及逐条原文引文。建图后，
别名被映射到稳定节点，重复逻辑边合并。例如 `THE DEAN`、`Augustin`、`Gus`
和 `Dr. Jedd` 归一为 `Augustin Jedd`，但每条边仍保留原始 chunk 与引文。

## 30 条关系人工抽查

- 抽样：{review['sampling_method']}（seed={review['random_seed']}）
- 样本：{len(review['sample'])}；已审：{reviewed}；通过：{passed}；通过率：{f'{rate:.2%}' if rate is not None else '待审'}
- 类别分布：`{review['category_counts']}`
- 只依据 phase2 原文支持性判断，不参考任何问题、答案或监督证据。

| # | 类别 | 关系 | Chunk | 原文引文 | 结论 |
|---:|---|---|---|---|---|
{chr(10).join(review_lines)}

## 失败、异常与质量结论

- 抽取失败：{quality['outcome_counts'].get('failure', 0)} 个 chunk。
- 无事实：{quality['outcome_counts'].get('no_fact', 0)} 个 chunk。
- 未解决别名：{len(unresolved)} 条。
- 后处理警告：{quality['postprocess_warning_count']} 条，详见 `quality_report.json`。
- 后处理警告分布：`{quality['postprocess_warning_counts']}`。其中大部分是保守规则主动拒绝弱关系或语义不足关系的审计记录，并非运行失败。
- 自动质量检查：`{quality['status']}`。
- Graph JSON 与 GraphML 均已重新加载验证。

下一步是“实现三个共享数据基础上的检索器”；本阶段未自行开始。
"""


def build_and_write(
    chunks: Sequence[dict[str, Any]],
    outcomes: Sequence[dict[str, Any]],
    config: dict[str, Any],
    alias_config: dict[str, Any],
    output_dir: Path,
    interim_dir: Path,
    report_path: Path,
    input_path: Path,
    prompt_path: Path,
    config_path: Path,
    alias_path: Path,
    runtime: dict[str, str],
    run_summary: dict[str, Any],
) -> dict[str, Any]:
    first = merge_graph_data(chunks, outcomes, config, alias_config)
    second = merge_graph_data(chunks, outcomes, config, alias_config)
    first_digest = sha256_text(
        canonical_json({"nodes": first["nodes"], "edges": first["edges"]})
    )
    second_digest = sha256_text(
        canonical_json({"nodes": second["nodes"], "edges": second["edges"]})
    )
    if first_digest != second_digest:
        raise Phase3Error("相同缓存的后处理结果不确定")

    nodes, edges = first["nodes"], first["edges"]
    graph = build_networkx(nodes, edges)
    graph_data = graph_json_data(nodes, edges)
    output_dir.mkdir(parents=True, exist_ok=True)
    graphml_path = output_dir / "graph.graphml"
    with tempfile.NamedTemporaryFile(
        suffix=".graphml", dir=output_dir, delete=False
    ) as handle:
        graphml_temp = Path(handle.name)
    nx.write_graphml(graph, graphml_temp, encoding="utf-8")
    graphml_temp.replace(graphml_path)

    chunk_map = build_chunk_entity_map(chunks, outcomes, nodes, edges)
    usage = aggregate_usage(outcomes, run_summary)
    failures = [
        {
            "chunk_id": row["chunk_id"],
            "failure_reason": row.get("failure_reason", "unknown failure"),
            "request_audit": row["request_audit"],
        }
        for row in outcomes
        if row["outcome"] == "failure"
    ]
    review_path = interim_dir / "manual_review.json"
    review = review_sample(
        edges,
        chunks,
        config["manual_review_sample_size"],
        config["manual_review_seed"],
        review_path,
    )
    write_text_atomic(review_path, canonical_json(review, pretty=True))

    stats = graph_statistics(graph)
    entity_counts = dict(sorted(collections.Counter(n["type"] for n in nodes).items()))
    relation_counts = dict(
        sorted(collections.Counter(e["relation"] for e in edges).items())
    )
    manifest = {
        "schema_version": config["schema_version"],
        "corpus_name": EXPECTED_CORPUS,
        "title": EXPECTED_TITLE,
        "input_file": input_path.as_posix(),
        "input_sha256": sha256_file(input_path),
        "input_chunk_count": len(chunks),
        "prompt_file": prompt_path.as_posix(),
        "prompt_sha256": sha256_file(prompt_path),
        "config_file": config_path.as_posix(),
        "config_sha256": sha256_file(config_path),
        "alias_overrides_file": alias_path.as_posix(),
        "alias_overrides_sha256": sha256_file(alias_path),
        "model": runtime["LLM_MODEL"],
        "base_url": runtime["LLM_BASE_URL"].rstrip("/"),
        "temperature": config["temperature"],
        "max_retries": config["max_retries"],
        "node_count": len(nodes),
        "edge_count": len(edges),
        "entity_type_counts": entity_counts,
        "relation_type_counts": relation_counts,
        "graph_statistics": stats,
        "outcome_counts": dict(
            sorted(collections.Counter(row["outcome"] for row in outcomes).items())
        ),
        "postprocess_deterministic": True,
        "postprocess_digest": first_digest,
    }

    write_text_atomic(output_dir / "nodes.jsonl", jsonl_dump(nodes))
    write_text_atomic(output_dir / "edges.jsonl", jsonl_dump(edges))
    write_text_atomic(output_dir / "graph.json", canonical_json(graph_data, pretty=True))
    write_text_atomic(
        output_dir / "chunk_entity_map.json", canonical_json(chunk_map, pretty=True)
    )
    write_text_atomic(
        output_dir / "alias_map.json", canonical_json(first["alias_map"], pretty=True)
    )
    write_text_atomic(
        output_dir / "unresolved_aliases.json",
        canonical_json(first["unresolved_aliases"], pretty=True),
    )
    write_text_atomic(output_dir / "extraction_failures.jsonl", jsonl_dump(failures))
    write_text_atomic(output_dir / "llm_usage.json", canonical_json(usage, pretty=True))
    write_text_atomic(output_dir / "kg_manifest.json", canonical_json(manifest, pretty=True))

    quality = quality_report(
        chunks,
        outcomes,
        first,
        graph,
        output_dir / "graph.json",
        graphml_path,
        config,
        review,
    )
    quality["postprocess_digest_first"] = first_digest
    quality["postprocess_digest_second"] = second_digest
    quality["postprocess_hashes_identical"] = first_digest == second_digest
    quality["manual_review"] = {
        "sample_size": review["sample_size"],
        "reviewed": sum(row["manual_verdict"] is not None for row in review["sample"]),
        "passed": sum(row["manual_verdict"] is True for row in review["sample"]),
    }
    write_text_atomic(
        output_dir / "quality_report.json", canonical_json(quality, pretty=True)
    )
    report = render_report(
        manifest,
        quality,
        first["alias_map"],
        first["unresolved_aliases"],
        review,
        usage,
        config,
    )
    write_text_atomic(report_path, report)
    if quality["status"] != "pass":
        failures_detail = [
            item for item in quality["checks"] if not item["passed"]
        ]
        raise Phase3Error(f"阶段3质量检查失败：{failures_detail}")
    return {
        "manifest": manifest,
        "quality_status": quality["status"],
        "manual_review_path": str(review_path),
        "report_path": str(report_path),
    }


def prepare_phase3(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input.resolve()
    config_path = args.config.resolve()
    prompt_path = args.prompt.resolve()
    alias_path = args.aliases.resolve()
    output_dir = args.output_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    interim_dir = cache_dir.parent
    report_path = args.report.resolve()
    config = load_config(config_path)
    prompt = prompt_path.read_text(encoding="utf-8")
    if "JSON" not in prompt and "json" not in prompt:
        raise Phase3Error("抽取提示词必须明确要求 JSON")
    alias_config = load_json(alias_path)
    chunks = load_chunks(input_path, config["expected_chunk_count"])
    input_hash_before = sha256_file(input_path)

    if args.offline:
        runtime = runtime_from_env(required=False)
        if not runtime["LLM_MODEL"]:
            cache_files = sorted(cache_dir.glob("*.json"))
            if not cache_files:
                raise Phase3Error("离线重建没有缓存，且 LLM_MODEL 未设置")
            first_cache = load_json(cache_files[0])
            runtime["LLM_MODEL"] = first_cache.get("model", "")
        if not runtime["LLM_BASE_URL"]:
            old_manifest = output_dir / "kg_manifest.json"
            runtime["LLM_BASE_URL"] = (
                load_json(old_manifest).get("base_url", "offline-cache")
                if old_manifest.exists()
                else "offline-cache"
            )
    else:
        runtime = runtime_from_env(required=True)

    outcomes, run_summary = extract_chunks(
        chunks,
        config,
        prompt,
        runtime,
        cache_dir,
        offline=args.offline,
        retry_failures=args.retry_failures,
    )
    result = build_and_write(
        chunks,
        outcomes,
        config,
        alias_config,
        output_dir,
        interim_dir,
        report_path,
        args.input,
        args.prompt,
        args.config,
        args.aliases,
        runtime,
        run_summary,
    )
    if sha256_file(input_path) != input_hash_before:
        raise Phase3Error("阶段2 chunks.jsonl 在阶段3执行期间被修改")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/processed/phase2/chunks.jsonl")
    )
    parser.add_argument("--config", type=Path, default=Path("configs/kg_config.json"))
    parser.add_argument(
        "--aliases", type=Path, default=Path("configs/alias_overrides.json")
    )
    parser.add_argument(
        "--prompt", type=Path, default=Path("prompts/kg_extraction.txt")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/phase3")
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/interim/phase3/llm_responses"),
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/phase3_knowledge_graph.md")
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = prepare_phase3(parse_args(argv))
    except Phase3Error as exc:
        print(f"阶段3失败：{exc}", file=sys.stderr)
        return 1
    print(canonical_json(result, pretty=True), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
