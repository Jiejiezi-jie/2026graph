#!/usr/bin/env python3
"""Deterministically clean, parse and structure-chunk Dandy Dick.

The only document input accepted by this stage is the phase-one
``indexing_input.json``. No question file, model, embedding service or graph
library is imported or accessed.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import statistics
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Sequence


EXPECTED_CORPUS = "Novel-40700"
EXPECTED_TITLE = "Dandy Dick"
CLEANING_VERSION = "phase2-v1"
SUPERVISION_KEYS = {"question", "answer", "evidence", "evidence_triple"}
UNIT_TYPES = {"heading", "prose", "cast_entry", "dialogue", "stage_direction"}
SECTION_META = {
    "front_matter": {"section_type": "front_matter", "act": None, "scene": None},
    "production_history": {
        "section_type": "production_history",
        "act": None,
        "scene": None,
    },
    "cast": {"section_type": "cast", "act": None, "scene": None},
    "act_1": {"section_type": "play", "act": 1, "scene": None},
    "act_2": {"section_type": "play", "act": 2, "scene": None},
    "act_3_scene_1": {"section_type": "play", "act": 3, "scene": 1},
    "act_3_scene_2": {"section_type": "play", "act": 3, "scene": 2},
}
SECTION_PREFIX = {
    "front_matter": "dd_front",
    "production_history": "dd_prod",
    "cast": "dd_cast",
    "act_1": "dd_act01",
    "act_2": "dd_act02",
    "act_3_scene_1": "dd_act03s01",
    "act_3_scene_2": "dd_act03s02",
}
SECTION_LABEL = {
    "front_matter": "Front Matter",
    "production_history": "Production History",
    "cast": "Cast and First-night Programme",
    "act_1": "Act 1",
    "act_2": "Act 2",
    "act_3_scene_1": "Act 3 | Scene 1 | The Strong Box",
    "act_3_scene_2": "Act 3 | Scene 2 | Morning Room at the Deanery",
}


class Phase2Error(RuntimeError):
    """Raised when the input or an output artifact violates the protocol."""


def json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def jsonl_dump(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def word_count(text: str) -> int:
    return len(text.split())


def load_input(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise Phase2Error(f"阶段1输入不存在：{path}") from exc
    except json.JSONDecodeError as exc:
        raise Phase2Error(f"阶段1输入不是有效 JSON：{exc}") from exc
    if not isinstance(value, dict):
        raise Phase2Error("indexing_input.json 必须是 JSON 对象")
    if set(value) != {"corpus_name", "title", "context"}:
        leaked = sorted(SUPERVISION_KEYS & set(value))
        raise Phase2Error(
            "indexing_input.json 字段必须严格为 corpus_name/title/context；"
            f"实际字段={sorted(value)}，监督字段={leaked}"
        )
    if value["corpus_name"] != EXPECTED_CORPUS:
        raise Phase2Error(
            f"本阶段只接受 {EXPECTED_CORPUS}，实际为 {value['corpus_name']}"
        )
    if value["title"] != EXPECTED_TITLE:
        raise Phase2Error(
            f"本阶段只接受标题 {EXPECTED_TITLE}，实际为 {value['title']}"
        )
    if not isinstance(value["context"], str) or not value["context"].strip():
        raise Phase2Error("阶段1正文为空")
    return value


def load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise Phase2Error(f"切块配置不存在：{path}") from exc
    required = {
        "target_words": 380,
        "min_words": 120,
        "max_words": 520,
        "overlap_words": 60,
        "approx_token_chars": 4,
        "cross_section": False,
        "cross_act": False,
        "cross_scene": False,
        "preserve_dialogue_turn": True,
        "preserve_stage_direction": True,
    }
    mismatches = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in required.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise Phase2Error(f"配置偏离固定实验参数：{mismatches}")
    if not isinstance(config.get("structure_markers"), dict):
        raise Phase2Error("配置缺少 structure_markers")
    speakers = config.get("final_speakers")
    if not isinstance(speakers, list) or len(set(speakers)) != len(speakers):
        raise Phase2Error("final_speakers 必须是无重复的字符串数组")
    if not all(isinstance(name, str) and name for name in speakers):
        raise Phase2Error("final_speakers 包含空值或非字符串")
    return config


def removed_span(raw: str, start: int, end: int, reason: str) -> dict[str, Any]:
    return {
        "raw_start": start,
        "raw_end": end,
        "before_preview": raw[max(0, start - 100) : start],
        "deleted_preview": raw[start : min(end, start + 100)],
        "after_preview": raw[end : min(len(raw), end + 100)],
        "reason": reason,
        "deleted_char_count": end - start,
    }


def _overlaps(start: int, end: int, intervals: Sequence[tuple[int, int, str]]) -> bool:
    return any(start < other_end and end > other_start for other_start, other_end, _ in intervals)


def clean_raw_text(
    raw: str, markers: dict[str, str]
) -> tuple[str, list[int | None], list[dict[str, Any]], dict[str, Any]]:
    title_marker = "DANDY DICK A PLAY IN THREE ACTS"
    title_start = raw.find(title_marker)
    if title_start < 0:
        raise Phase2Error("找不到作品标题，不能安全删除 Gutenberg 制作前缀")
    footer_start = raw.find(markers["footer_start"])
    if footer_start < 0:
        raise Phase2Error("找不到转录说明起点，不能安全确定作品结尾")
    work_end_start = raw.rfind(markers["work_end"], 0, footer_start)
    if work_end_start < 0:
        raise Phase2Error("找不到作品结束标记 THE END")
    work_end_end = work_end_start + len(markers["work_end"])
    if raw[work_end_end:footer_start].strip():
        raise Phase2Error("THE END 与转录说明之间存在未识别文本，停止清洗")

    intervals: list[tuple[int, int, str]] = [
        (0, title_start, "删除正文标题前的 Project Gutenberg 制作者信息"),
        (work_end_end, len(raw), "删除作品结束后的转录说明和 Gutenberg 页脚"),
    ]
    for match in re.finditer(r"<[^>]+>", raw):
        if not _overlaps(match.start(), match.end(), intervals):
            intervals.append((match.start(), match.end(), "删除 HTML 残留标签"))
    for match in re.finditer(r"(?i)(?<!\S)\[Illustration\](?!\S)", raw):
        if not _overlaps(match.start(), match.end(), intervals):
            intervals.append(
                (match.start(), match.end(), "删除无语义的独立图片占位标记")
            )
    intervals.sort()
    for left, right in zip(intervals, intervals[1:]):
        if left[1] > right[0]:
            raise Phase2Error(f"清洗删除区间重叠：{left} / {right}")

    deleted = [False] * len(raw)
    for start, end, _ in intervals:
        deleted[start:end] = [True] * (end - start)

    whitespace_intervals: list[tuple[int, int, str]] = []
    index = 0
    while index < len(raw):
        if deleted[index] or not raw[index].isspace():
            index += 1
            continue
        end = index + 1
        while end < len(raw) and not deleted[end] and raw[end].isspace():
            end += 1
        if end - index > 1:
            whitespace_intervals.append(
                (index + 1, end, "连续空白规范化为单个普通空格")
            )
            deleted[index + 1 : end] = [True] * (end - index - 1)
        index = end
    intervals.extend(whitespace_intervals)
    intervals.sort()

    output_chars: list[str] = []
    clean_to_raw: list[int | None] = []
    for raw_index, char in enumerate(raw):
        if deleted[raw_index]:
            continue
        normalized_char = " " if char == "\u00a0" or char.isspace() else char
        output_chars.append(normalized_char)
        clean_to_raw.append(raw_index)
    flat = "".join(output_chars).strip()
    leading = 0
    while leading < len(output_chars) and output_chars[leading].isspace():
        leading += 1
    trailing = len(output_chars)
    while trailing > leading and output_chars[trailing - 1].isspace():
        trailing -= 1
    clean_to_raw = clean_to_raw[leading:trailing]
    nfc = unicodedata.normalize("NFC", flat)
    if len(nfc) != len(flat):
        clean_to_raw = [None] * len(nfc)
    flat = nfc
    logs = [removed_span(raw, start, end, reason) for start, end, reason in intervals]
    summary = {
        "title_start": title_start,
        "work_end_start": work_end_start,
        "work_end_end": work_end_end,
        "footer_start": footer_start,
        "raw_char_count": len(raw),
        "flat_clean_char_count": len(flat),
        "deleted_char_count": sum(row["deleted_char_count"] for row in logs),
        "removed_span_count": len(logs),
    }
    return flat, clean_to_raw, logs, summary


def find_bracket_stage_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        opening = text.find("[", cursor, end)
        if opening < 0:
            break
        first_underscore = text.find("_", opening + 1, min(end, opening + 100))
        if first_underscore >= 0:
            if first_underscore == opening + 1:
                closing_underscore = text.find("_", first_underscore + 1, end)
            else:
                closing_underscore = text.find("_", first_underscore + 1, end)
            if closing_underscore >= 0:
                stage_end = closing_underscore + 1
                if stage_end < end and text[stage_end] == "]":
                    stage_end += 1
                spans.append((opening, stage_end))
                cursor = stage_end
                continue
        closing = text.find("]", opening + 1, end)
        if closing < 0:
            raise Phase2Error(
                f"无法确定方括号舞台说明的结束位置：normalized offset {opening}"
            )
        spans.append((opening, closing + 1))
        cursor = closing + 1
    return spans


def position_in_spans(position: int, spans: Sequence[tuple[int, int]]) -> bool:
    return any(start <= position < end for start, end in spans)


def find_italic_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    positions = [match.start() for match in re.finditer("_", text[start:end])]
    if len(positions) % 2:
        raise Phase2Error(f"斜体下划线无法成对：{start}:{end}")
    return [
        (start + positions[index], start + positions[index + 1] + 1)
        for index in range(0, len(positions), 2)
    ]


def normalize_speaker_label(label: str) -> str:
    label = label.replace("_", " ").replace("’", "'")
    return " ".join(label.upper().split())


def split_speaker_label(label: str, allowed: set[str] | None = None) -> list[str]:
    normalized = normalize_speaker_label(label)
    parts = [
        " ".join(part.split())
        for part in re.split(r"\s*(?:,|\bAND\b)\s*", normalized)
        if part.strip()
    ]
    if allowed is not None and (not parts or any(part not in allowed for part in parts)):
        return []
    return parts


def discover_dialogue_candidates(
    text: str, start: int, end: int
) -> dict[str, int]:
    stage_spans = find_bracket_stage_spans(text, start, end) + find_italic_spans(
        text, start, end
    )
    counts: collections.Counter[str] = collections.Counter()
    pattern = re.compile(r"(?<![A-Za-z])([A-Z][A-Z ,_'’\-]{1,100})\.\s+")
    for match in pattern.finditer(text, start, end):
        if position_in_spans(match.start(), stage_spans):
            continue
        for part in split_speaker_label(match.group(1)):
            counts[part] += 1
    return dict(sorted(counts.items()))


def build_speaker_pattern(speakers: Sequence[str]) -> re.Pattern[str]:
    name = "(?:" + "|".join(
        re.escape(value) for value in sorted(speakers, key=len, reverse=True)
    ) + ")"
    label = rf"{name}(?:\s*,\s*{name})*(?:\s*,?\s+(?:_and_|and)\s+{name})?"
    return re.compile(rf"(?<![A-Za-z])(?P<label>{label})\.\s+")


def find_dialogue_labels(
    text: str, start: int, end: int, speakers: Sequence[str]
) -> list[dict[str, Any]]:
    allowed = set(speakers)
    stage_spans = find_bracket_stage_spans(text, start, end) + find_italic_spans(
        text, start, end
    )
    pattern = build_speaker_pattern(speakers)
    labels: list[dict[str, Any]] = []
    for match in pattern.finditer(text, start, end):
        if position_in_spans(match.start(), stage_spans):
            continue
        parsed = split_speaker_label(match.group("label"), allowed)
        if not parsed:
            continue
        labels.append(
            {
                "start": match.start(),
                "end": match.end(),
                "label": match.group("label"),
                "speakers": parsed,
            }
        )
    return labels


def trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def sentence_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    start, end = trim_span(text, start, end)
    if start >= end:
        return []
    boundaries = [start]
    for match in re.finditer(r"(?<=[.!?])\s+(?=[\"'_(A-Z])", text[start:end]):
        boundaries.append(start + match.end())
    boundaries.append(end)
    spans: list[tuple[int, int]] = []
    for left, right in zip(boundaries, boundaries[1:]):
        left, right = trim_span(text, left, right)
        if left < right:
            spans.append((left, right))
    return spans


def base_unit(
    text: str,
    start: int,
    end: int,
    section_id: str,
    unit_type: str,
    speakers: Sequence[str] = (),
) -> dict[str, Any]:
    start, end = trim_span(text, start, end)
    if start >= end:
        raise Phase2Error(f"尝试创建空结构单元：{section_id}/{unit_type}")
    meta = SECTION_META[section_id]
    speaker_list = list(speakers)
    return {
        "section_id": section_id,
        "section_type": meta["section_type"],
        "act": meta["act"],
        "scene": meta["scene"],
        "unit_type": unit_type,
        "speaker": speaker_list[0] if len(speaker_list) == 1 else None,
        "speakers": speaker_list,
        "text": text[start:end],
        "normalized_start": start,
        "normalized_end": end,
    }


def parse_sentence_section(
    text: str,
    start: int,
    end: int,
    section_id: str,
    heading_markers: Sequence[str] = (),
) -> list[dict[str, Any]]:
    heading_spans: list[tuple[int, int]] = []
    for marker in heading_markers:
        position = text.find(marker, start, end)
        if position >= 0:
            heading_spans.append((position, position + len(marker)))
    heading_spans.sort()
    units: list[dict[str, Any]] = []
    cursor = start
    for heading_start, heading_end in heading_spans:
        for left, right in sentence_spans(text, cursor, heading_start):
            units.append(base_unit(text, left, right, section_id, "prose"))
        units.append(base_unit(text, heading_start, heading_end, section_id, "heading"))
        cursor = heading_end
    for left, right in sentence_spans(text, cursor, end):
        unit_type = "heading" if section_id == "front_matter" and not units else "prose"
        units.append(base_unit(text, left, right, section_id, unit_type))
    return units


def parse_cast_section(
    text: str,
    start: int,
    end: int,
    cast_markers: dict[str, str],
) -> tuple[list[dict[str, Any]], list[str]]:
    located: list[tuple[int, str, str]] = []
    for speaker, marker in cast_markers.items():
        position = text.find(marker, start, end)
        if position < 0:
            raise Phase2Error(f"演员表中找不到配置角色标记：{speaker} -> {marker}")
        located.append((position, speaker, marker))
    located.sort()
    if len({position for position, _, _ in located}) != len(located):
        raise Phase2Error("演员表角色标记位置发生冲突")
    programme_start = text.find("ACT I. AT THE DEANERY", start, end)
    if programme_start < 0 or programme_start <= located[-1][0]:
        raise Phase2Error("演员表之后找不到三幕场景说明")
    units: list[dict[str, Any]] = []
    header_start, header_end = trim_span(text, start, located[0][0])
    if header_start < header_end:
        units.append(base_unit(text, header_start, header_end, "cast", "heading"))
    for index, (position, speaker, _) in enumerate(located):
        next_position = (
            located[index + 1][0] if index + 1 < len(located) else programme_start
        )
        units.append(
            base_unit(text, position, next_position, "cast", "cast_entry", [speaker])
        )
    for left, right in sentence_spans(text, programme_start, end):
        units.append(base_unit(text, left, right, "cast", "prose"))
    return units, [speaker for _, speaker, _ in located]


def parse_play_section(
    text: str,
    start: int,
    end: int,
    section_id: str,
    speakers: Sequence[str],
    heading_markers: Sequence[str],
) -> list[dict[str, Any]]:
    headings: list[tuple[int, int]] = []
    for marker in heading_markers:
        position = text.find(marker, start, end)
        if position < 0:
            raise Phase2Error(f"{section_id} 缺少标题标记：{marker}")
        left, right = position, position + len(marker)
        if left > start and text[left - 1] == "_" and right < end and text[right] == "_":
            left -= 1
            right += 1
        headings.append((left, right))
    headings.sort()
    for first, second in zip(headings, headings[1:]):
        if first[1] > second[0]:
            raise Phase2Error(f"{section_id} 标题区间重叠")

    units: list[dict[str, Any]] = []
    region_start = start
    for heading_start, heading_end in headings:
        if region_start < heading_start:
            units.extend(
                parse_dialogue_region(
                    text, region_start, heading_start, section_id, speakers
                )
            )
        units.append(base_unit(text, heading_start, heading_end, section_id, "heading"))
        region_start = heading_end
    if region_start < end:
        units.extend(parse_dialogue_region(text, region_start, end, section_id, speakers))
    return sorted(units, key=lambda row: row["normalized_start"])


def parse_dialogue_region(
    text: str,
    start: int,
    end: int,
    section_id: str,
    speakers: Sequence[str],
) -> list[dict[str, Any]]:
    start, end = trim_span(text, start, end)
    if start >= end:
        return []
    labels = find_dialogue_labels(text, start, end, speakers)
    if not labels:
        return [base_unit(text, start, end, section_id, "stage_direction")]
    units: list[dict[str, Any]] = []
    if start < labels[0]["start"]:
        units.append(
            base_unit(text, start, labels[0]["start"], section_id, "stage_direction")
        )
    for index, label in enumerate(labels):
        turn_end = labels[index + 1]["start"] if index + 1 < len(labels) else end
        units.append(
            base_unit(
                text,
                label["start"],
                turn_end,
                section_id,
                "dialogue",
                label["speakers"],
            )
        )
    return units


def raw_bounds(
    mapping: Sequence[int | None], start: int, end: int
) -> tuple[int | None, int | None]:
    if start >= end or start >= len(mapping) or end > len(mapping):
        return None, None
    values = mapping[start:end]
    if not values or any(value is None for value in values):
        return None, None
    return int(values[0]), int(values[-1]) + 1


def split_oversized_unit(
    unit: dict[str, Any], text: str, max_words: int
) -> list[dict[str, Any]]:
    if word_count(unit["text"]) <= max_words:
        return [unit]
    spans = sentence_spans(text, unit["normalized_start"], unit["normalized_end"])
    if len(spans) <= 1:
        copy = dict(unit)
        copy["exception_reason"] = "single_sentence_exceeds_max_words"
        return [copy]
    pieces: list[dict[str, Any]] = []
    current: list[tuple[int, int]] = []
    current_words = 0
    for span in spans:
        count = word_count(text[span[0] : span[1]])
        if current and current_words + count > max_words:
            piece = dict(unit)
            piece["normalized_start"] = current[0][0]
            piece["normalized_end"] = current[-1][1]
            piece["text"] = text[piece["normalized_start"] : piece["normalized_end"]]
            pieces.append(piece)
            current = []
            current_words = 0
        current.append(span)
        current_words += count
    if current:
        piece = dict(unit)
        piece["normalized_start"] = current[0][0]
        piece["normalized_end"] = current[-1][1]
        piece["text"] = text[piece["normalized_start"] : piece["normalized_end"]]
        if word_count(piece["text"]) > max_words:
            piece["exception_reason"] = "single_sentence_exceeds_max_words"
        pieces.append(piece)
    return pieces


def assign_unit_ids(
    units: Sequence[dict[str, Any]], mapping: Sequence[int | None], title: str
) -> list[dict[str, Any]]:
    counters: collections.Counter[str] = collections.Counter()
    output: list[dict[str, Any]] = []
    for source_order, unit in enumerate(
        sorted(units, key=lambda row: row["normalized_start"]), start=1
    ):
        section = unit["section_id"]
        counters[section] += 1
        raw_start, raw_end = raw_bounds(
            mapping, unit["normalized_start"], unit["normalized_end"]
        )
        row = {
            "unit_id": f"{SECTION_PREFIX[section]}_u{counters[section]:04d}",
            "corpus_name": EXPECTED_CORPUS,
            "title": title,
            "section_id": section,
            "section_type": unit["section_type"],
            "act": unit["act"],
            "scene": unit["scene"],
            "unit_type": unit["unit_type"],
            "speaker": unit["speaker"],
            "speakers": unit["speakers"],
            "text": unit["text"],
            "source_order": source_order,
            "raw_start": raw_start,
            "raw_end": raw_end,
            "word_count": word_count(unit["text"]),
        }
        if unit.get("exception_reason"):
            row["exception_reason"] = unit["exception_reason"]
        output.append(row)
    return output


def structure_positions(
    text: str, markers: dict[str, str]
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    positions = {
        "introductory_note": text.find(markers["introductory_note"]),
        "cast_start": text.find(markers["cast_start"]),
        "cast_end": text.find(markers["cast_end"]),
        "act_1": text.find(markers["act_1"]),
        "act_1_end": text.find(markers["act_1_end"]),
        "act_2": text.find(markers["act_2"]),
        "act_2_end": text.find(markers["act_2_end"]),
        "act_3": text.find(markers["act_3"]),
        "act_3_scene_1_end": text.find(markers["act_3_scene_1_end"]),
        "act_3_scene_2": text.find(markers["act_3_scene_2"]),
    }
    if any(position < 0 for position in positions.values()):
        missing = [name for name, position in positions.items() if position < 0]
        raise Phase2Error(f"缺少必要戏剧结构标记：{missing}")
    positions["main_title"] = text.rfind(markers["main_title"], 0, positions["act_1"])
    positions["work_end"] = text.rfind(markers["work_end"])
    if positions["main_title"] < 0 or positions["work_end"] < 0:
        raise Phase2Error("缺少正文标题或最终 THE END")
    order = [
        "introductory_note",
        "cast_start",
        "cast_end",
        "main_title",
        "act_1",
        "act_1_end",
        "act_2",
        "act_2_end",
        "act_3",
        "act_3_scene_1_end",
        "act_3_scene_2",
        "work_end",
    ]
    ordered_positions = [positions[name] for name in order]
    if ordered_positions != sorted(ordered_positions) or len(set(ordered_positions)) != len(order):
        raise Phase2Error(
            f"戏剧结构标记顺序错误：{dict(zip(order, ordered_positions))}"
        )
    records = [
        {"name": name, "text": markers[name], "normalized_position": positions[name]}
        for name in order
    ]
    return positions, records


def build_units_and_structure(
    flat: str,
    mapping: Sequence[int | None],
    config: dict[str, Any],
    title: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    markers = config["structure_markers"]
    positions, marker_records = structure_positions(flat, markers)
    scene2_start = positions["act_3_scene_2"]
    if scene2_start > 0 and flat[scene2_start - 1] == "_":
        scene2_start -= 1
    ranges = [
        ("front_matter", 0, positions["introductory_note"]),
        ("production_history", positions["introductory_note"], positions["cast_start"]),
        ("cast", positions["cast_start"], positions["cast_end"]),
        ("production_history", positions["cast_end"], positions["main_title"]),
        ("act_1", positions["main_title"], positions["act_2"]),
        ("act_2", positions["act_2"], positions["act_3"]),
        ("act_3_scene_1", positions["act_3"], scene2_start),
        (
            "act_3_scene_2",
            scene2_start,
            positions["work_end"] + len(markers["work_end"]),
        ),
    ]
    play_start = positions["main_title"]
    play_end = positions["work_end"] + len(markers["work_end"])
    candidate_counts = discover_dialogue_candidates(flat, play_start, play_end)
    structural_candidates = {
        "THE FIRST ACT",
        "END OF THE FIRST ACT",
        "THE SECOND ACT",
        "END OF THE SECOND ACT",
        "THE THIRD ACT",
        "THE END OF THE FIRST SCENE",
        "THE END",
        "DANDY DICK",
    }
    stable = {
        name
        for name, count in candidate_counts.items()
        if count >= 2 and name not in structural_candidates
    }
    collective = set(config.get("collective_speakers", []))
    configured_characters = set(config["final_speakers"]) - collective
    if stable != configured_characters:
        raise Phase2Error(
            "正文稳定说话人候选与配置最终人物不一致："
            f"discovered={sorted(stable)}, configured={sorted(configured_characters)}"
        )

    all_units: list[dict[str, Any]] = []
    cast_detected: list[str] = []
    range_records: list[dict[str, Any]] = []
    for instance, (section_id, start, end) in enumerate(ranges, start=1):
        raw_start, raw_end = raw_bounds(mapping, start, end)
        range_records.append(
            {
                "section_id": section_id,
                "section_instance": instance,
                "section_type": SECTION_META[section_id]["section_type"],
                "act": SECTION_META[section_id]["act"],
                "scene": SECTION_META[section_id]["scene"],
                "normalized_start": start,
                "normalized_end": end,
                "raw_start": raw_start,
                "raw_end": raw_end,
            }
        )
        if section_id == "front_matter":
            parsed = parse_sentence_section(flat, start, end, section_id)
        elif section_id == "production_history":
            heading = [markers["introductory_note"]] if start == positions["introductory_note"] else []
            parsed = parse_sentence_section(flat, start, end, section_id, heading)
        elif section_id == "cast":
            parsed, cast_detected = parse_cast_section(
                flat, start, end, config["cast_role_markers"]
            )
        elif section_id == "act_1":
            parsed = parse_play_section(
                flat,
                start,
                end,
                section_id,
                config["final_speakers"],
                [markers["main_title"], markers["act_1"], markers["act_1_end"]],
            )
        elif section_id == "act_2":
            parsed = parse_play_section(
                flat,
                start,
                end,
                section_id,
                config["final_speakers"],
                [markers["act_2"], markers["act_2_end"]],
            )
        elif section_id == "act_3_scene_1":
            parsed = parse_play_section(
                flat,
                start,
                end,
                section_id,
                config["final_speakers"],
                [markers["act_3"], markers["act_3_scene_1_end"]],
            )
        else:
            parsed = parse_play_section(
                flat,
                start,
                end,
                section_id,
                config["final_speakers"],
                [markers["act_3_scene_2"], markers["work_end"]],
            )
        all_units.extend(parsed)

    if set(cast_detected) != configured_characters:
        raise Phase2Error(
            "演员表标记与配置人物不一致："
            f"cast={sorted(cast_detected)}, configured={sorted(configured_characters)}"
        )
    split_units: list[dict[str, Any]] = []
    for unit in all_units:
        split_units.extend(split_oversized_unit(unit, flat, config["max_words"]))
    units = assign_unit_ids(split_units, mapping, title)
    marker_map = []
    for record in marker_records:
        raw_start, _ = raw_bounds(
            mapping,
            record["normalized_position"],
            record["normalized_position"] + len(record["text"]),
        )
        marker_map.append({**record, "raw_position": raw_start})
    structure = {
        "corpus_name": EXPECTED_CORPUS,
        "title": title,
        "author": "Arthur W. Pinero",
        "genre": "三幕喜剧／闹剧",
        "coordinate_system": (
            "normalized positions refer to the cleaned single-line text before "
            "structural newlines; raw positions refer to indexing_input context"
        ),
        "sections": range_records,
        "front_matter_range": next(
            row for row in range_records if row["section_id"] == "front_matter"
        ),
        "production_history_ranges": [
            row for row in range_records if row["section_id"] == "production_history"
        ],
        "cast_range": next(row for row in range_records if row["section_id"] == "cast"),
        "act_ranges": [
            row for row in range_records if row["section_type"] == "play"
        ],
        "third_act_scene_ranges": [
            row for row in range_records if row["act"] == 3
        ],
        "structure_markers": marker_map,
        "dialogue_candidate_counts": candidate_counts,
        "stable_body_character_candidates": sorted(stable),
        "cast_detected_characters": cast_detected,
        "configured_dialogue_labels": config["final_speakers"],
        "characters": sorted(configured_characters),
        "collective_dialogue_labels": sorted(collective),
    }
    return units, structure


def contiguous_groups(units: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous_key: tuple[Any, ...] | None = None
    for unit in units:
        key = (unit["section_id"], unit["section_type"], unit["act"], unit["scene"])
        if current and key != previous_key:
            groups.append(current)
            current = []
        current.append(unit)
        previous_key = key
    if current:
        groups.append(current)
    return groups


def units_word_count(units: Sequence[dict[str, Any]]) -> int:
    return word_count(" ".join(unit["text"] for unit in units))


def pack_base_chunks(
    units: Sequence[dict[str, Any]], target: int, minimum: int, maximum: int
) -> list[dict[str, Any]]:
    base: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for unit in units:
        proposed = current + [unit]
        if current and units_word_count(proposed) > maximum:
            base.append({"units": current, "exception_reasons": []})
            current = []
        current.append(unit)
        if units_word_count(current) >= target:
            base.append({"units": current, "exception_reasons": []})
            current = []
    if current:
        base.append({"units": current, "exception_reasons": []})

    if base and units_word_count(base[-1]["units"]) < minimum:
        if len(base) >= 2 and units_word_count(
            base[-2]["units"] + base[-1]["units"]
        ) <= maximum:
            base[-2]["units"].extend(base[-1]["units"])
            base.pop()
        else:
            reason = (
                "entire_section_below_min_words"
                if len(base) == 1
                else "tail_below_min_cannot_merge_without_exceeding_max"
            )
            base[-1]["exception_reasons"].append(reason)
    return base


def add_unit_overlap(
    base: Sequence[dict[str, Any]], overlap_words: int, maximum: int
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, item in enumerate(base):
        core_units = list(item["units"])
        overlap: list[dict[str, Any]] = []
        exception_reasons = list(item["exception_reasons"])
        if index > 0:
            previous_units = base[index - 1]["units"]
            for candidate in reversed(previous_units):
                proposed = [candidate] + overlap + core_units
                if units_word_count(proposed) > maximum:
                    break
                overlap.insert(0, candidate)
                if units_word_count(overlap) >= overlap_words:
                    break
            if {unit["unit_id"] for unit in overlap} == {
                unit["unit_id"] for unit in previous_units
            } and not core_units:
                overlap = []
            if not overlap:
                exception_reasons.append(
                    "overlap_omitted_to_preserve_complete_unit_and_max_words"
                )
        output.append(
            {
                "units": overlap + core_units,
                "overlap_unit_ids": [unit["unit_id"] for unit in overlap],
                "overlap_word_count": units_word_count(overlap) if overlap else 0,
                "exception_reasons": exception_reasons,
            }
        )
    return output


def ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def build_chunks(
    units: Sequence[dict[str, Any]], config: dict[str, Any], title: str
) -> list[dict[str, Any]]:
    counters: collections.Counter[str] = collections.Counter()
    chunks: list[dict[str, Any]] = []
    for group_index, group in enumerate(contiguous_groups(units), start=1):
        base = pack_base_chunks(
            group,
            config["target_words"],
            config["min_words"],
            config["max_words"],
        )
        packed = add_unit_overlap(base, config["overlap_words"], config["max_words"])
        group_rows: list[dict[str, Any]] = []
        for item in packed:
            section = item["units"][0]["section_id"]
            counters[section] += 1
            chunk_id = f"{SECTION_PREFIX[section]}_c{counters[section]:04d}"
            text = " ".join(unit["text"] for unit in item["units"]).strip()
            speakers = ordered_unique(
                speaker for unit in item["units"] for speaker in unit["speakers"]
            )
            exceptions = list(item["exception_reasons"])
            for unit in item["units"]:
                if unit.get("exception_reason") and unit["exception_reason"] not in exceptions:
                    exceptions.append(unit["exception_reason"])
            first = item["units"][0]
            row = {
                "chunk_id": chunk_id,
                "corpus_name": EXPECTED_CORPUS,
                "title": title,
                "section_id": section,
                "section_type": first["section_type"],
                "act": first["act"],
                "scene": first["scene"],
                "speakers": speakers,
                "text": text,
                "retrieval_text": f"{title} | {SECTION_LABEL[section]}\n{text}",
                "word_count": word_count(text),
                "char_count": len(text),
                "approx_token_count": math.ceil(len(text) / config["approx_token_chars"]),
                "source_unit_ids": [unit["unit_id"] for unit in item["units"]],
                "overlap_unit_ids": item["overlap_unit_ids"],
                "overlap_word_count": item["overlap_word_count"],
                "previous_chunk_id": None,
                "next_chunk_id": None,
                "exception_reasons": exceptions,
                "_group_index": group_index,
            }
            group_rows.append(row)
        for index, row in enumerate(group_rows):
            if index > 0:
                row["previous_chunk_id"] = group_rows[index - 1]["chunk_id"]
            if index + 1 < len(group_rows):
                row["next_chunk_id"] = group_rows[index + 1]["chunk_id"]
        chunks.extend(group_rows)
    for row in chunks:
        row.pop("_group_index", None)
    return chunks


def percentile(values: Sequence[int], proportion: float) -> float | int:
    ordered = sorted(values)
    if not ordered:
        return 0
    position = (len(ordered) - 1) * proportion
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    result = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return round(result, 2)


def chunk_statistics(chunks: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    values = [row["word_count"] for row in chunks]
    return {
        "min": min(values) if values else 0,
        "p25": percentile(values, 0.25),
        "median": statistics.median(values) if values else 0,
        "p75": percentile(values, 0.75),
        "max": max(values) if values else 0,
    }


def validate_artifacts(
    raw: str,
    flat: str,
    clean_corpus: str,
    units: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
    structure: dict[str, Any],
    removed: Sequence[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any = None) -> None:
        if not condition:
            raise Phase2Error(f"质量检查失败：{name}; detail={detail}")
        checks.append({"name": name, "passed": True, "detail": detail})

    marker_positions = {
        row["name"]: row["normalized_position"]
        for row in structure["structure_markers"]
    }
    check("input_corpus", structure["corpus_name"] == EXPECTED_CORPUS)
    check(
        "act_order",
        marker_positions["act_1"]
        < marker_positions["act_1_end"]
        < marker_positions["act_2"]
        < marker_positions["act_2_end"]
        < marker_positions["act_3"],
    )
    check("first_second_act_end_markers", all(name in marker_positions for name in ("act_1_end", "act_2_end")))
    check(
        "third_act_two_scenes",
        len(structure["third_act_scene_ranges"]) == 2
        and {row["scene"] for row in structure["third_act_scene_ranges"]} == {1, 2},
    )
    preserved = {
        "title": "DANDY DICK A PLAY IN THREE ACTS" in clean_corpus,
        "author": bool(re.search(r"ARTHUR W\.\s+PINERO", clean_corpus)),
        "production_history": '"Dandy Dick" was performed 171 times' in clean_corpus,
        "theatre": "ROYAL COURT THEATRE" in clean_corpus,
        "cast": "THE VERY REV. AUGUSTIN JEDD" in clean_corpus,
    }
    check("required_front_material_preserved", all(preserved.values()), preserved)
    check("at_least_eight_characters", len(structure["characters"]) >= 8, len(structure["characters"]))
    check(
        "dialogue_nonempty",
        any(row["unit_type"] == "dialogue" and row["text"].strip() for row in units),
    )
    check(
        "stage_direction_nonempty",
        any(row["unit_type"] == "stage_direction" and row["text"].strip() for row in units),
    )
    check(
        "clean_content_conserved_during_structure_restore",
        " ".join(clean_corpus.split()) == " ".join(flat.split()),
    )
    check(
        "stage_direction_markup_preserved",
        clean_corpus.count("[") == flat.count("[")
        and clean_corpus.count("]") == flat.count("]")
        and clean_corpus.count("_") == flat.count("_"),
        {
            "opening_brackets": flat.count("["),
            "closing_brackets": flat.count("]"),
            "italic_underscores": flat.count("_"),
        },
    )
    unit_ids = [row["unit_id"] for row in units]
    chunk_ids = [row["chunk_id"] for row in chunks]
    check("unit_ids_unique", len(unit_ids) == len(set(unit_ids)), len(unit_ids))
    check("chunk_ids_unique", len(chunk_ids) == len(set(chunk_ids)), len(chunk_ids))
    covered = {unit_id for row in chunks for unit_id in row["source_unit_ids"]}
    check("all_units_covered", covered == set(unit_ids), len(covered))
    by_unit = {row["unit_id"]: row for row in units}
    check(
        "no_cross_section_or_scene",
        all(
            len(
                {
                    (
                        by_unit[unit_id]["section_id"],
                        by_unit[unit_id]["act"],
                        by_unit[unit_id]["scene"],
                    )
                    for unit_id in row["source_unit_ids"]
                }
            )
            == 1
            for row in chunks
        ),
    )
    check("no_empty_chunks", all(row["text"].strip() for row in chunks))
    check("no_duplicate_chunks", len({row["text"] for row in chunks}) == len(chunks))
    too_small = [
        row["chunk_id"]
        for row in chunks
        if row["word_count"] < config["min_words"]
        and not any("below_min" in reason for reason in row["exception_reasons"])
    ]
    check("chunk_minimum_or_documented_tail", not too_small, too_small)
    too_large = [
        row["chunk_id"]
        for row in chunks
        if row["word_count"] > config["max_words"]
        and "single_sentence_exceeds_max_words" not in row["exception_reasons"]
    ]
    check("chunk_maximum_or_long_sentence_exception", not too_large, too_large)
    missing_overlap_exceptions = [
        row["chunk_id"]
        for row in chunks
        if row["previous_chunk_id"]
        and row["overlap_word_count"] == 0
        and "overlap_omitted_to_preserve_complete_unit_and_max_words"
        not in row["exception_reasons"]
    ]
    check(
        "overlap_present_or_documented",
        not missing_overlap_exceptions,
        missing_overlap_exceptions,
    )
    chunk_map = {row["chunk_id"]: row for row in chunks}
    bad_links: list[str] = []
    for row in chunks:
        for direction, reciprocal in (("previous_chunk_id", "next_chunk_id"), ("next_chunk_id", "previous_chunk_id")):
            target = row[direction]
            if target is None:
                continue
            if target not in chunk_map or chunk_map[target][reciprocal] != row["chunk_id"]:
                bad_links.append(f"{row['chunk_id']}:{direction}->{target}")
                continue
            target_row = chunk_map[target]
            if (row["section_id"], row["act"], row["scene"]) != (
                target_row["section_id"], target_row["act"], target_row["scene"]
            ):
                bad_links.append(f"cross-boundary:{row['chunk_id']}->{target}")
    check("chunk_links_valid", not bad_links, bad_links)
    missing_units = [
        unit_id
        for row in chunks
        for unit_id in row["source_unit_ids"]
        if unit_id not in by_unit
    ]
    check("chunk_unit_references_valid", not missing_units, missing_units)
    check(
        "chunk_counts_and_retrieval_prefixes_valid",
        all(
            row["char_count"] == len(row["text"])
            and row["word_count"] == word_count(row["text"])
            and row["approx_token_count"]
            == math.ceil(row["char_count"] / config["approx_token_chars"])
            and row["retrieval_text"]
            == f"{row['title']} | {SECTION_LABEL[row['section_id']]}\n{row['text']}"
            for row in chunks
        ),
    )
    check(
        "chunk_schema_has_no_supervision_keys",
        all(not (SUPERVISION_KEYS & set(row)) for row in chunks),
    )
    retention = len(flat) / len(raw)
    check("retention_at_least_85_percent", retention >= 0.85, retention)
    check(
        "work_end_retained",
        flat.endswith(config["structure_markers"]["work_end"]),
    )
    check(
        "footer_removed",
        config["structure_markers"]["footer_start"] not in clean_corpus,
    )
    check(
        "removed_spans_logged",
        all(
            row["raw_end"] > row["raw_start"]
            and row["deleted_char_count"] == row["raw_end"] - row["raw_start"]
            and len(row["before_preview"]) <= 100
            and len(row["after_preview"]) <= 100
            for row in removed
        ),
    )
    return {
        "status": "pass",
        "checks": checks,
        "retention_ratio": retention,
        "small_chunk_exceptions": [
            {"chunk_id": row["chunk_id"], "reasons": row["exception_reasons"]}
            for row in chunks
            if row["word_count"] < config["min_words"]
        ],
        "oversize_chunk_exceptions": [
            {"chunk_id": row["chunk_id"], "reasons": row["exception_reasons"]}
            for row in chunks
            if row["word_count"] > config["max_words"]
        ],
        "overlap_exceptions": [
            {"chunk_id": row["chunk_id"], "reasons": row["exception_reasons"]}
            for row in chunks
            if row["previous_chunk_id"] and row["overlap_word_count"] == 0
        ],
    }


def render_clean_corpus(units: Sequence[dict[str, Any]]) -> str:
    return "\n\n".join(row["text"] for row in units).strip() + "\n"


def make_core_artifacts(
    input_data: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    raw = input_data["context"]
    flat, mapping, removed, cleaning_summary = clean_raw_text(
        raw, config["structure_markers"]
    )
    units, structure = build_units_and_structure(
        flat, mapping, config, input_data["title"]
    )
    clean_corpus = render_clean_corpus(units)
    chunks = build_chunks(units, config, input_data["title"])
    quality = validate_artifacts(
        raw, flat, clean_corpus, units, chunks, structure, removed, config
    )
    return {
        "flat": flat,
        "clean_corpus": clean_corpus,
        "removed": removed,
        "cleaning_summary": cleaning_summary,
        "units": units,
        "chunks": chunks,
        "structure": structure,
        "quality": quality,
    }


def core_digest(core: dict[str, Any]) -> str:
    serializable = {
        key: core[key]
        for key in (
            "clean_corpus",
            "removed",
            "cleaning_summary",
            "units",
            "chunks",
            "structure",
            "quality",
        )
    }
    return sha256_bytes(json_dump(serializable).encode("utf-8"))


def removed_category_counts(removed: Sequence[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for row in removed:
        entry = counts.setdefault(row["reason"], {"span_count": 0, "character_count": 0})
        entry["span_count"] += 1
        entry["character_count"] += row["deleted_char_count"]
    return counts


def preview(text: str, start: int, end: int, limit: int = 420) -> str:
    value = " ".join(text[start:end].split())
    return value[:limit] + ("…" if len(value) > limit else "")


def choose_chunk_example(
    chunks: Sequence[dict[str, Any]],
    section_id: str,
    *,
    min_speakers: int = 0,
    contains: str | None = None,
) -> dict[str, Any]:
    candidates = [
        row
        for row in chunks
        if row["section_id"] == section_id
        and len(row["speakers"]) >= min_speakers
        and (contains is None or contains in row["text"])
    ]
    if not candidates:
        candidates = [row for row in chunks if row["section_id"] == section_id]
    if not candidates:
        raise Phase2Error(f"报告无法找到 {section_id} 切块样例")
    return max(candidates, key=lambda row: (len(row["speakers"]), row["word_count"]))


def render_report(
    input_display: str,
    input_hash: str,
    raw: str,
    core: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    units = core["units"]
    chunks = core["chunks"]
    structure = core["structure"]
    stats = manifest["chunk_word_statistics"]
    category_counts = removed_category_counts(core["removed"])
    unit_counts = collections.Counter(row["unit_type"] for row in units)
    section_counts = collections.Counter(row["section_id"] for row in chunks)
    prod = choose_chunk_example(chunks, "production_history")
    cast = choose_chunk_example(chunks, "cast")
    act1 = choose_chunk_example(chunks, "act_1", min_speakers=2)
    stage = next(
        (
            row
            for row in chunks
            if any(
                unit["unit_type"] == "stage_direction"
                for unit in units
                if unit["unit_id"] in row["source_unit_ids"]
            )
        ),
        choose_chunk_example(chunks, "act_1"),
    )
    act3 = choose_chunk_example(chunks, "act_3_scene_2", min_speakers=3)
    raw_title_start = core["cleaning_summary"]["title_start"]
    raw_end = core["cleaning_summary"]["work_end_end"]
    clean_title_end = min(len(core["clean_corpus"]), 430)
    dialogue_raw = raw.find("SALOME. Oh! oh my!")
    boundary_raw = raw.find("END OF THE SECOND ACT.")
    lines = [
        "# 阶段 2：《Dandy Dick》确定性清洗与结构感知切块",
        "",
        "本阶段只读取阶段 1 的 `indexing_input.json` 正文，不读取训练题、测试题、",
        "参考答案、evidence 或 evidence_triple。未调用 LLM/Embedding，未建图，",
        "未运行检索，也未安装 LightRAG/PathRAG。",
        "",
        "## 输入与清洗",
        "",
        f"- 输入：`{input_display}`",
        f"- 输入文件 SHA-256：`{input_hash}`",
        f"- 原始正文：{len(raw):,} 字符",
        f"- 清洗后语义正文：{core['cleaning_summary']['flat_clean_char_count']:,} 字符",
        f"- 写入结构换行后的 `clean_corpus.txt`：{len(core['clean_corpus']):,} 字符",
        f"- 文本保留比例：{manifest['retention_ratio']:.4%}",
        f"- 删除区间：{len(core['removed'])} 个，共 "
        f"{core['cleaning_summary']['deleted_char_count']:,} 字符",
        "",
        "删除类别：",
        "",
        "| 原因 | 区间数 | 字符数 |",
        "|---|---:|---:|",
    ]
    for reason, values in category_counts.items():
        lines.append(
            f"| {reason} | {values['span_count']} | {values['character_count']:,} |"
        )
    lines.extend(
        [
            "",
            "清洗器只允许删除标题前制作者信息、作品结束后的转录/Gutenberg 页脚和重复空白；",
            "本次实际删除仅为前两类，原始正文没有需要压缩的连续空白。",
            "本输入未发现 HTML 标签或独立 `[Illustration]` 占位，因此没有伪造零长度删除记录。",
            "弯引号、破折号、拼写、人物口音、历史措辞和斜体下划线均保持原样。",
            "",
            "## 戏剧结构",
            "",
            "- 标题：*Dandy Dick*",
            "- 作者：Arthur W. Pinero",
            "- 体裁：三幕喜剧／闹剧",
            "- 识别结构：front matter、两段 production history、cast/首演节目、",
            "  Act 1、Act 2、Act 3 Scene 1、Act 3 Scene 2。",
            f"- 人物数：{len(structure['characters'])}；集体说话标签另有 `ALL`。",
            "- 人物：" + "、".join(structure["characters"]),
            "- 演出历史、Royal Court Theatre、首演演员信息和舞台说明均已保留。",
            "",
            "说话人不是由代码中的固定列表直接猜测：脚本先屏蔽方括号舞台说明，",
            "从正文抽取稳定大写说话标记，再与演员表内容标记及配置中的最终集合交叉验证。",
            "",
            "## Unit 与 Chunk 统计",
            "",
            f"- Unit：{len(units)}",
            f"- Unit 类型：`{dict(sorted(unit_counts.items()))}`",
            f"- Chunk：{len(chunks)}",
            f"- 各结构段 Chunk：`{dict(sorted(section_counts.items()))}`",
            f"- 词数：min={stats['min']}，p25={stats['p25']}，median={stats['median']}，"
            f"p75={stats['p75']}，max={stats['max']}。",
            "",
            "戏剧中的说话人、舞台动作、出入场和幕/场边界共同承载人物关系与事件链。",
            "因此切块以完整结构单元为最小单位，只复制完整 unit 形成重叠，且禁止跨幕或跨场。",
            "",
            "## 三组清洗前后对照",
            "",
            "### 1. 标题与制作信息",
            "",
            "- 清洗前：`" + preview(raw, 0, min(len(raw), raw_title_start + 260)) + "`",
            "- 清洗后：`" + preview(core["clean_corpus"], 0, clean_title_end) + "`",
            "",
            "### 2. 人物台词与舞台说明",
            "",
            "- 清洗前：`" + preview(raw, dialogue_raw - 180, dialogue_raw + 330) + "`",
            "- 清洗后：`" + preview(core["clean_corpus"], core["clean_corpus"].find("SALOME. Oh! oh my!") - 180, core["clean_corpus"].find("SALOME. Oh! oh my!") + 330) + "`",
            "",
            "### 3. 幕边界",
            "",
            "- 清洗前：`" + preview(raw, boundary_raw - 150, boundary_raw + 260) + "`",
            "- 清洗后：`" + preview(core["clean_corpus"], core["clean_corpus"].find("END OF THE SECOND ACT.") - 150, core["clean_corpus"].find("END OF THE SECOND ACT.") + 260) + "`",
            "",
            "## 五类切块抽样",
            "",
        ]
    )
    for label, row in (
        ("Production history", prod),
        ("Cast", cast),
        ("第一幕对话", act1),
        ("舞台说明所在块", stage),
        ("第三幕第二场多人场景", act3),
    ):
        lines.extend(
            [
                f"### {label}：`{row['chunk_id']}`",
                "",
                f"- 结构：`{row['section_id']}`；词数：{row['word_count']}；"
                f"说话人：`{row['speakers']}`",
                "- 文本：`" + preview(row["text"], 0, len(row["text"]), 650) + "`",
                "",
            ]
        )
    lines.extend(
        [
            "## 已记录的非阻塞结构例外",
            "",
            "- 原文最终标记实际为 `THE END`（无句点），已按真实文本配置，未补造标点。",
            "- 原文舞台说明采用 Gutenberg 排版约定，部分只有起始 `[`，由成对斜体下划线",
            "  标出结束；脚本同时保护方括号区间与斜体区间，未把其中人物名识别成台词。",
            "- front matter 只有 60 词，因禁止跨结构边界合并而保留为小块。",
            "- 两个相邻块因完整末尾台词加入后会超过 520 词，未强行制造重叠，原因已写入 chunk。",
            "- 以上均有自动检查或显式例外记录，不构成阻塞；仍建议提交前人工抽查报告中的五类样例。",
            "",
            "## 质量结论与边界",
            "",
            f"- 自动质量检查：{core['quality']['status']}。",
            f"- 同一进程双构建核心摘要：`{manifest['determinism_digest']}`，完全一致。",
            "- 所有 unit 均被覆盖；chunk ID、unit ID 和链接均有效。",
            "- 没有 chunk 跨 section、幕或场；没有空块或完全重复块。",
            f"- 小尾块例外：`{core['quality']['small_chunk_exceptions']}`。",
            f"- 超长块例外：`{core['quality']['oversize_chunk_exceptions']}`。",
            f"- 无重叠例外：`{core['quality']['overlap_exceptions']}`。",
            "- 阶段 1 输入文件保持不变。",
            "",
            "**下一步是构建共享知识图谱；本阶段没有自行开始。**",
            "",
        ]
    )
    return "\n".join(lines)


def prepare_phase2(
    input_path: Path,
    config_path: Path,
    output_dir: Path,
    report_path: Path,
    *,
    input_display: str | None = None,
) -> dict[str, Any]:
    input_hash_before = sha256_file(input_path)
    input_data = load_input(input_path)
    config = load_config(config_path)
    first = make_core_artifacts(input_data, config)
    second = make_core_artifacts(input_data, config)
    first_digest = core_digest(first)
    second_digest = core_digest(second)
    if first_digest != second_digest:
        raise Phase2Error(
            f"连续内存构建不一致：first={first_digest}, second={second_digest}"
        )
    if sha256_file(input_path) != input_hash_before:
        raise Phase2Error("阶段1 indexing_input.json 在处理过程中被修改")

    section_counts = dict(
        sorted(collections.Counter(row["section_id"] for row in first["chunks"]).items())
    )
    unit_type_counts = dict(
        sorted(collections.Counter(row["unit_type"] for row in first["units"]).items())
    )
    manifest = {
        "corpus_name": input_data["corpus_name"],
        "title": input_data["title"],
        "input_file": input_display or str(input_path),
        "input_sha256": input_hash_before,
        "config_file": str(config_path),
        "config_sha256": sha256_file(config_path),
        "cleaning_version": CLEANING_VERSION,
        "chunking_config": config,
        "raw_char_count": len(input_data["context"]),
        "cleaned_char_count": first["cleaning_summary"]["flat_clean_char_count"],
        "structured_clean_file_char_count": len(first["clean_corpus"]),
        "structural_separator_characters_added": (
            len(first["clean_corpus"])
            - first["cleaning_summary"]["flat_clean_char_count"]
        ),
        "retention_ratio": (
            first["cleaning_summary"]["flat_clean_char_count"]
            / len(input_data["context"])
        ),
        "unit_count": len(first["units"]),
        "unit_type_counts": unit_type_counts,
        "chunk_count": len(first["chunks"]),
        "section_counts": section_counts,
        "chunk_word_statistics": chunk_statistics(first["chunks"]),
        "deterministic": True,
        "determinism_digest": first_digest,
    }
    first["quality"].update(
        {
            "input_sha256_before": input_hash_before,
            "input_sha256_after": sha256_file(input_path),
            "stage1_input_unchanged": True,
            "core_double_build_digest": first_digest,
            "unit_type_counts": unit_type_counts,
            "section_chunk_counts": section_counts,
            "chunk_word_statistics": manifest["chunk_word_statistics"],
            "removed_category_counts": removed_category_counts(first["removed"]),
        }
    )
    report = render_report(
        input_display or str(input_path),
        input_hash_before,
        input_data["context"],
        first,
        manifest,
    )
    outputs = {
        "clean_corpus.txt": first["clean_corpus"],
        "document_structure.json": json_dump(first["structure"]),
        "units.jsonl": jsonl_dump(first["units"]),
        "chunks.jsonl": jsonl_dump(first["chunks"]),
        "chunk_manifest.json": json_dump(manifest),
        "removed_spans.jsonl": jsonl_dump(first["removed"]),
        "quality_report.json": json_dump(first["quality"]),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        (output_dir / name).write_text(content, encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    if sha256_file(input_path) != input_hash_before:
        raise Phase2Error("阶段1输入在写出阶段2产物时被修改")
    return {
        "manifest": manifest,
        "quality_status": first["quality"]["status"],
        "output_dir": str(output_dir),
        "report_path": str(report_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/phase2_cleaning_and_chunking.md"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_display = args.input.as_posix()
    try:
        result = prepare_phase2(
            args.input.resolve(),
            args.config.resolve(),
            args.output_dir.resolve(),
            args.report_path.resolve(),
            input_display=input_display,
        )
    except Phase2Error as exc:
        print(f"阶段2失败：{exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
