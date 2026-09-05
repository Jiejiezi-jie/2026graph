#!/usr/bin/env python3
"""Prepare the leak-safe GraphRAG-Bench Novel phase-one dataset.

This script performs data inspection, deterministic work selection and a
stratified train/test split only. It deliberately does not chunk text, build a
graph, run retrieval, call an LLM or train a router.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


OFFICIAL_URL = "https://github.com/GraphRAG-Bench/GraphRAG-Benchmark"
CORPUS_RELATIVE_PATH = Path("Datasets/Corpus/novel.json")
QUESTIONS_RELATIVE_PATH = Path("Datasets/Questions/novel_questions.json")
ALLOWED_TYPES = ("Fact Retrieval", "Complex Reasoning")
EXCLUDED_TYPES = ("Contextual Summarize", "Creative Generation")
QUESTION_FIELDS = (
    "id",
    "source",
    "question",
    "answer",
    "question_type",
    "evidence",
    "evidence_triple",
)
FORBIDDEN_TITLES = (
    "红楼梦",
    "Dream of the Red Chamber",
    "The Story of the Stone",
    "Hong Lou Meng",
)
SELECTION_RULE = (
    "先满足非《红楼梦》、单一连贯叙事作品、Fact Retrieval + Complex "
    "Reasoning >= 70、Complex Reasoning >= 25、文本非空且 source 严格匹配；"
    "再优先单一完整作品；以近似词数最少为主，距最短作品不超过 10% 时优先"
    "Complex Reasoning 更多者，最后按 corpus_name 字典序。"
)


class DatasetPreparationError(RuntimeError):
    """Raised when source data or prepared output violates the protocol."""


@dataclass(frozen=True)
class WorkAuditRule:
    fingerprint: str
    title: str
    genre: str
    coherent_narrative: bool
    classification_reason: str


# Human-audited against at least the first 2,000 characters of every official
# Novel corpus. Matching content fingerprints rather than corpus IDs keeps the
# selection data-driven and makes an upstream text change fail visibly.
WORK_AUDIT_RULES = (
    WorkAuditRule(
        "VESTIGES OF THE MAYAS",
        "Vestiges of the Mayas",
        "考古学论著",
        False,
        "考古与跨文明联系论著，不是人物和情节连续的叙事作品",
    ),
    WorkAuditRule(
        "DR. ELSIE INGLIS",
        "Dr. Elsie Inglis",
        "传记",
        True,
        "围绕 Elsie Inglis 生平展开的单一连贯传记",
    ),
    WorkAuditRule(
        "AN ASTRONOMER'S WIFE",
        "An Astronomer's Wife",
        "传记",
        True,
        "围绕 Angeline Hall 生平展开的单一连贯传记",
    ),
    WorkAuditRule(
        "IMPRESSIONS OF THEOPHRASTUS SUCH",
        "Impressions of Theophrastus Such",
        "讽刺随笔／人物素描集",
        False,
        "由独立社会观察和人物素描组成，不是连续情节叙事",
    ),
    WorkAuditRule(
        "LAURENCE STERNE IN GERMANY",
        "Laurence Sterne in Germany",
        "文学研究专著",
        False,
        "学术研究专著，不是叙事性文学作品",
    ),
    WorkAuditRule(
        "THE 'PATRIOTES' OF '37",
        "The 'Patriotes' of '37",
        "历史叙事",
        True,
        "围绕 1837 年下加拿大叛乱展开的单一连贯历史叙事",
    ),
    WorkAuditRule(
        "IF JAMES M. GOODHUE COULD REVISIT THE EARTH",
        "Reminiscences of Pioneer Days in St. Paul",
        "地方史文章合集",
        False,
        "由报刊文章汇编而成，涉及多组地方史主题，不是单一完整作品叙事",
    ),
    WorkAuditRule(
        "THE AMORES; OR, AMOURS",
        "The Amores; or, Amours",
        "诗歌／哀歌集",
        False,
        "由多首独立哀歌组成，缺少贯穿全书的连续情节",
    ),
    WorkAuditRule(
        "DRAGON'S BLOOD BY HENRY MILNER RIDEOUT",
        "Dragon's Blood",
        "小说",
        True,
        "章节连续、人物和情节统一的单一小说",
    ),
    WorkAuditRule(
        "SCIENTIFIC AMERICAN SUPPLEMENT NO. 360",
        "Scientific American Supplement No. 360",
        "科学期刊合集",
        False,
        "同一期包含工程、化学、自然史等多篇文章，不是单一叙事作品",
    ),
    WorkAuditRule(
        "PEN PICTURES OF EVENTFUL SCENES AND STRUGGLES OF LIFE",
        "Pen Pictures of Eventful Scenes and Struggles of Life",
        "回忆性事件合集",
        False,
        "作者明确汇集多人和多段事件经历，不是单一连续叙事",
    ),
    WorkAuditRule(
        "CHILD'S HEALTH PRIMER",
        "Child's Health Primer",
        "健康教材",
        False,
        "面向初级课堂的生理与健康教材，不是叙事作品",
    ),
    WorkAuditRule(
        "GALLEGHER AND OTHER STORIES",
        "Gallegher and Other Stories",
        "短篇小说集",
        False,
        "目录包含多篇互不连续的短篇故事",
    ),
    WorkAuditRule(
        "FROM SAND HILL TO PINE BY BRET HARTE",
        "From Sand Hill to Pine",
        "短篇小说集",
        False,
        "目录包含多篇不同人物和情节的短篇故事",
    ),
    WorkAuditRule(
        "TOTO'S MERRY WINTER",
        "Toto's Merry Winter",
        "儿童小说",
        True,
        "围绕 Toto 冬日经历展开的章节式连续儿童叙事",
    ),
    WorkAuditRule(
        "MUSICAL INSTRUMENTS SOUTH KENSINGTON MUSEUM ART HANDBOOKS",
        "Musical Instruments",
        "博物馆艺术手册",
        False,
        "乐器历史与藏品说明手册，不是叙事作品",
    ),
    WorkAuditRule(
        "TRAVELS IN MOROCCO",
        "Travels in Morocco, Volume II",
        "旅行叙事",
        True,
        "围绕同一次摩洛哥旅行展开的连续非虚构叙事（第二卷）",
    ),
    WorkAuditRule(
        "DANDY DICK A PLAY IN THREE ACTS",
        "Dandy Dick",
        "三幕喜剧／闹剧",
        True,
        "人物、冲突和情节贯穿三幕的单一戏剧作品",
    ),
    WorkAuditRule(
        "THE DIARY OF SAMUEL PEPYS",
        "The Diary of Samuel Pepys: June-August 1661",
        "日记体叙事",
        True,
        "围绕 Samuel Pepys 连续三个月经历展开的日记体叙事",
    ),
    WorkAuditRule(
        "AN UNSENTIMENTAL JOURNEY THROUGH CORNWALL",
        "An Unsentimental Journey Through Cornwall",
        "旅行叙事",
        True,
        "按连续行程日期组织的单一 Cornwall 旅行叙事",
    ),
)


def load_json_array(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetPreparationError(f"缺少{label}文件：{path}") from exc
    except json.JSONDecodeError as exc:
        raise DatasetPreparationError(f"{label}不是有效 JSON：{path}: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise DatasetPreparationError(f"{label}必须是由对象组成的 JSON 数组：{path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_opening(text: str, length: int = 4000) -> str:
    punctuation = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})
    return " ".join(text[:length].translate(punctuation).upper().split())


def classify_work(text: str) -> WorkAuditRule:
    opening = normalize_opening(text)
    matches = [rule for rule in WORK_AUDIT_RULES if rule.fingerprint in opening]
    if len(matches) != 1:
        fingerprints = [rule.fingerprint for rule in matches]
        raise DatasetPreparationError(
            "无法根据文本开头唯一识别作品；"
            f"匹配到 {len(matches)} 条内容规则：{fingerprints}。请人工复核上游文本。"
        )
    return matches[0]


def find_duplicates(values: Iterable[str]) -> list[str]:
    counts = collections.Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def forbidden_hits(*values: str) -> list[str]:
    joined = "\n".join(values).casefold()
    return [name for name in FORBIDDEN_TITLES if name.casefold() in joined]


def inspect_raw_data(
    corpora: Sequence[dict[str, Any]], questions: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    corpus_required = {"corpus_name", "context"}
    question_required = set(QUESTION_FIELDS)
    corpus_fields = sorted(set().union(*(row.keys() for row in corpora))) if corpora else []
    question_fields = (
        sorted(set().union(*(row.keys() for row in questions))) if questions else []
    )
    for index, row in enumerate(corpora):
        missing = corpus_required - row.keys()
        if missing:
            raise DatasetPreparationError(
                f"语料第 {index} 行字段与预期不一致，缺少：{sorted(missing)}；"
                f"实际字段：{sorted(row)}"
            )
        if not isinstance(row["corpus_name"], str) or not isinstance(row["context"], str):
            raise DatasetPreparationError(f"语料第 {index} 行字段类型不正确")
    for index, row in enumerate(questions):
        missing = question_required - row.keys()
        if missing:
            raise DatasetPreparationError(
                f"问题第 {index} 行字段与预期不一致，缺少：{sorted(missing)}；"
                f"实际字段：{sorted(row)}"
            )
        if any(not isinstance(row[field], str) for field in QUESTION_FIELDS):
            raise DatasetPreparationError(f"问题第 {index} 行存在非字符串必需字段")

    corpus_names = [row["corpus_name"] for row in corpora]
    corpus_duplicates = find_duplicates(corpus_names)
    if corpus_duplicates:
        raise DatasetPreparationError(f"corpus_name 不唯一：{corpus_duplicates}")
    empty_corpora = [row["corpus_name"] for row in corpora if not row["context"].strip()]
    if empty_corpora:
        raise DatasetPreparationError(f"存在空语料：{empty_corpora}")
    empty_questions = [row["id"] for row in questions if not row["question"].strip()]
    empty_answers = [row["id"] for row in questions if not row["answer"].strip()]
    if empty_questions or empty_answers:
        raise DatasetPreparationError(
            f"存在空问题或空答案：empty_questions={empty_questions}, "
            f"empty_answers={empty_answers}"
        )
    unmatched_sources = sorted({row["source"] for row in questions} - set(corpus_names))
    if unmatched_sources:
        raise DatasetPreparationError(f"问题 source 无法匹配语料：{unmatched_sources}")

    duplicate_ids = find_duplicates(row["id"] for row in questions)
    duplicate_texts = find_duplicates(row["question"].strip() for row in questions)
    duplicate_id_rows = {
        duplicate_id: [
            {
                "source": row["source"],
                "question_type": row["question_type"],
                "question": row["question"],
            }
            for row in questions
            if row["id"] == duplicate_id
        ]
        for duplicate_id in duplicate_ids
    }
    type_counts = dict(sorted(collections.Counter(row["question_type"] for row in questions).items()))
    all_forbidden: dict[str, list[str]] = {}
    for row in corpora:
        rule = classify_work(row["context"])
        hits = forbidden_hits(row["corpus_name"], rule.title, row["context"])
        if hits:
            all_forbidden[row["corpus_name"]] = hits

    return {
        "corpus_json_type": "array",
        "questions_json_type": "array",
        "corpus_fields": corpus_fields,
        "question_fields": question_fields,
        "corpus_count": len(corpora),
        "question_count": len(questions),
        "corpus_name_unique": True,
        "question_id_unique": not duplicate_ids,
        "duplicate_question_ids": duplicate_id_rows,
        "duplicate_question_text_count": len(duplicate_texts),
        "unmatched_sources": unmatched_sources,
        "empty_corpus_count": 0,
        "empty_question_count": 0,
        "empty_answer_count": 0,
        "question_type_counts": type_counts,
        "forbidden_work_hits": all_forbidden,
        "warning": (
            "官方全量问题中存在跨作品重复 ID；原始文件保持不变。后续只允许在选中"
            "作品内部 ID 唯一时继续。"
            if duplicate_ids
            else None
        ),
    }


def build_candidate_statistics(
    corpora: Sequence[dict[str, Any]], questions: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    questions_by_source: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in questions:
        questions_by_source[row["source"]].append(row)

    candidates: list[dict[str, Any]] = []
    for corpus in corpora:
        name = corpus["corpus_name"]
        text = corpus["context"]
        rule = classify_work(text)
        rows = questions_by_source[name]
        counts = collections.Counter(row["question_type"] for row in rows)
        usable = counts[ALLOWED_TYPES[0]] + counts[ALLOWED_TYPES[1]]
        hits = forbidden_hits(name, rule.title, text)
        reasons: list[str] = []
        if hits:
            reasons.append(f"命中禁用《红楼梦》名称：{hits}")
        if not rule.coherent_narrative:
            reasons.append(rule.classification_reason)
        if usable < 70:
            reasons.append(f"两类可用问题仅 {usable} 道，少于 70")
        if counts["Complex Reasoning"] < 25:
            reasons.append(
                f"Complex Reasoning 仅 {counts['Complex Reasoning']} 道，少于 25"
            )
        if not text.strip():
            reasons.append("原始文本为空")
        source_match = all(row["source"] == name for row in rows)
        if not source_match:
            reasons.append("问题 source 与 corpus_name 不严格匹配")
        candidates.append(
            {
                "corpus_name": name,
                "title": rule.title,
                "genre": rule.genre,
                "character_count": len(text),
                "approximate_word_count": len(text.split()),
                "question_count": len(rows),
                "fact_retrieval_count": counts["Fact Retrieval"],
                "complex_reasoning_count": counts["Complex Reasoning"],
                "usable_question_count": usable,
                "contextual_summarize_count": counts["Contextual Summarize"],
                "creative_generation_count": counts["Creative Generation"],
                "opening_inspected_characters": min(2000, len(text)),
                "single_coherent_narrative": rule.coherent_narrative,
                "classification_reason": rule.classification_reason,
                "forbidden_work": bool(hits),
                "source_match": source_match,
                "meets_candidate_standard": not reasons,
                "exclusion_reasons": reasons,
            }
        )
    return sorted(candidates, key=lambda row: row["corpus_name"])


def select_candidate(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in candidates if row["meets_candidate_standard"]]
    if not eligible:
        closest = sorted(
            candidates,
            key=lambda row: (
                len(row["exclusion_reasons"]),
                -row["usable_question_count"],
                -row["complex_reasoning_count"],
                row["corpus_name"],
            ),
        )[:3]
        detail = "; ".join(
            f"{row['corpus_name']}: {', '.join(row['exclusion_reasons'])}"
            for row in closest
        )
        raise DatasetPreparationError(f"没有作品满足候选标准。最接近的三项：{detail}")

    shortest = min(row["approximate_word_count"] for row in eligible)
    near_shortest = [
        row for row in eligible if row["approximate_word_count"] <= shortest * 1.10
    ]
    return min(
        near_shortest,
        key=lambda row: (-row["complex_reasoning_count"], row["corpus_name"]),
    )


def stratified_split(
    questions: Sequence[dict[str, Any]], seed: int, test_size: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for question_type in ALLOWED_TYPES:
        group = sorted(
            (row for row in questions if row["question_type"] == question_type),
            key=lambda row: row["id"],
        )
        random.Random(seed).shuffle(group)
        test_count = int(len(group) * test_size + 0.5)
        test.extend(group[:test_count])
        train.extend(group[test_count:])
    return (
        sorted(train, key=lambda row: row["id"]),
        sorted(test, key=lambda row: row["id"]),
    )


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    content = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    path.write_text(content, encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def detect_git_source(dataset_root: Path) -> tuple[str, str]:
    try:
        remote = subprocess.run(
            ["git", "-C", str(dataset_root), "remote", "get-url", "origin"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        revision = subprocess.run(
            ["git", "-C", str(dataset_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DatasetPreparationError(
            "dataset-root 不是可验证的官方 Git checkout；如使用 Hugging Face "
            "备用源，需先扩展脚本记录其 revision。"
        ) from exc
    normalized = remote.rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    if normalized.casefold() != OFFICIAL_URL.casefold():
        raise DatasetPreparationError(f"数据远程不是官方仓库：{remote}")
    if len(revision) != 40:
        raise DatasetPreparationError(f"无法获得完整 Git commit：{revision}")
    return remote, revision


def split_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = collections.Counter(row["question_type"] for row in rows)
    return {question_type: counts[question_type] for question_type in ALLOWED_TYPES}


def validate_prepared_output(
    output_dir: Path,
    selected_name: str,
    filtered: Sequence[dict[str, Any]],
    train: Sequence[dict[str, Any]],
    test: Sequence[dict[str, Any]],
    test_size: float,
) -> None:
    filtered_ids = {row["id"] for row in filtered}
    train_ids = {row["id"] for row in train}
    test_ids = {row["id"] for row in test}
    if len(filtered_ids) != len(filtered):
        raise DatasetPreparationError("选中作品的筛选问题存在重复 ID")
    if train_ids & test_ids:
        raise DatasetPreparationError("训练集和测试集 ID 有交集")
    if train_ids | test_ids != filtered_ids:
        raise DatasetPreparationError("训练/测试 ID 并集不等于全部筛选问题")
    if any(row["source"] != selected_name for row in filtered):
        raise DatasetPreparationError("筛选后存在 source 不匹配的问题")
    if {row["question_type"] for row in filtered} != set(ALLOWED_TYPES):
        raise DatasetPreparationError("筛选后问题类型不等于允许的两类")
    if any(not row["question"].strip() or not row["answer"].strip() for row in filtered):
        raise DatasetPreparationError("筛选后存在空问题或空答案")
    for question_type in ALLOWED_TYPES:
        total = sum(row["question_type"] == question_type for row in filtered)
        test_total = sum(row["question_type"] == question_type for row in test)
        if abs(test_total - total * test_size) > 1:
            raise DatasetPreparationError(f"{question_type} 测试集取整误差超过 1 条")

    public_test = read_jsonl(output_dir / "test_queries.jsonl")
    if any(set(row) != {"id", "source", "question", "question_type"} for row in public_test):
        raise DatasetPreparationError("test_queries.jsonl 字段未严格脱敏")
    indexing = json.loads((output_dir / "indexing_input.json").read_text(encoding="utf-8"))
    if set(indexing) != {"corpus_name", "title", "context"}:
        raise DatasetPreparationError("indexing_input.json 含监督字段或缺少作品字段")
    forbidden_keys = {"question", "answer", "evidence", "evidence_triple"}
    if forbidden_keys & set(indexing):
        raise DatasetPreparationError("indexing_input.json 泄漏问题、答案或证据")
    if list(output_dir.glob("*validation*")):
        raise DatasetPreparationError("输出目录中出现了禁止的验证集文件")


def render_report(
    manifest: dict[str, Any],
    raw_audit: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
    selected: dict[str, Any],
    filtered: Sequence[dict[str, Any]],
    train: Sequence[dict[str, Any]],
    test: Sequence[dict[str, Any]],
) -> str:
    headers = (
        "corpus_name",
        "标题",
        "体裁",
        "字符",
        "近似词数",
        "全部题",
        "Fact",
        "Complex",
        "两类合计",
        "Summarize",
        "Creative",
        "连续叙事",
        "合格",
        "排除原因",
    )
    lines = [
        "# 阶段 1：GraphRAG-Bench Novel 数据选择报告",
        "",
        "本报告只覆盖官方数据下载、检查、选书、筛选和 8:2 分层划分。",
        "未进行文本切块、知识图谱构建、检索、LLM 调用或分类器训练。",
        "",
        "## 官方来源与版本",
        "",
        f"- 官方仓库：{manifest['official_url']}",
        f"- 实际 remote：`{manifest['source_remote']}`",
        f"- Git commit：`{manifest['revision']}`",
        f"- 下载日期：{manifest['download_date']}（Asia/Shanghai）",
    ]
    for relative, digest in manifest["raw_file_hashes"].items():
        size = manifest["raw_file_sizes_bytes"][relative]
        lines.append(f"- `{relative}`：{size:,} bytes，SHA-256 `{digest}`")
    lines.extend(
        [
            "",
            "## 原始结构与完整性",
            "",
            f"- 语料文件：JSON 数组，共 {raw_audit['corpus_count']} 项；字段 "
            f"`{raw_audit['corpus_fields']}`。",
            f"- 问题文件：JSON 数组，共 {raw_audit['question_count']} 行；字段 "
            f"`{raw_audit['question_fields']}`。",
            f"- 实际 question_type：`{raw_audit['question_type_counts']}`。",
            "- corpus_name 全部唯一；无空文本、空问题、空答案或无法匹配的 source。",
            "- 未在标题或全文中发现《红楼梦》及列出的中英文变体。",
        ]
    )
    if raw_audit["duplicate_question_ids"]:
        duplicate_summary = "; ".join(
            f"`{key}` 出现在 {[row['source'] for row in rows]}"
            for key, rows in raw_audit["duplicate_question_ids"].items()
        )
        lines.extend(
            [
                f"- **官方原始异常：**全量问题存在重复 ID：{duplicate_summary}。",
                "  原始文件未被修改；该重复项不属于最终选中的作品。选中作品、训练集和",
                "  测试集均已重新断言 ID 唯一。",
            ]
        )

    lines.extend(
        [
            "",
            "## 全部 20 个候选语料",
            "",
            "每项至少检查文本开头 2,000 字符；表中的作品类型不是因位于 Novel "
            "目录而自动判定。",
            "",
            "| " + " | ".join(headers) + " |",
            "|" + "|".join(["---"] * len(headers)) + "|",
        ]
    )
    for row in candidates:
        reason = "；".join(row["exclusion_reasons"]) or "—"
        values = (
            row["corpus_name"],
            row["title"],
            row["genre"],
            f"{row['character_count']:,}",
            f"{row['approximate_word_count']:,}",
            row["question_count"],
            row["fact_retrieval_count"],
            row["complex_reasoning_count"],
            row["usable_question_count"],
            row["contextual_summarize_count"],
            row["creative_generation_count"],
            "是" if row["single_coherent_narrative"] else "否",
            "是" if row["meets_candidate_standard"] else "否",
            reason,
        )
        lines.append("| " + " | ".join(str(value).replace("|", "／") for value in values) + " |")

    filtered_counts = split_counts(filtered)
    train_counts = split_counts(train)
    test_counts = split_counts(test)
    fact_example = next(row for row in filtered if row["question_type"] == "Fact Retrieval")
    complex_example = next(
        row for row in filtered if row["question_type"] == "Complex Reasoning"
    )
    lines.extend(
        [
            "",
            "## 选择结果",
            "",
            f"- 选中：`{selected['corpus_name']}`，*{selected['title']}*。",
            f"- 体裁：{selected['genre']}。",
            f"- 长度：{selected['character_count']:,} 字符，约 "
            f"{selected['approximate_word_count']:,} 个空白分词。",
            f"- 原始问题 {selected['question_count']} 道；筛选后 "
            f"{selected['usable_question_count']} 道（Fact "
            f"{selected['fact_retrieval_count']}，Complex "
            f"{selected['complex_reasoning_count']}）。",
            "- 选择理由：它是单一且情节连续的三幕戏剧，满足两类题量硬门槛，"
            "不属于禁用作品，并且是所有合格候选中近似词数最少的作品。",
            f"- 完整确定性规则：{manifest['selection_rule']}",
            "",
            "## 分层划分",
            "",
            "| 集合 | Fact Retrieval | Complex Reasoning | 合计 |",
            "|---|---:|---:|---:|",
            f"| 筛选后 | {filtered_counts['Fact Retrieval']} | "
            f"{filtered_counts['Complex Reasoning']} | {len(filtered)} |",
            f"| 训练集 | {train_counts['Fact Retrieval']} | "
            f"{train_counts['Complex Reasoning']} | {len(train)} |",
            f"| 测试集 | {test_counts['Fact Retrieval']} | "
            f"{test_counts['Complex Reasoning']} | {len(test)} |",
            "",
            "划分前按 ID 排序，再在每种 question_type 内以 seed=42 独立打乱；"
            "训练/测试 ID 无交集且并集等于全部筛选问题。未创建验证集。",
            "",
            "## 数据泄漏防护",
            "",
            "- 原始官方 JSON 保持只读，不进行修正或去重。",
            "- `test_queries.jsonl` 只含 id、source、question、question_type。",
            "- 完整测试标签只保存在 `test_questions_labeled.jsonl`，供未来最终评估读取。",
            "- `indexing_input.json` 只含 corpus_name、title、context。",
            "- 后续索引或建图不得读取 questions、answer、evidence 或 evidence_triple。",
            "",
            "## 数据检查示例",
            "",
            "> 以下答案和证据仅用于数据检查，不能进入建图输入。",
            "",
            "### Fact Retrieval",
            "",
            f"- ID：`{fact_example['id']}`",
            f"- 问题：{fact_example['question']}",
            f"- 答案：{fact_example['answer']}",
            f"- 证据：{fact_example['evidence']}",
            "",
            "### Complex Reasoning",
            "",
            f"- ID：`{complex_example['id']}`",
            f"- 问题：{complex_example['question']}",
            f"- 答案：{complex_example['answer']}",
            f"- 证据：{complex_example['evidence']}",
            "",
            "## 后续边界",
            "",
            "**下一步只能从文本清洗与切块开始，且只能读取 "
            "`data/processed/phase1/indexing_input.json`。当前阶段到此停止。**",
            "",
        ]
    )
    return "\n".join(lines)


def prepare_dataset(
    dataset_root: Path,
    output_dir: Path,
    report_path: Path,
    seed: int = 42,
    test_size: float = 0.2,
    *,
    source_remote: str | None = None,
    revision: str | None = None,
    download_date: str | None = None,
) -> dict[str, Any]:
    if seed != 42:
        raise DatasetPreparationError("本实验随机种子固定为 42")
    if abs(test_size - 0.2) > 1e-12:
        raise DatasetPreparationError("本实验测试集比例固定为 0.2")
    corpus_path = dataset_root / CORPUS_RELATIVE_PATH
    questions_path = dataset_root / QUESTIONS_RELATIVE_PATH
    corpora = load_json_array(corpus_path, "语料")
    questions = load_json_array(questions_path, "问题")
    raw_audit = inspect_raw_data(corpora, questions)
    candidates = build_candidate_statistics(corpora, questions)
    selected = select_candidate(candidates)

    selected_corpus = next(
        row for row in corpora if row["corpus_name"] == selected["corpus_name"]
    )
    filtered = sorted(
        (
            dict(row)
            for row in questions
            if row["source"] == selected["corpus_name"]
            and row["question_type"] in ALLOWED_TYPES
        ),
        key=lambda row: row["id"],
    )
    train, test = stratified_split(filtered, seed, test_size)
    if len({row["id"] for row in filtered}) != len(filtered):
        raise DatasetPreparationError("选中作品内存在重复 ID，停止划分")

    if source_remote is None or revision is None:
        source_remote, revision = detect_git_source(dataset_root)
    existing_manifest = output_dir / "dataset_manifest.json"
    if download_date is None and existing_manifest.exists():
        try:
            download_date = json.loads(existing_manifest.read_text(encoding="utf-8"))[
                "download_date"
            ]
        except (KeyError, json.JSONDecodeError, TypeError):
            download_date = None
    download_date = download_date or dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).date().isoformat()
    raw_paths = (CORPUS_RELATIVE_PATH, QUESTIONS_RELATIVE_PATH)
    manifest = {
        "dataset_name": "GraphRAG-Bench",
        "subset": "Novel",
        "official_url": OFFICIAL_URL,
        "source_remote": source_remote,
        "revision": revision,
        "download_date": download_date,
        "raw_file_hashes": {
            str(path): sha256_file(dataset_root / path) for path in raw_paths
        },
        "raw_file_sizes_bytes": {
            str(path): (dataset_root / path).stat().st_size for path in raw_paths
        },
        "selected_corpus": selected["corpus_name"],
        "selected_title": selected["title"],
        "selected_genre": selected["genre"],
        "selection_rule": SELECTION_RULE,
        "included_question_types": list(ALLOWED_TYPES),
        "excluded_question_types": list(EXCLUDED_TYPES),
        "random_seed": seed,
        "test_size": test_size,
        "validation_set": False,
        "raw_question_id_unique": raw_audit["question_id_unique"],
        "selected_question_ids_unique": True,
    }
    selection_statistics = {
        "raw_data_audit": raw_audit,
        "selection_rule": SELECTION_RULE,
        "candidates": candidates,
        "selected": selected,
        "filtered_question_counts": split_counts(filtered),
        "train_question_counts": split_counts(train),
        "test_question_counts": split_counts(test),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    stale_validation = list(output_dir.glob("*validation*"))
    if stale_validation:
        raise DatasetPreparationError(
            f"输出目录已有禁止的验证集文件，请人工处理：{stale_validation}"
        )
    corpus_output = {
        "corpus_name": selected["corpus_name"],
        "title": selected["title"],
        "genre": selected["genre"],
        "context": selected_corpus["context"],
    }
    write_json(output_dir / "corpus.json", corpus_output)
    write_jsonl(output_dir / "questions_filtered.jsonl", filtered)
    write_jsonl(output_dir / "train_questions.jsonl", train)
    write_jsonl(output_dir / "test_questions_labeled.jsonl", test)
    write_jsonl(
        output_dir / "test_queries.jsonl",
        (
            {
                field: row[field]
                for field in ("id", "source", "question", "question_type")
            }
            for row in test
        ),
    )
    write_json(
        output_dir / "indexing_input.json",
        {
            "corpus_name": selected["corpus_name"],
            "title": selected["title"],
            "context": selected_corpus["context"],
        },
    )
    write_json(output_dir / "dataset_manifest.json", manifest)
    write_json(output_dir / "selection_statistics.json", selection_statistics)
    validate_prepared_output(
        output_dir, selected["corpus_name"], filtered, train, test, test_size
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_report(
            manifest, raw_audit, candidates, selected, filtered, train, test
        ),
        encoding="utf-8",
    )
    return {
        "selected": selected,
        "filtered_count": len(filtered),
        "train_count": len(train),
        "test_count": len(test),
        "output_dir": str(output_dir),
        "report_path": str(report_path),
        "raw_duplicate_ids": sorted(raw_audit["duplicate_question_ids"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/phase1_data_selection.md"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = prepare_dataset(
            args.dataset_root.resolve(),
            args.output_dir.resolve(),
            args.report_path.resolve(),
            args.seed,
            args.test_size,
        )
    except DatasetPreparationError as exc:
        print(f"数据准备失败：{exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
