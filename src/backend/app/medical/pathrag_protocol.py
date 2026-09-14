"""Protocol compatibility for the pinned PathRAG keyword extraction callback."""
import json
import re

from app.domain.errors import AppError


# The pinned upstream uses this text for early failures even in context-only mode.
FAIL_RESPONSE = "Sorry, I'm not able to provide an answer to that question."


def keyword_prompt(prompt, examples):
    # Upstream interpolates examples into another template, leaving their escaped
    # braces doubled. Replace only known examples, never the user's query text.
    for example in examples:
        prompt = prompt.replace(example, example.replace("{{", "{").replace("}}", "}"), 1)
    return prompt


def combined_source_texts(context):
    """Read the pinned combiner's id-comma-tab format, which is not quoted CSV."""
    match = re.search(r"-----Sources-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL)
    if not match:
        return None
    lines = match.group(1).splitlines()
    if not lines or lines[0] != "id,\tcontent":
        return None
    records, current = [], None
    for line in lines[1:]:
        start = re.match(r"^\d+,\t(.*)$", line)
        if start:
            if current is not None:
                records.append("\n".join(current))
            current = [start.group(1)]
        elif current is not None:
            current.append(line)
        else:
            return None
    if current is not None:
        records.append("\n".join(current))
    return records


def normalize_keywords(content):
    text = content.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Recover only the observed extra outer brace pair, not arbitrary JSON.
        try:
            if not (text.startswith("{{") and text.endswith("}}")):
                raise ValueError
            data = json.loads(text[1:-1])
        except (ValueError, json.JSONDecodeError) as exc:
            raise AppError("PATHRAG_KEYWORDS_INVALID",
                           "PathRAG 关键词 JSON 解析失败，尚未开始图谱召回。请重试。") from exc
    keys = ("high_level_keywords", "low_level_keywords")
    if not isinstance(data, dict) or any(
        not isinstance(data.get(key), list) or any(not isinstance(v, str) for v in data[key])
        for key in keys
    ):
        raise AppError("PATHRAG_KEYWORDS_INVALID",
                       "PathRAG 关键词格式不正确，需要高层与低层关键词数组；尚未开始图谱召回。")
    if any(not data[key] or any(not v.strip() for v in data[key]) for key in keys):
        raise AppError("PATHRAG_KEYWORDS_EMPTY",
                       "PathRAG 未提取到完整的高层与低层关键词，尚未开始图谱召回。")
    return json.dumps(data, ensure_ascii=False)
