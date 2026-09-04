from __future__ import annotations

import re
from collections import Counter

import numpy as np


STOPWORDS = set(
    "the a an and or of to in on for with is are was were be been being what which why "
    "how when where who whom whose does do did has have had from at by as that this these "
    "those it its patient patients".split()
)


def content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in STOPWORDS
    }


def evidence_scores(evidence: str, contexts: list[str], hit_threshold: float) -> dict:
    context_terms = content_tokens(" ".join(contexts))
    statements = [part.strip() for part in evidence.split(";") if part.strip()]
    recalls: list[float] = []
    for statement in statements:
        terms = content_tokens(statement)
        recalls.append(len(terms & context_terms) / max(1, len(terms)))
    return {
        "evidence_coverage": float(np.mean(recalls)) if recalls else 0.0,
        "evidence_hit_rate": float(np.mean([value >= hit_threshold for value in recalls]))
        if recalls
        else 0.0,
        "evidence_count": len(statements),
    }


def rouge_l_f1(reference: str, prediction: str) -> float:
    ref = re.findall(r"[a-z0-9]+", reference.lower())
    pred = re.findall(r"[a-z0-9]+", prediction.lower())
    if not ref or not pred:
        return 0.0
    previous = [0] * (len(pred) + 1)
    for ref_token in ref:
        current = [0]
        for index, pred_token in enumerate(pred, start=1):
            if ref_token == pred_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    lcs = previous[-1]
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def choose_silver_label(method_rows: dict[str, dict], tolerance: float) -> str:
    ordered = ("vector", "light_proxy", "path_proxy")
    best_quality = max(method_rows[method]["evidence_coverage"] for method in ordered)
    eligible = {
        method
        for method in ordered
        if method_rows[method]["evidence_coverage"] >= best_quality - tolerance
    }
    return next(method for method in ordered if method in eligible)


def label_counts(labels: list[str]) -> dict[str, int]:
    return dict(Counter(labels))

