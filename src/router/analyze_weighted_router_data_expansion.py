from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.router.analyze_router_data_expansion import build_labels, resolve, route_policy, validate_and_merge
from src.router.run_weighted_router_experiment import (
    classify_weight,
    estimator,
    fit_estimator,
    metrics,
    sha256,
    tune,
    write_predictions,
)
from src.backend.common.base import METHODS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the prior fixed sample-weighting scheme to expanded router data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/router_weighted_data_expansion.json"),
    )
    return parser.parse_args()


def run_weighted(
    name: str,
    development_ids: list[str],
    test_ids: list[str],
    labels: dict[str, dict[str, Any]],
    rows_by_id: dict[str, dict[str, dict[str, Any]]],
    embeddings: dict[str, np.ndarray],
    diagnostics: dict[str, dict[str, Any]],
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    x = np.stack([embeddings[qid] for qid in development_ids])
    y = np.asarray([labels[qid]["silver_label"] for qid in development_ids])
    weights = np.asarray([diagnostics[qid]["weight"] for qid in development_ids])
    strata = np.asarray([labels[qid]["question_type"] for qid in development_ids])
    folds = list(
        StratifiedKFold(
            n_splits=int(config["folds"]),
            shuffle=True,
            random_state=int(config["seed"]),
        ).split(x, strata)
    )
    candidates, best, oof_predictions, oof_probabilities = tune(
        config, x, y, weights, folds, weighted=True
    )
    model = estimator(config, best["C"], best["class_weight"])
    fit_estimator(model, x, y, weights)

    test_x = np.stack([embeddings[qid] for qid in test_ids])
    test_y = [labels[qid]["silver_label"] for qid in test_ids]
    test_predictions = model.predict(test_x).tolist()
    raw_probabilities = model.predict_proba(test_x)
    class_to_column = {label: index for index, label in enumerate(model.classes_)}
    test_probabilities = np.stack(
        [raw_probabilities[:, class_to_column[method]] for method in METHODS], axis=1
    )

    target_dir = output_dir / name
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "tuning.json").write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_predictions(
        target_dir / "oof_predictions.jsonl",
        development_ids,
        [labels[qid]["question"] for qid in development_ids],
        y.tolist(),
        oof_predictions,
        oof_probabilities,
        diagnostics,
    )
    write_predictions(
        target_dir / "test_predictions.jsonl",
        test_ids,
        [labels[qid]["question"] for qid in test_ids],
        test_y,
        test_predictions,
        test_probabilities,
        diagnostics,
    )
    joblib.dump(model, target_dir / "router.joblib")
    return {
        "development_count": len(development_ids),
        "test_count": len(test_ids),
        "development_label_distribution": dict(Counter(y.tolist())),
        "development_weight_category_distribution": dict(
            Counter(diagnostics[qid]["category"] for qid in development_ids)
        ),
        "selected_hyperparameters": {
            "C": best["C"],
            "class_weight": best["class_weight"],
        },
        "cross_validation": best["cross_validation"],
        "sample_weighted_cross_validation": best[
            "sample_weighted_cross_validation"
        ],
        "test": metrics(test_y, test_predictions),
        "test_policy": route_policy(test_ids, test_predictions, rows_by_id),
    }


def metric_delta(after: dict[str, Any], before: dict[str, Any]) -> dict[str, float]:
    return {
        "oof_accuracy": after["cross_validation"]["accuracy"]
        - before["cross_validation"]["accuracy"],
        "oof_macro_f1": after["cross_validation"]["macro_f1"]
        - before["cross_validation"]["macro_f1"],
        "test_accuracy": after["test"]["accuracy"] - before["test"]["accuracy"],
        "test_macro_f1": after["test"]["macro_f1"] - before["test"]["macro_f1"],
        "test_answer_correctness": after["test_policy"]["answer_correctness"]
        - before["test_policy"]["answer_correctness"],
        "test_evidence_recall": after["test_policy"]["evidence_recall"]
        - before["test_policy"]["evidence_recall"],
    }


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    config_path = resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expansion_config_path = resolve(config["expansion_config"])
    expansion_config = json.loads(expansion_config_path.read_text(encoding="utf-8"))
    rows_by_id, baseline_ids, expansion_ids, protocol, input_hashes = validate_and_merge(
        expansion_config
    )
    labels = build_labels(
        rows_by_id,
        float(config["correctness_threshold"]),
        float(config["best_margin"]),
    )
    diagnostics = {
        qid: classify_weight(
            method_rows,
            float(config["correctness_threshold"]),
            float(config["best_margin"]),
            config["weights"],
        )
        for qid, method_rows in rows_by_id.items()
    }

    embedding_path = resolve(config["embedding_cache"])
    embedding_data = np.load(embedding_path)
    embeddings = {
        str(qid): vector
        for qid, vector in zip(
            embedding_data["question_ids"].tolist(), embedding_data["embeddings"]
        )
    }
    if not set(rows_by_id).issubset(embeddings):
        raise ValueError("The existing embedding cache is incomplete")

    test_ids = sorted(qid for qid in baseline_ids if labels[qid]["split"] == "test")
    baseline_development = sorted(baseline_ids - set(test_ids))
    expanded_development = sorted(set(baseline_development) | expansion_ids)
    if (len(baseline_development), len(expanded_development), len(test_ids)) != (96, 276, 24):
        raise AssertionError("Expected 96 -> 276 development rows and a frozen 24-row test")

    output_dir = resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = run_weighted(
        "baseline_weighted",
        baseline_development,
        test_ids,
        labels,
        rows_by_id,
        embeddings,
        diagnostics,
        config,
        output_dir,
    )
    expanded = run_weighted(
        "expanded_weighted",
        expanded_development,
        test_ids,
        labels,
        rows_by_id,
        embeddings,
        diagnostics,
        config,
        output_dir,
    )

    prior = json.loads(resolve(config["prior_weighted_summary"]).read_text(encoding="utf-8"))[
        "experiments"
    ]["weighted"]
    for section in ("cross_validation", "test"):
        for metric in ("accuracy", "macro_f1"):
            if not np.isclose(
                baseline[section][metric], prior[section][metric], atol=1e-12
            ):
                raise ValueError(
                    f"Baseline weighted reproduction failed: {section}.{metric}"
                )
    for metric in ("answer_correctness", "evidence_recall"):
        if not np.isclose(
            baseline["test_policy"][metric], prior["adaptive_policy"][metric], atol=1e-12
        ):
            raise ValueError(f"Baseline policy reproduction failed: {metric}")

    unweighted_expanded = json.loads(
        resolve(config["expanded_unweighted_summary"]).read_text(encoding="utf-8")
    )["experiments"]["expanded_bge_m3"]
    summary = {
        "status": "completed entirely offline from frozen generations, evaluations, and embeddings",
        "protocol": {
            "feature": "BGE-M3 CLS embedding",
            "correctness_threshold": config["correctness_threshold"],
            "best_margin": config["best_margin"],
            "weights": config["weights"],
            "folds": config["folds"],
            "fold_stratification": "question_type",
            "seed": config["seed"],
            "scikit_learn_version": sklearn.__version__,
            "evaluation_protocol": protocol,
            "frozen_test_count": len(test_ids),
            "new_api_calls": 0,
            "input_sha256": input_hashes
            | {
                "embedding_cache": sha256(embedding_path),
                "expansion_config": sha256(expansion_config_path),
            },
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        },
        "experiments": {
            "baseline_weighted": baseline,
            "expanded_weighted": expanded,
        },
        "delta_expanded_minus_baseline_weighted": metric_delta(expanded, baseline),
        "delta_expanded_weighted_minus_expanded_unweighted": metric_delta(
            expanded, unweighted_expanded
        ),
        "baseline_reproduces_prior_weighted_experiment": True,
        "elapsed_seconds": time.monotonic() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "sample_weights.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "question_id": qid,
                    "split": labels[qid]["split"],
                    "silver_label": labels[qid]["silver_label"],
                    **diagnostics[qid],
                },
                ensure_ascii=False,
            )
            + "\n"
            for qid in sorted(diagnostics)
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
