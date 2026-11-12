"""Reproducible, group-aware audit for catcher stance classifiers.

This script evaluates the checked-in model and inexpensive local alternatives without
overwriting production artifacts. Group-aware splits expose leakage hidden by row-wise splits.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import platform
import re
import time
import warnings
from pathlib import Path

import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from project_paths import CLASSIFIER_METADATA_PATH, PROCESSED_DATASET_DIR, RAW_LABELED_VIDEOS_DIR
from stance_pipeline.model import StanceClassifier

GROUP_PATTERN = re.compile(r"pitch-pix-([^-]+)-\d+-\d+\.mp4$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_for_video(video: str, row_id: str) -> str:
    match = GROUP_PATTERN.search(video)
    return match.group(1) if match else Path(video).name.split("_")[0] or f"row-{row_id}"


def load_rows(dataset_path: Path, labels_path: Path):
    labels = {str(row["id"]): row for row in csv.DictReader(labels_path.open(newline="", encoding="utf-8"))}
    features, targets, groups = [], [], []
    statuses: dict[str, int] = {}
    with dataset_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            status = row.get("status", "")
            statuses[status] = statuses.get(status, 0) + 1
            if status != "ok" or not row.get("features", "").strip():
                continue
            try:
                vector = ast.literal_eval(row["features"])
            except (ValueError, SyntaxError):
                continue
            if len(vector) != 238:
                continue
            label = labels.get(str(row.get("id", "")), {})
            features.append(vector)
            targets.append(str(row.get("choice") or label.get("choice") or ""))
            groups.append(group_for_video(str(label.get("video", "")), str(row.get("id", ""))))
    if not features:
        raise ValueError(f"No usable feature rows found in {dataset_path}")
    return np.asarray(features, dtype=np.float32), np.asarray(targets), np.asarray(groups), statuses


def metrics(y_true, y_pred, labels: list[str]) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
            "classification_report": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        }


def build_group_splits(X, y, groups, *, n_splits=20, test_size=0.20, random_state=42):
    """Return deterministic group splits and never allow source-group leakage."""
    splitter = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    return list(splitter.split(X, y, groups))


def evaluate_model(name, model_factory, X, y, groups, labels, splits):
    split_results = []
    for split_number, (train, test) in enumerate(splits, start=1):
        started = time.perf_counter()
        model = model_factory()
        model.fit(X[train], y[train])
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        predictions = model.predict(X[test])
        predict_ms_per_row = (time.perf_counter() - started) * 1000 / len(test)
        split_metrics = metrics(y[test], predictions, labels)
        split_metrics.update({
            "split": split_number,
            "fit_seconds": fit_seconds,
            "predict_ms_per_row": predict_ms_per_row,
            "train_rows": len(train),
            "test_rows": len(test),
            "train_groups": len(set(groups[train])),
            "test_groups": len(set(groups[test])),
            "test_class_counts": {label: int(np.sum(y[test] == label)) for label in labels},
            "all_classes_in_test": set(labels).issubset(set(y[test])),
        })
        split_results.append(split_metrics)

    metric_names = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "fit_seconds", "predict_ms_per_row"]
    valid_splits = [item for item in split_results if item["all_classes_in_test"]]
    aggregate_source = valid_splits or split_results
    aggregate = {
        metric: {
            "mean": float(np.mean([item[metric] for item in aggregate_source])),
            "std": float(np.std([item[metric] for item in aggregate_source])),
        }
        for metric in metric_names
    }
    return {
        "name": name,
        "aggregate": aggregate,
        "aggregate_scope": "all_classes_in_test" if valid_splits else "all_splits_with_missing_classes",
        "valid_split_count": len(valid_splits),
        "split_count": len(split_results),
        "splits": split_results,
    }


def evaluate_artifact(X, y, labels):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        classifier = StanceClassifier()
    started = time.perf_counter()
    predictions = [classifier.predict(row)[0] for row in X]
    result = metrics(y, np.asarray(predictions), labels)
    result.update({"name": "checked_in_mlp_full_dataset", "predict_ms_per_row": (time.perf_counter() - started) * 1000 / len(X)})
    return result


def build_models():
    return {
        "dummy_majority": lambda: DummyClassifier(strategy="most_frequent"),
        "logistic_regression_balanced": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
        "linear_svm_balanced": lambda: make_pipeline(StandardScaler(), LinearSVC(class_weight="balanced", C=0.1, max_iter=5000)),
        "extra_trees_balanced": lambda: ExtraTreesClassifier(n_estimators=200, class_weight="balanced", max_features="sqrt", random_state=42, n_jobs=1),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=15, random_state=42),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROCESSED_DATASET_DIR / "keypoints-labeled.csv")
    parser.add_argument("--labels", type=Path, default=RAW_LABELED_VIDEOS_DIR / "labeled-videos-2026-04-22-00-05-e3d836a2.csv")
    parser.add_argument("--output", type=Path, default=Path("research/outputs/model_audit_summary.json"))
    args = parser.parse_args()
    X, y, groups, statuses = load_rows(args.dataset, args.labels)
    labels = sorted(set(y))
    splits = build_group_splits(X, y, groups)
    results = [evaluate_artifact(X, y, labels)]
    for name, model_factory in build_models().items():
        results.append(evaluate_model(name, model_factory, X, y, groups, labels, splits))
    summary = {
        "audit_version": 2,
        "dataset": {"path": str(args.dataset), "sha256": sha256_file(args.dataset), "usable_rows": int(len(X)), "feature_count": int(X.shape[1]), "class_counts": {label: int(np.sum(y == label)) for label in labels}, "source_groups": int(len(set(groups))), "row_statuses": statuses},
        "split": {"method": "GroupShuffleSplit", "n_splits": len(splits), "test_size": 0.20, "random_state": 42, "group_key": "source video prefix", "group_overlap_checked": True},
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__, "classifier_metadata": json.loads(CLASSIFIER_METADATA_PATH.read_text(encoding="utf-8"))},
        "results": results,
        "limitations": ["Squat has insufficient source groups for reliable grouped evaluation; metrics are reported with explicit class-coverage scope.", "Event timing requires a separate frame-level annotation set.", "This audit does not overwrite production model artifacts."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "usable_rows": len(X), "models": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
