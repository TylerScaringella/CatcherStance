from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from curator.features import process_video
from .config import StatusCallback
from .model import StanceClassifier
from .schemas import PitchDetection, PitchFeature
from .yolo import load_yolo_once


def read_manifest_rows(manifest_path: Path) -> list[dict]:
    if not manifest_path.exists():
        return []
    with open(manifest_path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def downloaded_video_rows(manifest_path: Path) -> list[dict]:
    rows = []
    for row in read_manifest_rows(manifest_path):
        path = row.get("saved_path", "")
        if row.get("status") == "downloaded" and path and Path(path).exists():
            rows.append(row)
    return rows


def detect_stances_for_manifest(
    manifest_path: Path,
    status_callback: StatusCallback | None = None,
) -> tuple[list[PitchDetection], list[PitchFeature]]:
    load_yolo_once()
    classifier = StanceClassifier()
    detections: list[PitchDetection] = []
    feature_rows: list[PitchFeature] = []
    video_rows = downloaded_video_rows(manifest_path)
    total = len(video_rows)

    if status_callback is not None:
        status_callback("Running catcher detection and stance classifier", 0, total)

    progress = tqdm(video_rows, desc="Catcher detection", unit="pitch", total=total)
    for idx, row in enumerate(progress, start=1):
        clip_id = row.get("clip_id") or Path(row.get("saved_path", "")).stem
        video_path = row.get("saved_path", "")
        try:
            features = process_video(video_path)
            if features is None:
                feature_rows.append(PitchFeature("", clip_id, "", "no_valid_catcher"))
                detections.append(PitchDetection(idx, clip_id, video_path, "", 0.0, "no_valid_catcher"))
                continue

            feature_rows.append(PitchFeature("", clip_id, features.tolist(), "ok"))
            stance, confidence = classifier.predict(features)
            detections.append(PitchDetection(idx, clip_id, video_path, stance, confidence, "ok"))
        except Exception as exc:
            feature_rows.append(PitchFeature("", clip_id, "", f"error:{type(exc).__name__}"))
            detections.append(
                PitchDetection(
                    idx,
                    clip_id,
                    video_path,
                    "",
                    0.0,
                    f"error:{type(exc).__name__}",
                    str(exc),
                )
            )

        if status_callback is not None:
            status_callback(f"Processed {idx} of {total} pitches", idx, total)

    return detections, feature_rows


def write_detection_outputs(
    run_dir: Path,
    detections: Iterable[PitchDetection],
    feature_rows: Iterable[PitchFeature],
) -> list[dict]:
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(item) for item in detections]
    features = [asdict(item) for item in feature_rows]

    json_path = run_dir / "detections.json"
    csv_path = run_dir / "detections.csv"
    feature_csv_path = run_dir / "pitch_features.csv"

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(PitchDetection.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(feature_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(PitchFeature.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(features)

    return rows
