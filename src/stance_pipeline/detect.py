from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from backend.storage import atomic_write_text, run_lock
from .analyzer import PitchStanceConfig, analyze_pitch_clip
from .config import StatusCallback
from .schemas import PitchDetection, PitchFeature


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
    detections: list[PitchDetection] = []
    feature_rows: list[PitchFeature] = []
    video_rows = downloaded_video_rows(manifest_path)
    total = len(video_rows)
    pipeline_config = PitchStanceConfig()

    if status_callback is not None:
        status_callback("Running catcher detection and stance classifier", 0, total)

    progress = tqdm(video_rows, desc="Catcher detection", unit="pitch", total=total)
    for idx, row in enumerate(progress, start=1):
        clip_id = row.get("clip_id") or Path(row.get("saved_path", "")).stem
        video_path = row.get("saved_path", "")
        try:
            result = analyze_pitch_clip(video_path, config=pipeline_config)
            status = "ok" if result.accepted else result.rejection_reason or "rejected"
            fps = float((result.diagnostics or {}).get("fps") or 0)
            impact_seconds = (
                result.impact_frame / fps
                if result.impact_frame is not None and fps
                else ""
            )
            window_start_seconds = (
                result.window_start_frame / fps
                if result.window_start_frame is not None and fps
                else ""
            )
            window_end_seconds = (
                result.window_end_frame / fps
                if result.window_end_frame is not None and fps
                else ""
            )
            features = (
                result.feature_vector.tolist()
                if result.feature_vector is not None
                else ""
            )
            feature_rows.append(PitchFeature("", clip_id, features, status))
            detections.append(
                PitchDetection(
                    pitch_index=idx,
                    clip_id=clip_id,
                    video_path=video_path,
                    stance=result.label or "",
                    confidence=result.confidence,
                    status=status,
                    impact_frame=result.impact_frame if result.impact_frame is not None else "",
                    window_start_frame=(
                        result.window_start_frame
                        if result.window_start_frame is not None
                        else ""
                    ),
                    window_end_frame=(
                        result.window_end_frame
                        if result.window_end_frame is not None
                        else ""
                    ),
                    valid_frame_count=result.valid_frame_count,
                    vote_distribution=json.dumps(result.vote_distribution, sort_keys=True),
                    detector_provenance=",".join(result.detector_provenance),
                    quality_flags=",".join(result.quality_flags),
                    accepted=result.accepted,
                    rejection_reason=result.rejection_reason or "",
                    camera_quality=result.camera_quality,
                    fps=fps or "",
                    impact_seconds=impact_seconds,
                    window_start_seconds=window_start_seconds,
                    window_end_seconds=window_end_seconds,
                )
            )
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

    csv_buffer = io.StringIO(newline="")
    with csv_buffer as f:
        fieldnames = list(PitchDetection.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        csv_content = f.getvalue()

    feature_buffer = io.StringIO(newline="")
    with feature_buffer as f:
        fieldnames = list(PitchFeature.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(features)
        feature_content = f.getvalue()

    with run_lock(run_dir.name):
        atomic_write_text(json_path, json.dumps(rows, indent=2) + "\n")
        atomic_write_text(csv_path, csv_content)
        atomic_write_text(feature_csv_path, feature_content)

    return rows
