"""Audit current pitch-level abstentions and produce a review manifest.

This intentionally measures the staged analyzer separately from the historical
frame-level detector summary. It never changes model artifacts or production data.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from project_paths import RAW_LABELED_VIDEOS_DIR
from stance_pipeline import PitchStanceConfig, PitchStanceAnalyzer
from stance_pipeline.assets import resolve_model_asset

SOURCE_RE = re.compile(r"pitch-pix-([^-]+)-\d+-\d+\.mp4$", re.IGNORECASE)
LABELS = {"Knee-down Left": "LKD", "Knee-down Right": "RKD", "Squat": "Squat"}


def source_group(video: str) -> str:
    name = Path(video).name
    match = SOURCE_RE.search(name)
    return match.group(1) if match else name.split("_")[0]


def resolve_video(video: str, roots: list[Path]) -> Path | None:
    name = Path(video.split("?d=")[-1]).name
    for root in roots:
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def load_manifest(labels_path: Path, roots: list[Path]) -> list[dict]:
    rows = []
    matched_paths = set()
    with labels_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            path = resolve_video(row.get("video", ""), roots)
            if path is None:
                continue
            matched_paths.add(path.resolve())
            rows.append({
                "id": str(row.get("id", "")),
                "choice": row.get("choice", ""),
                "video": row.get("video", ""),
                "source_group": source_group(row.get("video", "")),
                "path": str(path),
            })
    # The local 803-clip cache may be newer than the exported annotation CSV.
    # Keep those clips in the audit with an empty expected label instead of
    # treating an unmatched label join as evidence that no clips exist.
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("*.mp4")):
            if path.resolve() in matched_paths:
                continue
            rows.append({
                "id": path.stem,
                "choice": "",
                "video": "",
                "source_group": source_group(path.name),
                "path": str(path),
            })
    return rows


def video_metadata(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    return {
        "fps": round(fps, 3),
        "frame_count": frames,
        "width": width,
        "height": height,
        "duration_seconds": round(frames / fps, 3) if fps else 0.0,
    }


def flatten_diagnostics(result) -> dict:
    diagnostics = result.diagnostics or {}
    event = diagnostics.get("event_anchor") or {}
    pose = diagnostics.get("pose_extractor") or {}
    return {
        "abstention_stage": diagnostics.get("abstention_stage"),
        "abstention_reason": diagnostics.get("abstention_reason") or result.rejection_reason,
        "event_available": event.get("available"),
        "event_type": event.get("event_type"),
        "event_confidence": event.get("confidence"),
        "event_uncertainty_frames": event.get("uncertainty_frames"),
        "pose_observations": diagnostics.get("pose_observations"),
        "classifier_sequences": diagnostics.get("classifier_sequences"),
        "vote_share": diagnostics.get("vote_share"),
        "vote_distribution": diagnostics.get("vote_distribution", {}),
        "window_coverage": diagnostics.get("window_coverage") or diagnostics.get("coverage"),
        "pose_sampled_frames": pose.get("sampled_frames"),
        "pose_misses": pose.get("pose_misses"),
        "identity_jumps": pose.get("identity_jumps"),
        "catcher_rejection_reasons": pose.get("catcher_rejection_reasons", {}),
    }


def audit(rows: list[dict], config: PitchStanceConfig) -> tuple[list[dict], dict]:
    analyzer = PitchStanceAnalyzer(config)
    results = []
    started = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        clip_started = time.perf_counter()
        metadata = video_metadata(Path(row["path"]))
        try:
            result = analyzer.analyze(row["path"])
            diagnostics = flatten_diagnostics(result)
            output = {
                **row,
                **metadata,
                "stance": result.label or "",
                "expected_stance": LABELS.get(row["choice"], ""),
                "accepted": result.accepted,
                "confidence": result.confidence,
                "rejection_reason": result.rejection_reason or "",
                "quality_flags": ",".join(result.quality_flags),
                "detector_provenance": ",".join(result.detector_provenance),
                "impact_frame": result.impact_frame,
                "window_start_frame": result.window_start_frame,
                "window_end_frame": result.window_end_frame,
                "runtime_seconds": round(time.perf_counter() - clip_started, 3),
                **diagnostics,
            }
        except Exception as exc:  # Keep the audit complete and classify failures.
            output = {
                **row,
                **metadata,
                "stance": "",
                "expected_stance": LABELS.get(row["choice"], ""),
                "accepted": False,
                "confidence": 0.0,
                "rejection_reason": f"error:{type(exc).__name__}",
                "abstention_stage": "exception",
                "abstention_reason": f"error:{type(exc).__name__}",
                "quality_flags": "",
                "detector_provenance": "",
                "runtime_seconds": round(time.perf_counter() - clip_started, 3),
                "error": str(exc),
            }
        results.append(output)
        if index % 10 == 0 or index == len(rows):
            print(f"audited {index}/{len(rows)} clips", flush=True)

    abstained = [row for row in results if not row["accepted"]]
    stage_counts = Counter(row.get("abstention_stage") or "none" for row in abstained)
    reason_counts = Counter(row.get("abstention_reason") or "unknown" for row in abstained)
    catcher_reasons = Counter()
    for row in results:
        catcher_reasons.update(row.get("catcher_rejection_reasons") or {})
    groups = defaultdict(lambda: {"total": 0, "abstained": 0})
    for row in results:
        groups[row["source_group"]]["total"] += 1
        groups[row["source_group"]]["abstained"] += int(not row["accepted"])
    expected = [row for row in results if row["expected_stance"]]
    labeled_accepted = [row for row in expected if row["accepted"]]
    summary = {
        "audit_version": 1,
        "pipeline": "current_staged_pitch_stance_analyzer",
        "clips": len(results),
        "accepted": sum(row["accepted"] for row in results),
        "abstained": len(abstained),
        "acceptance_rate": sum(row["accepted"] for row in results) / max(len(results), 1),
        "abstention_rate": len(abstained) / max(len(results), 1),
        "labeled_clips": len(expected),
        "labeled_accepted": len(labeled_accepted),
        "labeled_label_matches": sum(row["stance"] == row["expected_stance"] for row in labeled_accepted),
        "abstention_stages": dict(stage_counts),
        "abstention_reasons": dict(reason_counts),
        "catcher_candidate_rejection_reasons": dict(catcher_reasons),
        "event_types": dict(Counter(row.get("event_type") or "none" for row in results)),
        "source_groups": len(groups),
        "highest_abstention_groups": sorted(
            ({"source_group": key, **value} for key, value in groups.items()),
            key=lambda item: (item["abstained"], item["total"]),
            reverse=True,
        )[:25],
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "label_join": {
            "matched_local_labels": sum(bool(row["choice"]) for row in results),
            "unmatched_local_clips": sum(not bool(row["choice"]) for row in results),
            "label_source": "data/raw/labeled_videos/labeled-videos-2026-04-22-00-05-e3d836a2.csv",
        },
    }
    return results, summary


def write_review_manifest(results: list[dict], path: Path, limit: int) -> None:
    candidates = [row for row in results if not row["accepted"]]
    candidates.sort(key=lambda row: (row.get("abstention_reason", ""), row["source_group"], row["id"]))
    selected = candidates[:limit]
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "choice", "video", "path", "source_group", "abstention_stage", "abstention_reason", "review_status", "review_notes"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            writer.writerow({**{field: row.get(field, "") for field in fields}, "review_status": "pending"})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=RAW_LABELED_VIDEOS_DIR / "labeled-videos-2026-04-22-00-05-e3d836a2.csv")
    parser.add_argument("--video-root", type=Path, action="append", default=[Path("data/raw/downloads"), Path("data/examples/duke-2026-04-21-liberty-sample/downloads")])
    parser.add_argument("--limit", type=int, help="Audit only the first N manifest rows")
    parser.add_argument("--output", type=Path, default=Path("research/outputs/generated/abstention_audit_summary.json"))
    parser.add_argument("--report", type=Path, default=Path("research/outputs/generated/abstention_clip_report.jsonl"))
    parser.add_argument("--review-output", type=Path, default=Path("research/outputs/generated/abstention_review_manifest.csv"))
    parser.add_argument("--review-limit", type=int, default=60)
    parser.add_argument("--device")
    args = parser.parse_args()
    rows = load_manifest(args.labels, args.video_root)
    if args.limit:
        rows = rows[: args.limit]
    config = PitchStanceConfig(
        phc_model_path=resolve_model_asset("baseballcv_phc"),
        event_model_path=resolve_model_asset("baseballcv_glove"),
        device=args.device,
    )
    results, summary = audit(rows, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row) + "\n")
    write_review_manifest(results, args.review_output, args.review_limit)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
