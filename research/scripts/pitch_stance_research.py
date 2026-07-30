from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Iterable

import cv2
import numpy as np

from catcher_detection import detect_catcher_from_res_item
from curator.features import load_yolo_once

try:
    import baseballcv  # type: ignore  # noqa: F401

    BASEBALLCV_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    BASEBALLCV_AVAILABLE = False


OPTION_MATRIX = [
    {
        "path": "Current detector + low-motion window",
        "role": "works now",
        "use_when": "baseline production path and notebook exploration",
        "pros": "already installed, strong catcher rejection, no extra dependency",
        "cons": "no explicit event anchor, still relies on clip-local motion heuristics",
    },
    {
        "path": "BaseballCV ball_tracking.pt / glove_tracking.pt",
        "role": "event anchoring",
        "use_when": "ball or glove timing can anchor release / contact",
        "pros": "gives a physical event to look backward from",
        "cons": "needs BaseballCV installed and benchmarked on broadcast footage",
    },
    {
        "path": "BaseballCV pitcher_hitter_catcher.pt",
        "role": "coarse candidate generation",
        "use_when": "you want an external person detector before pose gating",
        "pros": "simple pitcher/hitter/catcher triage",
        "cons": "does not solve set-stance timing by itself",
    },
    {
        "path": "RF-DETR rfdetr_glove_tracking",
        "role": "higher-accuracy broadcast detection",
        "use_when": "you can install RF-DETR and want a stronger glove/plate anchor",
        "pros": "promising accuracy-focused alternative",
        "cons": "extra dependency path and more integration work",
    },
]


def clip_metadata(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    duration = frames / fps if fps else 0.0
    return {
        "video": video_path.name,
        "path": str(video_path),
        "fps": round(float(fps), 3),
        "frames": frames,
        "duration_s": round(float(duration), 3),
    }


def load_frame_rows(video_path: Path, vid_stride: int = 2) -> list[dict]:
    model = load_yolo_once()
    results = model(
        str(video_path),
        show=False,
        stream=True,
        save=False,
        verbose=False,
        imgsz=512,
        vid_stride=vid_stride,
    )

    rows: list[dict] = []
    for frame_idx, result in enumerate(results):
        detection = detect_catcher_from_res_item(result)
        if detection is None:
            rows.append(
                {
                    "frame_idx": frame_idx,
                    "kept": False,
                    "frame_gate_reason": "abstain",
                    "rejected_reasons": ["abstain"],
                }
            )
            continue

        box = np.asarray(detection["box"], dtype=float)
        center_x = float((box[0] + box[2]) / 2.0)
        center_y = float((box[1] + box[3]) / 2.0)
        rows.append(
            {
                "frame_idx": frame_idx,
                "kept": True,
                "frame_gate_reason": detection["frame_gate"]["reason"],
                "score": float(detection["score"]),
                "margin": float(detection["margin"]),
                "confidence": float(detection["confidence"]),
                "anchor_dist_norm": float(detection["anchor_dist_norm"]),
                "search_overlap": float(detection["search_overlap"]),
                "box_x1": float(box[0]),
                "box_y1": float(box[1]),
                "box_x2": float(box[2]),
                "box_y2": float(box[3]),
                "box_w": float(box[2] - box[0]),
                "box_h": float(box[3] - box[1]),
                "center_x": center_x,
                "center_y": center_y,
                "rejected_reasons": [item["reason"] for item in detection["rejected_candidates"]],
            }
        )

    return rows


def _window_motion(rows: list[dict]) -> dict:
    centers = np.asarray([[r["center_x"], r["center_y"]] for r in rows], dtype=float)
    widths = np.asarray([r["box_w"] for r in rows], dtype=float)
    heights = np.asarray([r["box_h"] for r in rows], dtype=float)
    scores = np.asarray([r.get("score", 0.0) for r in rows], dtype=float)
    anchors = np.asarray([r.get("anchor_dist_norm", 0.0) for r in rows], dtype=float)

    center_deltas = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    area = widths * heights
    area_norm = np.std(area) / max(float(np.mean(area)), 1.0)

    return {
        "mean_center_delta": float(np.mean(center_deltas)) if len(center_deltas) else 0.0,
        "std_center_delta": float(np.std(center_deltas)) if len(center_deltas) else 0.0,
        "area_norm": float(area_norm),
        "mean_score": float(np.mean(scores)) if len(scores) else 0.0,
        "mean_anchor_dist_norm": float(np.mean(anchors)) if len(anchors) else 0.0,
        "motion_score": float(
            (np.mean(center_deltas) if len(center_deltas) else 0.0)
            + (np.std(center_deltas) if len(center_deltas) else 0.0)
            + area_norm
            - 0.2 * (np.mean(scores) if len(scores) else 0.0)
        ),
    }


def best_low_motion_window(rows: list[dict], window_frames: int = 45) -> dict | None:
    valid = [row for row in rows if row.get("kept")]
    if len(valid) < window_frames:
        return None

    best: dict | None = None
    for start in range(0, len(valid) - window_frames + 1):
        chunk = valid[start : start + window_frames]
        motion = _window_motion(chunk)
        candidate = {
            "start_frame": int(chunk[0]["frame_idx"]),
            "end_frame": int(chunk[-1]["frame_idx"]),
            "window_frames": window_frames,
            **motion,
        }
        if best is None or candidate["motion_score"] < best["motion_score"]:
            best = candidate
    return best


def summarize_clip(video_path: Path, vid_stride: int = 2, window_frames: int = 45) -> dict:
    meta = clip_metadata(video_path)
    rows = load_frame_rows(video_path, vid_stride=vid_stride)
    valid = [row for row in rows if row.get("kept")]
    gate_counts = Counter(row.get("frame_gate_reason", "") for row in rows)
    reject_counts = Counter(
        reason
        for row in rows
        for reason in row.get("rejected_reasons", [])
        if reason and reason != "abstain"
    )
    abstain_count = sum(1 for row in rows if not row.get("kept"))

    first_valid = valid[0]["frame_idx"] if valid else None
    last_valid = valid[-1]["frame_idx"] if valid else None
    gaps = [b["frame_idx"] - a["frame_idx"] - 1 for a, b in zip(valid, valid[1:])]
    best = best_low_motion_window(rows, window_frames=window_frames)

    return {
        **meta,
        "vid_stride": vid_stride,
        "sampled_frames": len(rows),
        "valid_detections": len(valid),
        "abstentions": abstain_count,
        "first_valid_frame": first_valid,
        "last_valid_frame": last_valid,
        "max_gap_in_valid_frames": max(gaps) if gaps else 0,
        "median_valid_motion": round(
            median(
                [
                    float(np.linalg.norm(
                        np.asarray([b["center_x"], b["center_y"]], dtype=float)
                        - np.asarray([a["center_x"], a["center_y"]], dtype=float)
                    ))
                    for a, b in zip(valid, valid[1:])
                ]
            ),
            3,
        ) if len(valid) > 1 else None,
        "frame_gate_reasons": dict(gate_counts),
        "candidate_rejection_reasons": dict(reject_counts),
        "best_window": best,
    }


def analyze_directory(video_dir: Path, output_path: Path | None = None) -> list[dict]:
    summaries = [summarize_clip(video_path) for video_path in sorted(video_dir.glob("*.mp4"))]
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return summaries


def majority_vote(labels: Iterable[str | None]) -> str | None:
    counts = Counter(label for label in labels if label)
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def rolling_majority_vote(labels: list[str | None], window: int = 5) -> list[str | None]:
    if window <= 1:
        return labels[:]

    half = window // 2
    voted: list[str | None] = []
    for idx in range(len(labels)):
        start = max(0, idx - half)
        end = min(len(labels), idx + half + 1)
        voted.append(majority_vote(labels[start:end]))
    return voted


def event_anchor_window(anchor_frame_idx: int, fps: float, pre_seconds: float = 1.5, post_seconds: float = 0.0) -> dict:
    frames_per_second = max(fps, 1.0)
    pre_frames = int(round(pre_seconds * frames_per_second))
    post_frames = int(round(post_seconds * frames_per_second))
    return {
        "start_frame": max(0, anchor_frame_idx - pre_frames),
        "end_frame": max(anchor_frame_idx, anchor_frame_idx + post_frames),
        "pre_seconds": pre_seconds,
        "post_seconds": post_seconds,
    }


def option_matrix() -> list[dict]:
    return OPTION_MATRIX[:]


def main() -> int:
    parser = argparse.ArgumentParser(description="Explore pitch stance windowing and catcher detection on sample clips.")
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("data/examples/duke-2026-04-21-liberty-sample/downloads"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/outputs/sample_pitch_stance_summary.json"),
    )
    parser.add_argument("--vid-stride", type=int, default=2)
    parser.add_argument("--window-frames", type=int, default=45)
    args = parser.parse_args()

    summaries = [summarize_clip(video_path, vid_stride=args.vid_stride, window_frames=args.window_frames) for video_path in sorted(args.video_dir.glob("*.mp4"))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Wrote {len(summaries)} summaries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
