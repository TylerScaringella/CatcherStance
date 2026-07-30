from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from stance_pipeline import PitchStanceConfig, analyze_pitch_clip
from stance_pipeline.assets import resolve_model_asset


SAMPLE_EXPECTATIONS = {
    "pitch-pix-69e939e0ee33ff11241e0939-356-369.mp4": {
        "label": "LKD",
        "impact_seconds": 10.76,
    },
    "pitch-pix-69e939e0ee33ff11241e0939-378-388.mp4": {
        "label": "LKD",
        "impact_seconds": 7.68,
    },
    "pitch-pix-69e939e0ee33ff11241e0939-395-409.mp4": {
        "label": "RKD",
        "impact_seconds": 10.76,
    },
    "pitch-pix-69e939e0ee33ff11241e0939-428-444.mp4": {
        "label": "LKD",
        "impact_seconds": 12.01,
    },
    "pitch-pix-69e939e0ee33ff11241e0939-449-462.mp4": {
        "label": "LKD",
        "impact_seconds": 10.01,
    },
}


def benchmark_clip(video_path: Path, config: PitchStanceConfig) -> dict:
    started = time.perf_counter()
    result = analyze_pitch_clip(video_path, config)
    elapsed = time.perf_counter() - started
    capture = cv2.VideoCapture(str(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    capture.release()
    impact_seconds = result.impact_frame / fps if result.impact_frame is not None and fps else None
    expected = SAMPLE_EXPECTATIONS.get(video_path.name, {})
    expected_impact = expected.get("impact_seconds")
    return {
        "video": video_path.name,
        "expected_label": expected.get("label"),
        "predicted_label": result.label,
        "label_match": result.label == expected.get("label"),
        "expected_impact_seconds": expected_impact,
        "impact_seconds": impact_seconds,
        "impact_error_seconds": (
            abs(impact_seconds - expected_impact)
            if impact_seconds is not None and expected_impact is not None
            else None
        ),
        "confidence": result.confidence,
        "rejection_reason": result.rejection_reason,
        "window_start_frame": result.window_start_frame,
        "window_end_frame": result.window_end_frame,
        "valid_frame_count": result.valid_frame_count,
        "vote_distribution": result.vote_distribution,
        "detector_provenance": result.detector_provenance,
        "quality_flags": result.quality_flags,
        "elapsed_seconds": elapsed,
        "diagnostics": result.diagnostics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the staged pitch stance analyzer.")
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("data/examples/duke-2026-04-21-liberty-sample/downloads"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/outputs/generated/staged_pipeline_benchmark.json"),
    )
    parser.add_argument("--phc-model", type=Path)
    parser.add_argument("--event-model", type=Path)
    parser.add_argument("--device")
    args = parser.parse_args()

    phc_model = args.phc_model or resolve_model_asset("baseballcv_phc")
    event_model = args.event_model or resolve_model_asset("baseballcv_glove")
    if phc_model is None or event_model is None:
        raise SystemExit(
            "BaseballCV assets are missing. Run: "
            "python src/download_models.py all"
        )

    config = PitchStanceConfig(
        phc_model_path=phc_model,
        event_model_path=event_model,
        device=args.device,
    )
    rows = [
        benchmark_clip(video_path, config)
        for video_path in sorted(args.video_dir.glob("*.mp4"))
    ]
    summary = {
        "clips": len(rows),
        "accepted": sum(row["rejection_reason"] is None for row in rows),
        "label_matches": sum(row["label_match"] for row in rows),
        "anchors_within_350ms": sum(
            row["impact_error_seconds"] is not None
            and row["impact_error_seconds"] <= 0.35
            for row in rows
        ),
        "total_elapsed_seconds": sum(row["elapsed_seconds"] for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "results": rows}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
