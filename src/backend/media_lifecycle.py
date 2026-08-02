from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from downloader.files import validate_mp4
from downloader.manifest import load_manifest, write_manifest

from .storage import job_temp_dir, resolve_manifest_media, resolve_run, run_lock


def _encode_review(source: Path, destination: Path, start: float, duration: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        subprocess.run(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{duration:.3f}",
                "-map", "0:v:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "25", "-movflags", "+faststart", str(destination),
            ],
            check=True,
            capture_output=True,
            timeout=180,
        )
    else:
        _encode_review_opencv(source, destination, start, duration)
    validate_mp4(str(destination))


def _encode_review_opencv(source: Path, destination: Path, start: float, duration: float) -> None:
    import cv2

    capture = cv2.VideoCapture(str(source))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if fps <= 0 or width <= 0 or height <= 0:
            raise RuntimeError("source video metadata is invalid")
        capture.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        writer = None
        for codec in ("avc1", "mp4v"):
            candidate = cv2.VideoWriter(
                str(destination), cv2.VideoWriter_fourcc(*codec), fps, (width, height)
            )
            if candidate.isOpened():
                writer = candidate
                break
            candidate.release()
            destination.unlink(missing_ok=True)
        if writer is None:
            raise RuntimeError("OpenCV could not initialize the MP4 review encoder")
        try:
            frames_remaining = max(1, int(round(duration * fps)))
            written = 0
            while written < frames_remaining:
                ok, frame = capture.read()
                if not ok:
                    break
                writer.write(frame)
                written += 1
            if written == 0:
                raise RuntimeError("no review frames could be decoded")
        finally:
            writer.release()
    finally:
        capture.release()


def _review_bounds(result: dict) -> tuple[float, float]:
    start = max(0.0, float(result.get("window_start_seconds") or 0) - 0.35)
    end_candidates = [
        float(result.get("window_end_seconds") or start + 1.5) + 0.5,
        float(result.get("impact_seconds") or 0) + 0.6,
    ]
    end = max(end_candidates)
    return start, max(1.0, min(4.0, end - start))


def storage_usage(run_id: str) -> dict[str, int]:
    location = resolve_run(run_id)
    if location is None:
        return {"downloaded_bytes": 0, "retained_bytes": 0}
    downloads = location.path / "downloads"
    artifacts = location.path / "artifacts"
    return {
        "downloaded_bytes": sum(path.stat().st_size for path in downloads.glob("*.mp4") if path.is_file()),
        "retained_bytes": sum(path.stat().st_size for path in artifacts.glob("review-*.mp4") if path.is_file()),
    }


def build_review_for_result(run_id: str, result: dict) -> Path:
    location = resolve_run(run_id)
    if location is None or location.read_only:
        raise ValueError("review generation requires a writable live run")
    clip_id = str(result.get("clip_id") or "")
    rows, _, _ = load_manifest(str(location.path / "video_manifest.csv"))
    row = next((item for item in rows if item.get("clip_id") == clip_id), None)
    if row is None:
        raise RuntimeError("result clip is missing from the manifest")
    source = resolve_manifest_media(location, str(row.get("saved_path") or ""))
    if not source.is_file():
        raise RuntimeError("source clip is missing")
    artifacts = location.path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    destination = artifacts / f"review-{clip_id}.mp4"
    if destination.is_file():
        validate_mp4(str(destination))
        return destination
    start, duration = _review_bounds(result)
    with job_temp_dir(run_id) as temporary_dir:
        temporary = temporary_dir / destination.name
        _encode_review(source, temporary, start, duration)
        with run_lock(run_id):
            os.replace(temporary, destination)
    return destination


def finalize_run_media(run_id: str, results: list[dict], retain_sources: bool) -> dict:
    location = resolve_run(run_id)
    if location is None or location.read_only:
        return {"status": "skipped", "sources_retained": True, **storage_usage(run_id)}
    manifest_path = location.path / "video_manifest.csv"
    rows, _, _ = load_manifest(str(manifest_path))
    result_by_clip = {str(result.get("clip_id")): result for result in results}
    artifacts = location.path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    try:
        for row in rows:
            if row.get("status") == "skipped" or row.get("skip_reason"):
                continue
            clip_id = str(row.get("clip_id") or "")
            result = result_by_clip.get(clip_id)
            if result is None:
                raise RuntimeError(f"No result metadata exists for clip {clip_id}")
            generated.append(build_review_for_result(run_id, result))
    except Exception as exc:
        return {
            "status": "warning",
            "sources_retained": True,
            "warning": "Compact review generation failed; source clips were retained.",
            "error_code": type(exc).__name__,
            **storage_usage(run_id),
        }

    if not retain_sources:
        with run_lock(run_id):
            for row in rows:
                if row.get("status") == "skipped" or row.get("skip_reason"):
                    continue
                source = resolve_manifest_media(location, str(row.get("saved_path") or ""))
                source.unlink(missing_ok=True)
                row["status"] = "cleaned"
                row["error"] = ""
            write_manifest(str(manifest_path), rows)

    return {
        "status": "retained" if retain_sources else "cleaned",
        "sources_retained": retain_sources,
        "review_clip_count": len(generated),
        **storage_usage(run_id),
    }
