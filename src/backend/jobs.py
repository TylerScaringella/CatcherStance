from __future__ import annotations

import csv
import json
import threading
import time
import traceback
from functools import lru_cache
from pathlib import Path

from .schedule import load_schedule
from .storage import (
    RunLocation,
    atomic_write_json,
    atomic_write_text,
    list_run_locations,
    live_run,
    resolve_manifest_media,
    resolve_run,
    run_lock,
)

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
ACTIVE_STATUSES = {"queued", "running", "downloading", "detecting", "finalizing"}


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def _split_csv(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _number(value, cast=float):
    if value in (None, ""):
        return None
    try:
        return cast(value)
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=256)
def _video_fps(path_text: str) -> float | None:
    try:
        import cv2

        capture = cv2.VideoCapture(path_text)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        capture.release()
        return fps or None
    except Exception:
        return None


def normalize_result(row: dict, location: RunLocation | None = None) -> dict:
    result = dict(row)
    status = str(result.get("status") or "unknown")
    rejection = result.get("rejection_reason") or (status if status not in {"ok", "complete"} else None)
    vote_distribution = result.get("vote_distribution") or {}
    if isinstance(vote_distribution, str):
        try:
            vote_distribution = json.loads(vote_distribution) if vote_distribution else {}
        except json.JSONDecodeError:
            vote_distribution = {}

    fps = _number(result.get("fps"))
    if fps is None and location is not None:
        try:
            media = resolve_manifest_media(location, str(result.get("video_path") or ""))
            fps = _video_fps(str(media))
        except (ValueError, OSError):
            fps = None

    impact_frame = _number(result.get("impact_frame"), int)
    start_frame = _number(result.get("window_start_frame"), int)
    end_frame = _number(result.get("window_end_frame"), int)

    result.update(
        {
            "pitch_index": _number(result.get("pitch_index"), int),
            "confidence": _number(result.get("confidence")) or 0.0,
            "accepted": bool(result.get("stance")) and rejection is None,
            "rejection_reason": rejection,
            "impact_frame": impact_frame,
            "window_start_frame": start_frame,
            "window_end_frame": end_frame,
            "fps": fps,
            "impact_seconds": _number(result.get("impact_seconds"))
            if result.get("impact_seconds") not in (None, "")
            else impact_frame / fps
            if impact_frame is not None and fps
            else None,
            "window_start_seconds": _number(result.get("window_start_seconds"))
            if result.get("window_start_seconds") not in (None, "")
            else start_frame / fps
            if start_frame is not None and fps
            else None,
            "window_end_seconds": _number(result.get("window_end_seconds"))
            if result.get("window_end_seconds") not in (None, "")
            else end_frame / fps
            if end_frame is not None and fps
            else None,
            "valid_frame_count": _number(result.get("valid_frame_count"), int) or 0,
            "camera_quality": _number(result.get("camera_quality")),
            "vote_distribution": vote_distribution,
            "detector_provenance": _split_csv(result.get("detector_provenance")),
            "quality_flags": _split_csv(result.get("quality_flags")),
        }
    )
    return result


def write_job_state(job: dict) -> None:
    location = live_run(job["id"], create=True)
    (location.path / "artifacts").mkdir(parents=True, exist_ok=True)
    with run_lock(job["id"]):
        atomic_write_json(location.path / "job.json", job)


def set_job(job_id: str, **updates) -> None:
    with JOBS_LOCK:
        current = JOBS.setdefault(job_id, {"id": job_id})
        current.update(updates)
        current["updated_at"] = time.time()
        job = dict(current)
    write_job_state(job)


def set_job_progress(
    job_id: str,
    message: str,
    current: int | None = None,
    total: int | None = None,
) -> None:
    phase = "detecting" if "download" not in message.lower() else "downloading"
    updates = {"status": phase, "phase": phase, "message": message}
    if current is not None and total is not None:
        updates["progress"] = {
            "phase": phase,
            "current": current,
            "total": total,
            "percent": round((current / total) * 100, 1) if total else 0,
        }
    set_job(job_id, **updates)


def load_results(location_or_id: RunLocation | str) -> list[dict]:
    location = (
        location_or_id
        if isinstance(location_or_id, RunLocation)
        else resolve_run(location_or_id)
    )
    if location is None:
        return []
    rows = _read_json(location.path / "detections.json", [])
    return [normalize_result(row, location) for row in rows if isinstance(row, dict)]


def manifest_rows(location_or_id: RunLocation | str) -> list[dict]:
    location = (
        location_or_id
        if isinstance(location_or_id, RunLocation)
        else resolve_run(location_or_id)
    )
    if location is None:
        return []
    path = location.path / "video_manifest.csv"
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def manifest_counts(location_or_id: RunLocation | str) -> dict[str, int]:
    location = (
        location_or_id
        if isinstance(location_or_id, RunLocation)
        else resolve_run(location_or_id)
    )
    counts = {"total": 0, "downloaded": 0, "pending": 0, "failed": 0}
    if location is None:
        return counts
    for row in manifest_rows(location):
        counts["total"] += 1
        status = row.get("status", "")
        try:
            exists = resolve_manifest_media(location, row.get("saved_path", "")).exists()
        except ValueError:
            exists = False
        if status == "downloaded" and not exists:
            counts["pending"] += 1
        elif status in counts:
            counts[status] += 1
    return counts


def manifest_row(job_id: str, clip_id: str) -> dict | None:
    location = resolve_run(job_id)
    if location is None:
        return None
    for row in manifest_rows(location):
        if row.get("clip_id") == clip_id:
            return row
    return None


def clip_media(job_id: str, clip_id: str) -> Path | None:
    location = resolve_run(job_id)
    row = manifest_row(job_id, clip_id)
    if location is None or row is None:
        return None
    try:
        path = resolve_manifest_media(location, row.get("saved_path", ""))
    except ValueError:
        return None
    return path if path.is_file() else None


def clip_result(job_id: str, clip_id: str) -> dict | None:
    for row in load_results(job_id):
        if row.get("clip_id") == clip_id:
            return row
    return None


def _game_for_location(location: RunLocation, saved_job: dict) -> dict | None:
    game = saved_job.get("game")
    if isinstance(game, dict) and game.get("id"):
        return game
    candidates = sorted(load_schedule().get("games", []), key=lambda game: len(game["id"]), reverse=True)
    return next((game for game in candidates if location.run_id.startswith(game["id"])), None)


def job_from_location(location: RunLocation) -> dict | None:
    saved = _read_json(location.path / "job.json", {})
    game = _game_for_location(location, saved)
    if game is None:
        return None

    results = load_results(location)
    counts = manifest_counts(location)
    created_at = float(saved.get("created_at") or location.path.stat().st_mtime)
    updated_at = max(
        float(saved.get("updated_at") or 0),
        location.path.stat().st_mtime,
    )
    job = {
        **saved,
        "id": location.run_id,
        "game": game,
        "source": location.source,
        "read_only": location.read_only,
        "created_at": created_at,
        "updated_at": updated_at,
        "manifest": counts,
        "results": results,
        "result_count": len(results),
    }

    if results:
        job.update(
            status="complete",
            phase="complete",
            message=f"Detection complete for {len(results)} pitches",
            progress={
                "phase": "complete",
                "current": len(results),
                "total": len(results),
                "percent": 100,
            },
        )
    elif counts["total"] > 0:
        if counts["downloaded"] == counts["total"] and counts["failed"] == 0:
            if saved.get("status") in ACTIVE_STATUSES and time.time() - updated_at < 600:
                job["status"] = saved.get("status")
                job["phase"] = saved.get("phase", "detecting")
                job["message"] = saved.get("message") or "Running catcher detection"
            else:
                job.update(
                    status="ready",
                    phase="ready",
                    message="Videos downloaded and ready for detection",
                )
            job["progress"] = {
                "phase": job["phase"],
                "current": counts["downloaded"],
                "total": counts["total"],
                "percent": 100,
            }
        else:
            recent = time.time() - updated_at < 600
            status = saved.get("status") if recent else "interrupted"
            if status not in ACTIVE_STATUSES:
                status = "interrupted"
            job.update(
                status=status,
                phase="downloading",
                message=(
                    f"Downloading videos: {counts['downloaded']} of {counts['total']}"
                    if status != "interrupted"
                    else f"Partial run: {counts['downloaded']} of {counts['total']} downloaded"
                ),
                progress={
                    "phase": "downloading",
                    "current": counts["downloaded"],
                    "total": counts["total"],
                    "percent": round(counts["downloaded"] / counts["total"] * 100, 1),
                },
            )
    elif location.source == "live" and saved:
        job.setdefault("status", "queued")
        job.setdefault("phase", job["status"])
        job.setdefault("message", "Queued")
    else:
        return None
    return job


def list_runs() -> list[dict]:
    jobs = [
        job
        for location in list_run_locations()
        if (job := job_from_location(location)) is not None
    ]
    return sorted(jobs, key=lambda job: float(job.get("updated_at") or 0), reverse=True)


def latest_job_for_game(game_id: str) -> dict | None:
    jobs = [job for job in list_runs() if job.get("game", {}).get("id") == game_id]
    live_jobs = [job for job in jobs if job.get("source") == "live"]
    candidates = live_jobs or jobs
    return max(candidates, key=lambda job: float(job.get("updated_at") or 0), default=None)


def game_status_summary() -> dict:
    summary = {}
    for game in load_schedule()["games"]:
        job = latest_job_for_game(game["id"])
        if job is None:
            summary[game["id"]] = {"status": "none", "label": "Not run"}
            continue
        status = job.get("status", "none")
        summary[game["id"]] = {
            "status": status,
            "label": {
                "complete": f"Detected {job.get('result_count', 0)}",
                "ready": f"Downloaded {job.get('manifest', {}).get('downloaded', 0)}",
                "interrupted": "Partial run",
                "failed": "Failed",
            }.get(status, "In progress" if status in ACTIVE_STATUSES else status),
            "job_id": job["id"],
            "result_count": job.get("result_count", 0),
            "manifest": job.get("manifest", {}),
            "progress": job.get("progress"),
            "source": job.get("source"),
            "read_only": job.get("read_only", False),
        }
    return summary


def hydrated_job(job_id: str) -> dict | None:
    location = resolve_run(job_id)
    if location is not None:
        return job_from_location(location)
    with JOBS_LOCK:
        return dict(JOBS[job_id]) if job_id in JOBS else None


def _record_failure(job_id: str, exc: Exception) -> None:
    location = live_run(job_id, create=True)
    artifacts = location.path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    atomic_write_text(artifacts / "job-error.log", traceback.format_exc())
    set_job(
        job_id,
        status="failed",
        phase="failed",
        message="The run failed. Review the run log for details.",
        error_code=type(exc).__name__,
    )


def run_existing_detection_job(job_id: str) -> None:
    try:
        from pipeline import run_detection_for_existing_run

        set_job(job_id, status="detecting", phase="detecting", message="Preparing stance detection")
        rows = run_detection_for_existing_run(
            run_id=job_id,
            status_callback=lambda message, current, total: set_job_progress(
                job_id, message, current, total
            ),
        )
        set_job(job_id, status="finalizing", phase="finalizing", message="Finalizing outputs")
        set_job(
            job_id,
            status="complete",
            phase="complete",
            message=f"Detection complete for {len(rows)} pitches",
            result_count=len(rows),
            results=rows,
        )
    except Exception as exc:
        _record_failure(job_id, exc)


def run_job(job_id: str, start_url: str) -> None:
    try:
        from pipeline import run_game_detection

        set_job(job_id, status="downloading", phase="downloading", message="Downloading pitch videos")
        rows = run_game_detection(
            run_id=job_id,
            start_url=start_url,
            status_callback=lambda message, current, total: set_job_progress(
                job_id, message, current, total
            ),
        )
        set_job(job_id, status="finalizing", phase="finalizing", message="Finalizing outputs")
        set_job(
            job_id,
            status="complete",
            phase="complete",
            message=f"Detection complete for {len(rows)} pitches",
            result_count=len(rows),
            results=rows,
        )
    except Exception as exc:
        _record_failure(job_id, exc)
