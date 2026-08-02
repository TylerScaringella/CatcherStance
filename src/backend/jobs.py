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
    pitch_state_dir,
    resolve_manifest_media,
    resolve_run,
    run_lock,
)

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
ACTIVE_STATUSES = {
    "queued", "running", "resolving_game", "discovering_pitches", "downloading",
    "detecting", "building_review", "cleaning_up", "finalizing",
}
PROGRESS_STAGES = (
    "discovering_pitches",
    "downloading",
    "detecting",
    "building_review",
)


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


def _performance_summary(
    progress: dict | None,
    result_count: int,
    expected: int,
    created_at: float,
    updated_at: float,
    status: str,
) -> dict:
    elapsed = max(
        0.0,
        (time.time() if status in ACTIVE_STATUSES else updated_at) - created_at,
    )
    detecting_stage = dict((progress or {}).get("stages", {}).get("detecting") or {})
    detected = int(
        detecting_stage.get("current")
        or ((progress or {}).get("current") if (progress or {}).get("phase") == "detecting" else 0)
        or result_count
    )
    detection_started = float(detecting_stage.get("started_at") or created_at)
    detection_ended = float(
        detecting_stage.get("completed_at")
        or (time.time() if status in ACTIVE_STATUSES else updated_at)
    )
    detection_elapsed = max(0.0, detection_ended - detection_started)
    rate = detected / (detection_elapsed / 60) if detected and detection_elapsed > 0 else None
    eta = ((expected - detected) / rate * 60) if rate and expected > detected else None
    return {
        "elapsed_seconds": round(elapsed, 1),
        "detection_elapsed_seconds": round(detection_elapsed, 1),
        "pitches_per_minute": round(rate, 2) if rate else None,
        "eta_seconds": round(eta, 1) if eta is not None else None,
    }


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
            fps = _video_fps(str(media)) if media.is_file() else None
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
    phase: str | None = None,
    activate: bool = True,
) -> None:
    lowered = message.lower()
    resolved_phase = phase or (
        "building_review" if "review" in lowered
        else "discovering_pitches" if "discover" in lowered
        else "downloading" if "download" in lowered
        else "detecting"
    )
    now = time.time()
    with JOBS_LOCK:
        current_job = JOBS.setdefault(job_id, {"id": job_id})
        previous = current_job.get("progress") if isinstance(current_job.get("progress"), dict) else {}
        stage_rank = {stage: index for index, stage in enumerate(PROGRESS_STAGES)}
        previous_active = previous.get("active_stage")
        if (
            activate
            and previous_active in stage_rank
            and resolved_phase in stage_rank
            and stage_rank[resolved_phase] < stage_rank[previous_active]
        ):
            activate = False
        stages = dict(previous.get("stages") or {})
        previous_stage = dict(stages.get(resolved_phase) or {})
        stage_current = current if current is not None else previous_stage.get("current")
        stage_total = total if total is not None else previous_stage.get("total")
        percent = (
            round((stage_current / stage_total) * 100, 1)
            if stage_current is not None and stage_total
            else None
        )
        stage_status = "complete" if percent is not None and percent >= 100 else "active"
        stages[resolved_phase] = {
            **previous_stage,
            "status": stage_status,
            "current": stage_current,
            "total": stage_total,
            "percent": percent,
            "started_at": previous_stage.get("started_at") or now,
            "updated_at": now,
            **({"completed_at": now} if stage_status == "complete" else {}),
        }
        active_stage = resolved_phase if activate else previous.get("active_stage", resolved_phase)
        active_data = stages.get(active_stage, {})
        current_job.update(
            status=active_stage,
            phase=active_stage,
            message=message if activate else current_job.get("message", message),
            progress={
                "active_stage": active_stage,
                "phase": active_stage,
                "current": active_data.get("current"),
                "total": active_data.get("total"),
                "percent": active_data.get("percent"),
                "stages": stages,
            },
            updated_at=now,
        )
        job = dict(current_job)
    write_job_state(job)


def begin_job_stage(job_id: str, phase: str, message: str, total: int | None = None) -> None:
    if phase not in PROGRESS_STAGES:
        raise ValueError("unknown progress phase")
    set_job_progress(job_id, message, 0 if total is not None else None, total, phase=phase)


def load_results(location_or_id: RunLocation | str) -> list[dict]:
    location = (
        location_or_id
        if isinstance(location_or_id, RunLocation)
        else resolve_run(location_or_id)
    )
    if location is None:
        return []
    rows = _read_json(location.path / "detections.json", [])
    if not rows and not location.read_only:
        state_directory = pitch_state_dir(location.run_id)
        partial_rows = []
        if state_directory.is_dir():
            for path in state_directory.glob("*.json"):
                payload = _read_json(path, {})
                if isinstance(payload.get("detection"), dict):
                    partial_rows.append(payload["detection"])
        rows = sorted(
            partial_rows,
            key=lambda row: int(row.get("pitch_index") or 0),
        )
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
    counts = {
        "total": 0, "downloaded": 0, "cleaned": 0, "pending": 0, "failed": 0,
        "skipped": 0,
    }
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


def _declared_manifest_counts(location: RunLocation) -> dict[str, int]:
    counts = {
        "total": 0, "downloaded": 0, "cleaned": 0, "pending": 0, "failed": 0,
        "skipped": 0,
    }
    for row in manifest_rows(location):
        counts["total"] += 1
        status = str(row.get("status") or "pending")
        counts[status if status in counts else "pending"] += 1
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
    if path.is_file():
        return path
    review = location.path / "artifacts" / f"review-{clip_id}.mp4"
    return review if review.is_file() else None


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
    rows_by_clip = {row.get("clip_id"): row for row in manifest_rows(location)}
    for result in results:
        media_mode = "unavailable"
        manifest = rows_by_clip.get(result.get("clip_id"))
        if manifest is not None:
            try:
                if resolve_manifest_media(location, manifest.get("saved_path", "")).is_file():
                    media_mode = "source"
            except ValueError:
                pass
        if media_mode == "unavailable" and (
            location.path / "artifacts" / f"review-{result.get('clip_id')}.mp4"
        ).is_file():
            media_mode = "review"
        result["media_mode"] = media_mode
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

    results_complete = (location.path / "detections.json").is_file()
    job["expected_result_count"] = int(
        saved.get("expected_result_count")
        or saved.get("discovery", {}).get("downloadable_pitches", 0)
        or counts["total"]
    )
    job["results_complete"] = results_complete
    if results and results_complete:
        saved_stages = dict((job.get("progress") or {}).get("stages") or {})
        job.update(
            status="complete",
            phase="complete",
            message=f"Detection complete for {len(results)} pitches",
            progress={
                "active_stage": "complete",
                "phase": "complete",
                "current": len(results),
                "total": len(results),
                "percent": 100,
                "stages": saved_stages,
            },
        )
    elif counts["total"] > 0:
        available = counts["downloaded"] + counts["cleaned"] + counts["skipped"]
        if available == counts["total"] and counts["failed"] == 0:
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
            saved_progress = saved.get("progress")
            if job["phase"] in {"detecting", "building_review"} and isinstance(saved_progress, dict):
                job["progress"] = saved_progress
            else:
                job["progress"] = {
                    "active_stage": job["phase"],
                    "phase": job["phase"],
                    "current": available,
                    "total": counts["total"],
                    "percent": 100,
                    "stages": dict((saved_progress or {}).get("stages") or {}),
                }
        else:
            recent = time.time() - updated_at < 600
            status = saved.get("status") if recent else "interrupted"
            if status not in ACTIVE_STATUSES and status != "auth_required":
                status = "interrupted"
            job.update(
                status=status,
                phase="auth_required" if status == "auth_required" else "downloading",
                message=(
                    "TruMedia authentication must be refreshed before this run can continue."
                    if status == "auth_required"
                    else f"Downloading videos: {counts['downloaded']} of {counts['total']}"
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
    job["performance"] = _performance_summary(
        job.get("progress"),
        len(results),
        job["expected_result_count"],
        created_at,
        updated_at,
        str(job.get("status") or "queued"),
    )
    return job


def list_runs() -> list[dict]:
    jobs = [
        job
        for location in list_run_locations()
        if (job := job_from_location(location)) is not None
    ]
    return sorted(jobs, key=lambda job: float(job.get("updated_at") or 0), reverse=True)


def _result_count_from_disk(location: RunLocation) -> int:
    canonical = location.path / "detections.json"
    if canonical.is_file():
        rows = _read_json(canonical, [])
        return len(rows) if isinstance(rows, list) else 0
    if location.read_only:
        return 0
    directory = pitch_state_dir(location.run_id)
    return len(list(directory.glob("*.json"))) if directory.is_dir() else 0


def job_summary_from_location(location: RunLocation) -> dict | None:
    saved = _read_json(location.path / "job.json", {})
    game = _game_for_location(location, saved)
    if game is None:
        return None
    counts = _declared_manifest_counts(location)
    result_count = _result_count_from_disk(location)
    created_at = float(saved.get("created_at") or location.path.stat().st_mtime)
    updated_at = max(float(saved.get("updated_at") or 0), location.path.stat().st_mtime)
    status = str(saved.get("status") or "queued")
    phase = str(saved.get("phase") or status)
    message = str(saved.get("message") or status)
    canonical_results = (location.path / "detections.json").is_file()
    results_complete = canonical_results

    if canonical_results or (result_count and location.read_only):
        status = phase = "complete"
        message = f"Detection complete for {result_count} pitches"
        results_complete = True
    elif status in ACTIVE_STATUSES and time.time() - updated_at >= 600:
        status = phase = "interrupted"
        message = "Processing was interrupted and can be resumed"

    progress = saved.get("progress") if isinstance(saved.get("progress"), dict) else None
    expected = int(
        saved.get("expected_result_count")
        or saved.get("discovery", {}).get("downloadable_pitches", 0)
        or counts["total"]
    )
    return {
        "id": location.run_id,
        "game": game,
        "source": location.source,
        "read_only": location.read_only,
        "status": status,
        "phase": phase,
        "message": message,
        "revision": saved.get("revision"),
        "retain_sources": bool(saved.get("retain_sources", False)),
        "created_at": created_at,
        "updated_at": updated_at,
        "manifest": counts,
        "result_count": result_count,
        "expected_result_count": expected,
        "results_complete": results_complete,
        "progress": progress,
        "performance": _performance_summary(
            progress, result_count, expected, created_at, updated_at, status
        ),
        "cleanup": saved.get("cleanup"),
    }


def list_run_summaries() -> list[dict]:
    summaries = [
        summary
        for location in list_run_locations()
        if (summary := job_summary_from_location(location)) is not None
    ]
    return sorted(summaries, key=lambda job: float(job.get("updated_at") or 0), reverse=True)


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

        begin_job_stage(job_id, "detecting", "Preparing stance detection")
        rows = run_detection_for_existing_run(
            run_id=job_id,
            progress_callback=lambda phase, message, current, total, activate: set_job_progress(
                job_id, message, current, total, phase=phase, activate=activate
            ),
        )
        from .media_lifecycle import finalize_run_media

        current = hydrated_job(job_id) or {}
        set_job(job_id, status="building_review", phase="building_review", message="Building compact review clips")
        cleanup = finalize_run_media(job_id, rows, bool(current.get("retain_sources", False)))
        set_job(job_id, status="finalizing", phase="finalizing", message="Finalizing outputs")
        set_job(
            job_id,
            status="complete",
            phase="complete",
            message=f"Detection complete for {len(rows)} pitches",
            result_count=len(rows),
            expected_result_count=len(rows),
            results_complete=True,
            cleanup=cleanup,
        )
    except Exception as exc:
        _record_failure(job_id, exc)


def run_job(job_id: str, start_url: str) -> None:
    try:
        from pipeline import run_game_detection

        begin_job_stage(
            job_id,
            "discovering_pitches",
            "Discovering TruMedia pitch videos",
        )
        rows = run_game_detection(
            run_id=job_id,
            start_url=start_url,
            discovery_callback=lambda stats: set_job(
                job_id,
                discovery=stats,
                expected_result_count=stats.get("downloadable_pitches", 0),
                pitch_scope="duke_catcher",
            ),
            progress_callback=lambda phase, message, current, total, activate: set_job_progress(
                job_id, message, current, total, phase=phase, activate=activate
            ),
        )
        from .media_lifecycle import finalize_run_media

        current = hydrated_job(job_id) or {}
        set_job(job_id, status="building_review", phase="building_review", message="Building compact review clips")
        cleanup = finalize_run_media(job_id, rows, bool(current.get("retain_sources", False)))
        if cleanup.get("status") == "cleaned":
            set_job(job_id, status="cleaning_up", phase="cleaning_up", message="Source clips removed")
        set_job(job_id, status="finalizing", phase="finalizing", message="Finalizing outputs")
        set_job(
            job_id,
            status="complete",
            phase="complete",
            message=f"Detection complete for {len(rows)} pitches",
            result_count=len(rows),
            expected_result_count=len(rows),
            results_complete=True,
            cleanup=cleanup,
        )
    except Exception as exc:
        if "authentication" in str(exc).lower() or "session is expired" in str(exc).lower():
            set_job(
                job_id,
                status="auth_required",
                phase="auth_required",
                message="TruMedia authentication must be refreshed before this run can continue.",
            )
        else:
            _record_failure(job_id, exc)
