from __future__ import annotations

import csv
import json
import threading
import time
from pathlib import Path

from .config import RUNS_DIR
from .schedule import find_game, load_schedule

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def set_job(job_id: str, **updates):
    with JOBS_LOCK:
        JOBS[job_id].update(updates)
        JOBS[job_id]["updated_at"] = time.time()
        job = dict(JOBS[job_id])

    write_job_state(job)


def set_job_progress(job_id: str, message: str, current: int | None = None, total: int | None = None):
    updates = {"status": "running", "message": message}
    if current is not None and total is not None:
        updates["progress"] = {
            "phase": "detection",
            "current": current,
            "total": total,
            "percent": round((current / total) * 100, 1) if total else 0,
        }
    set_job(job_id, **updates)


def write_job_state(job: dict):
    run_dir = RUNS_DIR / job["id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "job.json"
    state_path.write_text(json.dumps(job, indent=2), encoding="utf-8")


def load_results(job_id: str):
    results_path = RUNS_DIR / job_id / "detections.json"
    if not results_path.exists():
        return []
    return json.loads(results_path.read_text(encoding="utf-8"))


def manifest_rows(job_id: str):
    manifest_path = RUNS_DIR / job_id / "video_manifest.csv"
    if not manifest_path.exists():
        return []

    with open(manifest_path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def result_videos_exist(job_id: str, results: list[dict]):
    if not results:
        return False

    for row in results:
        status = row.get("status")
        video_path = row.get("video_path")
        if status == "ok" and video_path and not Path(video_path).exists():
            return False
    return True


def manifest_counts(job_id: str):
    counts = {"total": 0, "downloaded": 0, "pending": 0, "failed": 0}

    for row in manifest_rows(job_id):
        counts["total"] += 1
        status = row.get("status", "")
        saved_path = row.get("saved_path", "")
        file_exists = bool(saved_path and Path(saved_path).exists())

        if status == "downloaded" and not file_exists:
            counts["pending"] += 1
        elif status in counts:
            counts[status] += 1
    return counts


def manifest_row(job_id: str, clip_id: str):
    for row in manifest_rows(job_id):
        if row.get("clip_id") == clip_id:
            return row
    return None


def clip_result(job_id: str, clip_id: str):
    for row in load_results(job_id):
        if row.get("clip_id") == clip_id:
            return row
    return None


def job_from_run_dir(run_dir: Path):
    if not run_dir.exists() or not run_dir.is_dir():
        return None

    job_path = run_dir / "job.json"
    if job_path.exists():
        job = json.loads(job_path.read_text(encoding="utf-8"))
    else:
        game_id = run_dir.name.rsplit("-", 1)[0]
        game = find_game(game_id)
        if game is None:
            return None
        job = {
            "id": run_dir.name,
            "game": game,
            "created_at": run_dir.stat().st_mtime,
            "updated_at": run_dir.stat().st_mtime,
            "result_count": 0,
            "results": [],
        }

    results = load_results(run_dir.name)
    counts = manifest_counts(run_dir.name)
    job["manifest"] = counts
    valid_results = results if result_videos_exist(run_dir.name, results) else []
    job["results"] = valid_results
    job["result_count"] = len(valid_results)

    if valid_results:
        job["status"] = "complete"
        job["message"] = f"Detection complete for {len(valid_results)} pitches"
        job["progress"] = {
            "phase": "complete",
            "current": len(valid_results),
            "total": len(valid_results),
            "percent": 100,
        }
    elif counts["total"] > 0:
        if counts["downloaded"] == counts["total"] and counts["failed"] == 0:
            job_recent = time.time() - float(job.get("updated_at", 0) or 0) < 600
            if job.get("status") != "running" or not job_recent:
                job["status"] = "ready"
                job["message"] = "All videos downloaded; ready to run catcher detection"
                job["progress"] = {
                    "phase": "download",
                    "current": counts["downloaded"],
                    "total": counts["total"],
                    "percent": 100,
                }
            else:
                job["message"] = job.get("message") or "Running catcher detection and stance classifier"
            return job

        manifest_path = run_dir / "video_manifest.csv"
        recent = time.time() - max(manifest_path.stat().st_mtime, run_dir.stat().st_mtime) < 600
        job["status"] = job.get("status") or ("running" if recent else "interrupted")
        job["message"] = (
            f"Downloading videos: {counts['downloaded']} of {counts['total']} downloaded"
            if job["status"] == "running"
            else f"Run has a manifest but no detections yet: {counts['downloaded']} of {counts['total']} downloaded"
        )
        job["progress"] = {
            "phase": "download",
            "current": counts["downloaded"],
            "total": counts["total"],
            "percent": round((counts["downloaded"] / counts["total"]) * 100, 1) if counts["total"] else 0,
        }
    else:
        return None

    return job


def latest_job_for_game(game_id: str):
    candidates = [
        run_dir
        for run_dir in RUNS_DIR.glob(f"{game_id}-*")
        if run_dir.is_dir()
    ]
    if not candidates:
        return None

    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for candidate in candidates:
        job = job_from_run_dir(candidate)
        if job is not None:
            return job
    return None


def game_status_summary():
    summary = {}
    for game in load_schedule()["games"]:
        job = latest_job_for_game(game["id"])
        if job is None:
            summary[game["id"]] = {"status": "none", "label": "Not run"}
            continue

        status = job.get("status", "none")
        manifest = job.get("manifest", {})
        progress = job.get("progress")
        label = {
            "complete": f"Detected {job.get('result_count', 0)}",
            "running": "In progress",
            "queued": "Queued",
            "ready": f"Downloaded {manifest.get('downloaded', 0)}",
            "interrupted": "Partial run",
            "failed": "Failed",
        }.get(status, status)

        summary[game["id"]] = {
            "status": status,
            "label": label,
            "job_id": job.get("id"),
            "result_count": job.get("result_count", 0),
            "manifest": manifest,
            "progress": progress,
        }
    return summary


def hydrated_job(job_id: str):
    disk_job = job_from_run_dir(RUNS_DIR / job_id)
    if disk_job is not None:
        return disk_job
    with JOBS_LOCK:
        memory_job = JOBS.get(job_id)

    if memory_job is None:
        return None

    created_at = float(memory_job.get("created_at", 0) or 0)
    if memory_job.get("status") in {"queued", "running"} and time.time() - created_at < 60 * 60:
        return memory_job
    return None


def run_existing_detection_job(job_id: str):
    try:
        from pipeline import run_detection_for_existing_run

        rows = run_detection_for_existing_run(
            run_id=job_id,
            status_callback=lambda message, current, total: set_job_progress(job_id, message, current, total),
        )
        set_job(
            job_id,
            status="complete",
            message=f"Detection complete for {len(rows)} pitches",
            result_count=len(rows),
            results=rows,
        )
    except Exception as exc:
        set_job(
            job_id,
            status="failed",
            message=str(exc),
            error=f"{type(exc).__name__}: {exc}",
        )


def run_job(job_id: str, start_url: str):
    try:
        from pipeline import run_game_detection

        set_job(job_id, status="running", message="Downloading TruMedia pitch videos")
        rows = run_game_detection(
            run_id=job_id,
            start_url=start_url,
            status_callback=lambda message, current, total: set_job_progress(job_id, message, current, total),
        )
        set_job(
            job_id,
            status="complete",
            message=f"Detection complete for {len(rows)} pitches",
            result_count=len(rows),
            results=rows,
        )
    except Exception as exc:
        set_job(
            job_id,
            status="failed",
            message=str(exc),
            error=f"{type(exc).__name__}: {exc}",
        )
