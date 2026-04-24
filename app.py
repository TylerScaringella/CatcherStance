from __future__ import annotations

import json
import csv
import re
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen

from flask import Flask, Response, jsonify, request, send_file, send_from_directory, stream_with_context

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
RUNS_DIR = ROOT / "data" / "runs"
SCHEDULE_PATH = ROOT / "data" / "schedules" / "duke_baseball_2026.json"
TRUMEDIA_DEFAULT_URL = "https://duke-ncaabaseball.trumedianetworks.com/baseball/"
SCHEDULE_REFRESH_INTERVAL_SECONDS = 60 * 30

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()

app = Flask(__name__, static_folder=None)
_schedule_refresh_started = False


def load_schedule():
    return json.loads(SCHEDULE_PATH.read_text(encoding="utf-8"))


def find_game(game_id: str):
    for game in load_schedule()["games"]:
        if game["id"] == game_id:
            return game
    return None


def slugify(value: str):
    value = value.lower().replace("&", "and")
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def make_game_id(date: str, opponent: str, occurrence: int, total_occurrences: int):
    suffix = f"-{occurrence}" if total_occurrences > 1 else ""
    return f"duke-{date}-{slugify(opponent)}{suffix}"


def parse_acc_schedule_rows(html: str):
    pattern = re.compile(
        r"(?P<date>\d{1,2}/\d{1,2}/2026)\s+"
        r"(?:(?P<acc>\*)\s+)?"
        r"(?P<opponent>.*?)\s{2,}"
        r"(?P<location>.*?)\s{2,}"
        r"(?P<result>(?:[WL]\s+\d+[-–]\d+(?:\s+\([^)]*\))?|\d{1,2}:\d{2}\s+[AP]\.M\.\s+ET))"
    )
    raw_games = []
    totals: dict[tuple[str, str], int] = {}

    for match in pattern.finditer(html):
        parsed_date = datetime.strptime(match.group("date"), "%m/%d/%Y").date().isoformat()
        opponent = re.sub(r"\s+", " ", match.group("opponent")).strip()
        location = re.sub(r"\s+", " ", match.group("location")).strip()
        result = re.sub(r"\s+", " ", match.group("result")).strip().replace("–", "-")
        key = (parsed_date, opponent)
        totals[key] = totals.get(key, 0) + 1
        raw_games.append((key, parsed_date, opponent, location, bool(match.group("acc")), result))

    games = []
    occurrences: dict[tuple[str, str], int] = {}
    for key, parsed_date, opponent, location, conference, result in raw_games:
        occurrences[key] = occurrences.get(key, 0) + 1
        games.append(
            {
                "id": make_game_id(parsed_date, opponent, occurrences[key], totals[key]),
                "date": parsed_date,
                "opponent": opponent,
                "location": location,
                "conference": conference,
                "result": result,
                "trumedia_url": "",
            }
        )

    return games


def fetch_acc_schedule(source_url: str):
    request = Request(source_url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=20) as response:
        html = response.read().decode("utf-8", errors="replace")
    games = parse_acc_schedule_rows(html)
    if not games:
        raise RuntimeError("No games found on ACC schedule page")
    return games


def refresh_schedule_once():
    schedule = load_schedule()
    source_url = schedule.get("source")
    if not source_url:
        return

    fetched_games = fetch_acc_schedule(source_url)
    existing_by_id = {game["id"]: game for game in schedule.get("games", [])}
    for game in fetched_games:
        existing = existing_by_id.get(game["id"], {})
        game["trumedia_url"] = existing.get("trumedia_url", "")

    schedule["games"] = fetched_games
    schedule["source_checked"] = datetime.now().date().isoformat()
    SCHEDULE_PATH.write_text(json.dumps(schedule, indent=2), encoding="utf-8")
    print(f"Updated schedule from ACC: {len(fetched_games)} games")


def schedule_refresh_loop():
    while True:
        try:
            refresh_schedule_once()
        except Exception as exc:
            print(f"Schedule refresh failed: {type(exc).__name__}: {exc}")
        time.sleep(SCHEDULE_REFRESH_INTERVAL_SECONDS)


def start_schedule_refresh_job():
    global _schedule_refresh_started
    if _schedule_refresh_started:
        return
    _schedule_refresh_started = True
    thread = threading.Thread(target=schedule_refresh_loop, daemon=True)
    thread.start()


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


def manifest_has_existing_downloads(job_id: str):
    rows = manifest_rows(job_id)
    return bool(rows) and any(Path(row.get("saved_path", "")).exists() for row in rows)


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

        recent = time.time() - max((run_dir / "video_manifest.csv").stat().st_mtime, run_dir.stat().st_mtime) < 600
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


@app.get("/api/schedule")
def schedule():
    return jsonify(load_schedule())


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    job = hydrated_job(job_id)
    if job is None:
        return jsonify({"error": "job not found"}), 404
    return jsonify(job)


@app.get("/api/games/<game_id>/latest-job")
def latest_game_job(game_id: str):
    job = latest_job_for_game(game_id)
    if job is None:
        return jsonify({"error": "job not found"}), 404
    return jsonify(job)


@app.get("/api/game-status")
def game_statuses():
    return jsonify(game_status_summary())


@app.get("/api/results/<job_id>/<fmt>")
def export_results(job_id: str, fmt: str):
    if fmt not in {"json", "csv"}:
        return jsonify({"error": "expected json or csv"}), 400

    file_path = RUNS_DIR / job_id / f"detections.{fmt}"
    if not file_path.exists():
        return jsonify({"error": "results not found"}), 404

    mimetype = "application/json" if fmt == "json" else "text/csv"
    return send_file(
        file_path,
        mimetype=mimetype,
        as_attachment=True,
        download_name=f"{job_id}-detections.{fmt}",
    )


@app.get("/api/jobs/<job_id>/clips/<clip_id>/overlay.mjpg")
def overlay_clip(job_id: str, clip_id: str):
    row = manifest_row(job_id, clip_id)
    if row is None:
        return jsonify({"error": "clip not found"}), 404

    video_path = Path(row.get("saved_path", ""))
    if not video_path.exists():
        return jsonify({"error": "video file not found"}), 404

    result = clip_result(job_id, clip_id) or {}
    label = ""
    if result.get("stance"):
        confidence = float(result.get("confidence") or 0)
        label = f"{result['stance']} ({confidence:.0%})"

    from pipeline import overlay_mjpeg_frames

    return Response(
        stream_with_context(overlay_mjpeg_frames(video_path, pitch_label=label)),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/api/run")
def run_detection():
    payload = request.get_json(silent=True) or {}
    game = find_game(payload.get("game_id", ""))
    if game is None:
        return jsonify({"error": "unknown game_id"}), 400

    start_url = (payload.get("trumedia_url") or game.get("trumedia_url") or TRUMEDIA_DEFAULT_URL).strip()
    if not re.match(r"^https://.+", start_url):
        return jsonify({"error": "trumedia_url must be an https URL"}), 400

    force_redownload = bool(payload.get("force_redownload"))
    latest_job = None if force_redownload else latest_job_for_game(game["id"])

    if latest_job and latest_job.get("status") == "complete":
        return jsonify(latest_job), 200

    if latest_job and latest_job.get("status") in {"queued", "running"}:
        return jsonify(latest_job), 200

    if latest_job and latest_job.get("status") == "ready":
        job_id = latest_job["id"]
        with JOBS_LOCK:
            JOBS[job_id] = {
                **latest_job,
                "status": "queued",
                "message": "Queued catcher detection",
                "updated_at": time.time(),
            }
            write_job_state(JOBS[job_id])

        thread = threading.Thread(target=run_existing_detection_job, args=(job_id,), daemon=True)
        thread.start()
        return jsonify(JOBS[job_id]), 202

    if latest_job and latest_job.get("manifest", {}).get("total", 0) > 0:
        job_id = latest_job["id"]
        with JOBS_LOCK:
            JOBS[job_id] = {
                **latest_job,
                "status": "queued",
                "message": "Queued download resume",
                "updated_at": time.time(),
            }
            write_job_state(JOBS[job_id])

        thread = threading.Thread(target=run_job, args=(job_id, start_url), daemon=True)
        thread.start()
        return jsonify(JOBS[job_id]), 202

    job_id = f"{game['id']}-{uuid.uuid4().hex[:8]}"
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "game": game,
            "status": "queued",
            "message": "Queued",
            "result_count": 0,
            "results": [],
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        write_job_state(JOBS[job_id])

    thread = threading.Thread(target=run_job, args=(job_id, start_url), daemon=True)
    thread.start()
    return jsonify(JOBS[job_id]), 202


@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/<path:path>")
def static_files(path: str):
    return send_from_directory(WEB_DIR, path)


def main():
    start_schedule_refresh_job()
    print("Catcher Stance web app running at http://127.0.0.1:8000")
    app.run(host="127.0.0.1", port=8000, threaded=True)


if __name__ == "__main__":
    main()
