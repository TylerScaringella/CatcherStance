from __future__ import annotations

# Codex attribution: OpenAI Codex generated the Flask route structure for the
# web API, with project-specific review and edits.
import re
import threading
import time
import uuid
from pathlib import Path

from flask import Response, jsonify, request, send_file, send_from_directory, stream_with_context

from .config import RUNS_DIR, TRUMEDIA_DEFAULT_URL, WEB_DIR
from .jobs import (
    JOBS,
    JOBS_LOCK,
    clip_result,
    game_status_summary,
    hydrated_job,
    latest_job_for_game,
    manifest_row,
    run_existing_detection_job,
    run_job,
    write_job_state,
)
from .schedule import find_game, load_schedule


def register_routes(app):
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
