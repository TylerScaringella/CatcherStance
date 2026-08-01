from __future__ import annotations

import hashlib
import os
import re
import threading
import time
import uuid

from flask import Response, jsonify, request, send_file, send_from_directory, stream_with_context
from project_paths import POSE_MODEL_PATH

from .config import TRUMEDIA_DEFAULT_URL, WEB_DIR
from .jobs import (
    ACTIVE_STATUSES,
    JOBS,
    JOBS_LOCK,
    clip_media,
    clip_result,
    game_status_summary,
    hydrated_job,
    latest_job_for_game,
    list_runs,
    run_existing_detection_job,
    run_job,
    write_job_state,
)
from .schedule import find_game, load_schedule
from .storage import job_temp_dir, resolve_run, run_file, run_lock, validate_identifier


def _error(message: str, status: int):
    return jsonify({"error": message}), status


def _overlay_cache_key(video_path, label: str) -> str:
    video_stat = video_path.stat()
    model_stat = POSE_MODEL_PATH.stat()
    signature = (
        f"{video_stat.st_size}:{video_stat.st_mtime_ns}:"
        f"{model_stat.st_size}:{model_stat.st_mtime_ns}:{label}"
    )
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]


def _file_chunks(path, chunk_size: int = 1024 * 1024):
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            yield chunk


def register_routes(app):
    @app.get("/api/schedule")
    def schedule():
        return jsonify(load_schedule())

    @app.get("/api/runs")
    def runs():
        return jsonify({"runs": list_runs(), "updated_at": time.time()})

    @app.get("/api/runs/<run_id>")
    @app.get("/api/jobs/<run_id>")
    def run_status(run_id: str):
        try:
            validate_identifier(run_id, "run id")
        except ValueError:
            return _error("invalid run id", 400)
        job = hydrated_job(run_id)
        if job is None:
            return _error("run not found", 404)
        return jsonify(job)

    @app.get("/api/games/<game_id>/latest-job")
    def latest_game_job(game_id: str):
        try:
            validate_identifier(game_id, "game id")
        except ValueError:
            return _error("invalid game id", 400)
        job = latest_job_for_game(game_id)
        if job is None:
            return _error("run not found", 404)
        return jsonify(job)

    @app.get("/api/game-status")
    def game_statuses():
        return jsonify(game_status_summary())

    @app.get("/api/results/<run_id>/<fmt>")
    def export_results(run_id: str, fmt: str):
        if fmt not in {"json", "csv"}:
            return _error("expected json or csv", 400)
        try:
            location = resolve_run(run_id)
        except ValueError:
            return _error("invalid run id", 400)
        if location is None:
            return _error("run not found", 404)
        file_path = run_file(location, f"detections.{fmt}")
        if not file_path.is_file():
            return _error("results not found", 404)
        return send_file(
            file_path,
            mimetype="application/json" if fmt == "json" else "text/csv",
            as_attachment=True,
            download_name=f"{run_id}-detections.{fmt}",
            conditional=True,
        )

    @app.get("/api/runs/<run_id>/clips/<clip_id>/video")
    @app.get("/api/jobs/<run_id>/clips/<clip_id>/video")
    def source_clip(run_id: str, clip_id: str):
        try:
            validate_identifier(run_id, "run id")
            validate_identifier(clip_id, "clip id")
        except ValueError:
            return _error("invalid run or clip id", 400)
        video_path = clip_media(run_id, clip_id)
        if video_path is None:
            return _error("clip not found", 404)
        return send_file(video_path, mimetype="video/mp4", conditional=True)

    @app.get("/api/jobs/<run_id>/clips/<clip_id>/overlay.mjpg")
    @app.get("/api/runs/<run_id>/clips/<clip_id>/overlay.mjpg")
    def overlay_clip(run_id: str, clip_id: str):
        try:
            validate_identifier(run_id, "run id")
            validate_identifier(clip_id, "clip id")
        except ValueError:
            return _error("invalid run or clip id", 400)
        video_path = clip_media(run_id, clip_id)
        if video_path is None:
            return _error("clip not found", 404)
        result = clip_result(run_id, clip_id) or {}
        label = ""
        if result.get("stance"):
            label = f"{result['stance']} ({float(result.get('confidence') or 0):.0%})"

        from pipeline import overlay_mjpeg_frames

        location = resolve_run(run_id)
        if location is not None and not location.read_only:
            artifacts = location.path / "artifacts"
            artifacts.mkdir(parents=True, exist_ok=True)
            cache_path = artifacts / (
                f"overlay-{clip_id}-{_overlay_cache_key(video_path, label)}.mjpg"
            )
            if cache_path.is_file():
                return Response(
                    stream_with_context(_file_chunks(cache_path)),
                    mimetype="multipart/x-mixed-replace; boundary=frame",
                )

            def generate_and_cache():
                completed = False
                with job_temp_dir(run_id) as temporary_dir:
                    temporary_path = temporary_dir / cache_path.name
                    try:
                        with temporary_path.open("wb") as handle:
                            for chunk in overlay_mjpeg_frames(video_path, pitch_label=label):
                                handle.write(chunk)
                                yield chunk
                        completed = True
                    finally:
                        if completed:
                            with run_lock(run_id):
                                os.replace(temporary_path, cache_path)

            return Response(
                stream_with_context(generate_and_cache()),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        return Response(
            stream_with_context(overlay_mjpeg_frames(video_path, pitch_label=label)),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.post("/api/run")
    def run_detection():
        payload = request.get_json(silent=True) or {}
        game = find_game(payload.get("game_id", ""))
        if game is None:
            return _error("unknown game_id", 400)

        start_url = (
            payload.get("trumedia_url")
            or game.get("trumedia_url")
            or TRUMEDIA_DEFAULT_URL
        ).strip()
        if not re.match(r"^https://[^\s]+$", start_url):
            return _error("trumedia_url must be an https URL", 400)

        force_redownload = bool(payload.get("force_redownload"))
        latest_job = None if force_redownload else latest_job_for_game(game["id"])

        if latest_job and latest_job.get("read_only"):
            if latest_job.get("status") == "complete":
                return jsonify(latest_job), 200
            return _error("example runs are read-only", 409)
        if latest_job and latest_job.get("status") == "complete":
            return jsonify(latest_job), 200
        if latest_job and latest_job.get("status") in ACTIVE_STATUSES:
            return jsonify(latest_job), 200

        if latest_job and latest_job.get("status") == "ready":
            run_id = latest_job["id"]
            queued = {
                **latest_job,
                "status": "queued",
                "phase": "queued",
                "message": "Queued stance detection",
                "updated_at": time.time(),
            }
            with JOBS_LOCK:
                JOBS[run_id] = queued
            write_job_state(queued)
            threading.Thread(
                target=run_existing_detection_job,
                args=(run_id,),
                daemon=True,
            ).start()
            return jsonify(queued), 202

        if latest_job and latest_job.get("manifest", {}).get("total", 0) > 0:
            run_id = latest_job["id"]
            queued = {
                **latest_job,
                "status": "queued",
                "phase": "queued",
                "message": "Queued download resume",
                "updated_at": time.time(),
            }
            with JOBS_LOCK:
                JOBS[run_id] = queued
            write_job_state(queued)
            threading.Thread(
                target=run_job,
                args=(run_id, start_url),
                daemon=True,
            ).start()
            return jsonify(queued), 202

        run_id = f"{game['id']}-{uuid.uuid4().hex[:8]}"
        queued = {
            "id": run_id,
            "game": game,
            "source": "live",
            "read_only": False,
            "status": "queued",
            "phase": "queued",
            "message": "Queued",
            "result_count": 0,
            "results": [],
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        with JOBS_LOCK:
            JOBS[run_id] = queued
        write_job_state(queued)
        threading.Thread(target=run_job, args=(run_id, start_url), daemon=True).start()
        return jsonify(queued), 202

    @app.get("/")
    def index():
        return send_from_directory(WEB_DIR, "index.html")

    @app.get("/<path:path>")
    def static_files(path: str):
        return send_from_directory(WEB_DIR, path)
