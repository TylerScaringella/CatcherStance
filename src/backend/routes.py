from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path

from flask import Response, jsonify, request, send_file, send_from_directory, session, stream_with_context
from project_paths import AUTH_DIR, PLAYWRIGHT_STATE_PATH, POSE_MODEL_PATH

from .config import WEB_DIR
from .jobs import (
    ACTIVE_STATUSES,
    CONTROL_ACTIONS,
    JOBS,
    JOBS_LOCK,
    clip_media,
    clip_result,
    game_status_summary,
    hydrated_job,
    latest_job_for_game,
    list_runs,
    list_run_summaries,
    manifest_row,
    run_existing_detection_job,
    run_job,
    request_control,
    write_job_state,
)
from .schedule import (
    DEFAULT_SEASON,
    ScheduleUnavailableError,
    completed_schedule,
    find_game,
    load_schedule_settings,
    schedule_for_season,
    refresh_schedule,
    set_active_season,
    validate_season,
)
from .storage import job_temp_dir, resolve_manifest_media, resolve_run, run_file, run_lock, validate_identifier
from .review import build_xlsx, save_review
from .source_recovery import recover_source_clip
from .trumedia import (
    AuthenticationRequired,
    TruMediaProvider,
    admin_token_configured,
    check_admin_token,
    resolve_game,
    save_mapping,
    validate_storage_state_payload,
)

_ADMIN_FAILURES: dict[str, deque[float]] = defaultdict(deque)


def _error(message: str, status: int, code: str | None = None, **details):
    return jsonify({"error": message, **({"code": code} if code else {}), **details}), status


def _public_candidate(candidate: dict) -> dict:
    return {
        key: candidate.get(key)
        for key in (
            "id", "date", "start_time", "game_number", "opponent", "opponent_key",
            "site", "result", "status", "trackman_available",
        )
    }


def _persistable_job(job: dict) -> dict:
    return {key: value for key, value in job.items() if key != "results"}


def _admin_authorized() -> bool:
    return bool(session.get("trumedia_admin"))


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
        try:
            season = validate_season(request.args.get("season", DEFAULT_SEASON))
            if request.args.get("include_scheduled") in {"1", "true", "yes"}:
                return jsonify(schedule_for_season(season, include_scheduled=True))
            return jsonify(completed_schedule(season))
        except ValueError as exc:
            return _error(str(exc), 400)
        except ScheduleUnavailableError as exc:
            return jsonify({
                "team": "Duke",
                "team_id": "duke",
                "sport": "Baseball",
                "season": season,
                "games": [],
                "unpublished": True,
                "error": str(exc),
            })
        except Exception:
            return _error("schedule is temporarily unavailable", 503, "schedule_unavailable")

    @app.get("/api/teams")
    def teams():
        return jsonify({"teams": [{"id": "duke", "name": "Duke", "sport": "Baseball"}]})

    @app.get("/api/teams/duke/seasons")
    def seasons():
        current = time.gmtime().tm_year
        settings = load_schedule_settings()
        available = sorted(set(range(2026, current + 2)) | {settings["active_season"]})
        return jsonify({"team_id": "duke", "default": settings["active_season"], "seasons": available})

    @app.get("/api/teams/duke/schedule-settings")
    def schedule_settings():
        return jsonify(load_schedule_settings())

    @app.put("/api/teams/duke/schedule-settings")
    def update_schedule_settings():
        if not _admin_authorized():
            return _error("admin authorization is required", 401, "admin_required")
        try:
            payload = request.get_json(silent=True) or {}
            settings = set_active_season(payload.get("active_season"))
            return jsonify(settings)
        except ValueError as exc:
            return _error(str(exc), 400, "invalid_season")

    @app.post("/api/teams/duke/schedule/refresh")
    def refresh_team_schedule():
        if not _admin_authorized():
            return _error("admin authorization is required", 401, "admin_required")
        try:
            payload = request.get_json(silent=True) or {}
            season = validate_season(payload.get("season", load_schedule_settings()["active_season"]))
            schedule_payload = refresh_schedule(season)
            return jsonify({"settings": load_schedule_settings(), "schedule": schedule_payload})
        except ValueError as exc:
            return _error(str(exc), 400, "invalid_season")
        except ScheduleUnavailableError as exc:
            return _error(str(exc), 409, "schedule_unpublished")
        except Exception:
            return _error("schedule refresh failed; cached data was preserved", 503, "schedule_refresh_failed")

    @app.get("/api/teams/duke/schedule")
    def team_schedule():
        return schedule()

    @app.get("/api/integrations/trumedia/status")
    def trumedia_status():
        return jsonify(TruMediaProvider().status())

    @app.post("/api/admin/session")
    def admin_session():
        if not admin_token_configured():
            return _error("admin access is not configured", 503, "admin_not_configured")
        address = request.remote_addr or "unknown"
        now = time.time()
        failures = _ADMIN_FAILURES[address]
        while failures and now - failures[0] > 60:
            failures.popleft()
        if len(failures) >= 5:
            return _error("too many attempts; try again shortly", 429, "rate_limited")
        if not check_admin_token(str((request.get_json(silent=True) or {}).get("token") or "")):
            failures.append(now)
            return _error("invalid admin token", 401, "invalid_admin_token")
        failures.clear()
        session.clear()
        session["trumedia_admin"] = True
        return jsonify({"authorized": True})

    @app.post("/api/integrations/trumedia/session")
    def upload_trumedia_session():
        if not _admin_authorized():
            return _error("admin authorization is required", 401, "admin_required")
        uploaded = request.files.get("session")
        if uploaded is None:
            return _error("session JSON file is required", 400, "invalid_session")
        raw = uploaded.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            return _error("session file is too large", 413, "invalid_session")
        try:
            payload = validate_storage_state_payload(json.loads(raw.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return _error("invalid Playwright storage-state JSON", 400, "invalid_session")
        AUTH_DIR.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=AUTH_DIR, suffix=".json.part", delete=False
            ) as handle:
                json.dump(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
                temporary_path = Path(handle.name)
            temporary_path.chmod(0o600)
            TruMediaProvider().validate_live(temporary_path)
            os.replace(temporary_path, PLAYWRIGHT_STATE_PATH)
            PLAYWRIGHT_STATE_PATH.chmod(0o600)
        except AuthenticationRequired:
            return _error("the uploaded TruMedia session is expired", 400, "expired_session")
        except Exception:
            return _error("the TruMedia session could not be validated", 400, "invalid_session")
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return jsonify(TruMediaProvider().status())

    @app.post("/api/integrations/trumedia/validate")
    def validate_trumedia_session():
        if not _admin_authorized():
            return _error("admin authorization is required", 401, "admin_required")
        provider = TruMediaProvider()
        if not provider.state_path.is_file():
            return _error("TruMedia authentication is required", 409, "auth_required")
        try:
            provider.validate_live(provider.state_path)
        except AuthenticationRequired:
            return _error("the saved TruMedia session is expired", 409, "expired_session")
        except Exception:
            return _error("the TruMedia session could not be validated", 503, "validation_failed")
        return jsonify(provider.status())

    @app.get("/api/games/<game_id>/trumedia-match")
    def trumedia_match(game_id: str):
        game = find_game(game_id)
        if game is None:
            return _error("unknown game_id", 404)
        try:
            result = resolve_game(game)
        except AuthenticationRequired:
            return _error("TruMedia authentication is required", 409, "auth_required")
        return jsonify({
            "status": result.status,
            "match": _public_candidate(result.match) if result.match else None,
            "candidates": [_public_candidate(candidate) for candidate in result.candidates],
        })

    @app.post("/api/games/<game_id>/trumedia-match")
    def confirm_trumedia_match(game_id: str):
        game = find_game(game_id)
        if game is None:
            return _error("unknown game_id", 404)
        candidate_id = str((request.get_json(silent=True) or {}).get("candidate_id") or "")
        try:
            candidates = TruMediaProvider().discover_games(str(game["date"]))
        except AuthenticationRequired:
            return _error("TruMedia authentication is required", 409, "auth_required")
        candidate = next((item for item in candidates if item.get("id") == candidate_id), None)
        if candidate is None:
            return _error("unknown TruMedia candidate", 400, "invalid_match")
        saved = save_mapping(game_id, candidate)
        return jsonify({"status": "matched", "match": _public_candidate(saved)})

    @app.get("/api/runs")
    def runs():
        if request.args.get("view") == "summary":
            summaries = list_run_summaries()
            signature = hashlib.sha256(
                json.dumps(
                    [(item["id"], item.get("updated_at"), item.get("status")) for item in summaries],
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if request.if_none_match.contains(signature):
                return Response(status=304)
            response = jsonify({"runs": summaries, "updated_at": time.time()})
            response.set_etag(signature)
            response.headers["Cache-Control"] = "no-cache"
            return response
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

    @app.post("/api/runs/<run_id>/control")
    def control_run(run_id: str):
        try:
            validate_identifier(run_id, "run id")
            action = str((request.get_json(silent=True) or {}).get("action") or "")
            if action not in CONTROL_ACTIONS:
                return _error("action must be pause, stop, or cancel", 400, "invalid_control_action")
            job = request_control(run_id, action)
            return jsonify(job), 202
        except ValueError as exc:
            return _error(str(exc), 400, "invalid_control_action")
        except LookupError:
            return _error("run not found", 404)
        except PermissionError as exc:
            return _error(str(exc), 409, "read_only_run")
        except RuntimeError as exc:
            return _error(str(exc), 409, "invalid_run_state")

    @app.delete("/api/runs/<run_id>")
    def delete_run(run_id: str):
        if not _admin_authorized():
            return _error("admin authorization is required", 401, "admin_required")
        try:
            validate_identifier(run_id, "run id")
        except ValueError as exc:
            return _error(str(exc), 400)
        location = resolve_run(run_id)
        if location is None:
            return _error("run not found", 404)
        if location.read_only:
            return _error("sample runs are read-only", 409, "read_only_run")
        job = hydrated_job(run_id)
        if job is not None and job.get("status") in ACTIVE_STATUSES:
            return _error("stop or cancel the active run before deleting it", 409, "run_active")
        with run_lock(run_id):
            if location.path.exists():
                shutil.rmtree(location.path)
        with JOBS_LOCK:
            JOBS.pop(run_id, None)
        return jsonify({"deleted": True, "run_id": run_id})

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

    @app.put("/api/runs/<run_id>/clips/<clip_id>/review")
    def save_pitch_review(run_id: str, clip_id: str):
        if not _admin_authorized():
            return _error("admin authorization is required", 401, "admin_required")
        try:
            validate_identifier(run_id, "run id")
            validate_identifier(clip_id, "clip id")
        except ValueError:
            return _error("invalid run or clip id", 400)
        location = resolve_run(run_id)
        if location is None:
            return _error("run not found", 404)
        if location.read_only:
            return _error("sample runs are read-only", 409, "read_only_run")
        result = clip_result(run_id, clip_id)
        if result is None:
            return _error("pitch clip not found", 404)
        payload = request.get_json(silent=True) or {}
        try:
            entry = save_review(location, clip_id, str(payload.get("stance") or ""), payload.get("note", ""))
        except ValueError as exc:
            return _error(str(exc), 400, "invalid_review")
        refreshed_job = hydrated_job(run_id)
        refreshed = next(
            row for row in refreshed_job.get("results", [])
            if row.get("clip_id") == clip_id
        )
        return jsonify({"result": refreshed, "review": entry, "summary": refreshed_job.get("review", {})})

    @app.post("/api/runs/<run_id>/clips/<clip_id>/source-recovery")
    def recover_pitch_source(run_id: str, clip_id: str):
        if not _admin_authorized():
            return _error("admin authorization is required", 401, "admin_required")
        try:
            validate_identifier(run_id, "run id")
            validate_identifier(clip_id, "clip id")
        except ValueError:
            return _error("invalid run or clip id", 400)
        location = resolve_run(run_id)
        if location is None:
            return _error("run not found", 404)
        if location.read_only:
            return _error("sample runs are read-only", 409, "read_only_run")
        manifest = manifest_row(run_id, clip_id)
        if clip_media(run_id, clip_id) is not None and manifest is not None:
            row = manifest
            try:
                source = resolve_manifest_media(location, row.get("saved_path", ""))
                if source.is_file():
                    return jsonify({"status": "ready", "clip_id": clip_id})
            except ValueError:
                pass
        job = hydrated_job(run_id)
        recovery = job.get("source_recovery") if job else None
        if recovery and recovery.get("clip_id") == clip_id and recovery.get("status") == "running":
            return jsonify(recovery), 202
        set_job(run_id, source_recovery={"clip_id": clip_id, "status": "running"})

        def recover() -> None:
            try:
                recover_source_clip(run_id, clip_id)
                set_job(run_id, source_recovery={"clip_id": clip_id, "status": "ready"})
            except AuthenticationRequired as exc:
                set_job(run_id, source_recovery={"clip_id": clip_id, "status": "failed", "code": exc.code, "error": str(exc)})
            except Exception as exc:
                set_job(run_id, source_recovery={"clip_id": clip_id, "status": "failed", "code": type(exc).__name__, "error": str(exc)})

        threading.Thread(target=recover, daemon=True).start()
        return jsonify({"clip_id": clip_id, "status": "running"}), 202

    @app.get("/api/results/<run_id>/xlsx")
    def export_results_xlsx(run_id: str):
        if not _admin_authorized():
            return _error("admin authorization is required", 401, "admin_required")
        try:
            location = resolve_run(run_id)
        except ValueError:
            return _error("invalid run id", 400)
        if location is None:
            return _error("run not found", 404)
        job = hydrated_job(run_id)
        if job is None or not job.get("results_complete"):
            return _error("results are not finalized", 409, "results_incomplete")
        unresolved = [row.get("pitch_index") for row in job.get("results", []) if row.get("review_status") != "reviewed"]
        if unresolved:
            return _error(
                "review every pitch before exporting",
                409,
                "review_incomplete",
                reviewed=len(job.get("results", [])) - len(unresolved),
                unresolved=len(unresolved),
                unresolved_pitch_numbers=unresolved,
            )
        workbook = build_xlsx(job.get("results", []))
        return send_file(
            io.BytesIO(workbook),
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            as_attachment=True,
            download_name=f"{run_id}-catcher-review.xlsx",
        )

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
        job = hydrated_job(run_id)
        if job is None or not job.get("results_complete"):
            return _error("results are not finalized", 409, "results_incomplete")
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
        if request.args.get("mode") == "source":
            location = resolve_run(run_id)
            row = manifest_row(run_id, clip_id)
            video_path = None
            if location is not None and row is not None:
                try:
                    source = resolve_manifest_media(location, row.get("saved_path", ""))
                    video_path = source if source.is_file() else None
                except ValueError:
                    video_path = None
        if video_path is None:
            return _error("source clip is not currently available", 404, "source_unavailable")
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
    def run_detection(payload_override=None, reprocess: bool = False):
        payload = payload_override if payload_override is not None else (request.get_json(silent=True) or {})
        requested_game_id = str(payload.get("game_id") or "")
        raw_pitch_limit = payload.get("pitch_limit")
        if raw_pitch_limit in (None, ""):
            pitch_limit = None
        else:
            try:
                pitch_limit = int(raw_pitch_limit)
            except (TypeError, ValueError):
                return _error("pitch_limit must be a whole number from 1 to 25", 400, "invalid_pitch_limit")
            if not 1 <= pitch_limit <= 25:
                return _error("pitch_limit must be a whole number from 1 to 25", 400, "invalid_pitch_limit")
        force_redownload = bool(payload.get("force_redownload")) or reprocess
        latest_job = None if force_redownload else latest_job_for_game(requested_game_id)
        if latest_job and latest_job.get("status") == "canceled":
            latest_job = None
        if latest_job and latest_job.get("status") == "complete":
            return jsonify(latest_job), 200

        game = find_game(requested_game_id)
        if game is None:
            return _error("unknown game_id", 400)

        if latest_job and latest_job.get("read_only") and latest_job.get("status") != "complete":
            return _error("example runs are read-only", 409)
        if latest_job and latest_job.get("status") in ACTIVE_STATUSES:
            return jsonify(latest_job), 200

        try:
            match = resolve_game(game)
        except AuthenticationRequired:
            return _error("TruMedia authentication is required", 409, "auth_required")
        if match.match is None:
            return _error(
                "Select the matching TruMedia game before starting",
                409,
                "match_required",
                candidates=[_public_candidate(candidate) for candidate in match.candidates],
            )
        start_url = str(match.match["url"])

        if latest_job and latest_job.get("status") == "ready":
            run_id = latest_job["id"]
            queued = {
                **_persistable_job(latest_job),
                "status": "queued",
                "phase": "queued",
                "message": "Queued stance detection",
                "control_request": None,
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
                **_persistable_job(latest_job),
                "status": "queued",
                "phase": "queued",
                "message": "Queued download resume",
                "control_request": None,
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

        previous_runs = [run for run in list_runs() if run.get("game", {}).get("id") == game["id"] and run.get("source") == "live"]
        revision = max((int(run.get("revision") or 1) for run in previous_runs), default=0) + 1
        run_id = f"{game['id']}-r{revision}-{uuid.uuid4().hex[:8]}"
        queued = {
            "id": run_id,
            "game": game,
            "source": "live",
            "read_only": False,
            "status": "queued",
            "phase": "queued",
            "message": "Queued",
            "revision": revision,
            "reprocess_of": latest_job_for_game(game["id"])["id"] if reprocess and latest_job_for_game(game["id"]) else None,
            "retain_sources": bool(payload.get("retain_sources", False)),
            "pitch_limit": pitch_limit,
            "pitch_scope": "duke_catcher",
            "trumedia_match": _public_candidate(match.match),
            "result_count": 0,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        with JOBS_LOCK:
            JOBS[run_id] = queued
        write_job_state(queued)
        threading.Thread(target=run_job, args=(run_id, start_url), daemon=True).start()
        return jsonify(queued), 202

    @app.post("/api/games/<game_id>/reprocess")
    def reprocess_game(game_id: str):
        payload = {**(request.get_json(silent=True) or {}), "game_id": game_id}
        return run_detection(payload, reprocess=True)

    @app.get("/")
    def index():
        return send_from_directory(WEB_DIR, "index.html")

    @app.get("/<path:path>")
    def static_files(path: str):
        return send_from_directory(WEB_DIR, path)
