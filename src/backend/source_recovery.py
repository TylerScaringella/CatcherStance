from __future__ import annotations

import os
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

from downloader.crawler import get_logged_in_context, sign_media_ref
from downloader.files import download_with_retries, validate_mp4
from downloader.manifest import load_manifest, write_manifest
from project_paths import PLAYWRIGHT_STATE_PATH

from .jobs import manifest_row
from .storage import resolve_manifest_media, resolve_run, run_lock
from .trumedia import AuthenticationRequired, resolve_game


def recover_source_clip(run_id: str, clip_id: str) -> Path:
    location = resolve_run(run_id)
    if location is None or location.read_only:
        raise ValueError("source recovery requires a writable live run")
    row = manifest_row(run_id, clip_id)
    if row is None:
        raise ValueError("pitch clip was not found")
    source = resolve_manifest_media(location, row.get("saved_path", ""))
    if source.is_file():
        return source
    media_ref = row.get("media_ref") or ""
    if not media_ref:
        raise RuntimeError("this pitch has no recoverable TruMedia media reference")
    job = {}
    try:
        import json

        job = json.loads((location.path / "job.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    game = job.get("game") or {}
    match = resolve_game(game)
    if not match.match:
        raise RuntimeError("the TruMedia game could not be matched for source recovery")
    source.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{clip_id}.", suffix=".part", dir=source.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                _, page, _ = get_logged_in_context(
                    browser,
                    start_url=str(match.match["url"]),
                    storage_state_path=PLAYWRIGHT_STATE_PATH,
                )
                url = sign_media_ref(page, media_ref)
            finally:
                browser.close()
        ok, error = download_with_retries(url, str(temporary))
        if not ok:
            raise RuntimeError(error or "pitch video recovery failed")
        validate_mp4(str(temporary))
        os.replace(temporary, source)
        rows, _, _ = load_manifest(str(location.path / "video_manifest.csv"))
        for item in rows:
            if item.get("clip_id") == clip_id:
                item["status"] = "downloaded"
                item["error"] = ""
                break
        with run_lock(run_id):
            write_manifest(str(location.path / "video_manifest.csv"), rows)
        return source
    finally:
        temporary.unlink(missing_ok=True)
