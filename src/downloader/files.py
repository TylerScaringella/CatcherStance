from __future__ import annotations

import os
import tempfile
import threading
import time
from pathlib import Path

import requests

from .config import CHUNK_SIZE, REQUEST_TIMEOUT_SECONDS, RETRY_COUNT, RETRY_SLEEP_SECONDS

_THREAD_LOCAL = threading.local()


def _session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4)
        session.mount("https://", adapter)
        _THREAD_LOCAL.session = session
    return session


def validate_mp4(path: str) -> None:
    candidate = Path(path)
    if not candidate.is_file() or candidate.stat().st_size < 32:
        raise ValueError("downloaded video is empty")
    with candidate.open("rb") as handle:
        header = handle.read(64)
    if b"ftyp" not in header:
        raise ValueError("downloaded file is not an MP4")


def ensure_parent_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def download_file_once(url: str, filepath: str):
    ensure_parent_dir(filepath)
    destination = Path(filepath)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".part",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            with _session().get(url, stream=True, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").lower()
                if content_type and not any(kind in content_type for kind in ("video", "mp4", "octet-stream")):
                    raise ValueError("download response was not video media")
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        validate_mp4(str(temporary))
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def download_with_retries(url: str, filepath: str, retries: int = RETRY_COUNT):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            download_file_once(url, filepath)
            return True, ""
        except Exception as exc:
            last_err = f"download failed ({type(exc).__name__})"
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception:
                pass

            if attempt < retries:
                time.sleep(RETRY_SLEEP_SECONDS * attempt)

    return False, last_err or "unknown download error"


def download_one_row(row):
    clip_id = row["clip_id"]
    download_url = row.get("download_url") or row.get("s3_url") or ""
    saved_path = row["saved_path"]

    if os.path.exists(saved_path):
        return clip_id, True, "", saved_path

    if not download_url:
        return clip_id, False, "pitch video authorization is unavailable", saved_path

    ok, err = download_with_retries(download_url, saved_path, retries=RETRY_COUNT)
    return clip_id, ok, err, saved_path
