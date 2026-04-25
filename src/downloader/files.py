from __future__ import annotations

import os
import time
from pathlib import Path

import requests

from .config import CHUNK_SIZE, REQUEST_TIMEOUT_SECONDS, RETRY_COUNT, RETRY_SLEEP_SECONDS


def ensure_parent_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def download_file_once(url: str, filepath: str):
    ensure_parent_dir(filepath)
    with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        response.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)


def download_with_retries(url: str, filepath: str, retries: int = RETRY_COUNT):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            download_file_once(url, filepath)
            return True, ""
        except Exception as exc:
            last_err = str(exc)
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
    s3_url = row["s3_url"]
    saved_path = row["saved_path"]

    if os.path.exists(saved_path):
        return clip_id, True, "", saved_path

    ok, err = download_with_retries(s3_url, saved_path, retries=RETRY_COUNT)
    return clip_id, ok, err, saved_path
