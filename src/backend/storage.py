from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from project_paths import AUTH_DIR, EXAMPLES_DIR, PLAYWRIGHT_STATE_PATH, ROOT, RUNS_DIR, TEMP_DIR

IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
STALE_TEMP_SECONDS = 24 * 60 * 60
_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True)
class RunLocation:
    run_id: str
    path: Path
    source: str
    read_only: bool


def validate_identifier(value: str, kind: str = "identifier") -> str:
    if not isinstance(value, str) or not IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"invalid {kind}")
    return value


def _contained(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def ensure_contained(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    resolved_root = root.resolve()
    if not _contained(resolved, resolved_root):
        raise ValueError("path is outside the allowed storage root")
    return resolved


def initialize_storage() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    AUTH_DIR.mkdir(parents=True, exist_ok=True)
    AUTH_DIR.chmod(0o700)
    if PLAYWRIGHT_STATE_PATH.exists():
        PLAYWRIGHT_STATE_PATH.chmod(0o600)
    cleanup_stale_temp()


def cleanup_stale_temp(max_age_seconds: int = STALE_TEMP_SECONDS) -> list[Path]:
    if not TEMP_DIR.exists():
        return []
    cutoff = time.time() - max_age_seconds
    removed: list[Path] = []
    for candidate in TEMP_DIR.iterdir():
        try:
            if candidate.stat().st_mtime >= cutoff:
                continue
            if candidate.is_dir():
                shutil.rmtree(candidate)
            else:
                candidate.unlink()
            removed.append(candidate)
        except FileNotFoundError:
            continue
    return removed


@contextmanager
def job_temp_dir(run_id: str) -> Iterator[Path]:
    validate_identifier(run_id, "run id")
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{run_id}-", dir=TEMP_DIR) as directory:
        yield Path(directory)


def live_run(run_id: str, create: bool = False) -> RunLocation:
    validate_identifier(run_id, "run id")
    path = ensure_contained(RUNS_DIR / run_id, RUNS_DIR)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return RunLocation(run_id, path, "live", False)


def resolve_run(run_id: str) -> RunLocation | None:
    validate_identifier(run_id, "run id")
    live = ensure_contained(RUNS_DIR / run_id, RUNS_DIR)
    if live.is_dir():
        return RunLocation(run_id, live, "live", False)
    example = ensure_contained(EXAMPLES_DIR / run_id, EXAMPLES_DIR)
    if example.is_dir():
        return RunLocation(run_id, example, "example", True)
    return None


def list_run_locations() -> list[RunLocation]:
    locations: list[RunLocation] = []
    seen: set[str] = set()
    for root, source, read_only in (
        (RUNS_DIR, "live", False),
        (EXAMPLES_DIR, "example", True),
    ):
        if not root.exists():
            continue
        for path in root.iterdir():
            if not path.is_dir() or not IDENTIFIER_PATTERN.fullmatch(path.name):
                continue
            if path.name in seen:
                continue
            seen.add(path.name)
            locations.append(RunLocation(path.name, path.resolve(), source, read_only))
    return locations


def resolve_manifest_media(location: RunLocation, saved_path: str) -> Path:
    if not saved_path:
        raise ValueError("manifest media path is empty")
    candidate = Path(saved_path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return ensure_contained(candidate, location.path)


def run_file(location: RunLocation, filename: str) -> Path:
    validate_identifier(filename, "filename")
    return ensure_contained(location.path / filename, location.path)


def run_state_dir(run_id: str, create: bool = False) -> Path:
    location = live_run(run_id, create=create)
    path = ensure_contained(location.path / "state", location.path)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def pitch_state_dir(run_id: str, create: bool = False) -> Path:
    path = ensure_contained(run_state_dir(run_id, create=create) / "pitches", live_run(run_id).path)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def pitch_state_file(run_id: str, clip_id: str, create_parent: bool = False) -> Path:
    validate_identifier(clip_id, "clip id")
    directory = pitch_state_dir(run_id, create=create_parent)
    return ensure_contained(directory / f"{clip_id}.json", live_run(run_id).path)


def run_lock(run_id: str) -> threading.RLock:
    validate_identifier(run_id, "run id")
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(run_id, threading.RLock())


def atomic_write_text(path: Path, content: str, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".part",
            delete=False,
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        if mode is not None:
            temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: object) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2) + "\n")
