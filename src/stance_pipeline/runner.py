from __future__ import annotations

import queue
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from backend.media_lifecycle import build_review_for_result
from backend.resources import SCHEDULER
from downloader.main import run_download_pipeline
from project_paths import PLAYWRIGHT_STATE_PATH

from .analyzer import PitchStanceAnalyzer, PitchStanceConfig
from .config import RUNS_DIR, StatusCallback
from .detect import (
    detect_stance_for_row,
    downloaded_video_rows,
    load_pitch_states,
    persist_pitch_state,
    write_detection_outputs,
)


ProgressCallback = Callable[[str, str, int | None, int | None, bool], None]


def _emit(
    phase: str,
    message: str,
    current: int | None,
    total: int | None,
    status_callback: StatusCallback | None,
    progress_callback: ProgressCallback | None,
    activate: bool = True,
) -> None:
    if progress_callback is not None:
        progress_callback(phase, message, current, total, activate)
    if status_callback is not None and activate:
        status_callback(message, current, total)


def _process_ready_rows(
    run_id: str,
    ready_rows,
    expected_total: Callable[[], int],
    status_callback: StatusCallback | None,
    progress_callback: ProgressCallback | None,
) -> list[dict]:
    existing_detections, _ = load_pitch_states(run_id)
    completed_ids = {item.clip_id for item in existing_detections}
    processed = len(completed_ids)
    review_completed = 0
    review_lock = threading.Lock()
    review_futures = []

    def submit_review(detection) -> None:
        nonlocal review_completed
        future = SCHEDULER.submit_review(
            build_review_for_result,
            run_id,
            asdict(detection),
        )

        def finished(_future) -> None:
            nonlocal review_completed
            try:
                _future.result()
            except Exception:
                return
            with review_lock:
                review_completed += 1
                current = review_completed
            _emit(
                "building_review",
                f"Built {current} compact reviews",
                current,
                expected_total(),
                status_callback,
                progress_callback,
                activate=False,
            )

        future.add_done_callback(finished)
        review_futures.append(future)

    for detection in existing_detections:
        submit_review(detection)

    with SCHEDULER.inference_lease(run_id):
        analyzer = PitchStanceAnalyzer(PitchStanceConfig())
        _emit(
            "detecting",
            "Running catcher detection and stance classifier",
            processed,
            expected_total() or None,
            status_callback,
            progress_callback,
        )
        while True:
            row = ready_rows.get()
            if row is None:
                break
            clip_id = str(row.get("clip_id") or "")
            if not clip_id or clip_id in completed_ids:
                continue
            pitch_index = int(row.get("_analysis_index") or processed + 1)
            detection, feature = detect_stance_for_row(row, pitch_index, analyzer)
            persist_pitch_state(run_id, detection, feature)
            completed_ids.add(clip_id)
            processed += 1
            submit_review(detection)
            _emit(
                "detecting",
                f"Processed {processed} of {expected_total()} pitches",
                processed,
                expected_total(),
                status_callback,
                progress_callback,
            )

    _emit(
        "building_review",
        "Finalizing compact review clips",
        review_completed,
        expected_total(),
        status_callback,
        progress_callback,
    )
    for future in review_futures:
        try:
            future.result()
        except Exception:
            # Final media validation retries missing reviews and preserves sources on failure.
            pass

    detections, feature_rows = load_pitch_states(run_id)
    return write_detection_outputs(RUNS_DIR / run_id, detections, feature_rows)


def run_game_detection(
    run_id: str,
    start_url: str,
    headless: bool = True,
    download_workers: int = 8,
    status_callback: StatusCallback | None = None,
    discovery_callback=None,
    progress_callback: ProgressCallback | None = None,
) -> list[dict]:
    run_dir = RUNS_DIR / run_id
    download_dir = run_dir / "downloads"
    manifest_path = run_dir / "video_manifest.csv"
    ready_rows: queue.Queue = queue.Queue()
    producer_errors: list[BaseException] = []
    totals = {"expected": 0}

    def on_discovery(stats: dict) -> None:
        totals["expected"] = int(stats.get("downloadable_pitches") or 0)
        selected = int(stats.get("selected_pitches") or 0)
        _emit(
            "discovering_pitches",
            f"Discovered {selected} Duke-catching pitches",
            selected,
            selected,
            status_callback,
            progress_callback,
        )
        _emit(
            "downloading",
            "Preparing pitch video downloads",
            0,
            selected,
            status_callback,
            progress_callback,
        )
        if discovery_callback is not None:
            discovery_callback(stats)

    def download_status(message, current, total) -> None:
        if message.startswith("Discovered "):
            return
        _emit(
            "downloading",
            message,
            current,
            total,
            status_callback,
            progress_callback,
        )

    def produce() -> None:
        try:
            run_download_pipeline(
                start_url=start_url,
                download_dir=str(download_dir),
                manifest_path=str(manifest_path),
                storage_state_path=str(PLAYWRIGHT_STATE_PATH),
                headless=headless,
                download_workers=download_workers,
                status_callback=download_status,
                discovery_callback=on_discovery,
                clip_ready_callback=ready_rows.put,
            )
        except BaseException as exc:
            producer_errors.append(exc)
        finally:
            ready_rows.put(None)

    _emit(
        "discovering_pitches",
        "Discovering TruMedia pitch videos",
        None,
        None,
        status_callback,
        progress_callback,
    )
    producer = threading.Thread(target=produce, name=f"download-{run_id}", daemon=True)
    producer.start()
    rows = _process_ready_rows(
        run_id,
        ready_rows,
        lambda: totals["expected"],
        status_callback,
        progress_callback,
    )
    producer.join()
    if producer_errors:
        raise producer_errors[0]
    return rows


def run_detection_for_existing_run(
    run_id: str,
    status_callback: StatusCallback | None = None,
    progress_callback: ProgressCallback | None = None,
) -> list[dict]:
    run_dir = RUNS_DIR / run_id
    manifest_path = run_dir / "video_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest found for run: {run_id}")
    rows = downloaded_video_rows(manifest_path)
    ready_rows: queue.Queue = queue.Queue()
    for index, row in enumerate(rows, start=1):
        ready_rows.put({**row, "_analysis_index": index})
    ready_rows.put(None)
    return _process_ready_rows(
        run_id,
        ready_rows,
        lambda: len(rows),
        status_callback,
        progress_callback,
    )
