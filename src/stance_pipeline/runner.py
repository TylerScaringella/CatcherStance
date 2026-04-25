from __future__ import annotations

from .config import PROJECT_ROOT, RUNS_DIR, StatusCallback
from .detect import detect_stances_for_manifest, write_detection_outputs
from downloader.main import run_download_pipeline


def run_game_detection(
    run_id: str,
    start_url: str,
    headless: bool = False,
    download_workers: int = 8,
    status_callback: StatusCallback | None = None,
) -> list[dict]:
    run_dir = RUNS_DIR / run_id
    download_dir = run_dir / "downloads"
    manifest_path = run_dir / "video_manifest.csv"
    storage_state_path = PROJECT_ROOT / "data" / "downloader" / "playwright_state.json"

    run_download_pipeline(
        start_url=start_url,
        download_dir=str(download_dir),
        manifest_path=str(manifest_path),
        storage_state_path=str(storage_state_path),
        headless=headless,
        download_workers=download_workers,
    )
    if status_callback is not None:
        status_callback("Running catcher detection and stance classifier", None, None)

    detections, feature_rows = detect_stances_for_manifest(
        manifest_path,
        status_callback=status_callback,
    )
    return write_detection_outputs(run_dir, detections, feature_rows)


def run_detection_for_existing_run(
    run_id: str,
    status_callback: StatusCallback | None = None,
) -> list[dict]:
    run_dir = RUNS_DIR / run_id
    manifest_path = run_dir / "video_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest found for run: {run_id}")

    if status_callback is not None:
        status_callback("Running catcher detection and stance classifier", None, None)

    detections, feature_rows = detect_stances_for_manifest(
        manifest_path,
        status_callback=status_callback,
    )
    return write_detection_outputs(run_dir, detections, feature_rows)
