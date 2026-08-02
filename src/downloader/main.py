from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from playwright.sync_api import sync_playwright

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from downloader.config import DOWNLOAD_DIR, DOWNLOAD_WORKERS, MANIFEST_PATH, START_URL, STORAGE_STATE_PATH
from downloader.crawler import discover_pitch_media, get_logged_in_context, sign_media_ref
from downloader.files import download_one_row
from downloader.manifest import load_manifest, write_manifest


def run_download_pipeline(
    start_url=START_URL,
    download_dir=DOWNLOAD_DIR,
    manifest_path=MANIFEST_PATH,
    storage_state_path=STORAGE_STATE_PATH,
    headless=False,
    download_workers=DOWNLOAD_WORKERS,
    status_callback=None,
    discovery_callback=None,
    clip_ready_callback=None,
):
    os.makedirs(download_dir, exist_ok=True)
    rows, by_clip_id, by_media_ref = load_manifest(manifest_path)
    print(f"Loaded manifest rows: {len(rows)}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        try:
            _, page, _ = get_logged_in_context(
                browser,
                start_url=start_url,
                storage_state_path=storage_state_path,
            )
            collected_this_run, discovery = discover_pitch_media(
                page,
                start_url,
                rows,
                by_clip_id,
                by_media_ref,
                download_dir=download_dir,
            )
            write_manifest(manifest_path, rows)
            print(
                "Discovered "
                f"{discovery['total_pitches']} pitches; selected "
                f"{discovery['selected_pitches']} Duke-catching pitches."
            )
            if status_callback is not None:
                status_callback(
                    f"Discovered {discovery['selected_pitches']} Duke-catching pitches",
                    discovery["downloadable_pitches"],
                    discovery["selected_pitches"],
                )
            if discovery_callback is not None:
                discovery_callback(dict(discovery))

            pending_rows = []
            already_downloaded = 0
            analysis_indexes = {
                row.get("clip_id"): index
                for index, row in enumerate(rows, start=1)
                if row.get("clip_id")
            }
            for analysis_index, row in enumerate(rows, start=1):
                saved_path = row["saved_path"]
                if saved_path and os.path.exists(saved_path):
                    row["status"] = "downloaded"
                    row["error"] = ""
                    already_downloaded += 1
                    if clip_ready_callback is not None:
                        clip_ready_callback({**row, "_analysis_index": analysis_index})
                    continue
                if row.get("skip_reason"):
                    row["status"] = "skipped"
                    continue
                if row.get("media_ref") and row.get("clip_id"):
                    row["status"] = "pending"
                    pending_rows.append(row)

            write_manifest(manifest_path, rows)
            print(f"Already downloaded on disk: {already_downloaded}")
            print(f"Pending downloads: {len(pending_rows)}")

            completed = 0
            failed = 0
            row_lookup = {row["clip_id"]: row for row in rows if row.get("clip_id")}
            worker_count = max(1, int(download_workers))
            for offset in range(0, len(pending_rows), worker_count):
                batch = pending_rows[offset : offset + worker_count]
                signed_rows = []
                for row in batch:
                    try:
                        row["download_url"] = sign_media_ref(page, row["media_ref"])
                        signed_rows.append(row)
                    except Exception:
                        row["status"] = "failed"
                        row["error"] = "pitch video authorization failed"
                        row["attempts"] = str(int(row.get("attempts", "0") or "0") + 1)
                        failed += 1

                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    futures = {executor.submit(download_one_row, row): row for row in signed_rows}
                    for future in as_completed(futures):
                        row = futures[future]
                        clip_id = row["clip_id"]
                        try:
                            result_clip_id, ok, err, saved_path = future.result()
                        except Exception:
                            ok = False
                            err = "pitch video download failed"
                            saved_path = row["saved_path"]
                            result_clip_id = clip_id

                        manifest_row = row_lookup[result_clip_id]
                        manifest_row.pop("download_url", None)
                        attempts = int(manifest_row.get("attempts", "0") or "0") + 1
                        manifest_row["attempts"] = str(attempts)
                        if ok:
                            manifest_row["status"] = "downloaded"
                            manifest_row["error"] = ""
                            completed += 1
                            print(f"[OK]   {result_clip_id}")
                            if clip_ready_callback is not None:
                                clip_ready_callback({
                                    **manifest_row,
                                    "_analysis_index": analysis_indexes[result_clip_id],
                                })
                        else:
                            manifest_row["status"] = "failed"
                            manifest_row["error"] = "pitch video download failed"
                            failed += 1
                            print(f"[FAIL] {result_clip_id}")

                write_manifest(manifest_path, rows)
                processed = already_downloaded + completed + failed + discovery["skipped_pitches"]
                if status_callback is not None:
                    status_callback(
                        f"Downloading Duke-catching pitches: {processed} of {len(rows)}",
                        processed,
                        len(rows),
                    )
        finally:
            browser.close()

    print("\nDone.")
    print(f"Collected new pitch records this run: {collected_this_run}")
    print(f"Downloaded successfully this run: {completed}")
    print(f"Failed this run: {failed}")
    print(f"Manifest: {manifest_path}")
    print(f"Files saved in: {download_dir}")
    return rows


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Download pitch-by-pitch videos from TruMedia.")
    parser.add_argument("--start-url", default=START_URL)
    parser.add_argument("--download-dir", default=DOWNLOAD_DIR)
    parser.add_argument("--manifest-path", default=MANIFEST_PATH)
    parser.add_argument("--storage-state-path", default=STORAGE_STATE_PATH)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--workers", type=int, default=DOWNLOAD_WORKERS)
    return parser.parse_args()


def main():
    args = parse_args()
    run_download_pipeline(
        start_url=args.start_url,
        download_dir=args.download_dir,
        manifest_path=args.manifest_path,
        storage_state_path=args.storage_state_path,
        headless=args.headless,
        download_workers=args.workers,
    )


if __name__ == "__main__":
    main()
