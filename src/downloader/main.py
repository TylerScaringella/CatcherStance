from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from playwright.sync_api import sync_playwright

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from downloader.config import DOWNLOAD_DIR, DOWNLOAD_WORKERS, MANIFEST_PATH, START_URL, STORAGE_STATE_PATH
from downloader.crawler import collect_s3_urls, ensure_grid_loaded, get_logged_in_context
from downloader.files import download_one_row
from downloader.manifest import load_manifest, write_manifest


def run_download_pipeline(
    start_url=START_URL,
    download_dir=DOWNLOAD_DIR,
    manifest_path=MANIFEST_PATH,
    storage_state_path=STORAGE_STATE_PATH,
    headless=False,
    download_workers=DOWNLOAD_WORKERS,
):
    os.makedirs(download_dir, exist_ok=True)
    rows, by_clip_id, by_s3_url = load_manifest(manifest_path)
    print(f"Loaded manifest rows: {len(rows)}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context, page, new_login = get_logged_in_context(
            browser,
            start_url=start_url,
            storage_state_path=storage_state_path,
        )

        if not new_login:
            print(f"Using saved Playwright session: {storage_state_path}")
            print(
                "If the session is expired, log in manually in the opened browser tab,\n"
                "then press Enter to refresh the saved state and continue."
            )
            try:
                ensure_grid_loaded(page)
            except Exception:
                input("Session may be expired. Log in manually and navigate to the pitch cards page, then press Enter...")
                context.storage_state(path=storage_state_path)
                print(f"Updated saved authenticated session: {storage_state_path}")
                ensure_grid_loaded(page)

        collected_this_run = collect_s3_urls(page, rows, by_clip_id, by_s3_url, download_dir=download_dir)
        write_manifest(manifest_path, rows)
        print(f"\nCollected new URLs this run: {collected_this_run}")
        print(f"Manifest updated: {manifest_path}")

        browser.close()

    pending_rows = []
    already_downloaded = 0

    for row in rows:
        saved_path = row["saved_path"]
        if saved_path and os.path.exists(saved_path):
            row["status"] = "downloaded"
            row["error"] = ""
            already_downloaded += 1
            continue

        if row.get("s3_url") and row.get("clip_id"):
            row["status"] = "pending"
            pending_rows.append(row)

    write_manifest(manifest_path, rows)

    print(f"\nAlready downloaded on disk: {already_downloaded}")
    print(f"Pending downloads: {len(pending_rows)}")

    if not pending_rows:
        print("Nothing to download.")
        print(f"Files saved in: {download_dir}")
        return rows

    print(f"\nStarting parallel downloads with {download_workers} workers...")

    completed = 0
    failed = 0
    row_lookup = {row["clip_id"]: row for row in rows if row.get("clip_id")}

    with ThreadPoolExecutor(max_workers=download_workers) as executor:
        futures = {executor.submit(download_one_row, row): row for row in pending_rows}

        for future in as_completed(futures):
            row = futures[future]
            clip_id = row["clip_id"]

            try:
                result_clip_id, ok, err, saved_path = future.result()
            except Exception as exc:
                ok = False
                err = str(exc)
                saved_path = row["saved_path"]
                result_clip_id = clip_id

            manifest_row = row_lookup[result_clip_id]
            attempts = int(manifest_row.get("attempts", "0") or "0") + 1
            manifest_row["attempts"] = str(attempts)

            if ok:
                manifest_row["status"] = "downloaded"
                manifest_row["error"] = ""
                completed += 1
                print(f"[OK]   {result_clip_id} -> {saved_path}")
            else:
                manifest_row["status"] = "failed"
                manifest_row["error"] = err
                failed += 1
                print(f"[FAIL] {result_clip_id} -> {err}")

            write_manifest(manifest_path, rows)

    print("\nDone.")
    print(f"Collected new URLs this run: {collected_this_run}")
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
