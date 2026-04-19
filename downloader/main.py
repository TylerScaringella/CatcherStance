import os
import re
import csv
import time
import hashlib
import requests
from pathlib import Path
from urllib.parse import urlparse, unquote
from concurrent.futures import ThreadPoolExecutor, as_completed
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

# =========================
# CONFIG
# =========================
START_URL = "https://duke-ncaabaseball.trumedianetworks.com/baseball/"
DOWNLOAD_DIR = "downloads"
MANIFEST_PATH = "video_manifest.csv"
STORAGE_STATE_PATH = "playwright_state.json"

# Selectors
GRID_SELECTOR = "tmn-grid.contents-container"
VIEWPORT_SELECTOR = "tmn-grid.contents-container .viewport"
CARD_SELECTOR = "tmn-grid.contents-container tmn-paper"

# Crawl tuning
CLICK_TIMEOUT_MS = 5000
POST_CLICK_WAIT_MS = 150
SCROLL_WAIT_MS = 250
MAX_NO_PROGRESS_SCROLLS = 4

# Download tuning
REQUEST_TIMEOUT_SECONDS = 180
DOWNLOAD_WORKERS = 8
CHUNK_SIZE = 1024 * 1024
RETRY_COUNT = 3
RETRY_SLEEP_SECONDS = 2

# Optional safety cap; set to None for no cap
MAX_NEW_URLS_PER_RUN = None

os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# =========================
# HELPERS
# =========================
def is_s3_mp4_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        path = (parsed.path or "").lower()

        is_amazon_s3 = (
            host == "s3.amazonaws.com"
            or host.endswith(".s3.amazonaws.com")
            or (".s3." in host and "amazonaws.com" in host)
        )

        return (
            parsed.scheme in ("http", "https")
            and is_amazon_s3
            and path.endswith(".mp4")
        )
    except Exception:
        return False


def extract_clip_id(url: str) -> str:
    try:
        path = unquote(urlparse(url).path)
        filename = os.path.basename(path)
        if filename.lower().endswith(".mp4"):
            return filename[:-4]
        return filename
    except Exception:
        return hashlib.md5(url.encode()).hexdigest()


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[^\w.\-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name[:180] if name else "clip"


def ensure_parent_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def get_output_path(clip_id: str) -> str:
    return os.path.join(DOWNLOAD_DIR, f"{clip_id}.mp4")


def get_card_count(page) -> int:
    return page.locator(CARD_SELECTOR).count()


def click_card_and_capture_s3(page, card):
    """
    Click one card and wait for a NEW S3 mp4 response triggered by that click.
    """
    with page.expect_response(lambda resp: is_s3_mp4_url(resp.url), timeout=CLICK_TIMEOUT_MS) as response_info:
        card.click(force=True)
    response = response_info.value
    page.wait_for_timeout(POST_CLICK_WAIT_MS)
    return response.url


def scroll_viewport(page):
    page.eval_on_selector(
        VIEWPORT_SELECTOR,
        """el => {
            el.scrollTop = el.scrollTop + Math.floor(el.clientHeight * 0.85);
        }"""
    )
    page.wait_for_timeout(SCROLL_WAIT_MS)


# =========================
# MANIFEST
# =========================
MANIFEST_FIELDS = [
    "card_dom_index",
    "clip_id",
    "s3_url",
    "saved_path",
    "status",
    "attempts",
    "error",
]


def load_manifest(path: str):
    rows = []
    by_clip_id = {}
    by_s3_url = {}

    if os.path.exists(path):
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                normalized = {
                    "card_dom_index": row.get("card_dom_index", ""),
                    "clip_id": row.get("clip_id", ""),
                    "s3_url": row.get("s3_url", ""),
                    "saved_path": row.get("saved_path", ""),
                    "status": row.get("status", ""),
                    "attempts": row.get("attempts", "0"),
                    "error": row.get("error", ""),
                }
                rows.append(normalized)
                if normalized["clip_id"]:
                    by_clip_id[normalized["clip_id"]] = normalized
                if normalized["s3_url"]:
                    by_s3_url[normalized["s3_url"]] = normalized

    return rows, by_clip_id, by_s3_url


def write_manifest(path: str, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def upsert_manifest_row(rows, by_clip_id, by_s3_url, row):
    clip_id = row["clip_id"]
    s3_url = row["s3_url"]

    existing = None
    if clip_id in by_clip_id:
        existing = by_clip_id[clip_id]
    elif s3_url in by_s3_url:
        existing = by_s3_url[s3_url]

    if existing is None:
        rows.append(row)
        by_clip_id[clip_id] = row
        by_s3_url[s3_url] = row
        return row

    existing.update(row)
    by_clip_id[clip_id] = existing
    by_s3_url[s3_url] = existing
    return existing


# =========================
# DOWNLOAD
# =========================
def download_file_once(url: str, filepath: str):
    ensure_parent_dir(filepath)
    with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT_SECONDS) as r:
        r.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)


def download_with_retries(url: str, filepath: str, retries: int = RETRY_COUNT):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            download_file_once(url, filepath)
            return True, ""
        except Exception as e:
            last_err = str(e)
            # Clean up partial file if download failed
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


# =========================
# CRAWL
# =========================
def get_logged_in_context(browser, headless=False):
    """
    Reuse saved session if possible. If not, prompt for manual login and save it.
    """
    if os.path.exists(STORAGE_STATE_PATH):
        context = browser.new_context(storage_state=STORAGE_STATE_PATH, accept_downloads=False)
        page = context.new_page()
        page.goto(START_URL, wait_until="domcontentloaded")
        return context, page, False

    context = browser.new_context(accept_downloads=False)
    page = context.new_page()
    page.goto(START_URL, wait_until="domcontentloaded")

    input(
        "\nNo saved Playwright session found.\n"
        "Log in, complete email verification, navigate to the page with the pitch cards,\n"
        "then press Enter to save the session and continue..."
    )

    context.storage_state(path=STORAGE_STATE_PATH)
    print(f"Saved authenticated session to: {STORAGE_STATE_PATH}")
    return context, page, True


def ensure_grid_loaded(page):
    page.wait_for_selector(GRID_SELECTOR, timeout=30000)
    page.wait_for_selector(VIEWPORT_SELECTOR, timeout=30000)
    page.wait_for_selector(CARD_SELECTOR, timeout=30000)


def collect_s3_urls(page, rows, by_clip_id, by_s3_url):
    """
    Crawl the loaded TruMedia card grid, click cards, capture S3 mp4 URLs,
    and write/update manifest rows as pending.
    """
    ensure_grid_loaded(page)

    print("Grid located.")
    print(f"Initial visible card count: {get_card_count(page)}")

    processed_card_indices = set()
    no_progress_scrolls = 0
    collected_this_run = 0

    while True:
        current_count = get_card_count(page)
        print(f"\nCurrent DOM card count: {current_count}")

        progress_this_pass = False

        for i in range(current_count):
            if i in processed_card_indices:
                continue

            if MAX_NEW_URLS_PER_RUN is not None and collected_this_run >= MAX_NEW_URLS_PER_RUN:
                print(f"Reached MAX_NEW_URLS_PER_RUN={MAX_NEW_URLS_PER_RUN}, stopping crawl early.")
                return collected_this_run

            card = page.locator(CARD_SELECTOR).nth(i)

            try:
                card.scroll_into_view_if_needed()
                page.wait_for_timeout(300)
            except Exception:
                pass

            print(f"Processing card DOM index: {i}")

            try:
                s3_url = click_card_and_capture_s3(page, card)
            except PlaywrightTimeoutError:
                print(f"  No S3 mp4 captured for card {i}; marking processed and continuing.")
                processed_card_indices.add(i)
                continue
            except Exception as e:
                print(f"  Error clicking card {i}: {e}")
                processed_card_indices.add(i)
                continue

            if not is_s3_mp4_url(s3_url):
                print("  Captured URL was not an S3 MP4; skipping.")
                processed_card_indices.add(i)
                continue

            clip_id = sanitize_filename(extract_clip_id(s3_url))
            out_path = get_output_path(clip_id)

            existing = by_clip_id.get(clip_id) or by_s3_url.get(s3_url)
            if existing is not None:
                print(f"  Already in manifest: {clip_id}")
                processed_card_indices.add(i)
                continue

            row = {
                "card_dom_index": str(i),
                "clip_id": clip_id,
                "s3_url": s3_url,
                "saved_path": out_path,
                "status": "pending" if not os.path.exists(out_path) else "downloaded",
                "attempts": "0",
                "error": "",
            }

            upsert_manifest_row(rows, by_clip_id, by_s3_url, row)
            collected_this_run += 1
            progress_this_pass = True

            print(f"  Collected URL for: {clip_id}")

            processed_card_indices.add(i)

        before_count = get_card_count(page)
        scroll_viewport(page)
        after_count = get_card_count(page)

        if after_count > before_count:
            print(f"Loaded more cards after scroll: {before_count} -> {after_count}")
            no_progress_scrolls = 0
            continue

        if progress_this_pass:
            print("No new DOM cards from scroll, but we did collect URLs this pass. Trying another scroll.")
            no_progress_scrolls = 0
            continue

        no_progress_scrolls += 1
        print(f"No progress scroll count: {no_progress_scrolls}/{MAX_NO_PROGRESS_SCROLLS}")

        if no_progress_scrolls >= MAX_NO_PROGRESS_SCROLLS:
            print("Stopping crawl: no more new cards/URLs found.")
            break

    return collected_this_run


# =========================
# MAIN
# =========================
def main():
    rows, by_clip_id, by_s3_url = load_manifest(MANIFEST_PATH)
    print(f"Loaded manifest rows: {len(rows)}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context, page, new_login = get_logged_in_context(browser)

        if not new_login:
            print(f"Using saved Playwright session: {STORAGE_STATE_PATH}")
            print(
                "If the session is expired, log in manually in the opened browser tab,\n"
                "then press Enter to refresh the saved state and continue."
            )
            try:
                ensure_grid_loaded(page)
            except Exception:
                input("Session may be expired. Log in manually and navigate to the pitch cards page, then press Enter...")
                context.storage_state(path=STORAGE_STATE_PATH)
                print(f"Updated saved authenticated session: {STORAGE_STATE_PATH}")
                ensure_grid_loaded(page)

        collected_this_run = collect_s3_urls(page, rows, by_clip_id, by_s3_url)
        write_manifest(MANIFEST_PATH, rows)
        print(f"\nCollected new URLs this run: {collected_this_run}")
        print(f"Manifest updated: {MANIFEST_PATH}")

        browser.close()

    # Build list of pending downloads
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

    write_manifest(MANIFEST_PATH, rows)

    print(f"\nAlready downloaded on disk: {already_downloaded}")
    print(f"Pending downloads: {len(pending_rows)}")

    if not pending_rows:
        print("Nothing to download.")
        print(f"Files saved in: {DOWNLOAD_DIR}")
        return

    print(f"\nStarting parallel downloads with {DOWNLOAD_WORKERS} workers...")

    completed = 0
    failed = 0

    row_lookup = {row["clip_id"]: row for row in rows if row.get("clip_id")}

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        futures = {executor.submit(download_one_row, row): row for row in pending_rows}

        for future in as_completed(futures):
            row = futures[future]
            clip_id = row["clip_id"]

            try:
                result_clip_id, ok, err, saved_path = future.result()
            except Exception as e:
                ok = False
                err = str(e)
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

            write_manifest(MANIFEST_PATH, rows)

    print("\nDone.")
    print(f"Collected new URLs this run: {collected_this_run}")
    print(f"Downloaded successfully this run: {completed}")
    print(f"Failed this run: {failed}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Files saved in: {DOWNLOAD_DIR}")


if __name__ == "__main__":
    main()