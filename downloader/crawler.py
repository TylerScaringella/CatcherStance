from __future__ import annotations

import hashlib
import os
import re
from urllib.parse import unquote, urlparse

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from .config import (
    CARD_SELECTOR,
    CLICK_TIMEOUT_MS,
    DOWNLOAD_DIR,
    GRID_SELECTOR,
    MAX_NEW_URLS_PER_RUN,
    MAX_NO_PROGRESS_SCROLLS,
    POST_CLICK_WAIT_MS,
    SCROLL_WAIT_MS,
    START_URL,
    STORAGE_STATE_PATH,
    VIEWPORT_SELECTOR,
)
from .manifest import upsert_manifest_row


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


def get_output_path(clip_id: str, download_dir: str = DOWNLOAD_DIR) -> str:
    return os.path.join(download_dir, f"{clip_id}.mp4")


def get_card_count(page) -> int:
    return page.locator(CARD_SELECTOR).count()


def click_card_and_capture_s3(page, card):
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
        }""",
    )
    page.wait_for_timeout(SCROLL_WAIT_MS)


def get_logged_in_context(browser, start_url=START_URL, storage_state_path=STORAGE_STATE_PATH):
    if os.path.exists(storage_state_path):
        context = browser.new_context(storage_state=storage_state_path, accept_downloads=False)
        page = context.new_page()
        page.goto(start_url, wait_until="domcontentloaded")
        return context, page, False

    context = browser.new_context(accept_downloads=False)
    page = context.new_page()
    page.goto(start_url, wait_until="domcontentloaded")

    input(
        "\nNo saved Playwright session found.\n"
        "Log in, complete email verification, navigate to the page with the pitch cards,\n"
        "then press Enter to save the session and continue..."
    )

    context.storage_state(path=storage_state_path)
    print(f"Saved authenticated session to: {storage_state_path}")
    return context, page, True


def ensure_grid_loaded(page):
    page.wait_for_selector(GRID_SELECTOR, timeout=30000)
    page.wait_for_selector(VIEWPORT_SELECTOR, timeout=30000)
    page.wait_for_selector(CARD_SELECTOR, timeout=30000)


def collect_s3_urls(page, rows, by_clip_id, by_s3_url, download_dir=DOWNLOAD_DIR):
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
            except Exception as exc:
                print(f"  Error clicking card {i}: {exc}")
                processed_card_indices.add(i)
                continue

            if not is_s3_mp4_url(s3_url):
                print("  Captured URL was not an S3 MP4; skipping.")
                processed_card_indices.add(i)
                continue

            clip_id = sanitize_filename(extract_clip_id(s3_url))
            out_path = get_output_path(clip_id, download_dir=download_dir)

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
