from __future__ import annotations

import argparse
from pathlib import Path

from backend.config import TRUMEDIA_DEFAULT_URL
from backend.storage import atomic_write_json
from project_paths import AUTH_DIR


def export_session(output: Path) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        try:
            context = browser.new_context()
            page = context.new_page()
            page.goto(TRUMEDIA_DEFAULT_URL, wait_until="domcontentloaded")
            input(
                "Log in to TruMedia in the opened browser and finish any verification.\n"
                "After the baseball workspace loads, return here and press Enter to export the session: "
            )
            atomic_write_json(output, context.storage_state())
            output.chmod(0o600)
        finally:
            browser.close()
    print(f"Playwright session exported to {output}")
    print("Treat this file as a bearer credential. Upload it through the protected app screen, then remove transfers.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a TruMedia Playwright session for headless deployment.")
    parser.add_argument(
        "--output",
        type=Path,
        default=AUTH_DIR / "playwright_state.export.json",
        help="Ignored output path for the exported Playwright storage-state JSON.",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_session(args.output.resolve())


if __name__ == "__main__":
    main()
