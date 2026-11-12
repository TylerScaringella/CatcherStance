from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from playwright.sync_api import sync_playwright
from werkzeug.serving import make_server

from backend import jobs, storage
from backend.app import create_app
from downloader.manifest import write_manifest
from project_paths import EXAMPLES_DIR


@unittest.skipUnless(
    os.environ.get("RUN_BROWSER_TESTS") == "1",
    "Set RUN_BROWSER_TESTS=1 after installing Playwright Chromium.",
)
class BrowserPollingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run_id = "browser-progress-run"
        self.run_dir = self.root / "runs" / self.run_id
        self.run_dir.mkdir(parents=True)
        self.stack = ExitStack()
        self.stack.enter_context(patch.object(storage, "RUNS_DIR", self.root / "runs"))
        self.stack.enter_context(patch.object(storage, "TEMP_DIR", self.root / "tmp"))
        self.stack.enter_context(patch.object(storage, "AUTH_DIR", self.root / "auth"))
        jobs.JOBS.clear()

        sample_video = next(
            (EXAMPLES_DIR / "duke-2026-04-21-liberty-sample" / "downloads").glob("*.mp4")
        )
        downloads = self.run_dir / "downloads"
        downloads.mkdir()
        self.video_one = downloads / "clip-1.mp4"
        self.video_two = downloads / "clip-2.mp4"
        shutil.copyfile(sample_video, self.video_one)
        shutil.copyfile(sample_video, self.video_two)
        self.manifest = [
            {"clip_id": "clip-1", "saved_path": str(self.video_one), "status": "downloaded"},
            {"clip_id": "clip-2", "saved_path": str(self.video_two), "status": "pending"},
        ]
        write_manifest(str(self.run_dir / "video_manifest.csv"), self.manifest)
        self.job = {
            "id": self.run_id,
            "game": {
                "id": "duke-2026-04-21-liberty",
                "date": "2026-04-21",
                "opponent": "at Liberty",
                "location": "Lynchburg, Va.",
                "conference": False,
                "result": "L 10-13",
            },
            "source": "live",
            "status": "detecting",
            "phase": "detecting",
            "message": "Processed 0 of 2 pitches",
            "created_at": time.time(),
            "updated_at": time.time(),
            "expected_result_count": 2,
            "progress": {
                "active_stage": "detecting",
                "phase": "detecting",
                "current": 0,
                "total": 2,
                "percent": 0,
                "stages": {
                    "downloading": {"status": "active", "current": 1, "total": 2, "percent": 50},
                    "detecting": {"status": "active", "current": 0, "total": 2, "percent": 0},
                },
            },
        }
        storage.atomic_write_json(self.run_dir / "job.json", self.job)

        app = create_app()
        self.server = make_server("127.0.0.1", 0, app, threaded=True)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=True)
        self.page = self.browser.new_page(viewport={"width": 1280, "height": 800})
        self.requests = []
        self.summary_statuses = []
        self.console_errors = []
        self.page.on("request", lambda request: self.requests.append(request.url))
        self.page.on(
            "console",
            lambda message: self.console_errors.append(message.text) if message.type == "error" else None,
        )
        self.page.on("pageerror", lambda error: self.console_errors.append(str(error)))
        self.page.on(
            "response",
            lambda response: self.summary_statuses.append(response.status)
            if "/api/runs?view=summary" in response.url
            else None,
        )

    def tearDown(self):
        self.browser.close()
        self.playwright.stop()
        self.server.shutdown()
        self.server_thread.join(timeout=2)
        jobs.JOBS.clear()
        self.stack.close()
        self.temporary.cleanup()

    def _publish_first_result(self):
        state_dir = self.run_dir / "state" / "pitches"
        state_dir.mkdir(parents=True)
        storage.atomic_write_json(
            state_dir / "clip-1.json",
            {
                "detection": {
                    "pitch_index": 1,
                    "clip_id": "clip-1",
                    "video_path": str(self.video_one),
                    "stance": "LKD",
                    "confidence": 0.91,
                    "status": "ok",
                    "accepted": True,
                    "fps": 30.0,
                    "window_start_seconds": 1.0,
                    "window_end_seconds": 1.8,
                },
                "feature": {"choice": "", "id": "clip-1", "features": "", "status": "ok"},
            },
        )
        self.manifest[1]["status"] = "downloaded"
        write_manifest(str(self.run_dir / "video_manifest.csv"), self.manifest)
        self.job["message"] = "Processed 1 of 2 pitches"
        self.job["updated_at"] = time.time()
        self.job["progress"]["current"] = 1
        self.job["progress"]["percent"] = 50
        self.job["progress"]["stages"]["downloading"] = {
            "status": "complete", "current": 2, "total": 2, "percent": 100,
        }
        self.job["progress"]["stages"]["detecting"] = {
            "status": "active", "current": 1, "total": 2, "percent": 50,
        }
        storage.atomic_write_json(self.run_dir / "job.json", self.job)

    def test_polling_patches_partial_results_without_rebuilding_the_view(self):
        self.page.goto(f"{self.base_url}/#/runs", wait_until="networkidle")
        self.page.evaluate("window.__appRoot = document.querySelector('#appContent')")
        action = self.page.locator(f'[data-run-action="{self.run_id}"]')
        self.assertIn("View game", action.inner_text())
        self.assertIn("Pause", action.inner_text())

        self._publish_first_result()
        action = self.page.locator(f'[data-run-action="{self.run_id}"]')
        action.filter(has_text="Open results").wait_for(timeout=7000)
        self.assertTrue(self.page.evaluate("window.__appRoot === document.querySelector('#appContent')"))
        self.assertIn("2/2 clips downloaded", self.page.locator(f'[data-download-count="{self.run_id}"]').inner_text())

        action.focus()
        self.job["message"] = "Processed 1 of 2 pitches; polling"
        self.job["updated_at"] = time.time()
        storage.atomic_write_json(self.run_dir / "job.json", self.job)
        self.page.wait_for_timeout(3500)
        self.assertTrue(action.evaluate("node => document.activeElement === node"))

        action.get_by_text("Open results").click()
        self.page.locator('[data-clip-id="clip-1"]').wait_for(timeout=5000)
        self.page.locator('[data-clip-id="clip-1"]').click()
        self.page.locator("#pitchDialog[open]").wait_for()
        self.assertTrue(self.page.get_by_text("Full pitch source").is_visible())
        self.assertTrue(self.page.get_by_text("Jump to delivery").is_visible())
        self.assertTrue(self.page.get_by_text("Compact review").is_visible())
        self.page.evaluate(
            "window.__dialog = document.querySelector('#pitchDialog'); "
            "window.__video = document.querySelector('#pitchDialog video')"
        )
        self.job["message"] = "Processed 1 of 2 pitches; review active"
        self.job["updated_at"] = time.time()
        storage.atomic_write_json(self.run_dir / "job.json", self.job)
        self.page.wait_for_timeout(6500)
        self.assertTrue(self.page.evaluate("window.__dialog === document.querySelector('#pitchDialog')"))
        self.assertTrue(self.page.evaluate("window.__video === document.querySelector('#pitchDialog video')"))

        self.page.locator("#pitchDialog").evaluate("dialog => dialog.close()")
        self.page.goto(f"{self.base_url}/#/games", wait_until="domcontentloaded")
        search = self.page.locator("#gameSearch")
        search.fill("Liberty")
        search.focus()
        self.page.locator("#trumediaButton").click()
        self.page.locator("#workflowDialog[open]").wait_for()
        self.page.evaluate("window.__gamesRoot = document.querySelector('#appContent')")
        self.job["updated_at"] = time.time()
        storage.atomic_write_json(self.run_dir / "job.json", self.job)
        self.page.wait_for_timeout(3500)
        self.assertTrue(self.page.evaluate("window.__gamesRoot === document.querySelector('#appContent')"))
        self.assertEqual("Liberty", search.input_value())
        self.assertTrue(self.page.locator("#workflowDialog").evaluate("dialog => dialog.open"))

        stable_endpoints = (
            "/api/schedule", "/api/teams/duke/seasons", "/api/integrations/trumedia/status",
        )
        for endpoint in stable_endpoints:
            self.assertEqual(1, sum(endpoint in url for url in self.requests), endpoint)
        self.assertGreaterEqual(sum("/api/runs?view=summary" in url for url in self.requests), 3)
        self.assertIn(304, self.summary_statuses)
        self.assertEqual([], self.console_errors)


if __name__ == "__main__":
    unittest.main()
