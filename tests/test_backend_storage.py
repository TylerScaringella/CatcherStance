from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
import subprocess
from pathlib import Path
from unittest.mock import patch

from backend.app import create_app
from backend import jobs, storage


class StorageSecurityTests(unittest.TestCase):
    def test_identifiers_reject_paths(self):
        for value in ("../run", "/tmp/run", "run/clip", "", "a" * 201):
            with self.subTest(value=value), self.assertRaises(ValueError):
                storage.validate_identifier(value)

    def test_manifest_media_must_stay_inside_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            run_dir.mkdir()
            location = storage.RunLocation("run", run_dir, "live", False)
            with self.assertRaises(ValueError):
                storage.resolve_manifest_media(location, str(root / "outside.mp4"))

    def test_symlink_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            run_dir.mkdir()
            outside = root / "outside.mp4"
            outside.touch()
            link = run_dir / "clip.mp4"
            try:
                link.symlink_to(outside)
            except OSError:
                self.skipTest("symlinks unavailable")
            location = storage.RunLocation("run", run_dir, "live", False)
            with self.assertRaises(ValueError):
                storage.resolve_manifest_media(location, str(link))

    def test_atomic_write_preserves_existing_file_on_replace_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text('{"valid": true}\n', encoding="utf-8")
            with patch("backend.storage.os.replace", side_effect=OSError("interrupted")):
                with self.assertRaises(OSError):
                    storage.atomic_write_json(path, {"valid": False})
            self.assertEqual('{"valid": true}\n', path.read_text(encoding="utf-8"))
            self.assertEqual([], list(path.parent.glob("*.part")))

    def test_job_temp_directory_is_removed_after_failure(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            storage, "TEMP_DIR", Path(directory)
        ):
            with self.assertRaises(RuntimeError):
                with storage.job_temp_dir("safe-run") as path:
                    (path / "frame.jpg").touch()
                    raise RuntimeError("test")
            self.assertEqual([], list(Path(directory).iterdir()))

    def test_stale_temp_cleanup_leaves_recent_files(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            storage, "TEMP_DIR", Path(directory)
        ):
            stale = Path(directory) / "stale"
            recent = Path(directory) / "recent"
            stale.mkdir()
            recent.mkdir()
            old = time.time() - 100
            os.utime(stale, (old, old))
            removed = storage.cleanup_stale_temp(max_age_seconds=50)
            self.assertEqual([stale], removed)
            self.assertFalse(stale.exists())
            self.assertTrue(recent.exists())

    def test_concurrent_job_state_writes_remain_valid_json(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            storage, "RUNS_DIR", Path(directory)
        ):
            jobs.JOBS["concurrent-run"] = {
                "id": "concurrent-run",
                "status": "queued",
                "created_at": time.time(),
            }
            threads = [
                threading.Thread(target=jobs.set_job, args=("concurrent-run",), kwargs={"sequence": index})
                for index in range(20)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            payload = json.loads(
                (Path(directory) / "concurrent-run" / "job.json").read_text(encoding="utf-8")
            )
            self.assertIn(payload["sequence"], range(20))
            jobs.JOBS.pop("concurrent-run", None)

    def test_runtime_storage_paths_are_gitignored(self):
        for path in (
            "data/runs/check/job.json",
            "data/tmp/check/frame.jpg",
            "data/cache/duke_baseball_2026.json",
            "data/auth/playwright_state.json",
            "models/external/model.pt",
            "research/outputs/generated/report.json",
        ):
            with self.subTest(path=path):
                result = subprocess.run(
                    ["git", "check-ignore", "-q", "--no-index", path],
                    check=False,
                )
                self.assertEqual(0, result.returncode)

    def test_pitch_state_paths_are_contained_and_validated(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            storage, "RUNS_DIR", Path(directory)
        ):
            path = storage.pitch_state_file("safe-run", "clip-1", create_parent=True)
            self.assertEqual(
                (Path(directory) / "safe-run" / "state" / "pitches" / "clip-1.json").resolve(),
                path,
            )
            with self.assertRaises(ValueError):
                storage.pitch_state_file("safe-run", "../clip", create_parent=True)


class RunApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = create_app().test_client()

    def test_result_numeric_fields_are_normalized(self):
        result = jobs.normalize_result({"confidence": "0.75", "valid_frame_count": "12"})
        self.assertEqual(0.75, result["confidence"])
        self.assertEqual(12, result["valid_frame_count"])

    def test_example_run_is_discoverable_with_correct_results(self):
        response = self.client.get("/api/runs")
        self.assertEqual(200, response.status_code)
        sample = next(
            run
            for run in response.get_json()["runs"]
            if run["id"] == "duke-2026-04-21-liberty-sample"
        )
        self.assertTrue(sample["read_only"])
        self.assertEqual("example", sample["source"])
        self.assertEqual(
            ["LKD", "LKD", "RKD", "LKD", "LKD"],
            [row["stance"] for row in sample["results"]],
        )

    def test_run_summary_is_lightweight_and_conditional(self):
        response = self.client.get("/api/runs?view=summary")
        self.assertEqual(200, response.status_code)
        sample = next(
            run
            for run in response.get_json()["runs"]
            if run["id"] == "duke-2026-04-21-liberty-sample"
        )
        self.assertNotIn("results", sample)
        self.assertEqual(5, sample["result_count"])
        etag = response.headers["ETag"]
        unchanged = self.client.get("/api/runs?view=summary", headers={"If-None-Match": etag})
        self.assertEqual(304, unchanged.status_code)

    def test_detecting_progress_is_not_replaced_by_download_completion(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            storage, "RUNS_DIR", Path(directory)
        ), patch.object(jobs, "RUNS_DIR", Path(directory), create=True):
            run_dir = Path(directory) / "progress-run"
            run_dir.mkdir(parents=True)
            (run_dir / "downloads").mkdir()
            video_path = run_dir / "downloads" / "clip-1.mp4"
            video_path.write_bytes(b"video")
            (run_dir / "video_manifest.csv").write_text(
                f"clip_id,saved_path,status\nclip-1,{video_path},downloaded\n",
                encoding="utf-8",
            )
            payload = {
                "id": "progress-run",
                "game": {"id": "game-1", "opponent": "Test", "date": "2026-01-01"},
                "status": "detecting",
                "phase": "detecting",
                "created_at": time.time(),
                "updated_at": time.time(),
                "progress": {
                    "active_stage": "detecting",
                    "phase": "detecting",
                    "current": 0,
                    "total": 1,
                    "percent": 0,
                    "stages": {},
                },
            }
            (run_dir / "job.json").write_text(json.dumps(payload), encoding="utf-8")
            location = storage.RunLocation("progress-run", run_dir, "live", False)
            hydrated = jobs.job_from_location(location)
            self.assertEqual("detecting", hydrated["phase"])
            self.assertEqual(0, hydrated["progress"]["percent"])

    def test_stale_embedded_job_results_are_not_returned(self):
        response = self.client.get("/api/runs/duke-2026-04-21-liberty-sample")
        self.assertEqual(200, response.status_code)
        self.assertEqual("LKD", response.get_json()["results"][0]["stance"])

    def test_example_export_and_video_are_available(self):
        export = self.client.get(
            "/api/results/duke-2026-04-21-liberty-sample/json"
        )
        self.assertEqual(200, export.status_code)
        export.close()
        video = self.client.get(
            "/api/runs/duke-2026-04-21-liberty-sample/clips/"
            "pitch-pix-69e939e0ee33ff11241e0939-356-369/video",
            headers={"Range": "bytes=0-99"},
        )
        self.assertEqual(206, video.status_code)
        self.assertEqual(100, len(video.data))
        video.close()

    def test_invalid_identifier_does_not_expose_files(self):
        response = self.client.get("/api/runs/%2E%2E")
        self.assertIn(response.status_code, {400, 404})
        self.assertNotIn(b"Users/tylerscaringella", response.data)

    def test_opening_sample_run_does_not_modify_fixture(self):
        job_path = Path(
            "data/examples/duke-2026-04-21-liberty-sample/job.json"
        )
        before = job_path.read_bytes()
        response = self.client.post(
            "/api/run",
            json={
                "game_id": "duke-2026-04-21-liberty",
                "trumedia_url": "https://example.com",
            },
        )
        self.assertEqual(200, response.status_code)
        self.assertEqual(before, job_path.read_bytes())

    def test_live_overlay_is_atomically_cached_in_run_artifacts(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            storage, "RUNS_DIR", Path(directory)
        ):
            run_dir = Path(directory) / "safe-live-run"
            downloads = run_dir / "downloads"
            downloads.mkdir(parents=True)
            video = downloads / "clip-1.mp4"
            video.write_bytes(b"video")
            (run_dir / "video_manifest.csv").write_text(
                "clip_id,saved_path,status\n"
                f"clip-1,{video},downloaded\n",
                encoding="utf-8",
            )
            client = create_app().test_client()
            chunks = [b"--frame\r\nContent-Type: image/jpeg\r\n\r\nabc\r\n"]
            with patch("pipeline.overlay_mjpeg_frames", return_value=iter(chunks)):
                response = client.get(
                    "/api/runs/safe-live-run/clips/clip-1/overlay.mjpg"
                )
                self.assertEqual(200, response.status_code)
                self.assertEqual(b"".join(chunks), response.data)
                response.close()
            cached = list((run_dir / "artifacts").glob("overlay-*.mjpg"))
            self.assertEqual(1, len(cached))
            self.assertEqual(b"".join(chunks), cached[0].read_bytes())


if __name__ == "__main__":
    unittest.main()
