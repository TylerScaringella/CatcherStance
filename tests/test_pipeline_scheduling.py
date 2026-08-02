from __future__ import annotations

import csv
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from backend import jobs, storage
from backend.resources import ResourceScheduler
from stance_pipeline import runner
from stance_pipeline.schemas import PitchStanceResult


class ResourceSchedulerTests(unittest.TestCase):
    def test_inference_leases_are_fifo_and_exclusive(self):
        scheduler = ResourceScheduler(download_workers=2, review_workers=1)
        entered = []
        active = 0
        peak = 0
        guard = threading.Lock()
        release_first = threading.Event()

        def work(run_id):
            nonlocal active, peak
            with scheduler.inference_lease(run_id):
                with guard:
                    active += 1
                    peak = max(peak, active)
                    entered.append(run_id)
                if run_id == "run-1":
                    release_first.wait(timeout=2)
                time.sleep(0.01)
                with guard:
                    active -= 1

        first = threading.Thread(target=work, args=("run-1",))
        first.start()
        while scheduler.stats()["active_inference_run"] != "run-1":
            time.sleep(0.001)
        second = threading.Thread(target=work, args=("run-2",))
        third = threading.Thread(target=work, args=("run-3",))
        second.start()
        time.sleep(0.01)
        third.start()
        release_first.set()
        for thread in (first, second, third):
            thread.join(timeout=2)
        self.assertEqual(["run-1", "run-2", "run-3"], entered)
        self.assertEqual(1, peak)

    def test_download_slots_are_globally_bounded(self):
        scheduler = ResourceScheduler(download_workers=2, review_workers=1)
        threads = []

        def work():
            with scheduler.download_slot():
                time.sleep(0.02)

        for _ in range(8):
            thread = threading.Thread(target=work)
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join(timeout=2)
        self.assertEqual(2, scheduler.stats()["peak_downloads"])


class IncrementalPipelineTests(unittest.TestCase):
    @staticmethod
    def _result():
        return PitchStanceResult(
            label="LKD",
            confidence=0.9,
            impact_frame=30,
            window_start_frame=10,
            window_end_frame=20,
            vote_distribution={"LKD": 1.0},
            valid_frame_count=10,
            camera_quality=0.8,
            detector_provenance=["test"],
            quality_flags=[],
            diagnostics={"fps": 30.0},
        )

    def test_existing_run_persists_incrementally_and_resumes_without_duplicates(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            storage, "RUNS_DIR", Path(directory)
        ), patch.object(runner, "RUNS_DIR", Path(directory)), patch.object(
            runner, "build_review_for_result", return_value=Path(directory) / "review.mp4"
        ):
            run_dir = Path(directory) / "resume-run"
            downloads = run_dir / "downloads"
            downloads.mkdir(parents=True)
            video = downloads / "clip-1.mp4"
            video.write_bytes(b"video")
            with (run_dir / "video_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["clip_id", "saved_path", "status"])
                writer.writeheader()
                writer.writerow({"clip_id": "clip-1", "saved_path": str(video), "status": "downloaded"})

            calls = []

            class FakeAnalyzer:
                def __init__(self, _config):
                    pass

                def analyze(self, _path):
                    calls.append(_path)
                    return IncrementalPipelineTests._result()

            with patch.object(runner, "PitchStanceAnalyzer", FakeAnalyzer):
                first = runner.run_detection_for_existing_run("resume-run")
                second = runner.run_detection_for_existing_run("resume-run")

            self.assertEqual(1, len(calls))
            self.assertEqual(1, len(first))
            self.assertEqual(1, len(second))
            self.assertTrue(
                (run_dir / "state" / "pitches" / "clip-1.json").is_file()
            )
            self.assertTrue((run_dir / "detections.json").is_file())

    def test_stage_progress_does_not_regress_to_download_after_detection_starts(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            storage, "RUNS_DIR", Path(directory)
        ):
            run_id = "progress-run"
            jobs.JOBS[run_id] = {"id": run_id, "created_at": time.time()}
            jobs.set_job_progress(run_id, "Downloaded", 10, 10, phase="downloading")
            jobs.set_job_progress(run_id, "Detecting", 1, 10, phase="detecting")
            jobs.set_job_progress(run_id, "Downloaded", 10, 10, phase="downloading")
            current = jobs.JOBS[run_id]
            self.assertEqual("detecting", current["progress"]["active_stage"])
            self.assertEqual(10, current["progress"]["stages"]["downloading"]["current"])
            self.assertEqual(1, current["progress"]["stages"]["detecting"]["current"])
            jobs.JOBS.pop(run_id, None)

    def test_game_detection_starts_before_download_pipeline_finishes(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            storage, "RUNS_DIR", Path(directory)
        ), patch.object(runner, "RUNS_DIR", Path(directory)), patch.object(
            runner, "build_review_for_result", return_value=Path(directory) / "review.mp4"
        ):
            run_dir = Path(directory) / "stream-run"
            downloads = run_dir / "downloads"
            downloads.mkdir(parents=True)
            video = downloads / "clip-1.mp4"
            video.write_bytes(b"video")
            inference_started = threading.Event()
            downloader_returned = threading.Event()

            class FakeAnalyzer:
                def __init__(self, _config):
                    pass

                def analyze(self, _path):
                    inference_started.set()
                    self.assert_downloader_is_still_running()
                    return IncrementalPipelineTests._result()

                @staticmethod
                def assert_downloader_is_still_running():
                    if downloader_returned.is_set():
                        raise AssertionError("download pipeline returned before inference started")

            def fake_download_pipeline(**kwargs):
                kwargs["discovery_callback"]({
                    "selected_pitches": 1,
                    "downloadable_pitches": 1,
                })
                kwargs["clip_ready_callback"]({
                    "clip_id": "clip-1",
                    "saved_path": str(video),
                    "status": "downloaded",
                    "_analysis_index": 1,
                })
                self.assertTrue(inference_started.wait(timeout=2))
                downloader_returned.set()

            with patch.object(runner, "PitchStanceAnalyzer", FakeAnalyzer), patch.object(
                runner, "run_download_pipeline", side_effect=fake_download_pipeline
            ):
                rows = runner.run_game_detection("stream-run", "https://example.invalid")
            self.assertTrue(downloader_returned.is_set())
            self.assertEqual(1, len(rows))


if __name__ == "__main__":
    unittest.main()
