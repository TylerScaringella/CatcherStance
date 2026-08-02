from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend import storage
from backend.media_lifecycle import finalize_run_media
from downloader.manifest import load_manifest, write_manifest


def _fake_encode(source: Path, destination: Path, start: float, duration: float) -> None:
    destination.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"0" * 64)


class MediaLifecycleTests(unittest.TestCase):
    def _make_run(self, root: Path) -> tuple[Path, list[dict]]:
        run = root / "test-run"
        downloads = run / "downloads"
        downloads.mkdir(parents=True)
        video = downloads / "clip-1.mp4"
        video.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"1" * 128)
        rows = [{
            "card_dom_index": "0", "clip_id": "clip-1", "s3_url": "https://example.com/video.mp4",
            "saved_path": str(video), "status": "downloaded", "attempts": "1", "error": "",
        }]
        write_manifest(str(run / "video_manifest.csv"), rows)
        return run, [{
            "clip_id": "clip-1", "window_start_seconds": 1.0,
            "window_end_seconds": 2.0, "impact_seconds": 2.3,
        }]

    def test_review_is_created_before_source_cleanup(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            storage, "RUNS_DIR", Path(directory)
        ), patch("backend.media_lifecycle._encode_review", side_effect=_fake_encode):
            run, results = self._make_run(Path(directory))
            outcome = finalize_run_media("test-run", results, retain_sources=False)
            self.assertEqual("cleaned", outcome["status"])
            self.assertFalse((run / "downloads" / "clip-1.mp4").exists())
            self.assertTrue((run / "artifacts" / "review-clip-1.mp4").exists())
            rows, _, _ = load_manifest(str(run / "video_manifest.csv"))
            self.assertEqual("cleaned", rows[0]["status"])

    def test_encoding_failure_retains_source(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            storage, "RUNS_DIR", Path(directory)
        ), patch("backend.media_lifecycle._encode_review", side_effect=RuntimeError("encoder failed")):
            run, results = self._make_run(Path(directory))
            outcome = finalize_run_media("test-run", results, retain_sources=False)
            self.assertEqual("warning", outcome["status"])
            self.assertTrue(outcome["sources_retained"])
            self.assertTrue((run / "downloads" / "clip-1.mp4").exists())

    def test_retain_mode_keeps_source_and_review(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            storage, "RUNS_DIR", Path(directory)
        ), patch("backend.media_lifecycle._encode_review", side_effect=_fake_encode):
            run, results = self._make_run(Path(directory))
            outcome = finalize_run_media("test-run", results, retain_sources=True)
            self.assertEqual("retained", outcome["status"])
            self.assertTrue((run / "downloads" / "clip-1.mp4").exists())
            self.assertTrue((run / "artifacts" / "review-clip-1.mp4").exists())


if __name__ == "__main__":
    unittest.main()
