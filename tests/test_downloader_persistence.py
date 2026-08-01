from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from downloader.files import download_file_once
from downloader.manifest import load_manifest, write_manifest


class _Response:
    def __init__(self, body: bytes, content_type: str = "video/mp4") -> None:
        self.body = body
        self.headers = {"Content-Type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size: int):
        yield self.body


class DownloaderPersistenceTests(unittest.TestCase):
    def test_download_is_promoted_only_after_mp4_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "clip.mp4"
            session = MagicMock()
            session.get.return_value = _Response(b"\x00\x00\x00\x18ftypmp42" + b"0" * 64)
            with patch("downloader.files._session", return_value=session):
                download_file_once("https://example.com/clip.mp4", str(destination))
            self.assertTrue(destination.exists())
            self.assertEqual([], list(destination.parent.glob("*.part")))

    def test_invalid_media_does_not_replace_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "clip.mp4"
            destination.write_bytes(b"existing")
            session = MagicMock()
            session.get.return_value = _Response(b"not a video", "text/html")
            with patch("downloader.files._session", return_value=session), self.assertRaises(ValueError):
                download_file_once("https://example.com/clip.mp4", str(destination))
            self.assertEqual(b"existing", destination.read_bytes())
            self.assertEqual([], list(destination.parent.glob("*.part")))

    def test_manifest_round_trip_uses_atomic_replacement(self):
        rows = [{
            "card_dom_index": "0", "clip_id": "clip", "s3_url": "https://example.com",
            "saved_path": "downloads/clip.mp4", "status": "pending", "attempts": "0", "error": "",
        }]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            write_manifest(str(path), rows)
            loaded, by_id, _ = load_manifest(str(path))
            self.assertEqual(rows, loaded)
            self.assertEqual("clip", by_id["clip"]["clip_id"])
            self.assertEqual([], list(path.parent.glob("*.part")))


if __name__ == "__main__":
    unittest.main()
