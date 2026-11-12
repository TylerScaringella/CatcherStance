from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from backend import jobs, storage
from backend.app import create_app
from backend.review import build_xlsx, load_review_state, save_review
from backend.source_recovery import recover_source_clip


class PitchReviewTests(unittest.TestCase):
    def test_review_history_and_xlsx_are_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "review-run"
            run_dir.mkdir()
            location = storage.RunLocation("review-run", run_dir, "live", False)
            first = save_review(location, "clip-1", "KneeDown - Left", "model was wrong")
            second = save_review(location, "clip-1", "Squat", "confirmed on source")
            state = load_review_state(location)
            self.assertEqual("Squat", state["overrides"]["clip-1"][-1]["final_stance"])
            self.assertEqual(2, len(state["overrides"]["clip-1"]))
            self.assertEqual("KneeDown - Left", first["final_stance"])
            self.assertEqual("Squat", second["final_stance"])
            workbook = build_xlsx([
                {"pitch_index": 2, "final_stance": "KneeDown - Right"},
                {"pitch_index": 1, "final_stance": "KneeDown - Left"},
            ])
            with zipfile.ZipFile(io.BytesIO(workbook)) as archive:
                sheet = archive.read("xl/worksheets/sheet1.xml").decode("utf-8")
            self.assertIn("Pitch number", sheet)
            self.assertIn("Catcher Pose", sheet)
            self.assertLess(sheet.index("KneeDown - Left"), sheet.index("KneeDown - Right"))

    def test_api_blocks_unreviewed_export_and_preserves_model_stance(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(storage, "RUNS_DIR", Path(directory)), patch.dict(
            os.environ, {"CATCHER_STANCE_ADMIN_TOKEN": "test-token"}
        ):
            run_id = "review-api-run"
            run_dir = Path(directory) / run_id
            run_dir.mkdir()
            (run_dir / "job.json").write_text(json.dumps({
                "id": run_id,
                "game": {"id": "game-1", "opponent": "Test", "date": "2026-01-01"},
                "status": "complete",
                "created_at": 1,
                "updated_at": 2,
            }), encoding="utf-8")
            (run_dir / "detections.json").write_text(json.dumps([{
                "pitch_index": 1,
                "clip_id": "clip-1",
                "stance": "RKD",
                "status": "ok",
                "confidence": 0.5,
            }]), encoding="utf-8")
            app = create_app()
            client = app.test_client()
            self.assertEqual(401, client.get(f"/api/results/{run_id}/xlsx").status_code)
            self.assertEqual(200, client.post("/api/admin/session", json={"token": "test-token"}).status_code)
            blocked = client.get(f"/api/results/{run_id}/xlsx")
            self.assertEqual(409, blocked.status_code)
            self.assertEqual("review_incomplete", blocked.get_json()["code"])
            saved = client.put(
                f"/api/runs/{run_id}/clips/clip-1/review",
                json={"stance": "KneeDown - Left", "note": "confirmed"},
            )
            self.assertEqual(200, saved.status_code)
            result = saved.get_json()["result"]
            self.assertEqual("RKD", result["model_stance"])
            self.assertEqual("KneeDown - Left", result["final_stance"])
            exported = client.get(f"/api/results/{run_id}/xlsx")
            self.assertEqual(200, exported.status_code)
            self.assertEqual(
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                exported.mimetype,
            )

    def test_source_recovery_requires_a_media_reference_when_source_is_missing(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(storage, "RUNS_DIR", Path(directory)):
            run_dir = Path(directory) / "recovery-run"
            run_dir.mkdir()
            downloads = run_dir / "downloads"
            downloads.mkdir()
            (run_dir / "video_manifest.csv").write_text(
                f"clip_id,saved_path,media_ref,status\nclip-1,{downloads / 'clip-1.mp4'},,cleaned\n",
                encoding="utf-8",
            )
            (run_dir / "job.json").write_text(json.dumps({"game": {"id": "game-1"}}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "no recoverable"):
                recover_source_clip("recovery-run", "clip-1")


if __name__ == "__main__":
    unittest.main()
