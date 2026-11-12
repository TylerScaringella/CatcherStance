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
from backend.review import build_sheet_preview_html, build_xlsx, load_review_state, save_review
from backend.media_lifecycle import _review_bounds
from backend.source_recovery import recover_source_clip


class PitchReviewTests(unittest.TestCase):
    def test_attention_ignores_informational_flags_but_blocks_unsafe_evidence(self):
        clear = jobs.apply_attention_state({
            "accepted": True,
            "quality_flags": ["event_anchor:plate_arrival", "event_uncertainty_frames:5", "reduced_batch_pose_retry"],
        })
        self.assertFalse(clear["attention_required"])
        blocked = jobs.apply_attention_state({
            "accepted": True,
            "quality_flags": ["event_anchor:plate_arrival", "low_pose_coverage"],
        })
        self.assertTrue(blocked["attention_required"])
        reviewed = jobs.apply_attention_state({
            "accepted": False,
            "rejection_reason": "ambiguous_stance_vote",
            "quality_flags": ["low_vote_share"],
            "review_status": "reviewed",
            "final_stance": "KneeDown - Left",
        })
        self.assertFalse(reviewed["attention_required"])
        self.assertTrue(reviewed["model_attention_reasons"])

    def test_compact_review_bounds_include_set_and_delivery(self):
        start, duration = _review_bounds({
            "window_start_seconds": 5.0,
            "window_end_seconds": 5.8,
            "impact_seconds": 6.4,
        })
        self.assertLessEqual(start, 4.0)
        self.assertGreaterEqual(start + duration, 7.0)

    def test_sheet_preview_is_exactly_paste_ready(self):
        preview = build_sheet_preview_html([
            {"pitch_index": 2, "final_stance": "KneeDown - Right"},
            {"pitch_index": 1, "final_stance": "KneeDown - Left"},
        ])
        self.assertIn("<th>Pitch number</th><th>Catcher Pose</th>", preview)
        self.assertLess(preview.index("KneeDown - Left"), preview.index("KneeDown - Right"))
        self.assertIn("Copy table", preview)
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
            blocked_preview = client.get(f"/api/results/{run_id}/sheet-preview")
            self.assertEqual(409, blocked_preview.status_code)
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
            downloads = run_dir / "downloads"
            artifacts = run_dir / "artifacts"
            downloads.mkdir()
            artifacts.mkdir()
            source = downloads / "clip-1.mp4"
            source.write_bytes(b"source")
            compact = artifacts / "review-clip-1.mp4"
            compact.write_bytes(b"compact")
            (run_dir / "video_manifest.csv").write_text(
                f"clip_id,saved_path,status,skip_reason\nclip-1,{source},downloaded,\n",
                encoding="utf-8",
            )
            preview = client.get(f"/api/results/{run_id}/sheet-preview")
            self.assertEqual(200, preview.status_code)
            self.assertIn("Catcher Pose", preview.get_data(as_text=True))
            compact_response = client.get(f"/api/runs/{run_id}/clips/clip-1/video?mode=review")
            self.assertEqual(200, compact_response.status_code)
            purged = client.post(f"/api/runs/{run_id}/purge-sources")
            self.assertEqual(200, purged.status_code)
            self.assertFalse(source.exists())
            self.assertTrue(compact.exists())

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
