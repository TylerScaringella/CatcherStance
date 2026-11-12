from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.app import create_app
from backend import jobs, storage
from backend.schedule import ScheduleUnavailableError


def write_job(root: Path, run_id: str, status: str) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "job.json").write_text(
        json.dumps({
            "id": run_id,
            "game": {"id": "duke-2026-04-21-liberty", "opponent": "Liberty", "date": "2026-04-21"},
            "source": "live",
            "read_only": False,
            "status": status,
            "phase": status,
            "created_at": time.time(),
            "updated_at": time.time(),
        }),
        encoding="utf-8",
    )
    return run_dir


class RunControlTests(unittest.TestCase):
    def setUp(self):
        jobs.JOBS.clear()

    def test_pause_request_is_persisted_after_restart_like_reload(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(storage, "RUNS_DIR", Path(directory)):
            run_dir = write_job(Path(directory), "control-run", "detecting")
            client = create_app().test_client()
            response = client.post("/api/runs/control-run/control", json={"action": "pause"})
            self.assertEqual(202, response.status_code)
            self.assertEqual("pause", json.loads((run_dir / "job.json").read_text())["control_request"])
            jobs.JOBS.clear()
            self.assertEqual("pause", jobs.control_request("control-run"))

    def test_cancel_is_rejected_for_read_only_sample(self):
        client = create_app().test_client()
        response = client.post("/api/runs/duke-2026-04-21-liberty-sample/control", json={"action": "cancel"})
        self.assertEqual(409, response.status_code)
        self.assertEqual("read_only_run", response.get_json()["code"])

    def test_finished_run_delete_requires_admin_and_removes_run(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(storage, "RUNS_DIR", Path(directory)):
            run_dir = write_job(Path(directory), "finished-run", "complete")
            client = create_app().test_client()
            self.assertEqual(401, client.delete("/api/runs/finished-run").status_code)
            with client.session_transaction() as session:
                session["trumedia_admin"] = True
            response = client.delete("/api/runs/finished-run")
            self.assertEqual(200, response.status_code)
            self.assertFalse(run_dir.exists())

    def test_active_run_delete_requires_stop_or_cancel_first(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(storage, "RUNS_DIR", Path(directory)):
            write_job(Path(directory), "active-run", "detecting")
            client = create_app().test_client()
            with client.session_transaction() as session:
                session["trumedia_admin"] = True
            response = client.delete("/api/runs/active-run")
            self.assertEqual(409, response.status_code)
            self.assertEqual("run_active", response.get_json()["code"])

    def test_run_list_survives_unpublished_active_schedule(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(storage, "RUNS_DIR", Path(directory)):
            run_dir = write_job(Path(directory), "duke-2026-04-21-liberty-legacy", "complete")
            payload = {
                "games": [{
                    "id": "duke-2026-04-21-liberty",
                    "opponent": "Liberty",
                    "date": "2026-04-21",
                }],
            }
            job = json.loads((run_dir / "job.json").read_text())
            job.pop("game")
            (run_dir / "job.json").write_text(json.dumps(job), encoding="utf-8")
            with patch.object(
                jobs,
                "load_schedule",
                side_effect=[ScheduleUnavailableError("2027 is not published"), payload],
            ):
                response = create_app().test_client().get("/api/runs?view=summary")
            self.assertEqual(200, response.status_code)
            self.assertEqual("duke-2026-04-21-liberty", response.get_json()["runs"][0]["game"]["id"])


if __name__ == "__main__":
    unittest.main()
