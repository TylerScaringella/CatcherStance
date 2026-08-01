from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.app import create_app
from backend.trumedia import AuthenticationRequired, MatchResult, match_game, validate_storage_state_payload


class TruMediaMatchingTests(unittest.TestCase):
    def test_exact_match_ignores_rankings_and_uses_date(self):
        game = {"date": "2026-04-17", "opponent_key": "bostoncollege", "occurrence": 1}
        candidates = [
            {"id": "x", "date": "2026-04-17", "opponent_key": "bostoncollege", "opponent": "Boston College"},
            {"id": "y", "date": "2026-04-18", "opponent_key": "bostoncollege", "opponent": "Boston College"},
        ]
        result = match_game(game, candidates)
        self.assertEqual("matched", result.status)
        self.assertEqual("x", result.match["id"])

    def test_doubleheader_occurrence_selects_second_match(self):
        game = {"date": "2026-02-13", "opponent_key": "maine", "occurrence": 2}
        candidates = [
            {"id": "one", "date": "2026-02-13", "opponent_key": "maine"},
            {"id": "two", "date": "2026-02-13", "opponent_key": "maine"},
        ]
        self.assertEqual("two", match_game(game, candidates).match["id"])

    def test_ambiguous_date_requires_confirmation(self):
        game = {"date": "2026-03-01", "opponent_key": "princeton", "occurrence": 1}
        candidates = [{"id": "x", "date": "2026-03-01", "opponent_key": "unknown"}]
        result = match_game(game, candidates)
        self.assertEqual("match_required", result.status)
        self.assertIsNone(result.match)

    def test_storage_state_schema_is_strict(self):
        valid = {"cookies": [{"name": "sid", "value": "secret", "domain": ".example.com"}], "origins": []}
        self.assertEqual(valid, validate_storage_state_payload(valid))
        for invalid in ({}, [], {"cookies": [{}], "origins": []}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_storage_state_payload(invalid)


class TruMediaAdminApiTests(unittest.TestCase):
    def test_upload_requires_admin_authorization(self):
        client = create_app().test_client()
        response = client.post("/api/integrations/trumedia/session")
        self.assertEqual(401, response.status_code)
        self.assertEqual("admin_required", response.get_json()["code"])

    def test_valid_session_is_atomically_installed(self):
        payload = {"cookies": [{"name": "sid", "value": "secret", "domain": ".trumedianetworks.com"}], "origins": []}
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"CATCHER_STANCE_ADMIN_TOKEN": "test-token"}
        ), patch("backend.routes.AUTH_DIR", Path(directory)), patch(
            "backend.routes.PLAYWRIGHT_STATE_PATH", Path(directory) / "state.json"
        ), patch("backend.routes.TruMediaProvider.validate_live"):
            client = create_app().test_client()
            unlock = client.post("/api/admin/session", json={"token": "test-token"})
            self.assertEqual(200, unlock.status_code)
            upload = client.post(
                "/api/integrations/trumedia/session",
                data={"session": (io.BytesIO(json.dumps(payload).encode()), "state.json")},
                content_type="multipart/form-data",
            )
            self.assertEqual(200, upload.status_code)
            installed = Path(directory) / "state.json"
            self.assertEqual(payload, json.loads(installed.read_text(encoding="utf-8")))
            self.assertEqual(0o600, installed.stat().st_mode & 0o777)

    def test_run_reports_auth_required_without_starting_worker(self):
        client = create_app().test_client()
        with patch("backend.routes.resolve_game", side_effect=AuthenticationRequired("missing")):
            response = client.post("/api/run", json={"game_id": "duke-2026-02-13-maine-1"})
        self.assertEqual(409, response.status_code)
        self.assertEqual("auth_required", response.get_json()["code"])

    def test_match_candidates_do_not_expose_server_urls(self):
        candidate = {
            "id": "candidate-1", "date": "2026-02-13", "opponent": "Maine",
            "opponent_key": "maine", "url": "https://duke-ncaabaseball.trumedianetworks.com/private/game",
        }
        client = create_app().test_client()
        with patch(
            "backend.routes.resolve_game",
            return_value=MatchResult("match_required", None, [candidate]),
        ):
            response = client.get("/api/games/duke-2026-02-13-maine-1/trumedia-match")
        self.assertEqual(200, response.status_code)
        self.assertNotIn("url", response.get_json()["candidates"][0])

    def test_run_uses_server_match_and_defaults_to_source_cleanup(self):
        match = {
            "id": "candidate-1", "date": "2026-02-13", "opponent": "Maine",
            "opponent_key": "maine", "url": "https://duke-ncaabaseball.trumedianetworks.com/baseball/game/1",
        }
        client = create_app().test_client()
        with patch("backend.routes.resolve_game", return_value=MatchResult("matched", match, [match])), patch(
            "backend.routes.list_runs", return_value=[]
        ), patch("backend.routes.write_job_state"), patch("backend.routes.threading.Thread.start"):
            response = client.post("/api/run", json={"game_id": "duke-2026-02-13-maine-1"})
        self.assertEqual(202, response.status_code)
        payload = response.get_json()
        self.assertEqual(1, payload["revision"])
        self.assertFalse(payload["retain_sources"])
        self.assertNotIn("url", payload["trumedia_match"])


if __name__ == "__main__":
    unittest.main()
