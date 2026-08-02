from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.app import create_app
from backend.trumedia import (
    AuthenticationRequired,
    MatchResult,
    match_game,
    parse_score_payload,
    validate_storage_state_payload,
)
from downloader.crawler import parse_pitch_payload


def datapoint_payload(columns, rows):
    return {
        "header": [{"columnId": column} for column in columns],
        "rows": rows,
        "totalRows": len(rows),
        "status": "OK",
    }


class TruMediaMatchingTests(unittest.TestCase):
    def test_real_duke_doubleheader_ids_and_game_numbers_are_parsed(self):
        columns = [
            "gameId", "gameString", "gameDate", "gameStatus", "awayTeamId",
            "awayTeamName", "homeTeamId", "homeTeamName", "awayRunsScored",
            "homeRunsScored", "trackmanGameUID",
        ]
        payload = datapoint_payload(columns, [
            [463703220, "cs-duk01202602130", "2026-02-13 15:32:00 (1)", "Final", 730275840, "University of Maine", 730336256, "Duke University", 3, 8, "available"],
            [225496045, "cs-duk01202602132", "2026-02-13 18:49:00 (2)", "Final", 730275840, "University of Maine", 730336256, "Duke University", 4, 3, "available"],
            [111, "other", "2026-02-13 13:00:00", "Final", 1, "Other", 2, "Elsewhere", 1, 2, None],
        ])
        candidates = parse_score_payload(payload, "2026-02-13")
        self.assertEqual([463703220, 225496045], [item["trumedia_game_id"] for item in candidates])
        self.assertEqual([1, 2], [item["game_number"] for item in candidates])
        first = match_game(
            {"date": "2026-02-13", "opponent_key": "maine", "occurrence": 1},
            candidates,
        )
        second = match_game(
            {"date": "2026-02-13", "opponent_key": "maine", "occurrence": 2},
            candidates,
        )
        self.assertEqual("463703220", first.match["id"])
        self.assertEqual("225496045", second.match["id"])

    def test_pitch_payload_selects_all_duke_catching_rows_beyond_dom_limit(self):
        columns = [
            "videoAngles", "gameId", "pitchNumInGame", "uniqPitchId", "inn", "count",
            "pitcherAbbrevName", "batterAbbrevName", "catchingTeam", "catchingTeamId",
            "catcherId", "catcherAbbrevName", "pitchingTeam",
        ]
        rows = []
        for index in range(356):
            duke_pitch = index < 169
            rows.append([
                [{"view": "Broadcast", "url": f"s3://private/game/pitch-{index}.mp4"}],
                333925844, index + 1, f"333925844-{index + 1}-1", "Top 1", "0-0",
                "Pitcher", "Batter", "DUKE" if duke_pitch else "APP",
                730336256 if duke_pitch else 730240256, 1143069440, "Catcher", "DUKE",
            ])
        selected, stats = parse_pitch_payload(datapoint_payload(columns, rows), download_dir="downloads")
        self.assertEqual(356, stats["total_pitches"])
        self.assertEqual(169, stats["selected_pitches"])
        self.assertEqual(169, stats["downloadable_pitches"])
        self.assertEqual(169, len(selected))
        self.assertTrue(all(row["catching_team_id"] == "730336256" for row in selected))
        self.assertTrue(all(not row["s3_url"] for row in selected))

    def test_pitch_without_broadcast_is_retained_as_skipped(self):
        columns = ["videoAngles", "gameId", "pitchNumInGame", "uniqPitchId", "catchingTeamId"]
        payload = datapoint_payload(columns, [[
            [{"view": "High Home", "url": "s3://private/game/high-home.mp4"}],
            333925844, 1, "333925844-1-1", 730336256,
        ]])
        rows, stats = parse_pitch_payload(payload)
        self.assertEqual("skipped", rows[0]["status"])
        self.assertEqual("broadcast_video_unavailable", rows[0]["skip_reason"])
        self.assertEqual(1, stats["skipped_pitches"])
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

    def test_conflicting_doubleheader_result_requires_confirmation(self):
        game = {
            "date": "2026-02-13", "opponent_key": "maine", "occurrence": 2,
            "result": "L 3-4",
        }
        candidates = [{
            "id": "wrong", "date": "2026-02-13", "opponent_key": "universityofmaine",
            "game_number": 2, "result": "W 8-3",
        }]
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

    def test_revalidate_requires_admin_authorization(self):
        client = create_app().test_client()
        response = client.post("/api/integrations/trumedia/validate")
        self.assertEqual(401, response.status_code)
        self.assertEqual("admin_required", response.get_json()["code"])

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
