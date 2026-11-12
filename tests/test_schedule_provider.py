from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend import schedule
from backend.app import create_app


FIXTURE = """
<table>
  <thead><tr><th>Date</th><th>Time</th><th>At</th><th>Opponent</th><th>Location</th><th>Tournament</th><th>Result</th></tr></thead>
  <tbody>
    <tr><td>Feb 13 (Fri)</td><td>3:30 p.m.</td><td>Home</td><td>Maine</td><td>Durham, N.C.</td><td></td><td>W 8-3</td></tr>
    <tr><td>Feb 13 (Fri)</td><td>TBD</td><td>Home</td><td>Maine</td><td>Durham, N.C.</td><td></td><td>L 3-4</td></tr>
    <tr><td>Apr 17 (Fri)</td><td>3 p.m.</td><td>Away</td><td>#24 Boston College</td><td>Chestnut Hill, Mass.</td><td></td><td>L, 1-11</td></tr>
    <tr><td>May 19 (Tue)</td><td>9 a.m.</td><td>Neutral</td><td>NC State</td><td>Charlotte, N.C.</td><td>ACC Championship</td><td>W 21-12</td></tr>
    <tr><td>Nov 1 (Sun)</td><td>TBA</td><td>Home</td><td>Future Team</td><td>Durham, N.C.</td><td></td><td></td></tr>
  </tbody>
</table>
"""


class DukeScheduleProviderTests(unittest.TestCase):
    @staticmethod
    def payload(season=2026):
        return {
            "team": "Duke",
            "team_id": "duke",
            "season": season,
            "games": schedule.parse_duke_schedule_table(FIXTURE, season),
        }

    def test_parser_normalizes_completed_games_and_doubleheaders(self):
        games = schedule.parse_duke_schedule_table(FIXTURE, 2026)
        self.assertEqual(5, len(games))
        self.assertEqual("duke-2026-02-13-maine-1", games[0]["id"])
        self.assertEqual("duke-2026-02-13-maine-2", games[1]["id"])
        self.assertEqual("bostoncollege", games[2]["opponent_key"])
        self.assertTrue(games[2]["conference"])
        self.assertEqual("neutral", games[3]["site"])
        self.assertEqual("scheduled", games[4]["status"])

    def test_completed_schedule_filters_future_games(self):
        payload = {"team": "Duke", "season": 2026, "games": schedule.parse_duke_schedule_table(FIXTURE, 2026)}
        with patch("backend.schedule.load_schedule", return_value=payload):
            completed = schedule.completed_schedule()
        self.assertEqual(4, len(completed["games"]))

    def test_full_schedule_mode_preserves_scheduled_games(self):
        payload = {"team": "Duke", "season": 2026, "games": schedule.parse_duke_schedule_table(FIXTURE, 2026)}
        with patch("backend.routes.schedule_for_season", return_value=payload):
            response = create_app().test_client().get("/api/schedule?season=2026&include_scheduled=1")
        self.assertEqual(200, response.status_code)
        self.assertEqual(5, len(response.get_json()["games"]))

    def test_stale_cache_is_used_when_refresh_fails(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            schedule, "CACHE_DIR", Path(directory)
        ), patch.object(schedule.PROVIDERS["duke"], "fetch", side_effect=OSError("offline")):
            cache = Path(directory) / "duke_baseball_2026.json"
            cache.write_text(json.dumps(self.payload()), encoding="utf-8")
            payload = schedule.load_schedule(refresh=True)
        self.assertTrue(payload["stale"])

    def test_cross_team_schedule_is_rejected(self):
        payload = {
            "team": "Duke",
            "team_id": "duke",
            "season": 2026,
            "games": [
                {
                    "id": f"duke-2026-02-{day:02d}-hawaii",
                    "site": "home",
                    "location": "Portland, OR",
                    "conference": False,
                }
                for day in range(1, 22)
            ],
        }
        with self.assertRaisesRegex(schedule.ScheduleIdentityError, "not a Duke home schedule"):
            schedule.validate_duke_schedule_payload(payload, 2026)

    def test_invalid_cache_is_removed_and_refetched(self):
        bad_payload = {
            "team": "Duke",
            "season": 2026,
            "games": [{"site": "home", "location": "Portland, OR"}],
        }
        valid_payload = self.payload()
        with tempfile.TemporaryDirectory() as directory, patch.object(
            schedule, "CACHE_DIR", Path(directory)
        ), patch.object(schedule.PROVIDERS["duke"], "fetch", return_value=valid_payload) as fetch:
            cache = Path(directory) / "duke_baseball_2026.json"
            cache.write_text(json.dumps(bad_payload), encoding="utf-8")
            payload = schedule.load_schedule()
            self.assertEqual(valid_payload, payload)
            self.assertEqual(valid_payload, json.loads(cache.read_text(encoding="utf-8")))
            fetch.assert_called_once_with(2026)

    def test_season_validation_allows_next_season(self):
        self.assertEqual(2027, schedule.validate_season(2027))
        for invalid in ("bad", 1999, 9999):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                schedule.validate_season(invalid)

    def test_active_season_persists_and_reloads(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            schedule, "SCHEDULE_SETTINGS_PATH", Path(directory) / "settings.json"
        ):
            saved = schedule.set_active_season(2027)
            self.assertEqual(2027, saved["active_season"])
            self.assertEqual(2027, schedule.load_schedule_settings()["active_season"])

    def test_schedule_provider_rejects_upstream_wrong_season(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b"<title>2026 Baseball Schedule - Duke University</title>"

        with patch("backend.schedule.urlopen", return_value=Response()), self.assertRaisesRegex(
            schedule.ScheduleUnavailableError, "requested 2027"
        ):
            schedule.DukeScheduleProvider().fetch(2027)

    def test_schedule_management_requires_admin_and_refreshes(self):
        payload = self.payload(2027)
        with tempfile.TemporaryDirectory() as directory, patch.object(
            schedule, "SCHEDULE_SETTINGS_PATH", Path(directory) / "settings.json"
        ), patch.object(schedule, "CACHE_DIR", Path(directory) / "cache"), patch.object(
            schedule.PROVIDERS["duke"], "fetch", return_value=payload
        ):
            client = create_app().test_client()
            self.assertEqual(401, client.put("/api/teams/duke/schedule-settings", json={"active_season": 2027}).status_code)
            with client.session_transaction() as session:
                session["trumedia_admin"] = True
            response = client.put("/api/teams/duke/schedule-settings", json={"active_season": 2027})
            self.assertEqual(200, response.status_code)
            refreshed = client.post("/api/teams/duke/schedule/refresh", json={"season": 2027})
            self.assertEqual(200, refreshed.status_code)
            self.assertEqual(2027, refreshed.get_json()["settings"]["active_season"])
            self.assertEqual(2027, refreshed.get_json()["schedule"]["season"])

    def test_unpublished_schedule_returns_explicit_status(self):
        with patch("backend.routes.refresh_schedule", side_effect=schedule.ScheduleUnavailableError("not published")):
            client = create_app().test_client()
            with client.session_transaction() as session:
                session["trumedia_admin"] = True
            response = client.post("/api/teams/duke/schedule/refresh", json={"season": 2027})
        self.assertEqual(409, response.status_code)
        self.assertEqual("schedule_unpublished", response.get_json()["code"])


if __name__ == "__main__":
    unittest.main()
