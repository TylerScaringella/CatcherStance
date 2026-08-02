from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend import schedule


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

    def test_stale_cache_is_used_when_refresh_fails(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            schedule, "CACHE_DIR", Path(directory)
        ), patch.object(schedule.PROVIDERS["duke"], "fetch", side_effect=OSError("offline")):
            cache = Path(directory) / "duke_baseball_2026.json"
            cache.write_text(json.dumps({"season": 2026, "games": []}), encoding="utf-8")
            payload = schedule.load_schedule(refresh=True)
        self.assertTrue(payload["stale"])

    def test_season_validation_allows_next_season(self):
        self.assertEqual(2027, schedule.validate_season(2027))
        for invalid in ("bad", 1999, 9999):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                schedule.validate_season(invalid)


if __name__ == "__main__":
    unittest.main()
