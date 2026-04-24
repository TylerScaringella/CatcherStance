from __future__ import annotations

# Codex attribution: OpenAI Codex generated the schedule loading and refresh
# support code, with project-specific review and edits.
import json
import re
import threading
import time
from datetime import datetime
from urllib.request import Request, urlopen

from .config import SCHEDULE_PATH, SCHEDULE_REFRESH_INTERVAL_SECONDS

_schedule_refresh_started = False


def load_schedule():
    return json.loads(SCHEDULE_PATH.read_text(encoding="utf-8"))


def find_game(game_id: str):
    for game in load_schedule()["games"]:
        if game["id"] == game_id:
            return game
    return None


def slugify(value: str):
    value = value.lower().replace("&", "and")
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def make_game_id(date: str, opponent: str, occurrence: int, total_occurrences: int):
    suffix = f"-{occurrence}" if total_occurrences > 1 else ""
    return f"duke-{date}-{slugify(opponent)}{suffix}"


def parse_acc_schedule_rows(html: str):
    pattern = re.compile(
        r"(?P<date>\d{1,2}/\d{1,2}/2026)\s+"
        r"(?:(?P<acc>\*)\s+)?"
        r"(?P<opponent>.*?)\s{2,}"
        r"(?P<location>.*?)\s{2,}"
        r"(?P<result>(?:[WL]\s+\d+[-–]\d+(?:\s+\([^)]*\))?|\d{1,2}:\d{2}\s+[AP]\.M\.\s+ET))"
    )
    raw_games = []
    totals: dict[tuple[str, str], int] = {}

    for match in pattern.finditer(html):
        parsed_date = datetime.strptime(match.group("date"), "%m/%d/%Y").date().isoformat()
        opponent = re.sub(r"\s+", " ", match.group("opponent")).strip()
        location = re.sub(r"\s+", " ", match.group("location")).strip()
        result = re.sub(r"\s+", " ", match.group("result")).strip().replace("–", "-")
        key = (parsed_date, opponent)
        totals[key] = totals.get(key, 0) + 1
        raw_games.append((key, parsed_date, opponent, location, bool(match.group("acc")), result))

    games = []
    occurrences: dict[tuple[str, str], int] = {}
    for key, parsed_date, opponent, location, conference, result in raw_games:
        occurrences[key] = occurrences.get(key, 0) + 1
        games.append(
            {
                "id": make_game_id(parsed_date, opponent, occurrences[key], totals[key]),
                "date": parsed_date,
                "opponent": opponent,
                "location": location,
                "conference": conference,
                "result": result,
                "trumedia_url": "",
            }
        )

    return games


def fetch_acc_schedule(source_url: str):
    request = Request(source_url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=20) as response:
        html = response.read().decode("utf-8", errors="replace")
    games = parse_acc_schedule_rows(html)
    if not games:
        raise RuntimeError("No games found on ACC schedule page")
    return games


def refresh_schedule_once():
    schedule = load_schedule()
    source_url = schedule.get("source")
    if not source_url:
        return

    fetched_games = fetch_acc_schedule(source_url)
    existing_by_id = {game["id"]: game for game in schedule.get("games", [])}
    for game in fetched_games:
        existing = existing_by_id.get(game["id"], {})
        game["trumedia_url"] = existing.get("trumedia_url", "")

    schedule["games"] = fetched_games
    schedule["source_checked"] = datetime.now().date().isoformat()
    SCHEDULE_PATH.write_text(json.dumps(schedule, indent=2), encoding="utf-8")
    print(f"Updated schedule from ACC: {len(fetched_games)} games")


def schedule_refresh_loop():
    while True:
        try:
            refresh_schedule_once()
        except Exception as exc:
            print(f"Schedule refresh failed: {type(exc).__name__}: {exc}")
        time.sleep(SCHEDULE_REFRESH_INTERVAL_SECONDS)


def start_schedule_refresh_job():
    global _schedule_refresh_started
    if _schedule_refresh_started:
        return
    _schedule_refresh_started = True
    thread = threading.Thread(target=schedule_refresh_loop, daemon=True)
    thread.start()
