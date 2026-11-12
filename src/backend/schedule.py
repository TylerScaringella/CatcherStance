from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

from project_paths import CACHE_DIR, SCHEDULE_PATH, SCHEDULE_SETTINGS_PATH

from .storage import atomic_write_json

DEFAULT_TEAM = "duke"
DEFAULT_SEASON = 2026


class ScheduleIdentityError(RuntimeError):
    """Raised when an upstream response does not describe Duke baseball."""


class ScheduleUnavailableError(RuntimeError):
    """Raised when the requested season has not been published upstream."""


class ScheduleProvider(ABC):
    team_id: str

    @abstractmethod
    def fetch(self, season: int) -> dict:
        raise NotImplementedError


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._row is not None and self._cell is not None:
            self._row.append(re.sub(r"\s+", " ", "".join(self._cell)).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None


def slugify(value: str) -> str:
    value = re.sub(r"^#\d+\s+", "", value.strip().lower())
    value = value.replace("&", "and")
    return re.sub(r"[^a-z0-9]+", "-", value).strip("-")


def normalize_opponent(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", slugify(value))


def make_game_id(date: str, opponent: str, occurrence: int, total_occurrences: int) -> str:
    suffix = f"-{occurrence}" if total_occurrences > 1 else ""
    return f"duke-{date}-{slugify(opponent)}{suffix}"


def parse_duke_schedule_table(html: str, season: int) -> list[dict]:
    parser = _TableParser()
    parser.feed(html)
    rows = [row for row in parser.rows if len(row) >= 7]
    data_rows = [row for row in rows if row[0].lower() != "date"]
    parsed: list[dict] = []
    for row in data_rows:
        date_text, game_time, site, opponent, location, tournament, result = row[:7]
        match = re.match(r"([A-Z][a-z]{2})\s+(\d{1,2})", date_text)
        if not match:
            continue
        date = datetime.strptime(
            f"{match.group(1)} {match.group(2)} {season}", "%b %d %Y"
        ).date().isoformat()
        result = result.replace(",", "").replace("\u2013", "-").strip()
        completed = bool(re.match(r"^[WLT]\s+\d+\s*-\s*\d+", result))
        parsed.append(
            {
                "date": date,
                "time": game_time,
                "opponent": opponent,
                "opponent_key": normalize_opponent(opponent),
                "site": site.lower() if site.lower() in {"home", "away", "neutral"} else "neutral",
                "location": location,
                "tournament": tournament,
                "conference": tournament.lower().startswith("acc") or slugify(opponent) in {
                    "boston-college", "clemson", "florida-state", "georgia-tech", "louisville",
                    "miami", "nc-state", "north-carolina", "notre-dame", "pittsburgh",
                    "stanford", "virginia", "virginia-tech", "wake-forest",
                },
                "result": result,
                "status": "completed" if completed else "scheduled",
            }
        )

    totals = Counter((game["date"], game["opponent_key"]) for game in parsed)
    occurrences: dict[tuple[str, str], int] = defaultdict(int)
    for game in parsed:
        key = (game["date"], game["opponent_key"])
        occurrences[key] += 1
        game["occurrence"] = occurrences[key]
        game["id"] = make_game_id(
            game["date"], game["opponent"], occurrences[key], totals[key]
        )
    return parsed


class DukeScheduleProvider(ScheduleProvider):
    team_id = DEFAULT_TEAM
    source_template = "https://goduke.com/sports/baseball/schedule/text/{season}"

    def fetch(self, season: int) -> dict:
        source = self.source_template.format(season=season)
        request = Request(source, headers={"User-Agent": "CatcherStance/1.0"})
        with urlopen(request, timeout=20) as response:
            html = response.read().decode("utf-8", errors="replace")
        title_match = re.search(r"<title>\s*(\d{4})\s+Baseball\s+Schedule", html, re.IGNORECASE)
        if title_match and int(title_match.group(1)) != season:
            raise ScheduleUnavailableError(
                f"Duke published {title_match.group(1)} instead of requested {season}"
            )
        games = parse_duke_schedule_table(html, season)
        if not games:
            raise ScheduleUnavailableError(f"Duke has not published a schedule for {season}")
        return {
            "team": "Duke",
            "team_id": self.team_id,
            "sport": "Baseball",
            "season": season,
            "source": source,
            "source_checked": datetime.now().date().isoformat(),
            "stale": False,
            "games": games,
        }


PROVIDERS: dict[str, ScheduleProvider] = {DEFAULT_TEAM: DukeScheduleProvider()}


def validate_duke_schedule_payload(payload: object, season: int) -> dict:
    if not isinstance(payload, dict) or int(payload.get("season") or 0) != season:
        raise ScheduleIdentityError("schedule payload does not match the requested season")
    games = payload.get("games")
    if not isinstance(games, list) or not games:
        raise ScheduleIdentityError("schedule payload contains no games")
    if not any(
        game.get("site") == "home" and "durham" in str(game.get("location") or "").lower()
        for game in games
        if isinstance(game, dict)
    ):
        raise ScheduleIdentityError("schedule payload is not a Duke home schedule")
    if len(games) >= 20 and not any(
        bool(game.get("conference")) for game in games if isinstance(game, dict)
    ):
        raise ScheduleIdentityError("schedule payload has no Duke conference games")
    return payload


def validate_season(value: int | str) -> int:
    try:
        season = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid season") from exc
    if season < 2000 or season > datetime.now().year + 2:
        raise ValueError("invalid season")
    return season


def _default_schedule_settings() -> dict:
    return {
        "active_season": DEFAULT_SEASON,
        "last_successful_refresh_at": None,
        "last_attempted_refresh_at": None,
        "source_url": None,
        "stale": False,
        "error": None,
    }


def load_schedule_settings() -> dict:
    settings = _default_schedule_settings()
    try:
        saved = json.loads(SCHEDULE_SETTINGS_PATH.read_text(encoding="utf-8"))
        if isinstance(saved, dict):
            settings.update(saved)
    except (OSError, json.JSONDecodeError):
        pass
    try:
        settings["active_season"] = validate_season(settings.get("active_season", DEFAULT_SEASON))
    except ValueError:
        settings["active_season"] = DEFAULT_SEASON
    return settings


def save_schedule_settings(settings: dict) -> dict:
    normalized = _default_schedule_settings()
    normalized.update(settings)
    normalized["active_season"] = validate_season(normalized["active_season"])
    atomic_write_json(SCHEDULE_SETTINGS_PATH, normalized)
    return normalized


def set_active_season(season: int | str) -> dict:
    season = validate_season(season)
    settings = load_schedule_settings()
    settings.update({"active_season": season, "error": None})
    return save_schedule_settings(settings)


def schedule_cache_path(team_id: str, season: int) -> Path:
    return CACHE_DIR / f"{team_id}_baseball_{season}.json"


def _checked_in_fallback(season: int) -> dict | None:
    if season != DEFAULT_SEASON or not SCHEDULE_PATH.exists():
        return None
    payload = json.loads(SCHEDULE_PATH.read_text(encoding="utf-8"))
    payload.setdefault("team_id", DEFAULT_TEAM)
    payload.setdefault("stale", True)
    for game in payload.get("games", []):
        game.setdefault("opponent_key", normalize_opponent(game.get("opponent", "")))
        game.setdefault("site", "away" if game.get("opponent", "").lower().startswith("at ") else "home")
        game.setdefault("status", "completed" if game.get("result") else "scheduled")
        game.setdefault("occurrence", 1)
    return payload


def load_schedule(
    season: int | None = None,
    team_id: str = DEFAULT_TEAM,
    *,
    refresh: bool = False,
) -> dict:
    season = validate_season(
        load_schedule_settings()["active_season"] if season is None else season
    )
    provider = PROVIDERS.get(team_id)
    if provider is None:
        raise ValueError("unknown team")
    cache = schedule_cache_path(team_id, season)
    if cache.exists() and not refresh:
        try:
            return validate_duke_schedule_payload(
                json.loads(cache.read_text(encoding="utf-8")), season
            )
        except (OSError, json.JSONDecodeError, ScheduleIdentityError):
            cache.unlink(missing_ok=True)
    try:
        payload = validate_duke_schedule_payload(provider.fetch(season), season)
        atomic_write_json(cache, payload)
        return payload
    except ScheduleUnavailableError as exc:
        if cache.exists():
            try:
                payload = validate_duke_schedule_payload(
                    json.loads(cache.read_text(encoding="utf-8")), season
                )
                payload["stale"] = True
                payload["error"] = str(exc)
                return payload
            except (OSError, json.JSONDecodeError, ScheduleIdentityError):
                cache.unlink(missing_ok=True)
        fallback = _checked_in_fallback(season)
        if fallback is not None:
            fallback["stale"] = True
            fallback["error"] = str(exc)
            return validate_duke_schedule_payload(fallback, season)
        raise
    except Exception as exc:
        if cache.exists():
            try:
                payload = validate_duke_schedule_payload(
                    json.loads(cache.read_text(encoding="utf-8")), season
                )
                payload["stale"] = True
                payload["error"] = str(exc)
                return payload
            except (OSError, json.JSONDecodeError, ScheduleIdentityError):
                cache.unlink(missing_ok=True)
        fallback = _checked_in_fallback(season)
        if fallback is not None:
            fallback["stale"] = True
            fallback["error"] = str(exc)
            return validate_duke_schedule_payload(fallback, season)
        raise


def completed_schedule(season: int | None = None, team_id: str = DEFAULT_TEAM) -> dict:
    payload = load_schedule(season, team_id)
    return {**payload, "games": [game for game in payload["games"] if game["status"] == "completed"]}


def schedule_for_season(
    season: int | None = None,
    team_id: str = DEFAULT_TEAM,
    *,
    include_scheduled: bool = False,
) -> dict:
    payload = load_schedule(season, team_id)
    if include_scheduled:
        return payload
    return {**payload, "games": [game for game in payload["games"] if game["status"] == "completed"]}


def refresh_schedule(season: int | str | None = None) -> dict:
    season = validate_season(
        load_schedule_settings()["active_season"] if season is None else season
    )
    settings = load_schedule_settings()
    attempted = datetime.now().isoformat(timespec="seconds")
    settings.update({"last_attempted_refresh_at": attempted, "error": None})
    save_schedule_settings(settings)
    try:
        payload = load_schedule(season, refresh=True)
    except ScheduleUnavailableError as exc:
        settings.update({"source_url": DukeScheduleProvider.source_template.format(season=season), "stale": False, "error": str(exc)})
        save_schedule_settings(settings)
        raise
    except Exception as exc:
        settings.update({"source_url": DukeScheduleProvider.source_template.format(season=season), "stale": True, "error": str(exc)})
        save_schedule_settings(settings)
        raise
    successful = not bool(payload.get("stale"))
    settings.update({
        "source_url": payload.get("source"),
        "last_successful_refresh_at": attempted if successful else settings.get("last_successful_refresh_at"),
        "stale": bool(payload.get("stale")),
        "error": payload.get("error"),
    })
    save_schedule_settings(settings)
    return payload


def find_game(game_id: str) -> dict | None:
    year_match = re.match(r"^[a-z]+-(\d{4})-", game_id or "")
    season = int(year_match.group(1)) if year_match else load_schedule_settings()["active_season"]
    for game in load_schedule(season).get("games", []):
        if game["id"] == game_id:
            return game
    return None


def refresh_schedule_once(season: int | None = None) -> None:
    refresh_schedule(season)
