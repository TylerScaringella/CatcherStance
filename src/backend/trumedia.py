from __future__ import annotations

import hmac
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode, urlparse

from project_paths import CACHE_DIR, PLAYWRIGHT_STATE_PATH

from .config import TRUMEDIA_DEFAULT_URL
from .schedule import normalize_opponent
from .storage import atomic_write_json

MAX_SESSION_BYTES = 1024 * 1024
MAPPINGS_PATH = CACHE_DIR / "trumedia_game_mappings.json"
TRUMEDIA_HOST_SUFFIX = ".trumedianetworks.com"
TRUMEDIA_DUKE_TEAM_ID = 730336256
DISCOVERY_TIMEOUT_MS = 30_000


class TruMediaError(RuntimeError):
    code = "trumedia_error"


class AuthenticationRequired(TruMediaError):
    code = "auth_required"


class MatchRequired(TruMediaError):
    code = "match_required"

    def __init__(self, candidates: list[dict]):
        super().__init__("A TruMedia game must be selected")
        self.candidates = candidates


@dataclass(frozen=True)
class MatchResult:
    status: str
    match: dict | None
    candidates: list[dict]


def admin_token_configured() -> bool:
    return bool(os.environ.get("CATCHER_STANCE_ADMIN_TOKEN"))


def check_admin_token(candidate: str) -> bool:
    expected = os.environ.get("CATCHER_STANCE_ADMIN_TOKEN", "")
    return bool(expected) and hmac.compare_digest(expected, candidate or "")


def validate_storage_state_payload(payload: object) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("storage state must be a JSON object")
    cookies = payload.get("cookies")
    origins = payload.get("origins")
    if not isinstance(cookies, list) or not isinstance(origins, list):
        raise ValueError("storage state must contain cookies and origins arrays")
    for cookie in cookies:
        if not isinstance(cookie, dict) or not all(key in cookie for key in ("name", "value", "domain")):
            raise ValueError("storage state contains an invalid cookie")
    return payload


def _safe_trumedia_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and (
        host.endswith(TRUMEDIA_HOST_SUFFIX) or host == "trumedianetworks.com"
    )


def _dp_rows(payload: object) -> list[dict]:
    if not isinstance(payload, dict):
        return []
    header = payload.get("header")
    rows = payload.get("rows")
    if not isinstance(header, list) or not isinstance(rows, list):
        return []
    names = [
        str(column.get("columnId") or column.get("name") or column.get("label") or "")
        if isinstance(column, dict)
        else ""
        for column in header
    ]
    return [
        {names[index]: value for index, value in enumerate(row[: len(names)]) if names[index]}
        for row in rows
        if isinstance(row, list)
    ]


def _parse_game_datetime(value: object, fallback_date: str) -> tuple[str, str, int | None]:
    text = str(value or "").strip()
    number_match = re.search(r"\((\d+)\)\s*$", text)
    game_number = int(number_match.group(1)) if number_match else None
    clean = re.sub(r"\s*\(\d+\)\s*$", "", text)
    try:
        parsed = datetime.strptime(clean, "%Y-%m-%d %H:%M:%S")
        return parsed.date().isoformat(), parsed.time().isoformat(timespec="minutes"), game_number
    except ValueError:
        return fallback_date, "", game_number


def parse_score_payload(payload: object, requested_date: str) -> list[dict]:
    """Normalize the detailed Scores dataset emitted by TruMedia's DataPoint API."""
    candidates: list[dict] = []
    for row in _dp_rows(payload):
        required = {"gameId", "awayTeamId", "homeTeamId", "gameDate"}
        if not required.issubset(row):
            continue
        try:
            game_id = int(row["gameId"])
            away_id = int(row["awayTeamId"])
            home_id = int(row["homeTeamId"])
        except (TypeError, ValueError):
            continue
        if TRUMEDIA_DUKE_TEAM_ID not in {away_id, home_id}:
            continue

        date, start_time, game_number = _parse_game_datetime(row.get("gameDate"), requested_date)
        duke_home = home_id == TRUMEDIA_DUKE_TEAM_ID
        opponent = str(
            (row.get("awayTeamName") if duke_home else row.get("homeTeamName")) or ""
        ).strip()
        duke_runs = row.get("homeRunsScored") if duke_home else row.get("awayRunsScored")
        opponent_runs = row.get("awayRunsScored") if duke_home else row.get("homeRunsScored")
        result = ""
        if duke_runs is not None and opponent_runs is not None:
            try:
                duke_score = int(duke_runs)
                opponent_score = int(opponent_runs)
            except (TypeError, ValueError):
                pass
            else:
                outcome = "W" if duke_score > opponent_score else "L" if duke_score < opponent_score else "T"
                result = f"{outcome} {duke_score}-{opponent_score}"
        path = f"/baseball/game-pitch-log/null/{date}/{game_id}"
        default = urlparse(TRUMEDIA_DEFAULT_URL)
        candidates.append(
            {
                "id": str(game_id),
                "trumedia_game_id": game_id,
                "date": date,
                "start_time": start_time,
                "game_number": game_number,
                "opponent": opponent,
                "opponent_key": normalize_opponent(opponent),
                "site": "home" if duke_home else "away",
                "result": result,
                "status": str(row.get("gameStatus") or ""),
                "trackman_available": bool(row.get("trackmanGameUID")),
                "url": f"{default.scheme}://{default.netloc}{path}",
            }
        )
    return sorted(candidates, key=lambda item: (item["date"], item["game_number"] or 99, item["start_time"], item["id"]))


class TruMediaProvider:
    def __init__(self, state_path: Path = PLAYWRIGHT_STATE_PATH) -> None:
        self.state_path = state_path

    def status(self) -> dict:
        base = {"admin_upload_enabled": admin_token_configured()}
        if not self.state_path.is_file():
            return {**base, "status": "missing", "connected": False}
        try:
            validate_storage_state_payload(json.loads(self.state_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, ValueError):
            return {**base, "status": "invalid", "connected": False}
        return {
            **base,
            "status": "configured",
            "connected": True,
            "updated_at": self.state_path.stat().st_mtime,
        }

    @staticmethod
    def _assert_authenticated(page) -> None:
        body = page.locator("body").inner_text(timeout=10_000).lower()
        if "sign in" in body or "log in" in body or "login" in page.url.lower():
            raise AuthenticationRequired("The saved TruMedia session is expired")

    def validate_live(self, candidate_path: Path) -> None:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                context = browser.new_context(storage_state=str(candidate_path))
                page = context.new_page()
                page.goto(TRUMEDIA_DEFAULT_URL, wait_until="domcontentloaded", timeout=DISCOVERY_TIMEOUT_MS)
                self._assert_authenticated(page)
            finally:
                browser.close()

    def discover_games(self, game_date: str) -> list[dict]:
        if not re.fullmatch(r"20\d{2}-\d{2}-\d{2}", game_date or ""):
            raise ValueError("invalid game date")
        if not self.state_path.is_file():
            raise AuthenticationRequired("TruMedia authentication is required")
        from playwright.sync_api import sync_playwright

        score_payloads: list[dict] = []
        pc = json.dumps({"bbtl": "D1", "bgd": game_date}, separators=(",", ":"))
        scores_url = f"{TRUMEDIA_DEFAULT_URL.rstrip('/')}/scores?{urlencode({'pc': pc})}"

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                context = browser.new_context(storage_state=str(self.state_path))
                page = context.new_page()

                def capture(response) -> None:
                    if urlparse(response.url).path != "/dp-proxy" or response.request.method != "POST":
                        return
                    try:
                        payload = response.json()
                    except Exception:
                        return
                    rows = _dp_rows(payload)
                    if rows and {"gameString", "awayTeamId", "homeTeamId"}.issubset(rows[0]):
                        score_payloads.append(payload)

                page.on("response", capture)
                page.goto(scores_url, wait_until="domcontentloaded", timeout=DISCOVERY_TIMEOUT_MS)
                self._assert_authenticated(page)
                deadline = time.monotonic() + 12
                while not score_payloads and time.monotonic() < deadline:
                    page.wait_for_timeout(250)
            finally:
                browser.close()

        candidates: dict[str, dict] = {}
        for payload in score_payloads:
            for candidate in parse_score_payload(payload, game_date):
                candidates[candidate["id"]] = candidate
        return list(candidates.values())


def _opponent_matches(game: dict, candidate: dict) -> bool:
    expected = str(game.get("opponent_key") or "")
    actual = str(candidate.get("opponent_key") or "")
    return bool(expected and actual) and (expected == actual or expected in actual or actual in expected)


def _result_matches(game: dict, candidate: dict) -> bool:
    expected = re.sub(r"\s+", "", str(game.get("result") or "")).replace("\u2013", "-").upper()
    actual = re.sub(r"\s+", "", str(candidate.get("result") or "")).replace("\u2013", "-").upper()
    return bool(expected and actual and expected == actual)


def match_game(game: dict, candidates: list[dict]) -> MatchResult:
    same_date = [candidate for candidate in candidates if candidate.get("date") == game.get("date")]
    exact = [candidate for candidate in same_date if _opponent_matches(game, candidate)]
    occurrence = int(game.get("occurrence") or 1)
    numbered = [candidate for candidate in exact if candidate.get("game_number") == occurrence]
    if len(numbered) == 1:
        candidate = numbered[0]
        if game.get("result") and candidate.get("result") and not _result_matches(game, candidate):
            return MatchResult("match_required", None, exact)
        return MatchResult("matched", candidate, exact)
    if len(exact) == 1 and (exact[0].get("game_number") in {None, occurrence}):
        if game.get("result") and exact[0].get("result") and not _result_matches(game, exact[0]):
            return MatchResult("match_required", None, exact)
        return MatchResult("matched", exact[0], exact)
    result_matches = [candidate for candidate in exact if _result_matches(game, candidate)]
    if len(result_matches) == 1 and not any(candidate.get("game_number") for candidate in exact):
        return MatchResult("matched", result_matches[0], exact)
    if len(exact) > 1 and not any(candidate.get("game_number") for candidate in exact):
        ordered = sorted(exact, key=lambda item: (item.get("start_time") or "", item.get("id") or ""))
        if occurrence <= len(ordered):
            return MatchResult("matched", ordered[occurrence - 1], ordered)
    ranked = exact or same_date
    return MatchResult("match_required" if ranked else "not_found", None, ranked)


def load_mappings() -> dict:
    try:
        payload = json.loads(MAPPINGS_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}


_MAPPING_FIELDS = (
    "id", "trumedia_game_id", "date", "start_time", "game_number", "opponent",
    "opponent_key", "site", "result", "status", "trackman_available", "url",
)


def save_mapping(game_id: str, candidate: dict) -> dict:
    if not candidate.get("trumedia_game_id") or not _safe_trumedia_url(str(candidate.get("url") or "")):
        raise ValueError("invalid TruMedia match")
    mappings = load_mappings()
    mappings[game_id] = {
        **{key: candidate.get(key) for key in _MAPPING_FIELDS},
        "confirmed_at": datetime.now().isoformat(timespec="seconds"),
    }
    atomic_write_json(MAPPINGS_PATH, mappings)
    return mappings[game_id]


def resolve_game(game: dict, provider: TruMediaProvider | None = None) -> MatchResult:
    cached = load_mappings().get(game["id"])
    if (
        cached
        and cached.get("trumedia_game_id")
        and _safe_trumedia_url(str(cached.get("url") or ""))
    ):
        return MatchResult("matched", cached, [cached])
    provider = provider or TruMediaProvider()
    result = match_game(game, provider.discover_games(str(game["date"])))
    if result.match is not None:
        save_mapping(game["id"], result.match)
    return result
