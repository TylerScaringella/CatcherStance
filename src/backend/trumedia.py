from __future__ import annotations

import hmac
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

from project_paths import CACHE_DIR, PLAYWRIGHT_STATE_PATH

from .config import TRUMEDIA_DEFAULT_URL
from .schedule import normalize_opponent
from .storage import atomic_write_json

MAX_SESSION_BYTES = 1024 * 1024
MAPPINGS_PATH = CACHE_DIR / "trumedia_game_mappings.json"
TRUMEDIA_HOST_SUFFIX = ".trumedianetworks.com"


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
    return parsed.scheme == "https" and (host.endswith(TRUMEDIA_HOST_SUFFIX) or host == "trumedianetworks.com")


class TruMediaProvider:
    def __init__(self, state_path: Path = PLAYWRIGHT_STATE_PATH) -> None:
        self.state_path = state_path

    def status(self) -> dict:
        if not self.state_path.is_file():
            return {"status": "missing", "connected": False, "admin_upload_enabled": admin_token_configured()}
        try:
            validate_storage_state_payload(json.loads(self.state_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, ValueError):
            return {"status": "invalid", "connected": False, "admin_upload_enabled": admin_token_configured()}
        return {"status": "configured", "connected": True, "admin_upload_enabled": admin_token_configured()}

    def validate_live(self, candidate_path: Path) -> None:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                context = browser.new_context(storage_state=str(candidate_path))
                page = context.new_page()
                page.goto(TRUMEDIA_DEFAULT_URL, wait_until="domcontentloaded", timeout=30000)
                body = page.locator("body").inner_text(timeout=10000).lower()
                if "sign in" in body or "log in" in body or "login" in page.url.lower():
                    raise AuthenticationRequired("The uploaded TruMedia session is expired")
            finally:
                browser.close()

    def discover_games(self) -> list[dict]:
        if not self.state_path.is_file():
            raise AuthenticationRequired("TruMedia authentication is required")
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                context = browser.new_context(storage_state=str(self.state_path))
                page = context.new_page()
                page.goto(TRUMEDIA_DEFAULT_URL, wait_until="domcontentloaded", timeout=30000)
                body = page.locator("body").inner_text(timeout=10000).lower()
                if "sign in" in body or "log in" in body or "login" in page.url.lower():
                    raise AuthenticationRequired("The saved TruMedia session is expired")
                entries = page.locator("a[href]").evaluate_all(
                    "els => els.map(el => ({text: (el.innerText || el.textContent || '').trim(), href: el.href}))"
                )
            finally:
                browser.close()

        candidates: list[dict] = []
        seen: set[str] = set()
        date_pattern = re.compile(r"(?:([A-Z][a-z]{2})\s+(\d{1,2}),?\s+(20\d{2})|(20\d{2})-(\d{2})-(\d{2}))")
        for entry in entries:
            text = re.sub(r"\s+", " ", str(entry.get("text") or "")).strip()
            href = str(entry.get("href") or "")
            if not text or href in seen or not _safe_trumedia_url(href):
                continue
            match = date_pattern.search(text)
            if not match:
                continue
            if match.group(1):
                date = datetime.strptime(
                    f"{match.group(1)} {match.group(2)} {match.group(3)}", "%b %d %Y"
                ).date().isoformat()
            else:
                date = f"{match.group(4)}-{match.group(5)}-{match.group(6)}"
            opponent_text = text[match.end():].strip(" -|:") or text[:match.start()].strip(" -|:")
            candidates.append(
                {
                    "id": re.sub(r"[^A-Za-z0-9._-]+", "-", urlparse(href).path.strip("/"))[-180:] or f"game-{len(candidates) + 1}",
                    "date": date,
                    "opponent": opponent_text,
                    "opponent_key": normalize_opponent(opponent_text),
                    "url": urljoin(TRUMEDIA_DEFAULT_URL, href),
                }
            )
            seen.add(href)
        return candidates


def match_game(game: dict, candidates: list[dict]) -> MatchResult:
    same_date = [candidate for candidate in candidates if candidate.get("date") == game.get("date")]
    exact = [
        candidate for candidate in same_date
        if candidate.get("opponent_key") == game.get("opponent_key")
        or game.get("opponent_key") in str(candidate.get("opponent_key") or "")
        or str(candidate.get("opponent_key") or "") in str(game.get("opponent_key") or "")
    ]
    occurrence = int(game.get("occurrence") or 1)
    if len(exact) == 1:
        return MatchResult("matched", exact[0], exact)
    if len(exact) >= occurrence:
        return MatchResult("matched", exact[occurrence - 1], exact)
    ranked = exact or same_date
    return MatchResult("match_required" if ranked else "not_found", None, ranked)


def load_mappings() -> dict:
    try:
        payload = json.loads(MAPPINGS_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}


def save_mapping(game_id: str, candidate: dict) -> dict:
    if not _safe_trumedia_url(str(candidate.get("url") or "")):
        raise ValueError("invalid TruMedia match")
    mappings = load_mappings()
    mappings[game_id] = {
        "id": str(candidate.get("id") or ""),
        "date": str(candidate.get("date") or ""),
        "opponent": str(candidate.get("opponent") or ""),
        "opponent_key": str(candidate.get("opponent_key") or ""),
        "url": str(candidate["url"]),
        "confirmed_at": datetime.now().isoformat(timespec="seconds"),
    }
    atomic_write_json(MAPPINGS_PATH, mappings)
    return mappings[game_id]


def resolve_game(game: dict, provider: TruMediaProvider | None = None) -> MatchResult:
    cached = load_mappings().get(game["id"])
    if cached and _safe_trumedia_url(str(cached.get("url") or "")):
        return MatchResult("matched", cached, [cached])
    provider = provider or TruMediaProvider()
    result = match_game(game, provider.discover_games())
    if result.match is not None:
        save_mapping(game["id"], result.match)
    return result
