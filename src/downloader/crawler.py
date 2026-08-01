from __future__ import annotations

import hashlib
import os
import re
import time
from urllib.parse import unquote, urlparse

from .config import DOWNLOAD_DIR, START_URL, STORAGE_STATE_PATH
from .manifest import upsert_manifest_row

DUKE_TEAM_ID = 730336256
PITCH_DISCOVERY_TIMEOUT_SECONDS = 15


def is_s3_mp4_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        path = (parsed.path or "").lower()
        is_amazon_s3 = (
            host == "s3.amazonaws.com"
            or host.endswith(".s3.amazonaws.com")
            or (".s3." in host and "amazonaws.com" in host)
        )
        return parsed.scheme in ("http", "https") and is_amazon_s3 and path.endswith(".mp4")
    except Exception:
        return False


def is_stable_media_ref(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme == "s3" and bool(parsed.netloc) and parsed.path.lower().endswith(".mp4")


def extract_clip_id(url: str) -> str:
    try:
        path = unquote(urlparse(url).path)
        filename = os.path.basename(path)
        return filename[:-4] if filename.lower().endswith(".mp4") else filename
    except Exception:
        return hashlib.md5(url.encode(), usedforsecurity=False).hexdigest()


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[^\w.\-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name[:180] if name else "clip"


def get_output_path(clip_id: str, download_dir: str = DOWNLOAD_DIR) -> str:
    return os.path.join(download_dir, f"{clip_id}.mp4")


def _dp_rows(payload: object) -> list[dict]:
    if not isinstance(payload, dict) or not isinstance(payload.get("header"), list) or not isinstance(payload.get("rows"), list):
        return []
    names = [
        str(column.get("columnId") or column.get("name") or column.get("label") or "")
        if isinstance(column, dict)
        else ""
        for column in payload["header"]
    ]
    return [
        {names[index]: value for index, value in enumerate(row[: len(names)]) if names[index]}
        for row in payload["rows"]
        if isinstance(row, list)
    ]


def _broadcast_ref(video_angles: object) -> str:
    if not isinstance(video_angles, list):
        return ""
    for angle in video_angles:
        if not isinstance(angle, dict) or str(angle.get("view") or "").casefold() != "broadcast":
            continue
        reference = str(angle.get("url") or "")
        return reference if is_stable_media_ref(reference) else ""
    return ""


def parse_pitch_payload(
    payload: object,
    *,
    download_dir: str = DOWNLOAD_DIR,
    catching_team_id: int = DUKE_TEAM_ID,
) -> tuple[list[dict], dict]:
    """Convert TruMedia pitch rows into a durable, credential-free manifest schema."""
    selected: dict[str, dict] = {}
    total_rows = 0
    for pitch in _dp_rows(payload):
        if "videoAngles" not in pitch or "uniqPitchId" not in pitch:
            continue
        total_rows += 1
        try:
            row_team_id = int(pitch.get("catchingTeamId"))
        except (TypeError, ValueError):
            continue
        if row_team_id != catching_team_id:
            continue
        pitch_id = sanitize_filename(str(pitch.get("uniqPitchId") or ""))
        if not pitch_id:
            continue
        clip_id = sanitize_filename(f"tm-{pitch_id}")
        media_ref = _broadcast_ref(pitch.get("videoAngles"))
        skipped = not media_ref
        selected[pitch_id] = {
            "card_dom_index": "",
            "clip_id": clip_id,
            "pitch_id": pitch_id,
            "trumedia_game_id": str(pitch.get("gameId") or ""),
            "pitch_number": str(pitch.get("pitchNumInGame") or ""),
            "inning": str(pitch.get("inn") or ""),
            "count": str(pitch.get("count") or ""),
            "pitcher": str(pitch.get("pitcherAbbrevName") or pitch.get("pitcher") or ""),
            "batter": str(pitch.get("batterAbbrevName") or pitch.get("batter") or ""),
            "catcher_id": str(pitch.get("catcherId") or ""),
            "catcher": str(pitch.get("catcherAbbrevName") or ""),
            "catching_team": str(pitch.get("catchingTeam") or ""),
            "catching_team_id": str(row_team_id),
            "pitching_team": str(pitch.get("pitchingTeam") or ""),
            "video_angle": "Broadcast" if media_ref else "",
            "media_ref": media_ref,
            "s3_url": "",
            "saved_path": get_output_path(clip_id, download_dir=download_dir),
            "status": "skipped" if skipped else "pending",
            "attempts": "0",
            "error": "",
            "skip_reason": "broadcast_video_unavailable" if skipped else "",
        }
    rows = sorted(selected.values(), key=lambda row: int(row["pitch_number"] or 0))
    stats = {
        "total_pitches": total_rows,
        "selected_pitches": len(rows),
        "downloadable_pitches": sum(not row["skip_reason"] for row in rows),
        "skipped_pitches": sum(bool(row["skip_reason"]) for row in rows),
        "catching_team_id": catching_team_id,
    }
    return rows, stats


def get_logged_in_context(browser, start_url=START_URL, storage_state_path=STORAGE_STATE_PATH):
    if not os.path.exists(storage_state_path):
        raise RuntimeError("TruMedia authentication is required")
    context = browser.new_context(storage_state=storage_state_path, accept_downloads=False)
    return context, context.new_page(), False


def _assert_authenticated(page) -> None:
    body = page.locator("body").inner_text(timeout=10_000).lower()
    if "sign in" in body or "log in" in body or "login" in page.url.lower():
        raise RuntimeError("TruMedia authentication is required")


def discover_pitch_media(
    page,
    start_url: str,
    rows: list[dict],
    by_clip_id: dict,
    by_media_ref: dict,
    *,
    download_dir: str = DOWNLOAD_DIR,
) -> tuple[int, dict]:
    payloads: list[dict] = []

    def capture(response) -> None:
        if urlparse(response.url).path != "/dp-proxy" or response.request.method != "POST":
            return
        try:
            payload = response.json()
        except Exception:
            return
        parsed = _dp_rows(payload)
        if parsed and {"videoAngles", "uniqPitchId", "catchingTeamId"}.issubset(parsed[0]):
            payloads.append(payload)

    page.on("response", capture)
    page.goto(start_url, wait_until="domcontentloaded", timeout=30_000)
    _assert_authenticated(page)
    deadline = time.monotonic() + PITCH_DISCOVERY_TIMEOUT_SECONDS
    while not payloads and time.monotonic() < deadline:
        page.wait_for_timeout(250)
    if not payloads:
        raise RuntimeError("TruMedia pitch data is unavailable for this game")

    discovered: dict[str, dict] = {}
    stats = {}
    for payload in payloads:
        parsed_rows, stats = parse_pitch_payload(payload, download_dir=download_dir)
        for row in parsed_rows:
            discovered[row["pitch_id"]] = row
    if not discovered:
        raise RuntimeError("No Duke-catching pitches were found for this game")

    added = 0
    for row in discovered.values():
        existing = by_clip_id.get(row["clip_id"])
        if existing is None:
            added += 1
        elif existing.get("status") in {"downloaded", "cleaned"}:
            row["status"] = existing["status"]
            row["attempts"] = existing.get("attempts", "0")
        upsert_manifest_row(rows, by_clip_id, by_media_ref, row)
    return added, stats


def sign_media_ref(page, media_ref: str) -> str:
    if not is_stable_media_ref(media_ref):
        raise ValueError("invalid TruMedia media reference")
    origin = f"{urlparse(page.url).scheme}://{urlparse(page.url).netloc}"
    response = page.request.get(f"{origin}/s3bucket", params={"path": media_ref}, timeout=15_000)
    if not response.ok:
        raise RuntimeError("TruMedia could not authorize a pitch video")
    signed_url = response.text().strip()
    if not is_s3_mp4_url(signed_url) or not urlparse(signed_url).query:
        raise RuntimeError("TruMedia returned an invalid pitch video authorization")
    return signed_url


# Compatibility name for callers that still import the previous crawler function.
def collect_s3_urls(page, rows, by_clip_id, by_s3_url, download_dir=DOWNLOAD_DIR):
    added, _ = discover_pitch_media(
        page, page.url, rows, by_clip_id, by_s3_url, download_dir=download_dir
    )
    return added
