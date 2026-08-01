from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

MANIFEST_FIELDS = [
    "card_dom_index",
    "clip_id",
    "pitch_id",
    "trumedia_game_id",
    "pitch_number",
    "inning",
    "count",
    "pitcher",
    "batter",
    "catcher_id",
    "catcher",
    "catching_team",
    "catching_team_id",
    "pitching_team",
    "video_angle",
    "media_ref",
    "s3_url",
    "saved_path",
    "status",
    "attempts",
    "error",
    "skip_reason",
]


def load_manifest(path: str):
    rows = []
    by_clip_id = {}
    by_media_ref = {}

    if os.path.exists(path):
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                normalized = {field: row.get(field, "") for field in MANIFEST_FIELDS}
                normalized["attempts"] = normalized["attempts"] or "0"
                # Signed URLs from legacy manifests remain usable for this process only.
                legacy_url = normalized["s3_url"]
                if legacy_url:
                    normalized["download_url"] = legacy_url
                    normalized["s3_url"] = ""
                rows.append(normalized)
                if normalized["clip_id"]:
                    by_clip_id[normalized["clip_id"]] = normalized
                reference = normalized["media_ref"] or legacy_url
                if reference:
                    by_media_ref[reference] = normalized

    return rows, by_clip_id, by_media_ref


def write_manifest(path: str, rows):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".part",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def upsert_manifest_row(rows, by_clip_id, by_media_ref, row):
    clip_id = row["clip_id"]
    media_ref = row.get("media_ref") or row.get("s3_url") or ""

    existing = None
    if clip_id in by_clip_id:
        existing = by_clip_id[clip_id]
    elif media_ref and media_ref in by_media_ref:
        existing = by_media_ref[media_ref]

    if existing is None:
        rows.append(row)
        by_clip_id[clip_id] = row
        if media_ref:
            by_media_ref[media_ref] = row
        return row

    existing.update(row)
    by_clip_id[clip_id] = existing
    if media_ref:
        by_media_ref[media_ref] = existing
    return existing
