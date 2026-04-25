from __future__ import annotations

import csv
import os

MANIFEST_FIELDS = [
    "card_dom_index",
    "clip_id",
    "s3_url",
    "saved_path",
    "status",
    "attempts",
    "error",
]


def load_manifest(path: str):
    rows = []
    by_clip_id = {}
    by_s3_url = {}

    if os.path.exists(path):
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                normalized = {
                    "card_dom_index": row.get("card_dom_index", ""),
                    "clip_id": row.get("clip_id", ""),
                    "s3_url": row.get("s3_url", ""),
                    "saved_path": row.get("saved_path", ""),
                    "status": row.get("status", ""),
                    "attempts": row.get("attempts", "0"),
                    "error": row.get("error", ""),
                }
                rows.append(normalized)
                if normalized["clip_id"]:
                    by_clip_id[normalized["clip_id"]] = normalized
                if normalized["s3_url"]:
                    by_s3_url[normalized["s3_url"]] = normalized

    return rows, by_clip_id, by_s3_url


def write_manifest(path: str, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def upsert_manifest_row(rows, by_clip_id, by_s3_url, row):
    clip_id = row["clip_id"]
    s3_url = row["s3_url"]

    existing = None
    if clip_id in by_clip_id:
        existing = by_clip_id[clip_id]
    elif s3_url in by_s3_url:
        existing = by_s3_url[s3_url]

    if existing is None:
        rows.append(row)
        by_clip_id[clip_id] = row
        by_s3_url[s3_url] = row
        return row

    existing.update(row)
    by_clip_id[clip_id] = existing
    by_s3_url[s3_url] = existing
    return existing
