"""Prepare a balanced, reviewable manifest for stance and pitch-event annotation."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

GROUP_PATTERN = re.compile(r"pitch-pix-([^-]+)-\d+-\d+\.mp4$")


def group_for_video(video: str, row_id: str) -> str:
    match = GROUP_PATTERN.search(video)
    return match.group(1) if match else Path(video).name.split("_")[0] or f"row-{row_id}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("data/raw/labeled_videos/labeled-videos-2026-04-22-00-05-e3d836a2.csv"))
    parser.add_argument("--output", type=Path, default=Path("research/annotations/generated/pitch_event_annotation_manifest.csv"))
    parser.add_argument("--per-class", type=int, default=50)
    args = parser.parse_args()

    with args.labels.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("choice") in {"Knee-down Left", "Knee-down Right", "Squat"}]

    selected = []
    for label in ("Knee-down Left", "Knee-down Right", "Squat"):
        candidates = [row for row in rows if row.get("choice") == label]
        by_group: dict[str, list[dict]] = {}
        for row in candidates:
            by_group.setdefault(group_for_video(row.get("video", ""), row.get("id", "")), []).append(row)
        groups = sorted(by_group)
        cursor = 0
        while groups and len([row for row in selected if row["choice"] == label]) < args.per_class:
            group = groups[cursor % len(groups)]
            selected.append(by_group[group].pop(0))
            groups = [item for item in groups if by_group[item]]
            cursor += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "choice", "video", "source_group", "annotation_status", "set_start_frame", "set_end_frame", "release_frame", "arrival_frame", "event_reliable", "camera_quality", "notes"]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            writer.writerow({
                "id": row.get("id", ""), "choice": row.get("choice", ""), "video": row.get("video", ""),
                "source_group": group_for_video(row.get("video", ""), row.get("id", "")), "annotation_status": "pending",
                "set_start_frame": "", "set_end_frame": "", "release_frame": "", "arrival_frame": "",
                "event_reliable": "", "camera_quality": "", "notes": "",
            })
    print(f"wrote {len(selected)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
