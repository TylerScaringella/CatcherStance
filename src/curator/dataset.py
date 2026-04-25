from __future__ import annotations

import csv
import multiprocessing as mp
import os

from tqdm import tqdm

from .config import (
    CSV_FIELDNAMES,
    DATASET_OUTPUT_PATH,
    DEFAULT_NUM_WORKERS,
    DESIRED_COLUMNS,
    LABELED_VIDEO_PATH,
    MODEL_PATH,
    PROJECT_ROOT,
)
from .features import init_worker, process_video

filtered_videos = None


def load_filtered_videos():
    with open(LABELED_VIDEO_PATH, mode="r", newline="") as f:
        original_videos = [row for row in csv.DictReader(f)]

    filtered = []
    for video in original_videos:
        if video["choice"] == "Bad Video" or video["choice"] == "":
            continue

        filtered_video = {k: v for k, v in video.items() if k in DESIRED_COLUMNS}
        video_file_name = video["video"].replace("/data/local-files/?d=downloads/", "")
        filtered_video["video"] = str(PROJECT_ROOT / "data" / "downloader" / "downloads" / video_file_name)
        filtered.append(filtered_video)

    return filtered


def ensure_output_csv_schema(output_path):
    if not output_path.exists():
        return

    with open(output_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames == CSV_FIELDNAMES:
            return

        rows = []
        for row in reader:
            rows.append(
                {
                    "choice": row.get("choice", ""),
                    "id": row.get("id", ""),
                    "features": row.get("features", ""),
                    "status": row.get("status") or "ok",
                }
            )

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def load_completed_ids(output_path):
    done_ids = set()
    if not output_path.exists():
        return done_ids

    with open(output_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row.get("id")
            if video_id:
                done_ids.add(video_id)

    return done_ids


def build_pending_videos(videos, done_ids):
    pending = []
    queued_ids = set()

    for video in videos:
        video_id = video["id"]
        if video_id in done_ids or video_id in queued_ids:
            continue

        pending.append(video)
        queued_ids.add(video_id)

    return pending


def process_video_record(video_record):
    try:
        processed_video = process_video(video_record["video"])
    except Exception as exc:
        return {
            "choice": video_record["choice"],
            "id": video_record["id"],
            "features": "",
            "status": f"error:{type(exc).__name__}",
        }

    if processed_video is None:
        return {
            "choice": video_record["choice"],
            "id": video_record["id"],
            "features": "",
            "status": "no_valid_catcher",
        }

    return {
        "choice": video_record["choice"],
        "id": video_record["id"],
        "features": processed_video.tolist(),
        "status": "ok",
    }


def append_result(writer, output_file, row):
    writer.writerow(row)
    output_file.flush()
    os.fsync(output_file.fileno())


def generate_dataset(num_workers=DEFAULT_NUM_WORKERS):
    global filtered_videos

    os.makedirs(DATASET_OUTPUT_PATH.parent, exist_ok=True)
    ensure_output_csv_schema(DATASET_OUTPUT_PATH)

    if filtered_videos is None:
        filtered_videos = load_filtered_videos()

    done_ids = load_completed_ids(DATASET_OUTPUT_PATH)
    pending_videos = build_pending_videos(filtered_videos, done_ids)
    file_exists = DATASET_OUTPUT_PATH.exists()

    with open(DATASET_OUTPUT_PATH, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)

        if not file_exists:
            writer.writeheader()
            f.flush()
            os.fsync(f.fileno())

        if not pending_videos:
            print("No pending videos to process.")
            return

        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=num_workers,
            initializer=init_worker,
            initargs=(MODEL_PATH,),
        ) as pool:
            results = pool.imap_unordered(process_video_record, pending_videos)

            for row in tqdm(results, total=len(pending_videos), desc="Videos"):
                video_id = row["id"]
                if video_id in done_ids:
                    continue

                append_result(writer, f, row)
                done_ids.add(video_id)
