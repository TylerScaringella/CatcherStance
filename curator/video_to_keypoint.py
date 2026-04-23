from pathlib import Path
import argparse
import sys
import os
import csv
import multiprocessing as mp

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from notebook.catcher_detection import CatcherDetectionConfig, detect_catcher_from_res_item

LABELED_VIDEO_PATH = PROJECT_ROOT / "data/labeled_videos/labeled-videos-2026-04-22-00-05-e3d836a2.csv"
DATASET_OUTPUT_PATH = PROJECT_ROOT / "data/dataset/keypoints-labeled.csv"
MODEL_PATH = PROJECT_ROOT / "notebook/yolo26n-pose.pt"

DESIRED_COLUMNS = ['id', 'choice', 'video']
NUM_FRAMES = 7
CSV_FIELDNAMES = ["choice", "id", "features", "status"]
DEFAULT_NUM_WORKERS = 2

_worker_model = None
filtered_videos = None


def load_filtered_videos():
    """Load labeled videos and convert Label Studio paths to local file paths."""
    original_videos = []

    with open(LABELED_VIDEO_PATH, mode='r', newline='') as f:
        reader = csv.DictReader(f)
        original_videos = [row for row in reader]

    filtered_videos = []

    for video in original_videos:
        # If labeled as bad, or no label, do not include it in the dataset.
        if video['choice'] == 'Bad Video' or video['choice'] == '':
            continue

        filtered_video = {k: v for k, v in video.items() if k in DESIRED_COLUMNS}

        # Transform the local studio path into our relative directory path.
        video_file_name = video['video'].replace('/data/local-files/?d=downloads/', '')
        video_path = PROJECT_ROOT / "downloader/downloads" / video_file_name

        filtered_video['video'] = str(video_path)
        filtered_videos.append(filtered_video)

    return filtered_videos


cfg = CatcherDetectionConfig(
    search_roi_norm=(0.30, 0.42, 0.70, 0.98),
    frame_gate_roi_norm=(0.24, 0.34, 0.76, 1.00),
    anchor_norm=(0.50, 0.78),
    invalid_zones_norm=[
        (0.00, 0.00, 0.18, 1.00),
        (0.82, 0.00, 1.00, 1.00),
        (0.00, 0.00, 1.00, 0.18),
    ],
    max_anchor_dist_norm=0.30,
    max_final_anchor_dist_norm=0.30,
    min_score=0.48,
    min_margin=0,
    min_aspect_ratio=0.7,
    max_compactness=2.30,
)


def pad_last(items, target_len):
    """
    Transform list into a list of a fixed length by repeating items

    Args:
        - items (list): The list to mutate
        - target_len (int): The desired number of items in the list

    Returns:
        - items (list): The mutated list
    """

    if len(items) == 0:
        return None
    while len(items) < target_len:
        items.append(items[-1].copy() if hasattr(items[-1], "copy") else items[-1])
    
    return items


LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6

def normalize_keypoints(kpts):
    kpts = kpts.astype(np.float32).copy()

    # center = midpoint of hips
    hip_center = (kpts[LEFT_HIP] + kpts[RIGHT_HIP]) / 2.0

    # scale = shoulder-to-hip distance or box height fallback
    shoulder_center = (kpts[LEFT_SHOULDER] + kpts[RIGHT_SHOULDER]) / 2.0
    scale = np.linalg.norm(shoulder_center - hip_center)

    if scale < 1e-6:
        y_min, x_min = np.min(kpts[:, 1]), np.min(kpts[:, 0])
        y_max, x_max = np.max(kpts[:, 1]), np.max(kpts[:, 0])
        scale = max(y_max - y_min, x_max - x_min, 1.0)

    kpts = (kpts - hip_center) / scale
    return kpts


def init_worker(model_path):
    """
    Initialize a YOLO model inside each worker process.

    The model is intentionally not created in the parent and shared across
    processes. Each process owns its model instance, which avoids pickle/device
    issues and keeps multiprocessing behavior predictable on macOS.
    """
    global _worker_model
    _worker_model = YOLO(str(model_path))


def process_video(video):
    """
    Process a video by running it through YOLO pose detection model and then each frame through our catcher detection model.

    Args:
        - video (str): The path of the video to process
    
    Returns:
        - sample (list): The fixed-length vector of the processed video. If this is None, we were unable to detect at least 1 frame with a catcher in it to add padding ot make it valid.
    """

    # TODO: MAKE SURE THAT WE ARE NORMALIZING
    # ^ LOOK AT CHATGPT HISTORY TO SEE HOW THAT IS DONE

    if _worker_model is None:
        raise RuntimeError("Worker YOLO model has not been initialized.")

    # keypoints of detected catcher in each frame
    catcher_detections = []

    results = _worker_model(
        video, 
        show=False,
        stream=True,
        save=False,
        verbose=False,

        # Testing to see if we can get it to go faster
        imgsz=512,
        vid_stride=2
    )

    for result in results:
        # Detect the catcher at this frame

        detected_catcher = detect_catcher_from_res_item(result, cfg=cfg)

        # If we have no catcher detection, do nothing
        if detected_catcher is None:
            continue

        keypoints = detected_catcher['keypoints']
        keypoints = normalize_keypoints(keypoints)

        catcher_detections.append(keypoints)


    # none of the frames had a catcher detected, nothing that we can do
    if len(catcher_detections) == 0:
        return None
    
    # take the last 7 keypoints of catcher detection
    selected = catcher_detections[-NUM_FRAMES:]

    # add repeated padding if there are less than 7 valid frames
    selected = pad_last(selected, NUM_FRAMES)

    # TODO: see if we want to do this in the dataset, or right before we feed it into the model
    # convert the frames into a fixed length vector
    # 7 * 17 * 2 = 238 values
    sample = np.concatenate([frame.reshape(-1) for frame in selected], axis=0)

    return sample


def ensure_output_csv_schema(output_path):
    """
    Ensure the output CSV has the resumable schema.

    Older runs wrote only choice,id,features. If that file exists, rewrite it
    once with status=ok for those already-created samples so resume can treat
    them as completed IDs and future appends match the header.
    """
    if not output_path.exists():
        return

    with open(output_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames == CSV_FIELDNAMES:
            return

        rows = []
        for row in reader:
            rows.append({
                "choice": row.get("choice", ""),
                "id": row.get("id", ""),
                "features": row.get("features", ""),
                "status": row.get("status") or "ok",
            })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def load_completed_ids(output_path):
    """
    Load IDs already present in the CSV.

    Resume works by treating any row already written by the main process as
    completed, including status=no_valid_catcher. That prevents invalid videos
    from being retried forever after a restart.
    """
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
    """
    Return videos that still need work, de-duplicated by ID.

    This avoids duplicate work in two ways:
    1. IDs already saved in the output CSV are not dispatched on restart.
    2. Duplicate IDs in the source labels are collapsed before multiprocessing,
       so two workers are never given the same ID during one run.
    """
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
    """
    Worker entry point.

    Workers only compute and return a dataset row. The parent process is the
    only process that writes to disk, which keeps the CSV append path safe.
    """
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
    """Append one result and flush it immediately for crash-safe progress."""
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

        # Use spawn explicitly for macOS/laptop friendliness. tqdm is wrapped
        # around imap_unordered so progress advances as videos finish, not in
        # source order.
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a resumable catcher keypoint dataset from labeled videos."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of worker processes to use. Keep this small on a laptop.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    generate_dataset(num_workers=args.workers)
