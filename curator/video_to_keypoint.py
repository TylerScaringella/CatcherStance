from pathlib import Path
import sys
import os
import csv

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from notebook.catcher_detection import CatcherDetectionConfig, detect_catcher_from_res_item

# Load the CSV of the labeled videos to prep for creating the keypoint dataset

LABELED_VIDEO_PATH = PROJECT_ROOT / "data/labeled_videos/labeled-videos-2026-04-22-00-05-e3d836a2.csv"

original_videos = []

with open(LABELED_VIDEO_PATH, mode='r', newline='') as f:
    reader = csv.DictReader(f)
    original_videos = [row for row in reader]

# Filter out videos labeled as a bad video
# Transform the path to be relative to working directory instead of LabelStudio
# Also get rid of the columns that we don't want

DESIRED_COLUMNS = ['id', 'choice', 'video']

filtered_videos = []

for video in original_videos:
    # if labeled as bad, or no label
    if video['choice'] == 'Bad Video' or video['choice'] == '':
        continue

    filtered_video = {k: v for k, v in video.items() if k in DESIRED_COLUMNS}

    # Transform the local studio path into our relative directory path
    video_file_name = video['video'].replace('/data/local-files/?d=downloads/', '')
    video_path = PROJECT_ROOT / "downloader/downloads" / video_file_name

    filtered_video['video'] = str(video_path)
    
    filtered_videos.append(filtered_video)

# Load the pose model
model = YOLO(str(PROJECT_ROOT / 'notebook/yolo26n-pose.pt'))

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


# The amount of detected frames we are going to use to curate our dataset
NUM_FRAMES = 7

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

    print(video)

    # keypoints of detected catcher in each frame
    catcher_detections = []

    results = model(
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


# Process all of the videos and create the dataset

DATASET_OUTPUT_PATH = PROJECT_ROOT / "data/dataset/keypoints-labeled.csv"
os.makedirs(DATASET_OUTPUT_PATH.parent, exist_ok=True)

# Load already-processed IDs so you can resume after a crash
done_ids = set()
if DATASET_OUTPUT_PATH.exists():
    with open(DATASET_OUTPUT_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done_ids.add(row["id"])

fieldnames = ["choice", "id", "features"]
file_exists = DATASET_OUTPUT_PATH.exists()

with open(DATASET_OUTPUT_PATH, mode="a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)

    if not file_exists:
        writer.writeheader()

    for video in tqdm(filtered_videos, desc="Videos"):
        if video["id"] in done_ids:
            continue

        processed_video = process_video(video["video"])

        # not valid, so we skip it
        if processed_video is None:
            continue

        dataset_entry = {
            "choice": video["choice"],
            "id": video["id"],
            "features": processed_video.tolist(),  # CSV-safe
        }

        writer.writerow(dataset_entry)
        f.flush()
