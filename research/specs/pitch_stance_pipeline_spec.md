# Pitch Stance Pipeline Spec

## Summary

Build a frame-by-frame pipeline that turns each pitch clip into one pitch-level catcher stance label:
`Squat`, `LKD`, or `RKD`.

The pipeline should treat the clip as a short temporal sequence, not as a single-image classification task.

## Public Interface

`analyze_pitch_clip(video_path) -> PitchStanceResult`

`PitchStanceResult` should include:
- `label`
- `confidence`
- `window_start_frame`
- `window_end_frame`
- `camera_quality`
- `valid_frame_count`
- `rejection_reason` if the clip is unusable

## Implementation Shape

1. Decode the clip frame-by-frame.
2. Run catcher detection on every frame with the existing plate-anchor ROI and invalid-zone filters.
3. Collect only frames that pass the catcher gate.
4. Determine the set-stance window:
   - prefer an external event anchor if ball/glove tracking is available
   - otherwise score sliding windows and choose the lowest-motion stable segment
5. Normalize pose coordinates by torso or box scale.
6. Predict a frame-level stance label from keypoint geometry or a lightweight classifier.
7. Aggregate the frame labels with majority vote or rolling median.
8. Return the pitch-level label with quality metadata.

## Pseudocode

```text
function analyze_pitch_clip(video):
    frames = decode(video)
    candidates = []

    for frame in frames:
        det = detect_catcher(frame)
        if det is None:
            continue
        if is_bad_camera(frame, det):
            continue

        pose = normalize(det.keypoints, det.box)
        frame_label = classify_frame(pose)
        candidates.append({
            frame_index,
            frame_label,
            score,
            det_confidence,
            anchor_distance,
        })

    if too_few_candidates(candidates):
        return reject("no_stable_catcher_window")

    if event_anchor_available:
        window = anchor_window(contact_or_release_frame, pre_seconds=1.5)
    else:
        window = choose_low_motion_window(candidates, window_seconds=1.5)

    window_labels = labels_inside(window, candidates)
    pitch_label = majority_vote(window_labels)
    pitch_confidence = aggregate_confidence(window_labels)

    return result(pitch_label, pitch_confidence, window, quality_flags)
```

## Camera Filtering Rules

- reject frames with too few visible keypoints
- reject frames with obviously tall, upright, pitcher-like geometry
- reject frames that live too far from the home-plate anchor region
- reject clips where the catcher only appears in a tiny fraction of the clip

## Notes From Sample Exploration

- BaseballCV / RF-DETR utilities are not installed in this repo environment, so they should be treated as optional integrations rather than hard dependencies.
- The current detector already does useful disambiguation with anchor distance, invalid zones, and lower-body geometry.
- The measured sample clips show a stable low-motion window later in the clip, which supports a hybrid windowing strategy.
