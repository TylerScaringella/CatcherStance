# Pitch Stance Pipeline Spec

## Summary

Build a frame-by-frame pipeline that turns each pitch clip into one pitch-level catcher stance label:
`Squat`, anatomical `LKD`, or anatomical `RKD`.

The implemented analyzer treats the clip as a temporal sequence and preserves source frame indices.
BaseballCV PHC provides semantic catcher proposals, the glove model supplies a ball-trajectory impact
anchor, and the existing pose/MLP artifacts classify only the stationary pre-impact set stance.

## Public Interface

`analyze_pitch_clip(video_path, config=None) -> PitchStanceResult`

`PitchStanceResult` should include:
- `label`
- `confidence`
- `window_start_frame`
- `window_end_frame`
- `camera_quality`
- `valid_frame_count`
- `rejection_reason` if the clip is unusable
- vote distribution, detector provenance, and quality flags

## Implementation Shape

1. Detect camera cuts without compacting the source-frame timeline.
2. Run BaseballCV PHC at low frequency and retain the strongest contiguous catcher track.
3. Run BaseballCV glove/ball/plate inference only over that broadcast segment.
4. Select the longest plausible ball trajectory and use its plate-side endpoint as impact.
5. Search `1.75-0.45s` before impact for a contiguous low-motion set window.
6. Use PHC boxes to select the catcher from full-frame `512px` pose inference.
7. Predict overlapping seven-frame MLP sequences and aggregate quality-weighted votes.
8. Abstain on insufficient pose coverage, temporal discontinuity, or vote ambiguity.

## Optional Integrations

- Download the two selected BaseballCV assets with
  `python src/download_models.py all`.
- The runtime loads the YOLO weights through Ultralytics; installing the full BaseballCV package is not required.
- RF-DETR remains an optional event-anchor backend until it beats YOLO anchor success by at least five percentage points without exceeding twice its runtime.
- `rfdetr 1.9.0` installed successfully on Python 3.14, but the 344MB BaseballCV
  checkpoint transfer stalled at 99% and ended with a server reset. No RF-DETR
  accuracy claim is made, and it is not promoted over the validated YOLO backend.
- Without external assets, the analyzer uses the full-frame pose gate and motion-followed stable-window fallback.

## Pseudocode

```text
function analyze_pitch_clip(video):
    cuts = detect_camera_cuts(video)
    catcher_track = propose_catcher_track(video, cuts)
    impact = detect_ball_trajectory_endpoint(video, catcher_track.bounds)

    search_bounds = impact.pre_window(1.75, 0.45) if impact else catcher_track.bounds
    poses = extract_full_frame_pose(video, search_bounds, catcher_track)
    window = choose_contiguous_stable_window(poses, cuts)

    if window is missing:
        return reject("no_stable_set_stance_window")

    predictions = classify_rolling_7_frame_sequences(window)
    label, confidence = quality_weighted_vote(predictions)
    return result(label, confidence, impact, window, quality_flags)
```

## Camera Filtering Rules

- reject frames with too few visible keypoints
- reject frames with obviously tall, upright, pitcher-like geometry
- reject frames that live too far from the home-plate anchor region
- reject clips where the catcher only appears in a tiny fraction of the clip
- reject temporal identity jumps and windows that cross missing-frame gaps or camera cuts
- preserve anatomical COCO left/right semantics

## Notes From Sample Exploration

- Direct BaseballCV YOLO weights work with the existing Ultralytics dependency.
- The corrected five-clip benchmark accepted all clips, matched `LKD, LKD, RKD, LKD, LKD`,
  and placed every impact anchor within `86ms` of human verification.
- The accuracy-first staged pipeline took `173.6s` for five clips on Apple MPS.
- Crop-based pose inference changed the legacy MLP input distribution. PHC therefore guides
  full-frame pose selection until the classifier is retrained on crop-derived features.
- The previous last-seven-valid-frames path changed four clips from pre-impact `LKD` to post-pitch `RKD`.
