# Pitch event annotation

The pitch-event annotation manifest is a local generated review manifest, not a replacement for
the original label export. Generate it under `research/annotations/generated/` and keep it out of
version control. Reviewers fill in the frame fields using the local pitch clips:

- `set_start_frame` / `set_end_frame`: stationary catcher set stance.
- `release_frame`: pitcher release when visible.
- `arrival_frame`: ball reaches the plate/catcher receiving zone.
- `event_reliable`: `yes` or `no`.
- `camera_quality`: `good`, `usable`, or `poor`.

The manifest is generated deterministically from the raw labels. Pending rows must not be treated
as ground truth by the audit or production pipeline.
