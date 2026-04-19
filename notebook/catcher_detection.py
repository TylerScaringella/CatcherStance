from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# COCO keypoint indices used by Ultralytics pose models.
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


@dataclass
class CatcherDetectionConfig:
    """
    Notebook-friendly configuration for a detect-or-abstain catcher selector.

    All ROIs / zones are normalized to [0, 1] so the same logic can be reused
    across videos with different resolutions.
    """

    search_roi_norm: Tuple[float, float, float, float] = (0.30, 0.42, 0.70, 0.98)
    frame_gate_roi_norm: Tuple[float, float, float, float] = (0.24, 0.34, 0.76, 1.00)
    anchor_norm: Tuple[float, float] = (0.50, 0.78)

    # Typical broadcast failure case: dugout / bench people at left or right edge.
    # Leave empty if not needed; override per camera when you know those regions.
    invalid_zones_norm: List[Tuple[float, float, float, float]] = field(
        default_factory=lambda: [
            (0.00, 0.00, 0.18, 1.00),
            (0.82, 0.00, 1.00, 1.00),
            (0.00, 0.00, 1.00, 0.18),
        ]
    )

    min_box_conf: float = 0.20
    min_keypoints_present: int = 6
    min_lower_body_keypoints: int = 4

    # Frame-level gate: if nobody is even near the catcher region, abstain early.
    min_gate_overlap: float = 0.05
    max_gate_anchor_dist_norm: float = 0.22
    min_gate_bottom_norm: float = 0.52

    # Candidate hard rejections.
    min_search_overlap: float = 0.08
    max_anchor_dist_norm: float = 0.18
    min_bottom_norm: float = 0.55
    min_box_height_norm: float = 0.12
    max_box_height_norm: float = 0.65
    min_aspect_ratio: float = 0.90
    max_aspect_ratio: float = 4.20
    max_shoulder_to_ankle_dx_norm: float = 0.14
    max_hip_center_to_anchor_dx_norm: float = 0.16
    max_knee_angle: float = 162.0
    max_mean_leg_straightness: float = 0.93
    min_width_ratio: float = 0.78
    min_compactness: float = 0.18
    max_compactness: float = 1.75

    # Detect-or-abstain decision thresholds.
    min_score: float = 0.52
    min_margin: float = 0.08
    max_final_anchor_dist_norm: float = 0.16


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _clip01(v: float) -> float:
    return float(np.clip(v, 0.0, 1.0))


def _scale_norm_box(box_norm: Sequence[float], w: int, h: int) -> np.ndarray:
    x1, y1, x2, y2 = box_norm
    return np.array([x1 * w, y1 * h, x2 * w, y2 * h], dtype=float)


def _scale_norm_point(pt_norm: Sequence[float], w: int, h: int) -> np.ndarray:
    x, y = pt_norm
    return np.array([x * w, y * h], dtype=float)


def box_center(box: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)


def roi_overlap(box: Sequence[float], roi: Sequence[float]) -> float:
    x1, y1, x2, y2 = map(float, box)
    rx1, ry1, rx2, ry2 = map(float, roi)

    ix1 = max(x1, rx1)
    iy1 = max(y1, ry1)
    ix2 = min(x2, rx2)
    iy2 = min(y2, ry2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area = max(1.0, (x2 - x1) * (y2 - y1))
    return inter / area


def point_in_box(pt: Sequence[float], box: Sequence[float]) -> bool:
    return bool(box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3])


def angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Optional[float]:
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return None
    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def safe_mean(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def valid_point(pt: Sequence[float]) -> bool:
    return bool(
        pt is not None
        and len(pt) == 2
        and np.isfinite(pt[0])
        and np.isfinite(pt[1])
        and not (pt[0] == 0 and pt[1] == 0)
    )


def normalized_distance(a: Sequence[float], b: Sequence[float], w: int, h: int) -> float:
    dx = (a[0] - b[0]) / max(1.0, w)
    dy = (a[1] - b[1]) / max(1.0, h)
    return float(np.hypot(dx, dy))


def compute_catcher_features(
    kpts: np.ndarray,
    box: np.ndarray,
    frame_shape: Tuple[int, int],
) -> Dict[str, Optional[float]]:
    h, w = frame_shape
    box_w = max(1.0, float(box[2] - box[0]))
    box_h = max(1.0, float(box[3] - box[1]))
    center = box_center(box)

    points = {
        "ls": kpts[LEFT_SHOULDER],
        "rs": kpts[RIGHT_SHOULDER],
        "lh": kpts[LEFT_HIP],
        "rh": kpts[RIGHT_HIP],
        "lk": kpts[LEFT_KNEE],
        "rk": kpts[RIGHT_KNEE],
        "la": kpts[LEFT_ANKLE],
        "ra": kpts[RIGHT_ANKLE],
    }

    present = {name: valid_point(pt) for name, pt in points.items()}

    def mean_xy(names: Sequence[str]) -> Optional[np.ndarray]:
        vals = [points[n] for n in names if present[n]]
        if not vals:
            return None
        return np.mean(np.asarray(vals, dtype=float), axis=0)

    shoulder_c = mean_xy(["ls", "rs"])
    hip_c = mean_xy(["lh", "rh"])
    knee_c = mean_xy(["lk", "rk"])
    ankle_c = mean_xy(["la", "ra"])

    left_knee_angle = (
        angle_between(points["lh"], points["lk"], points["la"])
        if present["lh"] and present["lk"] and present["la"]
        else None
    )
    right_knee_angle = (
        angle_between(points["rh"], points["rk"], points["ra"])
        if present["rh"] and present["rk"] and present["ra"]
        else None
    )
    avg_knee_angle = safe_mean([left_knee_angle, right_knee_angle])
    min_knee_angle = (
        min(v for v in (left_knee_angle, right_knee_angle) if v is not None)
        if left_knee_angle is not None or right_knee_angle is not None
        else None
    )
    max_knee_angle = (
        max(v for v in (left_knee_angle, right_knee_angle) if v is not None)
        if left_knee_angle is not None or right_knee_angle is not None
        else None
    )
    knee_angle_diff = (
        float(abs(left_knee_angle - right_knee_angle))
        if left_knee_angle is not None and right_knee_angle is not None
        else None
    )

    shoulder_width = (
        float(abs(points["rs"][0] - points["ls"][0]))
        if present["ls"] and present["rs"]
        else None
    )
    hip_width = (
        float(abs(points["rh"][0] - points["lh"][0]))
        if present["lh"] and present["rh"]
        else None
    )
    knee_width = (
        float(abs(points["rk"][0] - points["lk"][0]))
        if present["lk"] and present["rk"]
        else None
    )
    ankle_width = (
        float(abs(points["ra"][0] - points["la"][0]))
        if present["la"] and present["ra"]
        else None
    )

    upper_body_width = safe_mean([shoulder_width, hip_width])
    lower_body_width = safe_mean([knee_width, ankle_width])
    width_ratio = (
        float(lower_body_width / max(1e-6, upper_body_width))
        if lower_body_width is not None and upper_body_width is not None
        else None
    )

    torso_height = (
        float(hip_c[1] - shoulder_c[1])
        if shoulder_c is not None and hip_c is not None
        else None
    )
    leg_height = (
        float(ankle_c[1] - hip_c[1])
        if hip_c is not None and ankle_c is not None
        else None
    )
    compactness = (
        float(torso_height / max(1e-6, leg_height))
        if torso_height is not None and leg_height is not None
        else None
    )

    shoulder_to_ankle_dx_norm = (
        float(abs(shoulder_c[0] - ankle_c[0]) / max(1.0, w))
        if shoulder_c is not None and ankle_c is not None
        else None
    )
    mean_leg_straightness = (
        float(np.cos(np.deg2rad(avg_knee_angle)))
        if avg_knee_angle is not None
        else None
    )

    lower_body_count = int(sum(present[k] for k in ("lh", "rh", "lk", "rk", "la", "ra")))
    total_present = int(sum(valid_point(pt) for pt in kpts))

    bottom_y_norm = float(box[3] / max(1.0, h))
    center_x_norm = float(center[0] / max(1.0, w))
    center_y_norm = float(center[1] / max(1.0, h))

    return {
        "total_present": total_present,
        "lower_body_count": lower_body_count,
        "left_knee_angle": left_knee_angle,
        "right_knee_angle": right_knee_angle,
        "avg_knee_angle": avg_knee_angle,
        "min_knee_angle": min_knee_angle,
        "max_knee_angle": max_knee_angle,
        "knee_angle_diff": knee_angle_diff,
        "torso_height": torso_height,
        "leg_height": leg_height,
        "compactness": compactness,
        "shoulder_width": shoulder_width,
        "hip_width": hip_width,
        "knee_width": knee_width,
        "ankle_width": ankle_width,
        "upper_body_width": upper_body_width,
        "lower_body_width": lower_body_width,
        "width_ratio": width_ratio,
        "aspect_ratio": float(box_h / box_w),
        "box_h_norm": float(box_h / max(1.0, h)),
        "box_w_norm": float(box_w / max(1.0, w)),
        "bottom_y_norm": bottom_y_norm,
        "center_x_norm": center_x_norm,
        "center_y_norm": center_y_norm,
        "shoulder_center_x_norm": (
            float(shoulder_c[0] / max(1.0, w)) if shoulder_c is not None else None
        ),
        "hip_center_x_norm": float(hip_c[0] / max(1.0, w)) if hip_c is not None else None,
        "hip_center_y_norm": float(hip_c[1] / max(1.0, h)) if hip_c is not None else None,
        "knee_center_x_norm": float(knee_c[0] / max(1.0, w)) if knee_c is not None else None,
        "knee_center_y_norm": float(knee_c[1] / max(1.0, h)) if knee_c is not None else None,
        "ankle_center_y_norm": float(ankle_c[1] / max(1.0, h)) if ankle_c is not None else None,
        "shoulder_to_ankle_dx_norm": shoulder_to_ankle_dx_norm,
        "mean_leg_straightness": mean_leg_straightness,
    }


def candidate_anchor_point(
    feats: Dict[str, Optional[float]],
    box: np.ndarray,
    frame_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Use pose-based lower-body center when available instead of box center.

    Box center is unstable for crouched poses because the top of the box can move
    a lot with arm / head position while the plate-relative lower body stays more
    consistent.
    """
    h, w = frame_shape

    if feats.get("hip_center_x_norm") is not None and feats.get("hip_center_y_norm") is not None:
        return np.array(
            [feats["hip_center_x_norm"] * w, feats["hip_center_y_norm"] * h],
            dtype=float,
        )

    if feats.get("knee_center_x_norm") is not None and feats.get("knee_center_y_norm") is not None:
        return np.array(
            [feats["knee_center_x_norm"] * w, feats["knee_center_y_norm"] * h],
            dtype=float,
        )

    return box_center(box)


def _score_band(value: Optional[float], low: float, high: float) -> float:
    if value is None:
        return 0.0
    if value < low:
        return 0.0
    if value > high:
        return 1.0
    return float((value - low) / max(1e-6, high - low))


def _score_inverse_band(value: Optional[float], low: float, high: float) -> float:
    if value is None:
        return 0.0
    if value <= low:
        return 1.0
    if value >= high:
        return 0.0
    return float((high - value) / max(1e-6, high - low))


def _score_peak(value: Optional[float], low: float, mid_low: float, mid_high: float, high: float) -> float:
    """
    Piecewise-linear peak score.

    Score is:
    - 0 outside [low, high]
    - ramps up from low -> mid_low
    - 1 inside [mid_low, mid_high]
    - ramps down from mid_high -> high
    """
    if value is None:
        return 0.0
    if value <= low or value >= high:
        return 0.0
    if mid_low <= value <= mid_high:
        return 1.0
    if value < mid_low:
        return float((value - low) / max(1e-6, mid_low - low))
    return float((high - value) / max(1e-6, high - mid_high))


def frame_is_valid_for_catcher(
    boxes: np.ndarray,
    confs: np.ndarray,
    keypoints: np.ndarray,
    frame_shape: Tuple[int, int],
    cfg: CatcherDetectionConfig,
    debug: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Scene-level gate.

    The main purpose is to reject frames where the catcher view is absent, which
    is the common broadcast-angle / dugout false-positive failure mode.
    """

    h, w = frame_shape
    gate_roi = _scale_norm_box(cfg.frame_gate_roi_norm, w, h)
    anchor = _scale_norm_point(cfg.anchor_norm, w, h)
    valid_like = []

    for i, box in enumerate(boxes):
        if confs[i] < cfg.min_box_conf:
            continue
        feats = compute_catcher_features(keypoints[i], box, frame_shape)
        overlap = roi_overlap(box, gate_roi)
        dist = normalized_distance(candidate_anchor_point(feats, box, frame_shape), anchor, w, h)
        near_gate = (
            overlap >= cfg.min_gate_overlap
            or dist <= cfg.max_gate_anchor_dist_norm
        )
        low_enough = feats["bottom_y_norm"] is not None and feats["bottom_y_norm"] >= cfg.min_gate_bottom_norm
        enough_pose = (
            feats["total_present"] >= cfg.min_keypoints_present
            and feats["lower_body_count"] >= cfg.min_lower_body_keypoints
        )
        if near_gate and low_enough and enough_pose:
            valid_like.append(
                {
                    "index": i,
                    "overlap": overlap,
                    "dist_norm": dist,
                    "bottom_y_norm": feats["bottom_y_norm"],
                }
            )

    reason = "ok" if valid_like else "no_one_near_catcher_region"
    meta = {"reason": reason, "gate_candidates": valid_like}
    if debug:
        print(f"Frame gate: {reason}, count={len(valid_like)}")
    return bool(valid_like), meta


def candidate_rejection_reason(
    box: np.ndarray,
    conf: float,
    kpts: np.ndarray,
    frame_shape: Tuple[int, int],
    cfg: CatcherDetectionConfig,
) -> Tuple[Optional[str], Dict[str, Any]]:
    h, w = frame_shape
    search_roi = _scale_norm_box(cfg.search_roi_norm, w, h)
    anchor = _scale_norm_point(cfg.anchor_norm, w, h)
    feats = compute_catcher_features(kpts, box, frame_shape)
    center = box_center(box)
    anchor_pt = candidate_anchor_point(feats, box, frame_shape)
    overlap = roi_overlap(box, search_roi)
    anchor_dist_norm = normalized_distance(anchor_pt, anchor, w, h)

    invalid_overlap = 0.0
    in_invalid_zone = False
    for zone_norm in cfg.invalid_zones_norm:
        zone = _scale_norm_box(zone_norm, w, h)
        invalid_overlap = max(invalid_overlap, roi_overlap(box, zone))
        in_invalid_zone = in_invalid_zone or point_in_box(center, zone)

    hip_anchor_dx_norm = None
    if feats["hip_center_x_norm"] is not None:
        hip_anchor_dx_norm = abs(feats["hip_center_x_norm"] - cfg.anchor_norm[0])

    meta = {
        "features": feats,
        "search_overlap": overlap,
        "anchor_dist_norm": anchor_dist_norm,
        "anchor_point": anchor_pt,
        "invalid_overlap": invalid_overlap,
        "in_invalid_zone": in_invalid_zone,
        "hip_anchor_dx_norm": hip_anchor_dx_norm,
    }

    # Low-confidence boxes often come from background clutter. Reject early.
    if conf < cfg.min_box_conf:
        return "low_box_conf", meta

    # If lower-body joints are missing, crouch geometry is too unreliable.
    if feats["total_present"] < cfg.min_keypoints_present:
        return "too_few_keypoints", meta
    if feats["lower_body_count"] < cfg.min_lower_body_keypoints:
        return "missing_lower_body", meta

    # Dugout / bench / broadcast clutter rejection.
    if in_invalid_zone and overlap < cfg.min_search_overlap:
        return "invalid_zone", meta
    if invalid_overlap > 0.45 and overlap < cfg.min_search_overlap:
        return "mostly_invalid_zone", meta

    # Catcher should live near the lower-center home-plate region.
    if overlap < cfg.min_search_overlap and anchor_dist_norm > cfg.max_anchor_dist_norm:
        return "too_far_from_search_region", meta
    if feats["bottom_y_norm"] is None or feats["bottom_y_norm"] < cfg.min_bottom_norm:
        return "not_low_enough", meta
    if feats["box_h_norm"] is None or feats["box_h_norm"] < cfg.min_box_height_norm:
        return "too_small", meta
    if feats["box_h_norm"] is not None and feats["box_h_norm"] > cfg.max_box_height_norm:
        return "too_large", meta

    # Broad geometry checks: exclude long upright batter/pitcher poses.
    if feats["aspect_ratio"] is None:
        return "missing_aspect_ratio", meta
    if feats["aspect_ratio"] < cfg.min_aspect_ratio or feats["aspect_ratio"] > cfg.max_aspect_ratio:
        return "bad_aspect_ratio", meta
    # Common false-positive pitcher pattern: very tall box, very low in frame.
    # This is intentionally global and geometry-based rather than camera-specific.
    if (
        feats["bottom_y_norm"] is not None
        and feats["box_h_norm"] is not None
        and feats["aspect_ratio"] is not None
        and feats["bottom_y_norm"] > 0.76
        and feats["box_h_norm"] > 0.40
        and feats["aspect_ratio"] > 1.90
    ):
        return "too_tall_and_low_for_catcher", meta
    if feats["avg_knee_angle"] is None:
        return "missing_knee_angle", meta
    min_knee_angle = feats.get("min_knee_angle")
    knee_angle_diff = feats.get("knee_angle_diff")
    # Allow one-knee catcher stances: one leg may be relatively straight while the
    # other is clearly bent. Reject only when both legs look too straight.
    if feats["avg_knee_angle"] > cfg.max_knee_angle:
        allow_one_knee_catcher = (
            overlap >= 0.45
            and anchor_dist_norm <= 0.36
            and feats["bottom_y_norm"] is not None
            and 0.54 <= feats["bottom_y_norm"] <= 0.72
            and feats["width_ratio"] is not None
            and feats["width_ratio"] >= 1.50
            and feats["aspect_ratio"] is not None
            and 0.75 <= feats["aspect_ratio"] <= 1.80
            and feats["compactness"] is not None
            and 1.00 <= feats["compactness"] <= 2.40
            and feats["shoulder_to_ankle_dx_norm"] is not None
            and feats["shoulder_to_ankle_dx_norm"] <= 0.03
        )
        if not allow_one_knee_catcher:
            if min_knee_angle is None or min_knee_angle > 150.0:
                return "knees_too_straight", meta
            if knee_angle_diff is None or knee_angle_diff < 12.0:
                return "knees_too_straight", meta
    if (
        feats["mean_leg_straightness"] is not None
        and feats["mean_leg_straightness"] > cfg.max_mean_leg_straightness
    ):
        return "legs_too_straight", meta
    if feats["compactness"] is None:
        return "missing_compactness", meta
    if feats["compactness"] < cfg.min_compactness or feats["compactness"] > cfg.max_compactness:
        return "bad_compactness", meta
    if feats["width_ratio"] is None or feats["width_ratio"] < cfg.min_width_ratio:
        return "stance_too_narrow", meta

    # Catcher tends to be roughly stacked over the plate area, unlike hitter/pitcher.
    if (
        feats["shoulder_to_ankle_dx_norm"] is not None
        and feats["shoulder_to_ankle_dx_norm"] > cfg.max_shoulder_to_ankle_dx_norm
    ):
        return "too_leaned", meta
    if hip_anchor_dx_norm is not None and hip_anchor_dx_norm > cfg.max_hip_center_to_anchor_dx_norm:
        return "hips_too_far_from_anchor", meta

    return None, meta


def score_candidate(
    meta: Dict[str, Any],
    conf: float,
    cfg: CatcherDetectionConfig,
) -> Dict[str, float]:
    feats = meta["features"]
    overlap = meta["search_overlap"]
    anchor_dist_norm = meta["anchor_dist_norm"]

    # Weighted score favors "catcher-like" evidence instead of just "most crouched".
    score_parts = {
        # Strong prior: candidate should overlap the plate-area ROI.
        "location_overlap": 0.24 * _score_band(overlap, 0.08, 0.45),
        # Still useful, but not so dominant that it overrules better stance geometry.
        "anchor_distance": 0.18 * _score_inverse_band(anchor_dist_norm, 0.05, 0.30),
        # Catcher should sit in a preferred lower-middle vertical band, not simply "lower is better".
        "bottom_position": 0.14 * _score_peak(feats["bottom_y_norm"], 0.50, 0.58, 0.70, 0.82),
        # Reward either a strong two-knee crouch or a one-knee asymmetric catcher stance.
        "knee_bend": 0.08 * _score_inverse_band(feats["avg_knee_angle"], 115.0, 162.0),
        "single_knee_bend": 0.08 * _score_inverse_band(feats.get("min_knee_angle"), 105.0, 150.0),
        "knee_asymmetry": 0.05 * _score_band(feats.get("knee_angle_diff"), 10.0, 45.0),
        # Short/wide catcher boxes should still receive credit instead of being zeroed out.
        "compactness": 0.10 * _score_peak(feats["compactness"], 0.45, 0.70, 2.20, 2.60),
        # Give more separation to very wide lower-body stances, which helps catcher vs pitcher.
        "lower_width": 0.15 * _score_band(feats["width_ratio"], 0.90, 2.40),
        "upright_stack": 0.04 * _score_inverse_band(feats["shoulder_to_ankle_dx_norm"], 0.02, 0.14),
        "det_conf": 0.03 * _score_band(conf, 0.20, 0.75),
    }
    total = float(sum(score_parts.values()))
    score_parts["total"] = total
    return score_parts


def detect_catcher(
    result: Any,
    frame: Optional[np.ndarray] = None,
    cfg: Optional[CatcherDetectionConfig] = None,
    debug: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Detect one catcher or abstain with None.

    Multi-stage pipeline:
    1. frame-level gate
    2. candidate rejection
    3. candidate scoring
    4. detect-or-abstain decision
    """

    cfg = cfg or CatcherDetectionConfig()

    if result is None or result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
        if debug:
            print("No pose detections available.")
        return None

    boxes = _to_numpy(result.boxes.xyxy)
    confs = _to_numpy(result.boxes.conf)
    keypoints = _to_numpy(result.keypoints.xy)

    if frame is not None:
        frame_h, frame_w = frame.shape[:2]
    else:
        # Fallback if you only pass the Ultralytics result.
        frame_w = int(np.max(boxes[:, [0, 2]]) + 1)
        frame_h = int(np.max(boxes[:, [1, 3]]) + 1)
    frame_shape = (frame_h, frame_w)

    valid_frame, gate_meta = frame_is_valid_for_catcher(
        boxes=boxes,
        confs=confs,
        keypoints=keypoints,
        frame_shape=frame_shape,
        cfg=cfg,
        debug=debug,
    )
    if not valid_frame:
        if debug:
            print(f"Abstain at frame gate: {gate_meta['reason']}")
        return None

    candidates: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for i, box in enumerate(boxes):
        reason, meta = candidate_rejection_reason(
            box=box,
            conf=float(confs[i]),
            kpts=keypoints[i],
            frame_shape=frame_shape,
            cfg=cfg,
        )

        if reason is not None:
            rejected.append({"index": i, "reason": reason, **meta})
            if debug:
                print(
                    f"reject i={i} reason={reason} "
                    f"overlap={meta['search_overlap']:.3f} "
                    f"anchor_dist={meta['anchor_dist_norm']:.3f}"
                )
            continue

        score_parts = score_candidate(meta=meta, conf=float(confs[i]), cfg=cfg)
        cand = {
            "index": i,
            "box": box,
            "keypoints": keypoints[i],
            "confidence": float(confs[i]),
            "features": meta["features"],
            "search_overlap": meta["search_overlap"],
            "anchor_dist_norm": meta["anchor_dist_norm"],
            "invalid_overlap": meta["invalid_overlap"],
            "score_parts": score_parts,
            "score": score_parts["total"],
        }
        candidates.append(cand)
        if debug:
            print(
                f"keep i={i} score={cand['score']:.3f} "
                f"overlap={cand['search_overlap']:.3f} "
                f"anchor_dist={cand['anchor_dist_norm']:.3f} "
                f"knee={cand['features']['avg_knee_angle']:.1f} "
                f"bottom={cand['features']['bottom_y_norm']:.3f}"
            )

    if not candidates:
        if debug:
            print("Abstain: no plausible catcher candidates after filtering.")
        return None

    candidates.sort(key=lambda c: c["score"], reverse=True)
    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None
    margin = float(best["score"] - second["score"]) if second is not None else float("inf")

    # Final abstention: weak score means nobody looks enough like a catcher.
    if best["score"] < cfg.min_score:
        if debug:
            print(f"Abstain: best score too low ({best['score']:.3f} < {cfg.min_score:.3f})")
        return None

    # Final abstention: even a decent shape should still be close to the plate anchor.
    if best["anchor_dist_norm"] > cfg.max_final_anchor_dist_norm:
        if debug:
            print(
                "Abstain: best candidate too far from catcher anchor "
                f"({best['anchor_dist_norm']:.3f})"
            )
        return None

    # Final abstention: if winner barely beats runner-up, treat as ambiguous.
    if margin < cfg.min_margin:
        if debug:
            print(f"Abstain: ambiguous winner (margin={margin:.3f} < {cfg.min_margin:.3f})")
        return None

    return {
        "index": best["index"],
        "box": best["box"],
        "keypoints": best["keypoints"],
        "confidence": best["confidence"],
        "features": best["features"],
        "score": best["score"],
        "score_parts": best["score_parts"],
        "margin": margin,
        "search_overlap": best["search_overlap"],
        "anchor_dist_norm": best["anchor_dist_norm"],
        "frame_gate": gate_meta,
        "all_candidates": candidates,
        "rejected_candidates": rejected,
    }


def detect_catcher_from_res_item(
    item: Dict[str, Any],
    cfg: Optional[CatcherDetectionConfig] = None,
    debug: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Convenience wrapper for your existing structure:
    res[i]["result"], res[i]["frame"]
    """

    return detect_catcher(
        result=item["result"],
        frame=item.get("frame"),
        cfg=cfg,
        debug=debug,
    )
