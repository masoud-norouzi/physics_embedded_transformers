from __future__ import annotations

import numpy as np


# Region codes match REGION_NAMES in the evaluation scripts (single_droplet_runtime_rollout.py,
# single_droplet_rollout.py) and the region_labels raster produced by the geometry pipeline.
INLET_CHANNEL = 1
OUTLET_CHANNEL = 2
LEFT_BRANCH = 3
RIGHT_BRANCH = 4
INLET_JUNCTION = 5
OUTLET_JUNCTION = 6

COMMIT_REGIONS = (LEFT_BRANCH, RIGHT_BRANCH)
PRE_JUNCTION_REGIONS = (INLET_CHANNEL, INLET_JUNCTION)


def region_codes_for_points(x: np.ndarray, y: np.ndarray, region_labels: np.ndarray) -> np.ndarray:
    """Vectorized region lookup, matching region_at_pixel's rounding/out-of-bounds convention.

    Non-finite (x, y) and out-of-bounds points are treated as region 0 ("outside"), same as
    region_at_pixel in the single-droplet evaluation scripts.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"x and y must share shape, got {x.shape} and {y.shape}")
    height, width = region_labels.shape
    finite = np.isfinite(x) & np.isfinite(y)
    col = np.zeros(x.shape, dtype=np.int64)
    row = np.zeros(x.shape, dtype=np.int64)
    col[finite] = np.round(x[finite]).astype(np.int64)
    row[finite] = np.round(y[finite]).astype(np.int64)
    in_bounds = finite & (row >= 0) & (row < height) & (col >= 0) & (col < width)
    codes = np.zeros(x.shape, dtype=np.int64)
    codes[in_bounds] = region_labels[row[in_bounds], col[in_bounds]]
    return codes


def derive_branch_decision_labels(
    Z: np.ndarray,
    mask: np.ndarray,
    feature_index: dict[str, int],
    region_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive per-(track, frame) junction branch-decision labels and pre-junction window mask.

    branch_label: float32 (n_tracks, n_frames). 1.0 = short/right branch, 0.0 = long/left branch,
    NaN outside any track's pre-junction window (including tracks that never reach a branch, or
    are never observed upstream of the junction before committing).

    in_window: bool (n_tracks, n_frames). True from a track's first observation in
    {inlet_channel, inlet_junction} up to (not including) the frame it first enters
    {left_branch, right_branch}.

    frames_until_commit: int32 (n_tracks, n_frames). Number of frames remaining until the
    commitment frame, valid only where in_window is True; -1 elsewhere. Used to bin calibration
    (predicted probability vs. observed frequency) by how far out the prediction is made, matching
    the frames-until-commitment structure already used in the Ch. 3 logistic regression baseline.
    """
    mask = np.asarray(mask, dtype=bool)
    if Z.shape[:2] != mask.shape:
        raise ValueError(f"Z and mask must share (n_tracks, n_frames), got {Z.shape[:2]} and {mask.shape}")
    n_tracks, n_frames = mask.shape

    x = Z[:, :, feature_index["x"]]
    y = Z[:, :, feature_index["y"]]
    codes = region_codes_for_points(x, y, region_labels)
    codes = np.where(mask, codes, -1)

    is_commit_region = np.isin(codes, COMMIT_REGIONS)
    is_pre_region = np.isin(codes, PRE_JUNCTION_REGIONS)

    branch_label = np.full((n_tracks, n_frames), np.nan, dtype=np.float32)
    in_window = np.zeros((n_tracks, n_frames), dtype=bool)
    frames_until_commit = np.full((n_tracks, n_frames), -1, dtype=np.int32)

    for track_index in range(n_tracks):
        commit_frames = np.flatnonzero(is_commit_region[track_index])
        if commit_frames.size == 0:
            continue
        commit_frame = int(commit_frames[0])
        commit_code = int(codes[track_index, commit_frame])
        label_value = 1.0 if commit_code == RIGHT_BRANCH else 0.0

        pre_frames = np.flatnonzero(is_pre_region[track_index, :commit_frame])
        if pre_frames.size == 0:
            continue
        first_pre_frame = int(pre_frames[0])

        window = slice(first_pre_frame, commit_frame)
        in_window[track_index, window] = mask[track_index, window]
        branch_label[track_index, window] = label_value
        frames_until_commit[track_index, window] = commit_frame - np.arange(first_pre_frame, commit_frame)

    return branch_label, in_window, frames_until_commit
