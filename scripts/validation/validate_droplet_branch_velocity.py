from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_experiment_config

BRANCH_INTERIOR_THRESHOLD = 0.95
MINIMUM_BRANCH_INTERIOR_SAMPLES = 60
STD_TOLERANCE = 1e-12
NORMALIZATION_DDOF = 1
OUTPUT_NAMES = {
    "selected_tracks": "selected_tracks.csv",
    "pooled_samples": "pooled_normalized_samples.csv",
    "summary": "droplet_branch_velocity_summary.json",
    "scatter": "droplet_branch_velocity_scatter.png",
}


def validate_unique_keys(df: pd.DataFrame, keys: list[str], label: str) -> None:
    missing = set(keys).difference(df.columns)
    if missing:
        raise ValueError(f"{label} is missing join key columns: {sorted(missing)}")
    duplicate = df.duplicated(keys, keep=False)
    if duplicate.any():
        examples = df.loc[duplicate, keys].head(5).to_dict("records")
        raise ValueError(f"{label} contains duplicate rows for keys {keys}: {examples}")


def _velocity_units_from_columns(tracks: pd.DataFrame) -> str | None:
    for column in ["tracked_velocity_units", "velocity_units"]:
        if column in tracks.columns:
            units = tracks[column].dropna().astype(str).unique()
            if len(units) == 1:
                return str(units[0])
            if len(units) > 1:
                raise ValueError(f"Tracked velocity units are not unique: {sorted(units)}")
    return None


def detect_centroid_columns(tracks: pd.DataFrame) -> tuple[str, str]:
    candidates = [("centroid_x", "centroid_y"), ("center_x", "center_y"), ("x", "y")]
    for x_col, y_col in candidates:
        if x_col in tracks.columns and y_col in tracks.columns:
            return x_col, y_col
    raise ValueError(
        "Tracked features do not contain a supported centroid column pair. "
        "Expected one of: centroid_x/centroid_y, center_x/center_y, x/y."
    )


def prepare_tracked_velocity(tracks: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    required_keys = {"frame", "track_id"}
    missing_keys = required_keys.difference(tracks.columns)
    if missing_keys:
        raise ValueError(f"Tracked velocity input is missing required key columns: {sorted(missing_keys)}")
    if {"vx", "vy"}.issubset(tracks.columns):
        validate_unique_keys(tracks, ["frame", "track_id"], "Tracked velocity input")
        result = tracks.copy()
        result["vx"] = pd.to_numeric(result["vx"], errors="coerce")
        result["vy"] = pd.to_numeric(result["vy"], errors="coerce")
        velocity = result[["vx", "vy"]].to_numpy(float)
        if not np.all(np.isfinite(velocity)):
            raise ValueError("Existing tracked velocity columns vx and vy must be finite")
        units = _velocity_units_from_columns(result) or "native tracked velocity units (unspecified)"
        result["tracked_velocity_units"] = units
        result["tracked_velocity_source"] = "existing_velocity_columns"
        result["velocity_x_column"] = "vx"
        result["velocity_y_column"] = "vy"
        return result, {
            "tracked_velocity_units": units,
            "tracked_velocity_source": "existing_velocity_columns",
            "centroid_columns_used": None,
        }

    x_col, y_col = detect_centroid_columns(tracks)
    validate_unique_keys(tracks, ["frame", "track_id"], "Tracked position input")
    result = tracks.copy()
    result["vx"] = np.nan
    result["vy"] = np.nan
    for _, group in result.sort_values(["track_id", "frame"]).groupby("track_id", sort=False):
        idx = group.index.to_numpy()
        frames = group["frame"].to_numpy(dtype=float)
        x = group[x_col].to_numpy(dtype=float)
        y = group[y_col].to_numpy(dtype=float)
        if not np.all(np.isfinite(frames)) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("Frame and centroid columns used for derived velocity must be finite")
        if len(group) < 3:
            continue
        prev_gap = frames[1:-1] - frames[:-2]
        next_gap = frames[2:] - frames[1:-1]
        valid = (prev_gap == 1) & (next_gap == 1)
        middle_idx = idx[1:-1][valid]
        result.loc[middle_idx, "vx"] = (x[2:][valid] - x[:-2][valid]) / 2.0
        result.loc[middle_idx, "vy"] = (y[2:][valid] - y[:-2][valid]) / 2.0
    result["tracked_velocity_units"] = "px/frame"
    result["tracked_velocity_source"] = "centered_position_difference"
    result["velocity_x_column"] = x_col
    result["velocity_y_column"] = y_col
    return result, {
        "tracked_velocity_units": "px/frame",
        "tracked_velocity_source": "centered_position_difference",
        "centroid_columns_used": [x_col, y_col],
    }


def validate_tracked_velocity(tracks: pd.DataFrame) -> str:
    _, metadata = prepare_tracked_velocity(tracks)
    return str(metadata["tracked_velocity_units"])


def validate_occupancy_input(occupancy: pd.DataFrame) -> None:
    required = {"frame", "track_id", "occupancy_computable", "w_left", "w_right"}
    missing = required.difference(occupancy.columns)
    if missing:
        raise ValueError(f"Occupancy input is missing columns: {sorted(missing)}")
    validate_unique_keys(occupancy, ["frame", "track_id"], "Occupancy input")


def validate_hydraulic_state_input(state: pd.DataFrame) -> None:
    required = {"frame", "left_velocity_um_s", "right_velocity_um_s"}
    missing = required.difference(state.columns)
    if missing:
        raise ValueError(f"Hydraulic state input is missing columns: {sorted(missing)}")
    validate_unique_keys(state, ["frame"], "Hydraulic state input")


def calculate_speeds(tracks: pd.DataFrame) -> pd.DataFrame:
    result = tracks.copy()
    result["droplet_speed"] = np.sqrt(result["vx"].astype(float) ** 2 + result["vy"].astype(float) ** 2)
    return result


def build_branch_interior_samples(
    tracks: pd.DataFrame,
    occupancy: pd.DataFrame,
    hydraulic_state: pd.DataFrame,
    branch_interior_threshold: float = BRANCH_INTERIOR_THRESHOLD,
) -> pd.DataFrame:
    if not 0 < branch_interior_threshold <= 1:
        raise ValueError("branch_interior_threshold must be in (0, 1]")
    tracks, velocity_metadata = prepare_tracked_velocity(tracks)
    validate_occupancy_input(occupancy)
    validate_hydraulic_state_input(hydraulic_state)

    tracks = calculate_speeds(tracks)
    merged = tracks[["frame", "track_id", "vx", "vy", "droplet_speed"]].merge(
        occupancy[["frame", "track_id", "occupancy_computable", "w_left", "w_right"]],
        on=["frame", "track_id"],
        how="inner",
        validate="one_to_one",
    )
    merged = merged.merge(
        hydraulic_state[["frame", "left_velocity_um_s", "right_velocity_um_s"]],
        on="frame",
        how="inner",
        validate="many_to_one",
    )
    finite_columns = ["vx", "vy", "droplet_speed", "w_left", "w_right", "left_velocity_um_s", "right_velocity_um_s"]
    finite = np.isfinite(merged[finite_columns].to_numpy(float)).all(axis=1)
    computable = merged["occupancy_computable"].astype(bool)
    left = merged["w_left"].astype(float) >= branch_interior_threshold
    right = merged["w_right"].astype(float) >= branch_interior_threshold
    exactly_one_branch = left ^ right
    filtered = merged.loc[finite & computable & exactly_one_branch].copy()
    filtered["branch"] = np.where(filtered["w_left"] >= branch_interior_threshold, "left", "right")
    filtered["calculated_branch_velocity_um_s"] = np.where(
        filtered["branch"] == "left",
        filtered["left_velocity_um_s"],
        filtered["right_velocity_um_s"],
    )
    filtered["tracked_velocity_units"] = velocity_metadata["tracked_velocity_units"]
    filtered["tracked_velocity_source"] = velocity_metadata["tracked_velocity_source"]
    return filtered[
        [
            "frame",
            "track_id",
            "branch",
            "droplet_speed",
            "calculated_branch_velocity_um_s",
            "tracked_velocity_units",
            "tracked_velocity_source",
            "w_left",
            "w_right",
        ]
    ].sort_values(["track_id", "branch", "frame"])


def pearson_r(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if len(x_arr) < 2:
        return np.nan
    return float(pd.Series(x_arr).corr(pd.Series(y_arr), method="pearson"))


def spearman_r(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if len(x_arr) < 2:
        return np.nan
    x_rank = pd.Series(x_arr).rank(method="average")
    y_rank = pd.Series(y_arr).rank(method="average")
    return pearson_r(x_rank, y_rank)


def normalize_candidate(candidate: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    droplet_mean = float(candidate["droplet_speed"].mean())
    droplet_std = float(candidate["droplet_speed"].std(ddof=NORMALIZATION_DDOF))
    branch_mean = float(candidate["calculated_branch_velocity_um_s"].mean())
    branch_std = float(candidate["calculated_branch_velocity_um_s"].std(ddof=NORMALIZATION_DDOF))
    if not np.isfinite(droplet_std) or not np.isfinite(branch_std):
        return candidate, None
    if abs(droplet_std) <= STD_TOLERANCE or abs(branch_std) <= STD_TOLERANCE:
        return candidate, None
    normalized = candidate.copy()
    normalized["droplet_speed_z"] = (normalized["droplet_speed"] - droplet_mean) / droplet_std
    normalized["branch_velocity_z"] = (
        normalized["calculated_branch_velocity_um_s"] - branch_mean
    ) / branch_std
    if not np.isfinite(normalized[["droplet_speed_z", "branch_velocity_z"]].to_numpy(float)).all():
        raise ValueError("Normalized candidate contains nonfinite values")
    stats = {
        "track_id": int(candidate["track_id"].iloc[0]),
        "branch": str(candidate["branch"].iloc[0]),
        "frame_start": int(candidate["frame"].min()),
        "frame_end": int(candidate["frame"].max()),
        "n_samples": int(len(candidate)),
        "mean_droplet_speed": droplet_mean,
        "std_droplet_speed": droplet_std,
        "mean_branch_velocity_um_s": branch_mean,
        "std_branch_velocity_um_s": branch_std,
        "pearson_r": pearson_r(normalized["branch_velocity_z"], normalized["droplet_speed_z"]),
        "spearman_r": spearman_r(normalized["branch_velocity_z"], normalized["droplet_speed_z"]),
        "tracked_velocity_units": str(candidate["tracked_velocity_units"].iloc[0]),
    }
    return normalized, stats


def build_candidate_table(
    branch_samples: pd.DataFrame,
    minimum_branch_interior_samples: int = MINIMUM_BRANCH_INTERIOR_SAMPLES,
) -> tuple[pd.DataFrame, dict[tuple[int, str], pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    normalized_by_key: dict[tuple[int, str], pd.DataFrame] = {}
    for (track_id, branch), group in branch_samples.groupby(["track_id", "branch"], sort=True):
        group = group.sort_values("frame").copy()
        if len(group) < minimum_branch_interior_samples:
            continue
        normalized, stats = normalize_candidate(group)
        if stats is None:
            continue
        key = (int(track_id), str(branch))
        rows.append(stats)
        normalized_by_key[key] = normalized
    return pd.DataFrame(rows), normalized_by_key


def select_candidates(candidates: pd.DataFrame, target_count: int = 10) -> pd.DataFrame:
    if len(candidates) < target_count:
        raise ValueError(f"Fewer than {target_count} eligible unique track segments exist overall: {len(candidates)}")
    ranked = candidates.sort_values(
        ["branch", "std_branch_velocity_um_s", "track_id"],
        ascending=[True, False, True],
    ).copy()
    left = ranked[ranked["branch"] == "left"]
    right = ranked[ranked["branch"] == "right"]
    selected_parts = []
    if len(left) >= 5 and len(right) >= 5:
        selected_parts = [left.head(5), right.head(5)]
    else:
        scarce = left if len(left) < 5 else right
        abundant = right if len(left) < 5 else left
        selected_parts = [scarce, abundant.head(target_count - len(scarce))]
    selected = pd.concat(selected_parts, ignore_index=True)
    selected = _enforce_one_segment_per_track(selected, ranked, target_count)
    selected = selected.sort_values(
        ["branch", "std_branch_velocity_um_s", "track_id"],
        ascending=[True, False, True],
    ).head(target_count)
    if len(selected) != target_count:
        raise ValueError(f"Expected {target_count} selected candidates, got {len(selected)}")
    selected = selected.reset_index(drop=True)
    selected.insert(0, "selection_rank", np.arange(1, len(selected) + 1))
    return selected


def _enforce_one_segment_per_track(selected: pd.DataFrame, ranked: pd.DataFrame, target_count: int) -> pd.DataFrame:
    if selected["track_id"].nunique() == len(selected):
        return selected
    unique_rows = []
    used_tracks: set[int] = set()
    for _, row in selected.iterrows():
        track_id = int(row["track_id"])
        if track_id not in used_tracks:
            unique_rows.append(row)
            used_tracks.add(track_id)
    for _, row in ranked.iterrows():
        if len(unique_rows) >= target_count:
            break
        track_id = int(row["track_id"])
        if track_id in used_tracks:
            continue
        unique_rows.append(row)
        used_tracks.add(track_id)
    if len(unique_rows) < target_count:
        return selected
    return pd.DataFrame(unique_rows)


def build_pooled_samples(
    selected: pd.DataFrame,
    normalized_by_key: dict[tuple[int, str], pd.DataFrame],
) -> pd.DataFrame:
    pieces = []
    selected_keys = set()
    for _, row in selected.iterrows():
        key = (int(row["track_id"]), str(row["branch"]))
        selected_keys.add(key)
        piece = normalized_by_key[key].copy()
        pieces.append(piece)
    pooled = pd.concat(pieces, ignore_index=True)
    actual_keys = set(zip(pooled["track_id"].astype(int), pooled["branch"].astype(str)))
    if actual_keys.difference(selected_keys):
        raise ValueError("Pooled output contains samples from a non-selected track segment")
    return pooled[
        [
            "track_id",
            "branch",
            "frame",
            "droplet_speed",
            "calculated_branch_velocity_um_s",
            "droplet_speed_z",
            "branch_velocity_z",
        ]
    ].sort_values(["track_id", "branch", "frame"])


def validate_normalized_samples(pooled: pd.DataFrame, selected: pd.DataFrame) -> None:
    selected_keys = set(zip(selected["track_id"].astype(int), selected["branch"].astype(str)))
    actual_keys = set(zip(pooled["track_id"].astype(int), pooled["branch"].astype(str)))
    if not actual_keys.issubset(selected_keys):
        raise ValueError("Pooled output contains samples from non-selected track segments")
    for (track_id, branch), group in pooled.groupby(["track_id", "branch"]):
        if len(group) < MINIMUM_BRANCH_INTERIOR_SAMPLES:
            raise ValueError(f"Selected track {track_id} {branch} has too few samples")
        for column in ["droplet_speed_z", "branch_velocity_z"]:
            values = group[column].to_numpy(float)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{column} contains nonfinite values")
            if abs(float(np.mean(values))) > 1e-10:
                raise ValueError(f"{column} does not have approximately zero mean")
            if abs(float(pd.Series(values).std(ddof=NORMALIZATION_DDOF)) - 1.0) > 1e-10:
                raise ValueError(f"{column} does not have approximately unit sample standard deviation")


def save_track_plot(row: pd.Series, samples: pd.DataFrame, output_dir: Path) -> Path:
    getter = row.__getitem__ if isinstance(row, pd.Series) else lambda key: getattr(row, key)
    track_id = int(getter("track_id"))
    branch = str(getter("branch"))
    path = output_dir / f"track_{track_id}_normalized_velocity.png"
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(samples["frame"], samples["droplet_speed_z"], label="Measured droplet speed", linewidth=1.4)
    ax.plot(samples["frame"], samples["branch_velocity_z"], label="Calculated branch velocity", linewidth=1.4)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("frame")
    ax.set_ylabel("normalized velocity, z-score")
    ax.set_title(
        f"Track {track_id} - {branch} branch\n"
        f"n={int(getter('n_samples'))}, "
        f"Pearson r={float(getter('pearson_r')):.3f}, "
        f"Spearman rho={float(getter('spearman_r')):.3f}"
    )
    ax.text(
        0.01,
        0.01,
        "Per-track normalized temporal variation; measured speed is px/frame, branch velocity is um/s.",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_scatter_plot(pooled: pd.DataFrame, selected: pd.DataFrame, output_path: Path) -> dict[str, float]:
    x = pooled["branch_velocity_z"].to_numpy(float)
    y = pooled["droplet_speed_z"].to_numpy(float)
    pooled_pearson = pearson_r(x, y)
    pooled_spearman = spearman_r(x, y)
    slope, intercept = np.polyfit(x, y, deg=1)
    low = float(np.nanmin([x.min(), y.min()]))
    high = float(np.nanmax([x.max(), y.max()]))
    pad = 0.05 * max(high - low, 1.0)
    low -= pad
    high += pad

    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    ax.scatter(x, y, s=8, alpha=0.35)
    ax.plot([low, high], [low, high], color="black", linewidth=1.0, linestyle="--", label="y=x")
    xs = np.array([low, high])
    ax.plot(xs, slope * xs + intercept, color="tab:red", linewidth=1.4, label="OLS fit")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("normalized calculated branch velocity")
    ax.set_ylabel("normalized measured droplet speed")
    ax.set_title(
        "Pooled zero-lag descriptive correlation\n"
        f"n={len(pooled)}, tracks={len(selected)}, Pearson r={pooled_pearson:.3f}, Spearman rho={pooled_spearman:.3f}"
    )
    ax.text(
        0.02,
        0.02,
        "Descriptive normalized temporal comparison; absolute units are not compared.",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return {
        "pooled_pearson_r": float(pooled_pearson),
        "pooled_spearman_r": float(pooled_spearman),
        "ols_slope": float(slope),
        "ols_intercept": float(intercept),
    }


def build_summary(
    config: dict[str, dict[str, Any]],
    tracked_path: Path,
    tracked_velocity_units: str,
    tracked_velocity_source: str,
    centroid_columns_used: list[str] | None,
    occupancy_path: Path,
    hydraulic_state_path: Path,
    branch_interior_threshold: float,
    minimum_branch_interior_samples: int,
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    pooled: pd.DataFrame,
    pooled_stats: dict[str, float],
) -> dict[str, Any]:
    experiment = config["experiment"]["experiment"]
    device = config["device"]["device"]
    return {
        "experiment_id": experiment["id"],
        "device_id": device["id"],
        "tracked_velocity_source_path": str(tracked_path),
        "tracked_velocity_units": tracked_velocity_units,
        "tracked_velocity_source": tracked_velocity_source,
        "centroid_columns_used": centroid_columns_used,
        "occupancy_source_path": str(occupancy_path),
        "hydraulic_state_source_path": str(hydraulic_state_path),
        "branch_interior_threshold": branch_interior_threshold,
        "minimum_branch_interior_samples": minimum_branch_interior_samples,
        "normalization_method": "per selected (track_id, branch) z-score",
        "normalization_ddof": NORMALIZATION_DDOF,
        "candidate_count_total": int(len(candidates)),
        "candidate_count_left": int((candidates["branch"] == "left").sum()) if len(candidates) else 0,
        "candidate_count_right": int((candidates["branch"] == "right").sum()) if len(candidates) else 0,
        "selected_track_count": int(len(selected)),
        "selected_left_count": int((selected["branch"] == "left").sum()),
        "selected_right_count": int((selected["branch"] == "right").sum()),
        "pooled_sample_count": int(len(pooled)),
        "pooled_pearson_r": pooled_stats["pooled_pearson_r"],
        "pooled_spearman_r": pooled_stats["pooled_spearman_r"],
        "median_track_pearson_r": float(selected["pearson_r"].median()),
        "minimum_track_pearson_r": float(selected["pearson_r"].min()),
        "maximum_track_pearson_r": float(selected["pearson_r"].max()),
        "median_track_spearman_r": float(selected["spearman_r"].median()),
        "selected_track_ids": [int(value) for value in selected["track_id"].tolist()],
        "selected_track_segments": [
            {"track_id": int(row.track_id), "branch": str(row.branch)}
            for row in selected.itertuples(index=False)
        ],
        "selection_rule": (
            "Deterministic branch-balanced selection: choose five left and five right candidates when available; "
            "within each branch rank by std_branch_velocity_um_s descending with track_id as tie-breaker. "
            "If one branch has fewer than five candidates, take all from that branch and fill from the other. "
            "Correlation is not used for candidate ranking."
        ),
        "unit_interpretation": (
            "Measured droplet speed is recorded in px/frame when derived from tracked positions, while "
            "hydraulic branch velocity is in um/s. The comparison standardizes each signal independently "
            "within each selected track segment, so it compares normalized temporal variation rather than "
            "absolute velocity magnitudes. No frame-rate or pixel-scale conversion is required for this "
            "normalized comparison."
        ),
        "limitations": [
            "This is a temporal co-variation validation, not proof that branch velocity equals droplet speed.",
            "The baseline velocity is a superficial mixture velocity.",
            "Individual droplet slip is not modeled.",
            "Droplet-droplet interactions are not modeled beyond additive resistance.",
            "Junction and transition frames are excluded.",
            "Repeated frames within a trajectory are temporally correlated.",
            "Per-track normalization removes absolute scale differences.",
        ],
    }


def resolve_tracked_path(config: dict[str, dict[str, Any]]) -> Path:
    experiment = config["experiment"]["experiment"]
    try:
        return Path(experiment["data"]["tracks"])
    except KeyError as exc:
        raise ValueError("Experiment config is missing experiment.data.tracks") from exc


def run_validation(
    experiment_path: Path,
    occupancy_path: Path,
    hydraulic_state_path: Path,
    output_dir: Path,
    overwrite: bool = False,
    branch_interior_threshold: float = BRANCH_INTERIOR_THRESHOLD,
    minimum_branch_interior_samples: int = MINIMUM_BRANCH_INTERIOR_SAMPLES,
) -> dict[str, Any]:
    config = load_experiment_config(experiment_path)
    tracked_path = resolve_tracked_path(config)
    for label, path in [
        ("tracked velocity", tracked_path),
        ("occupancy", occupancy_path),
        ("hydraulic state", hydraulic_state_path),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required {label} input file is missing: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / name for name in OUTPUT_NAMES.values()]
    if not overwrite:
        existing = [path for path in outputs if path.exists()]
        existing.extend(output_dir.glob("track_*_normalized_velocity.png"))
        if existing:
            raise FileExistsError(f"Output files already exist. Use --overwrite: {existing}")
    else:
        for stale_plot in output_dir.glob("track_*_normalized_velocity.png"):
            stale_plot.unlink()

    tracks = pd.read_csv(tracked_path)
    occupancy = pd.read_csv(occupancy_path)
    hydraulic_state = pd.read_csv(hydraulic_state_path)
    branch_samples = build_branch_interior_samples(
        tracks,
        occupancy,
        hydraulic_state,
        branch_interior_threshold=branch_interior_threshold,
    )
    if branch_samples.empty:
        raise ValueError("No finite branch-interior samples are available after velocity derivation and joins")
    tracked_velocity_units = str(branch_samples["tracked_velocity_units"].iloc[0])
    tracked_velocity_source = str(branch_samples["tracked_velocity_source"].iloc[0])
    centroid_columns_used = None
    if tracked_velocity_source == "centered_position_difference":
        x_col, y_col = detect_centroid_columns(tracks)
        centroid_columns_used = [x_col, y_col]
    candidates, normalized_by_key = build_candidate_table(branch_samples, minimum_branch_interior_samples)
    selected = select_candidates(candidates, target_count=10)
    pooled = build_pooled_samples(selected, normalized_by_key)
    validate_normalized_samples(pooled, selected)

    plot_paths = []
    for _, row in selected.iterrows():
        key = (int(row["track_id"]), str(row["branch"]))
        plot_path = save_track_plot(row, normalized_by_key[key], output_dir)
        plot_paths.append(plot_path)
    if len(plot_paths) != 10 or len(list(output_dir.glob("track_*_normalized_velocity.png"))) != 10:
        raise ValueError("Expected exactly ten individual track plot files")

    selected = selected.copy()
    selected["plot_path"] = [str(path) for path in plot_paths]
    pooled_stats = save_scatter_plot(pooled, selected, output_dir / OUTPUT_NAMES["scatter"])
    summary = build_summary(
        config,
        tracked_path,
        tracked_velocity_units,
        tracked_velocity_source,
        centroid_columns_used,
        occupancy_path,
        hydraulic_state_path,
        branch_interior_threshold,
        minimum_branch_interior_samples,
        candidates,
        selected,
        pooled,
        pooled_stats,
    )

    selected.to_csv(output_dir / OUTPUT_NAMES["selected_tracks"], index=False)
    pooled.to_csv(output_dir / OUTPUT_NAMES["pooled_samples"], index=False)
    (output_dir / OUTPUT_NAMES["summary"]).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"summary": summary, "selected": selected, "pooled": pooled, "output_dir": output_dir}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate droplet speed against baseline branch velocity.")
    parser.add_argument("--experiment", required=True, type=Path, help="Experiment YAML path.")
    parser.add_argument(
        "--occupancy",
        type=Path,
        default=Path("outputs/physics/video_2/droplet_occupancy.csv"),
        help="Normalized droplet occupancy CSV.",
    )
    parser.add_argument(
        "--hydraulic-state",
        type=Path,
        default=Path("outputs/physics/video_2/baseline_hydraulic_state.csv"),
        help="Baseline hydraulic state CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/physics/video_2/droplet_branch_velocity_validation"),
        help="Validation output directory.",
    )
    parser.add_argument("--branch-interior-threshold", type=float, default=BRANCH_INTERIOR_THRESHOLD)
    parser.add_argument("--minimum-branch-interior-samples", type=int, default=MINIMUM_BRANCH_INTERIOR_SAMPLES)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_validation(
        experiment_path=args.experiment,
        occupancy_path=args.occupancy,
        hydraulic_state_path=args.hydraulic_state,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        branch_interior_threshold=args.branch_interior_threshold,
        minimum_branch_interior_samples=args.minimum_branch_interior_samples,
    )
    summary = result["summary"]
    selected = result["selected"]
    print("Droplet branch-velocity validation")
    print(f"  tracked velocity source: {summary['tracked_velocity_source_path']}")
    print(f"  tracked velocity units: {summary['tracked_velocity_units']}")
    print(f"  eligible candidates: {summary['candidate_count_total']}")
    print(f"  left/right eligible: {summary['candidate_count_left']} / {summary['candidate_count_right']}")
    for row in selected.itertuples(index=False):
        print(
            "  selected "
            f"track={int(row.track_id)} branch={row.branch} n={int(row.n_samples)} "
            f"Pearson={row.pearson_r:.6f} Spearman={row.spearman_r:.6f}"
        )
    print(f"  pooled Pearson: {summary['pooled_pearson_r']:.6f}")
    print(f"  pooled Spearman: {summary['pooled_spearman_r']:.6f}")
    print(f"  pooled samples: {summary['pooled_sample_count']}")
    print(f"  output directory: {result['output_dir']}")


if __name__ == "__main__":
    main()
