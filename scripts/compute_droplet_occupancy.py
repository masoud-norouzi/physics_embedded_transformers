from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from src.config import load_experiment_config
from src.geometry.regions import RegionLabel
from src.occupancy import NORM_COLUMNS, RAW_COLUMNS, calculate_dataset_occupancy, summarize_occupancy

VIDEO_EXTENSIONS = (".avi", ".mp4", ".mov", ".mkv")
REGION_COLORS = {
    0: "#111111",
    1: "tab:blue",
    2: "tab:orange",
    3: "tab:green",
    4: "magenta",
    5: "cyan",
    6: "yellow",
}


def _resolve_video_path(experiment: dict[str, Any]) -> Path:
    data = experiment.get("data", {})
    candidates = [
        data.get("raw_video") if isinstance(data, dict) else None,
        data.get("raw_video_path") if isinstance(data, dict) else None,
        data.get("video_path") if isinstance(data, dict) else None,
        experiment.get("raw_video_path"),
        experiment.get("video_path"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            path = Path(candidate)
            if path.is_dir():
                return _select_video_from_directory(path, experiment)
            return path
    raise ValueError("Experiment config does not define a raw video path")


def _select_video_from_directory(directory: Path, experiment: dict[str, Any]) -> Path:
    data = experiment.get("data", {})
    hints = []
    for value in (experiment.get("id"), data.get("processed_root") if isinstance(data, dict) else None):
        if not value:
            continue
        tail = str(value).replace("\\", "/").rstrip("/").split("/")[-1]
        digits = "".join(ch for ch in tail if ch.isdigit())
        if digits:
            hints.append(str(int(digits)))
    videos = [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS]
    for hint in hints:
        matches = [path for path in videos if path.stem == hint]
        if len(matches) == 1:
            return matches[0]
    if len(videos) == 1:
        return videos[0]
    raise ValueError(f"Could not select one raw video from directory: {directory}")


def _read_video_frame(video_path: Path, frame_index: int) -> np.ndarray:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open raw video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame_bgr = capture.read()
    capture.release()
    if not ok or frame_bgr is None:
        raise ValueError(f"Could not read frame {frame_index} from {video_path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _load_inputs(config: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any], Path, Path, Path]:
    experiment = config["experiment"]["experiment"]
    device = config["device"]["device"]
    data = experiment.get("data", {})
    tracks_path = Path(data.get("tracks", ""))
    labels_path = Path(device["geometry"].get("region_labels_path", ""))
    metadata_path = Path(device["geometry"].get("region_metadata_path", ""))
    for label, path in [
        ("tracked features", tracks_path),
        ("region labels", labels_path),
        ("region metadata", metadata_path),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    tracks = pd.read_csv(tracks_path)
    labels = np.load(labels_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return tracks, labels, metadata, tracks_path, labels_path, metadata_path


def _save_diagnostics(summary: dict[str, Any], occupancy: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    coverage = occupancy["physical_coverage_raw"].to_numpy(float)
    axes[0].hist(coverage, bins=60, color="steelblue")
    axes[0].axvline(summary["configured_low_coverage_threshold"], color="red", linestyle="--")
    axes[0].set_title("Physical coverage")
    axes[0].set_xlabel("coverage")
    sorted_cov = np.sort(coverage)
    axes[1].plot(sorted_cov, np.arange(1, len(sorted_cov) + 1) / len(sorted_cov))
    axes[1].axvline(summary["configured_low_coverage_threshold"], color="red", linestyle="--")
    axes[1].set_title("Coverage ECDF")
    axes[1].set_xlabel("coverage")
    computable = occupancy[occupancy["occupancy_computable"]]
    computable["dominant_region"].value_counts().sort_index().plot(kind="bar", ax=axes[2], color="gray")
    axes[2].set_title("Dominant region")
    computable["number_active_regions"].value_counts().sort_index().plot(kind="bar", ax=axes[3], color="gray")
    axes[3].set_title("Active regions > 0.01")
    axes[4].axis("off")
    text = "\n".join(
        [
            f"low threshold: {summary['configured_low_coverage_threshold']}",
            f"samples: {summary['total_droplet_frame_samples']}",
            f"computable: {summary['occupancy_computable_count']}",
            f"computable frac: {summary['occupancy_computable_fraction']:.4f}",
            f"low coverage: {summary['low_physical_coverage_count']}",
            f"low cov frac: {summary['low_physical_coverage_fraction']:.4f}",
            f"coverage mean: {summary['physical_coverage_mean']:.4f}",
            f"coverage median: {summary['physical_coverage_median']:.4f}",
            f"boundary clipped: {summary['image_boundary_clipped_count']}",
            f"max norm sum err: {summary['maximum_normalized_sum_error']:.3e}",
        ]
    )
    axes[4].text(0, 1, text, va="top", family="monospace")
    axes[5].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _select_spot_checks(occupancy: pd.DataFrame) -> pd.DataFrame:
    selected = []
    computable = occupancy[occupancy["occupancy_computable"]]
    targets = [
        ("inlet", computable[computable["dominant_region"] == "inlet"].sort_values("w_inlet", ascending=False)),
        ("outlet", computable[computable["dominant_region"] == "outlet"].sort_values("w_outlet", ascending=False)),
        ("left_branch", computable[computable["dominant_region"] == "left_branch"].sort_values("w_left", ascending=False)),
        ("right_branch", computable[computable["dominant_region"] == "right_branch"].sort_values("w_right", ascending=False)),
    ]
    for _, frame in targets:
        if len(frame):
            selected.append(frame.iloc[0])
    pairs = [
        ("w_inlet", "w_upper_junction"),
        ("w_upper_junction", "w_left"),
        ("w_left", "w_lower_junction"),
        ("w_lower_junction", "w_outlet"),
    ]
    for a, b in pairs:
        candidates = computable[(computable[a] > 0.05) & (computable[b] > 0.05)].copy()
        if len(candidates):
            candidates["pair_score"] = candidates[a] + candidates[b]
            selected.append(candidates.sort_values("pair_score", ascending=False).iloc[0])
    for _, row in occupancy.sort_values("physical_coverage_raw").head(6).iterrows():
        selected.append(row)
    if not selected:
        return occupancy.head(0)
    return pd.DataFrame(selected).drop_duplicates(subset=["frame", "track_id"]).head(12)


def _save_spot_checks(
    spot_checks: pd.DataFrame,
    labels: np.ndarray,
    video_path: Path,
    output_path: Path,
    um_per_px: float,
) -> None:
    if len(spot_checks) == 0:
        return
    cmap = ListedColormap([REGION_COLORS[idx] for idx in range(7)])
    ncols = 3
    nrows = int(np.ceil(len(spot_checks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
    frame_cache: dict[int, np.ndarray] = {}
    for ax, row in zip(axes.ravel(), spot_checks.itertuples(index=False)):
        frame_index = int(row.frame)
        if frame_index not in frame_cache:
            frame_cache[frame_index] = _read_video_frame(video_path, frame_index)
        frame = frame_cache[frame_index]
        cx = float(row.center_x_um) / um_per_px
        cy = float(row.center_y_um) / um_per_px
        bw = float(row.bbox_width_um) / um_per_px
        bh = float(row.bbox_height_um) / um_per_px
        pad = 45
        x0 = max(0, int(np.floor(cx - bw / 2 - pad)))
        x1 = min(labels.shape[1], int(np.ceil(cx + bw / 2 + pad)))
        y0 = max(0, int(np.floor(cy - bh / 2 - pad)))
        y1 = min(labels.shape[0], int(np.ceil(cy + bh / 2 + pad)))
        ax.imshow(frame[y0:y1, x0:x1])
        mask = np.ma.masked_where(labels[y0:y1, x0:x1] == 0, labels[y0:y1, x0:x1])
        ax.imshow(mask, cmap=cmap, vmin=0, vmax=6, alpha=0.35)
        ax.add_patch(
            patches.Rectangle((cx - bw / 2 - x0, cy - bh / 2 - y0), bw, bh, fill=False, edgecolor="white", linewidth=1.5)
        )
        ax.add_patch(
            patches.Ellipse((cx - x0, cy - y0), bw, bh, fill=False, edgecolor="red", linewidth=1.5)
        )
        vector = [getattr(row, column) for column in NORM_COLUMNS]
        vector_text = "noncomputable" if not bool(row.occupancy_computable) else ", ".join(f"{v:.2f}" for v in vector)
        ax.set_title(
            f"f{frame_index} t{int(row.track_id)} cov={float(row.physical_coverage_raw):.2f} low={bool(row.low_physical_coverage)}\n{vector_text}",
            fontsize=8,
        )
        ax.axis("off")
    for ax in axes.ravel()[len(spot_checks) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute droplet fractional occupancy from device region labels.")
    parser.add_argument("--experiment", required=True, type=Path, help="Experiment YAML path.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/physics/video_2"), help="Output directory.")
    parser.add_argument("--minimum-physical-coverage", type=float, default=0.95, help="Diagnostic low-coverage threshold.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_dir / "droplet_occupancy.csv",
        output_dir / "droplet_occupancy_summary.json",
        output_dir / "occupancy_diagnostics.png",
        output_dir / "occupancy_spot_checks.png",
    ]
    if not args.overwrite:
        existing = [path for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(f"Output files already exist. Use --overwrite: {existing}")

    config = load_experiment_config(args.experiment)
    experiment = config["experiment"]["experiment"]
    device = config["device"]["device"]
    tracks, labels, _, tracks_path, labels_path, _ = _load_inputs(config)
    um_per_px = float(device["calibration"]["um_per_px"])
    occupancy = calculate_dataset_occupancy(
        tracks,
        labels,
        um_per_px=um_per_px,
        minimum_physical_coverage=args.minimum_physical_coverage,
    )
    summary = summarize_occupancy(
        occupancy,
        experiment_id=experiment["id"],
        device_id=device["id"],
        tracked_feature_path=str(tracks_path),
        region_label_path=str(labels_path),
        um_per_px=um_per_px,
        minimum_physical_coverage=args.minimum_physical_coverage,
    )
    occupancy.to_csv(outputs[0], index=False)
    outputs[1].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _save_diagnostics(summary, occupancy, outputs[2])
    video_path = _resolve_video_path(experiment)
    _save_spot_checks(_select_spot_checks(occupancy), labels, video_path, outputs[3], um_per_px)

    print("Droplet occupancy summary")
    print(f"  output CSV: {outputs[0]}")
    print(f"  summary JSON: {outputs[1]}")
    print(f"  samples: {summary['total_droplet_frame_samples']}")
    print(f"  occupancy computable: {summary['occupancy_computable_count']}")
    print(f"  occupancy noncomputable: {summary['occupancy_noncomputable_count']}")
    print(f"  occupancy computable fraction: {summary['occupancy_computable_fraction']:.6f}")
    print(f"  low physical coverage: {summary['low_physical_coverage_count']}")
    print(f"  low physical coverage fraction: {summary['low_physical_coverage_fraction']:.6f}")
    print(f"  coverage mean/median: {summary['physical_coverage_mean']:.6f} / {summary['physical_coverage_median']:.6f}")
    print(f"  >=0.95 / >=0.98 / >=0.99: {summary['coverage_ge_0.95_fraction']:.6f} / {summary['coverage_ge_0.98_fraction']:.6f} / {summary['coverage_ge_0.99_fraction']:.6f}")
    print(f"  boundary clipped: {summary['image_boundary_clipped_count']}")
    print(f"  max normalized sum error: {summary['maximum_normalized_sum_error']:.3e}")


if __name__ == "__main__":
    main()
