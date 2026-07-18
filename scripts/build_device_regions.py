from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from src.config import load_experiment_config
from src.geometry.regions import LABEL_NAMES, RegionLabel, build_region_label_map

VIDEO_EXTENSIONS = (".avi", ".mp4", ".mov", ".mkv")
REGION_COLORS = {
    RegionLabel.UNASSIGNED: "#111111",
    RegionLabel.INLET: "tab:blue",
    RegionLabel.OUTLET: "tab:orange",
    RegionLabel.LEFT_BRANCH: "tab:green",
    RegionLabel.RIGHT_BRANCH: "magenta",
    RegionLabel.UPPER_JUNCTION: "cyan",
    RegionLabel.LOWER_JUNCTION: "yellow",
}
SCRIPT_VERSION = "device-region-labels-v1"


def _resolve_raw_video_path(experiment: dict[str, Any]) -> Path:
    data = experiment.get("data", {})
    candidates = [
        experiment.get("raw_video_path"),
        experiment.get("video_path"),
        data.get("raw_video") if isinstance(data, dict) else None,
        data.get("raw_video_path") if isinstance(data, dict) else None,
        data.get("video_path") if isinstance(data, dict) else None,
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
            print(f"Resolved raw video directory to {matches[0]} using numeric hint {hint}.")
            return matches[0]
    if len(videos) == 1:
        return videos[0]
    raise ValueError(f"Could not select one video from directory: {directory}")


def _read_frame(video_path: Path, frame_index: int) -> tuple[np.ndarray, dict[str, int]]:
    if not video_path.exists():
        raise FileNotFoundError(f"Raw video file does not exist: {video_path}")
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required. Install requirements into the project virtual environment.") from exc
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open raw video: {video_path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame_bgr = capture.read()
    capture.release()
    if not ok or frame_bgr is None:
        raise ValueError(f"Could not read frame index {frame_index} from {video_path}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return frame_rgb, {"frame_index": frame_index, "frame_count": total, "width": width, "height": height}


def _resolve_sources(config: dict[str, dict[str, Any]]) -> tuple[Path, Path, Path]:
    experiment = config["experiment"]["experiment"]
    device = config["device"]["device"]
    geometry = device["geometry"]
    data = experiment.get("data", {})
    processed_root = data.get("processed_root") or data.get("root") if isinstance(data, dict) else None
    channel_mask_path = Path(geometry.get("channel_mask") or data.get("channel_mask"))
    centerline_path = Path(geometry.get("centerline") or Path(processed_root) / "centerlines.csv")
    raw_video_path = _resolve_raw_video_path(experiment)
    return channel_mask_path, centerline_path, raw_video_path


def _metadata(
    label_map: np.ndarray,
    channel_mask: np.ndarray,
    diagnostics: dict[str, Any],
    config: dict[str, dict[str, Any]],
    channel_mask_path: Path,
    centerline_path: Path,
    raw_video_path: Path,
) -> dict[str, Any]:
    device = config["device"]["device"]
    um_per_px = float(device["calibration"]["um_per_px"])
    pixel_area = um_per_px * um_per_px
    label_counts = diagnostics["label_counts"]
    region_areas = {name: count * pixel_area for name, count in label_counts.items()}
    junctions = diagnostics["junctions"]
    return {
        "device_id": device["id"],
        "image_shape": list(label_map.shape),
        "pixel_scale_um": um_per_px,
        "label_id_to_name": {str(int(label)): name for label, name in LABEL_NAMES.items()},
        "source_channel_mask_path": str(channel_mask_path),
        "source_centerline_path": str(centerline_path),
        "source_raw_video_path": str(raw_video_path),
        "upper_junction_definition": _junction_metadata(junctions["upper"], um_per_px),
        "lower_junction_definition": _junction_metadata(junctions["lower"], um_per_px),
        "pixel_count_by_region": label_counts,
        "physical_area_um2_by_region": region_areas,
        "background_pixel_count": diagnostics["background_pixel_count"],
        "unassigned_within_channel_pixel_count": diagnostics["unassigned_within_channel_pixels"],
        "total_channel_pixel_count": diagnostics["total_channel_pixels"],
        "region_generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "script_or_schema_version": SCRIPT_VERSION,
        "validation_results": {
            "assigned_pixels_outside_channel": diagnostics["assigned_pixels_outside_channel"],
            "overlap_pixels_before_precedence_resolution": diagnostics[
                "overlap_pixels_before_precedence_resolution"
            ],
            "assigned_channel_fraction": diagnostics["assigned_channel_fraction"],
            "connectivity": diagnostics["connectivity"],
        },
    }


def _junction_metadata(junction: Any, um_per_px: float) -> dict[str, Any]:
    return {
        "center_px": junction.center_px.tolist(),
        "center_um": (junction.center_px * um_per_px).tolist(),
        "size_px": junction.size_px.tolist(),
        "size_um": junction.size_um.tolist(),
    }


def _save_overlay(
    path: Path,
    frame: np.ndarray,
    label_map: np.ndarray,
    centerlines: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    colors = [REGION_COLORS[RegionLabel(i)] for i in range(7)]
    cmap = ListedColormap(colors)
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), gridspec_kw={"width_ratios": [1.1, 1.0, 0.9]})

    axes[0].imshow(frame)
    masked_labels = np.ma.masked_where(label_map == 0, label_map)
    axes[0].imshow(masked_labels, cmap=cmap, vmin=0, vmax=6, alpha=0.38)
    _draw_centerlines_and_junctions(axes[0], centerlines, metadata)
    axes[0].set_title("Raw frame with device regions")

    axes[1].imshow(label_map, cmap=cmap, vmin=0, vmax=6)
    _draw_centerlines_and_junctions(axes[1], centerlines, metadata)
    axes[1].set_title("Categorical region label map")

    text = _diagnostic_text(metadata)
    axes[2].axis("off")
    axes[2].text(0.0, 1.0, text, va="top", family="monospace", fontsize=9)
    axes[2].set_title("Diagnostics")

    for ax in axes[:2]:
        handles = [
            patches.Patch(color=REGION_COLORS[label], label=LABEL_NAMES[label])
            for label in RegionLabel
            if label != RegionLabel.UNASSIGNED
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x_px")
        ax.set_ylabel("y_px")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _draw_centerlines_and_junctions(ax: plt.Axes, centerlines: pd.DataFrame, metadata: dict[str, Any]) -> None:
    branch_colors = {"inlet": "blue", "outlet": "orange", "left": "green", "right": "magenta"}
    for branch, color in branch_colors.items():
        subset = centerlines[centerlines["channel"].astype(str) == branch]
        ax.plot(subset["x"], subset["y"], color=color, linewidth=1.3, alpha=0.9)
    for key, label in [("upper_junction_definition", "upper junction"), ("lower_junction_definition", "lower junction")]:
        info = metadata[key]
        center = np.asarray(info["center_px"], dtype=float)
        size = np.asarray(info["size_px"], dtype=float)
        ax.scatter(center[0], center[1], s=80, c="white", edgecolors="black", zorder=5)
        ax.add_patch(
            patches.Rectangle(
                center - size / 2.0,
                size[0],
                size[1],
                fill=False,
                linestyle="--",
                linewidth=1.8,
                edgecolor="white",
            )
        )
        ax.text(center[0] + 5, center[1] - 5, label, color="white", fontsize=8, weight="bold")


def _diagnostic_text(metadata: dict[str, Any]) -> str:
    counts = metadata["pixel_count_by_region"]
    areas = metadata["physical_area_um2_by_region"]
    lines = [
        f"device: {metadata['device_id']}",
        f"shape: {metadata['image_shape']}",
        f"um/px: {metadata['pixel_scale_um']}",
        "",
        "pixels / area_um2:",
    ]
    for name in ("inlet", "outlet", "left_branch", "right_branch", "upper_junction", "lower_junction"):
        lines.append(f"{name:16s} {counts[name]:7d} {areas[name]:10.1f}")
    lines.extend(
        [
            "",
            f"unassigned in channel: {metadata['unassigned_within_channel_pixel_count']}",
            f"background pixels:     {metadata['background_pixel_count']}",
            f"channel pixels:        {metadata['total_channel_pixel_count']}",
            "",
            "connectivity:",
        ]
    )
    for name, ok in metadata["validation_results"]["connectivity"].items():
        lines.append(f"{name}: {ok}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build device-level physical region labels.")
    parser.add_argument("--experiment", required=True, type=Path, help="Experiment YAML path.")
    parser.add_argument("--frame-index", type=int, default=0, help="Raw video frame index for visualization.")
    parser.add_argument("--output-root", type=Path, default=Path("data/geometry"), help="Device geometry output root.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing device region directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.experiment)
    device_id = config["device"]["device"]["id"]
    output_dir = args.output_root / device_id
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory exists. Use --overwrite: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    channel_mask_path, centerline_path, raw_video_path = _resolve_sources(config)
    if not channel_mask_path.exists():
        raise FileNotFoundError(f"Channel mask does not exist: {channel_mask_path}")
    if not centerline_path.exists():
        raise FileNotFoundError(f"Centerline file does not exist: {centerline_path}")

    channel_mask = np.load(channel_mask_path).astype(bool)
    centerlines = pd.read_csv(centerline_path)
    label_map, diagnostics = build_region_label_map(channel_mask, centerlines, config["device"])
    frame, frame_info = _read_frame(raw_video_path, args.frame_index)
    metadata = _metadata(label_map, channel_mask, diagnostics, config, channel_mask_path, centerline_path, raw_video_path)
    metadata["visualization_frame"] = frame_info

    label_path = output_dir / "region_labels.npy"
    metadata_path = output_dir / "region_metadata.json"
    overlay_path = output_dir / "region_overlay.png"
    np.save(label_path, label_map)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _save_overlay(overlay_path, frame, label_map, centerlines, metadata)

    print("Device region build summary")
    print(f"  output directory: {output_dir}")
    print(f"  region labels: {label_path}")
    print(f"  metadata: {metadata_path}")
    print(f"  overlay: {overlay_path}")
    print(f"  total channel pixels: {metadata['total_channel_pixel_count']}")
    for name in ("inlet", "outlet", "left_branch", "right_branch", "upper_junction", "lower_junction"):
        print(
            f"  {name}: {metadata['pixel_count_by_region'][name]} px, "
            f"{metadata['physical_area_um2_by_region'][name]:.1f} um^2"
        )
    print(f"  unassigned within channel: {metadata['unassigned_within_channel_pixel_count']}")
    print(f"  assigned channel fraction: {metadata['validation_results']['assigned_channel_fraction']:.4f}")
    print(f"  connectivity: {metadata['validation_results']['connectivity']}")


if __name__ == "__main__":
    main()
