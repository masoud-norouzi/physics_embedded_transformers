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

from src.config import load_experiment_config
from src.geometry.centerlines import cumulative_arc_length, order_branch_points

COLORS = {
    "inlet": "tab:blue",
    "outlet": "tab:orange",
    "left": "tab:green",
    "right": "magenta",
}
EXPECTED_COLUMNS = {"x", "y", "channel"}
EXPECTED_BRANCHES = ("inlet", "outlet", "left", "right")
VIDEO_EXTENSIONS = (".avi", ".mp4", ".mov", ".mkv")
UPPER_JUNCTION = np.array([330.0, 215.0])
LOWER_JUNCTION = np.array([330.0, 396.0])
JUNCTION_BOX_SIZE_PX = 25.0


def _resolve_raw_video_path(experiment: dict[str, Any]) -> Path:
    data = experiment.get("data", {})
    video = experiment.get("video", {})
    data_video = data.get("video", {}) if isinstance(data, dict) else {}
    data_raw = data.get("raw", {}) if isinstance(data, dict) else {}
    candidates = [
        experiment.get("raw_video_path"),
        experiment.get("raw_video"),
        experiment.get("video_path"),
        experiment.get("video") if isinstance(experiment.get("video"), str) else None,
        data.get("raw_video_path") if isinstance(data, dict) else None,
        data.get("raw_video") if isinstance(data, dict) else None,
        data.get("video_path") if isinstance(data, dict) else None,
        data.get("video") if isinstance(data.get("video") if isinstance(data, dict) else None, str) else None,
        video.get("path") if isinstance(video, dict) else None,
        video.get("raw_path") if isinstance(video, dict) else None,
        video.get("file") if isinstance(video, dict) else None,
        data_video.get("path") if isinstance(data_video, dict) else None,
        data_video.get("raw_path") if isinstance(data_video, dict) else None,
        data_video.get("file") if isinstance(data_video, dict) else None,
        data_raw.get("video_path") if isinstance(data_raw, dict) else None,
        data_raw.get("video") if isinstance(data_raw, dict) else None,
        data_raw.get("path") if isinstance(data_raw, dict) else None,
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            path = Path(candidate)
            if path.is_dir():
                return _select_video_from_directory(path, experiment)
            return path
    raise ValueError(
        "Experiment config does not define a raw video path. Add one under "
        "experiment.raw_video_path, experiment.video.path, or experiment.data.raw_video_path."
    )


def _select_video_from_directory(directory: Path, experiment: dict[str, Any]) -> Path:
    data = experiment.get("data", {})
    processed_root = data.get("processed_root") or data.get("root") if isinstance(data, dict) else None
    numeric_hints = []
    for value in (experiment.get("id"), processed_root):
        if not value:
            continue
        text = str(value).replace("\\", "/").rstrip("/").split("/")[-1]
        digits = "".join(ch for ch in text if ch.isdigit())
        if digits:
            numeric_hints.append(str(int(digits)))

    files = [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS]
    for hint in numeric_hints:
        matches = [path for path in files if path.stem == hint]
        if len(matches) == 1:
            print(f"Resolved raw video directory to {matches[0]} using numeric hint {hint}.")
            return matches[0]
    if len(files) == 1:
        print(f"Resolved raw video directory to only video file: {files[0]}")
        return files[0]
    raise ValueError(
        f"Raw video path is a directory with {len(files)} video files and no unambiguous match: {directory}"
    )


def _read_frame(video_path: Path, frame_index: int) -> tuple[np.ndarray, dict[str, int | float]]:
    if not video_path.exists():
        raise FileNotFoundError(f"Raw video file does not exist: {video_path}")
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required to read the raw video frame. Install opencv-python.") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open raw video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame_bgr = capture.read()
    capture.release()
    if not ok or frame_bgr is None:
        raise ValueError(f"Could not read zero-based frame index {frame_index} from {video_path}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    info = {
        "requested_frame_index": frame_index,
        "total_frame_count": frame_count,
        "frame_width": width,
        "frame_height": height,
    }
    print(
        f"Video frame: requested={frame_index}, total={frame_count}, "
        f"width={width}, height={height}"
    )
    return frame_rgb, info


def _load_centerlines(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Centerline file does not exist: {path}")
    df = pd.read_csv(path)
    missing = EXPECTED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Centerline file is missing required columns: {sorted(missing)}")
    labels = set(df["channel"].astype(str).unique())
    missing_labels = set(EXPECTED_BRANCHES).difference(labels)
    if missing_labels:
        raise ValueError(f"Centerline file is missing required labels: {sorted(missing_labels)}")
    return df


def _branch_points(df: pd.DataFrame, branch: str) -> np.ndarray:
    return df[df["channel"].astype(str) == branch][["x", "y"]].to_numpy(float)


def _line_points(points: np.ndarray) -> tuple[np.ndarray, bool]:
    try:
        ordered = order_branch_points(points)
        return ordered, not np.allclose(ordered, points)
    except ValueError:
        return points, False


def _duplicates(df: pd.DataFrame) -> list[dict[str, Any]]:
    duplicated = df[df.duplicated(subset=["x", "y"], keep=False)].copy()
    details = []
    for (x, y), group in duplicated.groupby(["x", "y"], sort=True):
        details.append(
            {
                "x_px": float(x),
                "y_px": float(y),
                "branches": group["channel"].astype(str).tolist(),
                "row_indices": [int(idx) for idx in group.index.tolist()],
            }
        )
    return details


def _raw_lengths(df: pd.DataFrame) -> dict[str, float]:
    lengths = {}
    for branch in EXPECTED_BRANCHES:
        points = _branch_points(df, branch)
        lengths[branch] = float(cumulative_arc_length(points)[-1])
    return lengths


def _correction_value(device: dict[str, Any]) -> float:
    return float(device.get("channel", {}).get("width_px", 0.0)) * 2.0 - 1.0


def _add_overlays(ax: plt.Axes, frame: np.ndarray, df: pd.DataFrame, include_boxes: bool) -> None:
    ax.imshow(frame)
    for branch in EXPECTED_BRANCHES:
        points = _branch_points(df, branch)
        line_points, _ = _line_points(points)
        color = COLORS[branch]
        ax.plot(line_points[:, 0], line_points[:, 1], color=color, alpha=0.65, linewidth=2.0, label=branch)
        ax.scatter(points[:, 0], points[:, 1], color=color, s=9, alpha=0.85)

    for xy, label, marker in [
        (UPPER_JUNCTION, "upper junction", "o"),
        (LOWER_JUNCTION, "lower junction", "s"),
    ]:
        ax.scatter(xy[0], xy[1], c="yellow", edgecolors="black", marker=marker, s=90, zorder=5)
        ax.text(xy[0] + 6, xy[1] - 6, label, color="yellow", fontsize=9, weight="bold")
        if include_boxes:
            half = JUNCTION_BOX_SIZE_PX / 2.0
            rect = patches.Rectangle(
                (xy[0] - half, xy[1] - half),
                JUNCTION_BOX_SIZE_PX,
                JUNCTION_BOX_SIZE_PX,
                fill=False,
                linestyle="--",
                linewidth=2.0,
                edgecolor="yellow",
            )
            ax.add_patch(rect)
            ax.text(xy[0] + half + 3, xy[1] + half, "25 px junction box", color="yellow", fontsize=8)
    ax.set_aspect("equal", adjustable="box")


def _save_summary(
    path: Path,
    frame_index: int,
    video_path: Path,
    frame_info: dict[str, int | float],
    centerline_path: Path,
    df: pd.DataFrame,
    correction_px: float,
    overwrite: bool,
) -> dict[str, Any]:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Summary already exists. Use --overwrite: {path}")

    raw_lengths = _raw_lengths(df)
    branch_counts = {branch: int((_branch_points(df, branch)).shape[0]) for branch in EXPECTED_BRANCHES}
    ranges = {}
    for branch in EXPECTED_BRANCHES:
        points = _branch_points(df, branch)
        ranges[branch] = {
            "x_min": float(points[:, 0].min()),
            "x_max": float(points[:, 0].max()),
            "y_min": float(points[:, 1].min()),
            "y_max": float(points[:, 1].max()),
        }
    length_after_correction = {
        branch: float(length - correction_px) if branch in {"left", "right"} else float(length)
        for branch, length in raw_lengths.items()
    }
    summary = {
        "frame_index": frame_index,
        "raw_video_path": str(video_path),
        "frame_dimensions": {
            "width": frame_info["frame_width"],
            "height": frame_info["frame_height"],
        },
        "source_centerline_path": str(centerline_path),
        "point_count_per_branch": branch_counts,
        "original_coordinate_range_per_branch": ranges,
        "duplicate_coordinates": _duplicates(df),
        "upper_junction_coordinate": UPPER_JUNCTION.tolist(),
        "lower_junction_coordinate": LOWER_JUNCTION.tolist(),
        "junction_box_size_px": JUNCTION_BOX_SIZE_PX,
        "current_correction_value_px": correction_px,
        "correction_representation": "applied only as a scalar subtraction",
        "identifiable_correction_segments": [],
        "total_raw_length_per_branch_px": raw_lengths,
        "length_after_current_correction_px": length_after_correction,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _make_plot(
    frame: np.ndarray,
    df: pd.DataFrame,
    output_path: Path,
    frame_index: int,
    correction_px: float,
    point_size: float,
    line_width: float,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output image already exists. Use --overwrite: {output_path}")

    global _add_overlays

    def add_scaled_overlays(ax: plt.Axes, include_boxes: bool) -> None:
        ax.imshow(frame)
        for branch in EXPECTED_BRANCHES:
            points = _branch_points(df, branch)
            line_points, _ = _line_points(points)
            color = COLORS[branch]
            ax.plot(
                line_points[:, 0],
                line_points[:, 1],
                color=color,
                alpha=0.65,
                linewidth=line_width,
                label=branch,
            )
            ax.scatter(points[:, 0], points[:, 1], color=color, s=point_size, alpha=0.85)
        for xy, label, marker in [
            (UPPER_JUNCTION, "upper junction", "o"),
            (LOWER_JUNCTION, "lower junction", "s"),
        ]:
            ax.scatter(xy[0], xy[1], c="yellow", edgecolors="black", marker=marker, s=100, zorder=5)
            ax.text(xy[0] + 6, xy[1] - 6, label, color="yellow", fontsize=9, weight="bold")
            if include_boxes:
                half = JUNCTION_BOX_SIZE_PX / 2.0
                ax.add_patch(
                    patches.Rectangle(
                        (xy[0] - half, xy[1] - half),
                        JUNCTION_BOX_SIZE_PX,
                        JUNCTION_BOX_SIZE_PX,
                        fill=False,
                        linestyle="--",
                        linewidth=2.0,
                        edgecolor="yellow",
                    )
                )
                ax.text(xy[0] + half + 3, xy[1] + half, "25 px junction box", color="yellow", fontsize=8)
        ax.set_aspect("equal", adjustable="box")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    add_scaled_overlays(axes[0], include_boxes=False)
    axes[0].set_title("Full frame: original labeled centerline regions")
    axes[0].legend(loc="upper right")

    add_scaled_overlays(axes[1], include_boxes=True)
    all_points = df[["x", "y"]].to_numpy(float)
    xmin, ymin = all_points.min(axis=0) - 40
    xmax, ymax = all_points.max(axis=0) + 40
    axes[1].set_xlim(xmin, xmax)
    axes[1].set_ylim(ymax, ymin)
    axes[1].set_title("Junction view: 49 px correction is scalar only")
    axes[1].text(
        0.02,
        0.02,
        f"Current code subtracts {correction_px:.1f} px as a scalar;\n"
        "no source segments are preserved to highlight.",
        transform=axes[1].transAxes,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.65, "pad": 4},
        fontsize=9,
    )

    for ax in axes:
        ax.set_xlabel("x_px")
        ax.set_ylabel("y_px")
    fig.suptitle(f"Centerline region overlay, frame {frame_index}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay source centerline regions on one raw video frame.")
    parser.add_argument("--experiment", required=True, type=Path, help="Experiment YAML path.")
    parser.add_argument("--frame-index", type=int, default=0, help="Zero-based frame index to read.")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path.")
    parser.add_argument("--point-size", type=float, default=8.0, help="Centerline marker size.")
    parser.add_argument("--line-width", type=float, default=2.0, help="Centerline line width.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.experiment)
    experiment = config["experiment"]["experiment"]
    device = config["device"]["device"]
    device_id = device["id"]
    video_path = _resolve_raw_video_path(experiment)
    data = experiment["data"]
    processed_root = data.get("processed_root") or data.get("root")
    if not processed_root:
        raise ValueError("Experiment config data block must define processed_root or root.")
    centerline_path = Path(processed_root) / "centerlines.csv"
    output_path = args.output or Path("outputs/geometry") / device_id / f"centerline_regions_frame_{args.frame_index}.png"
    summary_path = Path("outputs/geometry") / device_id / "centerline_region_summary.json"

    frame, frame_info = _read_frame(video_path, args.frame_index)
    df = _load_centerlines(centerline_path)
    correction_px = _correction_value(device)
    _make_plot(
        frame,
        df,
        output_path,
        args.frame_index,
        correction_px,
        args.point_size,
        args.line_width,
        args.overwrite,
    )
    summary = _save_summary(
        summary_path,
        args.frame_index,
        video_path,
        frame_info,
        centerline_path,
        df,
        correction_px,
        args.overwrite,
    )

    print("Centerline region visualization")
    print(f"  output path: {output_path}")
    print(f"  selected frame index: {args.frame_index}")
    print(f"  branch point counts: {summary['point_count_per_branch']}")
    print(f"  duplicate coordinate details: {summary['duplicate_coordinates']}")
    print(f"  upper junction: {summary['upper_junction_coordinate']}")
    print(f"  lower junction: {summary['lower_junction_coordinate']}")
    print(f"  correction implementation: {summary['correction_representation']} ({correction_px:.1f} px)")
    print("  exact corrected segments visualized: no")


if __name__ == "__main__":
    main()
