from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib-cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.physics.runtime import load_physics_runtime_context
from src.physics.runtime.state_transition import _image_points_to_library_frame


DEFAULT_CONFIG = Path("configs/experiments/video_2.yml")
DEFAULT_ENRICHED = Path("outputs/physics/video_2/enrichment/physics_enriched_tracked_features.csv")
DEFAULT_SUMMARY = Path("outputs/physics/video_2/enrichment/physics_enriched_tracking_summary.json")
DEFAULT_CFD_LIBRARY = Path("outputs/physics/full_device_cfd/library")
DEFAULT_OUTPUT = Path("outputs/evaluation/cfd_field_overlay/frame_cfd_overlay.png")
VIDEO_EXTENSIONS = (".avi", ".mp4", ".mov", ".mkv")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_yaml(args.config)
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    video_path = resolve_video_path(config)
    frame = read_video_frame(video_path, int(args.frame))
    context = load_physics_runtime_context(
        experiment_config_path=args.config,
        cfd_library_path=args.cfd_library,
    )
    left_fraction = float(args.left_fraction) if args.left_fraction is not None else median_left_fraction(args.enriched, int(args.frame))
    cutoff = image_cutoffs_from_summary(summary)
    vectors = sample_cfd_grid(
        context=context,
        frame_shape=frame.shape[:2],
        left_fraction=left_fraction,
        spacing_px=float(args.spacing_px),
        cutoff=cutoff,
    )
    droplets = load_frame_droplets(args.enriched, int(args.frame))
    save_overlay(frame, context.region_labels, vectors, droplets, cutoff, left_fraction, output)
    save_json(
        output.with_suffix(".json"),
        {
            "frame": int(args.frame),
            "video_path": str(video_path),
            "left_flow_fraction": left_fraction,
            "output": str(output),
            "vector_count": int(len(vectors["x"])),
            "cutoff": cutoff,
            "notes": "CFD vectors are sampled only where original_valid is true; y component is flipped for image-coordinate display.",
        },
    )
    print(f"Saved CFD field overlay: {output}")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay full-device CFD vectors on one real video frame.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--enriched", type=Path, default=DEFAULT_ENRICHED)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--cfd-library", type=Path, default=DEFAULT_CFD_LIBRARY)
    parser.add_argument("--frame", type=int, default=27415)
    parser.add_argument("--left-fraction", type=float, default=None)
    parser.add_argument("--spacing-px", type=float, default=18.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML config is empty or malformed: {path}")
    return data


def resolve_video_path(config: dict[str, Any]) -> Path:
    experiment = config["experiment"]
    data = experiment.get("data", {})
    raw = data.get("raw_video")
    if raw is None:
        raise KeyError("experiment.data.raw_video is missing")
    path = Path(str(raw))
    if path.is_dir():
        videos = sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS)
        experiment_id = str(experiment.get("id", ""))
        digits = "".join(ch for ch in experiment_id if ch.isdigit())
        if digits:
            preferred = [item for item in videos if item.stem == str(int(digits))]
            if len(preferred) == 1:
                return preferred[0]
        if len(videos) == 1:
            return videos[0]
        raise ValueError(f"Could not select one video from {path}; candidates: {[item.name for item in videos]}")
    return path


def read_video_frame(video_path: Path, frame: int) -> np.ndarray:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, bgr = capture.read()
    capture.release()
    if not ok or bgr is None:
        raise RuntimeError(f"Could not read frame {frame} from {video_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def median_left_fraction(enriched_path: Path, frame: int) -> float:
    table = pd.read_csv(enriched_path, usecols=["frame", "left_flow_fraction"])
    values = table.loc[table["frame"].astype(int) == int(frame), "left_flow_fraction"].dropna().to_numpy(float)
    if len(values) == 0:
        return float(table["left_flow_fraction"].dropna().median())
    return float(np.median(values))


def image_cutoffs_from_summary(summary: dict[str, Any]) -> dict[str, float]:
    transform = summary["coordinate_transform"]
    y_reference_px = float(transform["y_reference_px"])
    um_per_px = float(transform["um_per_px"])
    trim = summary["acquisition_domain_trim"]
    inlet_y_px = y_reference_px - float(trim["inlet"]["cutoff_y_device_um"]) / um_per_px
    outlet_y_px = y_reference_px - float(trim["outlet"]["cutoff_y_device_um"]) / um_per_px
    return {
        "inlet_y_px": float(inlet_y_px),
        "outlet_y_px": float(outlet_y_px),
        "y_reference_px": y_reference_px,
        "um_per_px": um_per_px,
    }


def sample_cfd_grid(context, frame_shape: tuple[int, int], left_fraction: float, spacing_px: float, cutoff: dict[str, float]) -> dict[str, np.ndarray]:
    height, width = frame_shape
    y_values = np.arange(max(cutoff["inlet_y_px"], 0.0), min(cutoff["outlet_y_px"], height - 1), spacing_px)
    x_values = np.arange(0.0, width - 1, spacing_px)
    xx, yy = np.meshgrid(x_values, y_values)
    points_px = np.column_stack([xx.ravel(), yy.ravel()])
    row = np.clip(np.rint(points_px[:, 1]).astype(int), 0, context.region_labels.shape[0] - 1)
    col = np.clip(np.rint(points_px[:, 0]).astype(int), 0, context.region_labels.shape[1] - 1)
    in_region_mask = context.region_labels[row, col] > 0
    points_px = points_px[in_region_mask]

    points_library = _image_points_to_library_frame(points_px, context)
    sample = context.cfd_library.interpolate(float(left_fraction)).sample_cfd(points_library)
    valid = sample.original_valid & np.isfinite(sample.cfd_u_norm) & np.isfinite(sample.cfd_v_norm)
    points_px = points_px[valid]
    u = sample.cfd_u_norm[valid]
    v_image = -sample.cfd_v_norm[valid]
    speed = np.hypot(u, v_image)
    nonzero = speed > 1.0e-12
    return {
        "x": points_px[nonzero, 0],
        "y": points_px[nonzero, 1],
        "u": u[nonzero] / speed[nonzero],
        "v": v_image[nonzero] / speed[nonzero],
        "speed_norm": speed[nonzero],
    }


def load_frame_droplets(enriched_path: Path, frame: int) -> pd.DataFrame:
    columns = ["frame", "track_id", "centroid_x", "centroid_y", "dominant_region"]
    table = pd.read_csv(enriched_path, usecols=columns)
    return table.loc[table["frame"].astype(int) == int(frame)].copy()


def save_overlay(
    frame: np.ndarray,
    region_labels: np.ndarray,
    vectors: dict[str, np.ndarray],
    droplets: pd.DataFrame,
    cutoff: dict[str, float],
    left_fraction: float,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 9), constrained_layout=True)
    ax.imshow(frame, origin="upper", cmap="gray")
    ax.contour(region_labels > 0, levels=[0.5], colors=["#22c55e"], linewidths=0.8, alpha=0.85)
    ax.axhspan(0, cutoff["inlet_y_px"], color="#ef4444", alpha=0.15, label="trimmed top/inlet")
    ax.axhspan(cutoff["outlet_y_px"], frame.shape[0], color="#ef4444", alpha=0.15, label="trimmed bottom/outlet")
    ax.axhline(cutoff["inlet_y_px"], color="#ef4444", linestyle="--", linewidth=1.2)
    ax.axhline(cutoff["outlet_y_px"], color="#ef4444", linestyle="--", linewidth=1.2)
    quiver = ax.quiver(
        vectors["x"],
        vectors["y"],
        vectors["u"],
        vectors["v"],
        vectors["speed_norm"],
        cmap="viridis",
        angles="xy",
        scale_units="xy",
        scale=0.12,
        width=0.003,
        alpha=0.9,
    )
    if len(droplets):
        ax.scatter(
            droplets["centroid_x"],
            droplets["centroid_y"],
            s=28,
            facecolors="none",
            edgecolors="#f97316",
            linewidths=1.0,
            label="tracked droplets",
        )
    ax.set_xlim(0, frame.shape[1])
    ax.set_ylim(frame.shape[0], 0)
    ax.set_xlabel("x (image px)")
    ax.set_ylabel("y (image px)")
    ax.set_title(
        f"CFD field overlay on frame | left_flow_fraction={left_fraction:.4f}\n"
        "Vectors use original-valid CFD points only; y flipped for image coordinates"
    )
    fig.colorbar(quiver, ax=ax, label="normalized CFD speed")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
