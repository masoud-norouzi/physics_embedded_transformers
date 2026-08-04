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

from src.physics.occupancy.ellipse import rasterize_bbox_ellipse
from src.physics.runtime import load_physics_runtime_context
from src.physics.runtime.state_transition import _image_points_to_library_frame


DEFAULT_TRACKS = Path("outputs/physics/video_2/enrichment/physics_enriched_tracked_features.csv")
DEFAULT_CONFIG = Path("configs/experiments/video_2.yml")
DEFAULT_CFD_LIBRARY = Path("outputs/physics/full_device_cfd/library")
DEFAULT_OUTPUT = Path("outputs/evaluation/junction_ellipse_cfd_average")
REGION_NAMES = {
    0: "outside",
    1: "inlet_channel",
    2: "outlet_channel",
    3: "left_branch",
    4: "right_branch",
    5: "inlet_junction",
    6: "outlet_junction",
}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    context = load_physics_runtime_context(
        experiment_config_path=args.config,
        cfd_library_path=args.cfd_library,
    )
    tracks = load_tracks(args.tracks_csv)
    selected = select_junction_droplets(
        tracks,
        context.region_labels,
        count=int(args.count),
        min_speed_mm_s=float(args.min_speed_mm_s),
        selection=str(args.selection),
    )
    rows = []
    for sample_id, row in enumerate(selected.itertuples(index=False), start=1):
        rows.append(analyze_droplet(row, sample_id, context))
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "junction_ellipse_cfd_average.csv", index=False)
    save_comparison_figure(results, context.region_labels, output_dir / "junction_ellipse_cfd_average.png")
    save_json(
        output_dir / "summary.json",
        {
            "tracks_csv": str(args.tracks_csv),
            "config": str(args.config),
            "cfd_library": str(args.cfd_library),
            "count": int(len(results)),
            "selection": str(args.selection),
            "mean_angle_error_deg": float(results["angle_error_deg"].mean()),
            "median_angle_error_deg": float(results["angle_error_deg"].median()),
            "mean_cfd_valid_pixel_fraction": float(results["cfd_valid_pixel_fraction"].mean()),
            "outputs": {
                "csv": str(output_dir / "junction_ellipse_cfd_average.csv"),
                "figure": str(output_dir / "junction_ellipse_cfd_average.png"),
            },
        },
    )
    print("Junction ellipse CFD average study complete")
    print(f"  output: {output_dir}")
    print(results[["sample_id", "frame", "track_id", "x", "y", "angle_error_deg", "cfd_valid_pixel_fraction"]].to_string(index=False))


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ellipse-averaged CFD direction to observed droplet velocity in the junction.")
    parser.add_argument("--tracks-csv", type=Path, default=DEFAULT_TRACKS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cfd-library", type=Path, default=DEFAULT_CFD_LIBRARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--min-speed-mm-s", type=float, default=5.0)
    parser.add_argument("--selection", choices=("center", "wall"), default="center")
    return parser.parse_args(argv)


def load_tracks(path: Path) -> pd.DataFrame:
    columns = [
        "frame",
        "track_id",
        "centroid_x",
        "centroid_y",
        "bbox_w",
        "bbox_h",
        "left_flow_fraction",
        "dominant_region",
        "observed_v_x_device_mm_per_s",
        "observed_v_y_device_mm_per_s",
        "observed_speed_mm_per_s",
    ]
    table = pd.read_csv(path, usecols=columns)
    table = table.rename(columns={"centroid_x": "x", "centroid_y": "y"})
    table = table.dropna()
    table["frame"] = table["frame"].astype(int)
    table["track_id"] = table["track_id"].astype(int)
    return table


def select_junction_droplets(
    tracks: pd.DataFrame,
    region_labels: np.ndarray,
    *,
    count: int,
    min_speed_mm_s: float,
    selection: str,
) -> pd.DataFrame:
    center = region_centroid(region_labels, region_id=5)
    distance_to_wall = channel_distance_to_wall(region_labels)
    candidates = tracks[
        (tracks["dominant_region"].astype(str) == "upper_junction")
        & (tracks["bbox_w"] > 0.0)
        & (tracks["bbox_h"] > 0.0)
        & (tracks["observed_speed_mm_per_s"] >= float(min_speed_mm_s))
    ].copy()
    if candidates.empty:
        raise RuntimeError("No upper_junction candidates found.")
    candidates["distance_to_junction_center_px"] = np.hypot(candidates["x"] - center[0], candidates["y"] - center[1])
    candidates["distance_to_wall_px"] = sample_distance_image(
        distance_to_wall,
        candidates["x"].to_numpy(float),
        candidates["y"].to_numpy(float),
    )
    candidates = candidates[np.isfinite(candidates["distance_to_wall_px"])]
    if str(selection) == "wall":
        candidates = candidates.sort_values(["distance_to_wall_px", "distance_to_junction_center_px", "frame", "track_id"])
    elif str(selection) == "center":
        candidates = candidates.sort_values(["distance_to_junction_center_px", "frame", "track_id"])
    else:
        raise ValueError(f"Unsupported selection: {selection!r}")
    selected = []
    used_tracks: set[int] = set()
    for row in candidates.itertuples(index=False):
        track_id = int(row.track_id)
        if track_id in used_tracks:
            continue
        selected.append(row._asdict())
        used_tracks.add(track_id)
        if len(selected) >= int(count):
            break
    if len(selected) < int(count):
        selected = candidates.head(int(count)).to_dict("records")
    return pd.DataFrame(selected)


def channel_distance_to_wall(region_labels: np.ndarray) -> np.ndarray:
    mask = (region_labels > 0).astype(np.uint8)
    try:
        import cv2

        return cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(float)
    except ModuleNotFoundError:
        from scipy import ndimage

        return ndimage.distance_transform_edt(mask).astype(float)


def sample_distance_image(distance: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    cols = np.rint(x).astype(int)
    rows = np.rint(y).astype(int)
    valid = (rows >= 0) & (rows < distance.shape[0]) & (cols >= 0) & (cols < distance.shape[1])
    out = np.full(len(x), np.nan, dtype=float)
    out[valid] = distance[rows[valid], cols[valid]]
    return out


def region_centroid(region_labels: np.ndarray, region_id: int) -> np.ndarray:
    rows, cols = np.where(region_labels == int(region_id))
    if len(rows) == 0:
        raise RuntimeError(f"Region id {region_id} is not present in region labels.")
    return np.asarray([float(cols.mean()), float(rows.mean())], dtype=float)


def analyze_droplet(row, sample_id: int, context) -> dict[str, Any]:
    ellipse = rasterize_bbox_ellipse(
        float(row.x),
        float(row.y),
        float(row.bbox_w),
        float(row.bbox_h),
        context.region_labels.shape,
    )
    yy, xx = np.indices(ellipse.mask.shape)
    points_px = np.column_stack([xx[ellipse.mask] + ellipse.x0, yy[ellipse.mask] + ellipse.y0]).astype(float)
    field = context.cfd_library.interpolate(float(row.left_flow_fraction))
    samples = field.sample_cfd(_image_points_to_library_frame(points_px, context))
    valid = (
        np.asarray(samples.original_valid, dtype=bool)
        & np.isfinite(samples.u_x_m_per_s)
        & np.isfinite(samples.u_y_m_per_s)
    )
    if not valid.any():
        raise RuntimeError(f"No valid CFD pixels for sample {sample_id}, frame={row.frame}, track={row.track_id}")
    cfd_ux_m_s = float(np.mean(samples.u_x_m_per_s[valid]))
    cfd_uy_m_s = float(np.mean(samples.u_y_m_per_s[valid]))
    cfd_vx_mm_s = cfd_ux_m_s * 1000.0
    cfd_vy_image_mm_s = -cfd_uy_m_s * 1000.0
    centroid_sample = field.sample_cfd(_image_points_to_library_frame(np.asarray([[float(row.x), float(row.y)]], dtype=float), context))
    centroid_cfd_vx_mm_s = float(centroid_sample.u_x_m_per_s[0] * 1000.0)
    centroid_cfd_vy_image_mm_s = float(-centroid_sample.u_y_m_per_s[0] * 1000.0)
    obs_vx_mm_s = float(row.observed_v_x_device_mm_per_s)
    obs_vy_image_mm_s = -float(row.observed_v_y_device_mm_per_s)
    angle_error = angle_between_deg(cfd_vx_mm_s, cfd_vy_image_mm_s, obs_vx_mm_s, obs_vy_image_mm_s)
    centroid_angle_error = angle_between_deg(centroid_cfd_vx_mm_s, centroid_cfd_vy_image_mm_s, obs_vx_mm_s, obs_vy_image_mm_s)
    return {
        "sample_id": int(sample_id),
        "frame": int(row.frame),
        "track_id": int(row.track_id),
        "x": float(row.x),
        "y": float(row.y),
        "bbox_w": float(row.bbox_w),
        "bbox_h": float(row.bbox_h),
        "left_flow_fraction": float(row.left_flow_fraction),
        "ellipse_pixel_count": int(ellipse.pixel_count),
        "cfd_valid_pixel_count": int(valid.sum()),
        "cfd_valid_pixel_fraction": float(valid.mean()),
        "cfd_avg_vx_mm_s": cfd_vx_mm_s,
        "cfd_avg_vy_image_mm_s": cfd_vy_image_mm_s,
        "cfd_avg_speed_mm_s": float(np.hypot(cfd_vx_mm_s, cfd_vy_image_mm_s)),
        "cfd_centroid_vx_mm_s": centroid_cfd_vx_mm_s,
        "cfd_centroid_vy_image_mm_s": centroid_cfd_vy_image_mm_s,
        "cfd_centroid_speed_mm_s": float(np.hypot(centroid_cfd_vx_mm_s, centroid_cfd_vy_image_mm_s)),
        "observed_vx_mm_s": obs_vx_mm_s,
        "observed_vy_image_mm_s": obs_vy_image_mm_s,
        "observed_speed_mm_s": float(row.observed_speed_mm_per_s),
        "angle_error_deg": angle_error,
        "centroid_cfd_angle_error_deg": centroid_angle_error,
        "avg_vs_centroid_cfd_angle_deg": angle_between_deg(
            cfd_vx_mm_s,
            cfd_vy_image_mm_s,
            centroid_cfd_vx_mm_s,
            centroid_cfd_vy_image_mm_s,
        ),
        "distance_to_junction_center_px": float(getattr(row, "distance_to_junction_center_px", np.nan)),
        "distance_to_wall_px": float(getattr(row, "distance_to_wall_px", np.nan)),
        "ellipse_x0": int(ellipse.x0),
        "ellipse_x1": int(ellipse.x1),
        "ellipse_y0": int(ellipse.y0),
        "ellipse_y1": int(ellipse.y1),
    }


def angle_between_deg(ax: float, ay: float, bx: float, by: float) -> float:
    a_norm = float(np.hypot(ax, ay))
    b_norm = float(np.hypot(bx, by))
    if a_norm <= 1.0e-12 or b_norm <= 1.0e-12:
        return float("nan")
    cos_value = np.clip((ax * bx + ay * by) / (a_norm * b_norm), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_value)))


def save_comparison_figure(results: pd.DataFrame, region_labels: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(18, 8), constrained_layout=True)
    axes = axes.ravel()
    for ax, row in zip(axes, results.itertuples(index=False)):
        crop = crop_bounds(row, region_labels.shape, margin=45)
        y0, y1, x0, x1 = crop
        ax.imshow(region_labels[y0:y1, x0:x1] > 0, cmap="gray", alpha=0.28, origin="upper", extent=[x0, x1, y1, y0])
        ax.contour(region_labels[y0:y1, x0:x1] == 5, levels=[0.5], colors=["#a855f7"], linewidths=1.0, extent=[x0, x1, y1, y0])
        ellipse = plt.matplotlib.patches.Ellipse(
            (row.x, row.y),
            width=row.bbox_w,
            height=row.bbox_h,
            angle=0.0,
            facecolor="none",
            edgecolor="#111827",
            linewidth=1.5,
        )
        ax.add_patch(ellipse)
        arrow_scale = 0.22
        ax.arrow(
            row.x,
            row.y,
            row.cfd_avg_vx_mm_s * arrow_scale,
            row.cfd_avg_vy_image_mm_s * arrow_scale,
            color="#dc2626",
            width=0.7,
            head_width=5,
            length_includes_head=True,
            label="ellipse avg CFD",
        )
        ax.arrow(
            row.x,
            row.y,
            row.cfd_centroid_vx_mm_s * arrow_scale,
            row.cfd_centroid_vy_image_mm_s * arrow_scale,
            color="#2563eb",
            width=0.5,
            head_width=4,
            length_includes_head=True,
            label="centroid CFD",
        )
        ax.arrow(
            row.x,
            row.y,
            row.observed_vx_mm_s * arrow_scale,
            row.observed_vy_image_mm_s * arrow_scale,
            color="#16a34a",
            width=0.7,
            head_width=5,
            length_includes_head=True,
            label="observed",
        )
        ax.scatter([row.x], [row.y], s=18, color="#111827", zorder=5)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_aspect("equal")
        ax.set_title(f"#{row.sample_id} frame {row.frame}\nangle={row.angle_error_deg:.1f} deg valid={row.cfd_valid_pixel_fraction:.2f}", fontsize=9)
    for ax in axes[len(results):]:
        ax.axis("off")
    handles = [
        plt.Line2D([0], [0], color="#dc2626", linewidth=3, label="ellipse-avg CFD"),
        plt.Line2D([0], [0], color="#2563eb", linewidth=3, label="centroid CFD"),
        plt.Line2D([0], [0], color="#16a34a", linewidth=3, label="observed velocity"),
        plt.Line2D([0], [0], color="#111827", linewidth=2, label="bbox ellipse"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4)
    fig.suptitle("Upper-junction droplets: ellipse-averaged CFD vector vs observed velocity", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def crop_bounds(row, image_shape: tuple[int, int], margin: int) -> tuple[int, int, int, int]:
    height, width = image_shape
    x0 = max(0, int(np.floor(row.x - margin)))
    x1 = min(width, int(np.ceil(row.x + margin)))
    y0 = max(0, int(np.floor(row.y - margin)))
    y1 = min(height, int(np.ceil(row.y + margin)))
    return y0, y1, x0, x1


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
