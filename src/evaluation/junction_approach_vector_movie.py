from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib-cache").resolve()))

import cv2
import numpy as np
import pandas as pd
import yaml

from src.evaluation.plot_cfd_field_on_frame import resolve_video_path
from src.physics.runtime import load_physics_runtime_context
from src.physics.runtime.state_transition import _image_points_to_library_frame


DEFAULT_CONFIG = Path("configs/experiments/video_2.yml")
DEFAULT_TRACKS = Path("outputs/physics/video_2/enrichment/physics_enriched_tracked_features.csv")
DEFAULT_CFD_LIBRARY = Path("outputs/physics/full_device_cfd/library")
DEFAULT_OUTPUT_DIR = Path("outputs/evaluation/junction_approach_vector_movie")

JUNCTION_REGION = "upper_junction"
BRANCH_REGIONS = {"left_branch", "right_branch"}
INLET_REGION = "inlet"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_yaml(args.config)
    video_path = resolve_video_path(config)
    context = load_physics_runtime_context(
        experiment_config_path=args.config,
        cfd_library_path=args.cfd_library,
    )
    tracks = load_tracks(args.tracks_csv)
    selected = select_sequence(
        tracks,
        track_id=args.track_id,
        pre_junction_frames=int(args.pre_junction_frames),
        post_branch_frames=int(args.post_branch_frames),
        min_frames=int(args.min_frames),
    )
    annotated = annotate_cfd(selected, context)

    track_id_value = int(selected["track_id"].iloc[0])
    output_stem = f"junction_approach_track_{track_id_value}"
    output_avi = output_dir / f"{output_stem}.avi"
    output_csv = output_dir / f"{output_stem}_vectors.csv"
    output_json = output_dir / f"{output_stem}_summary.json"
    metadata = write_movie(
        rows=annotated,
        video_path=video_path,
        output_avi=output_avi,
        movie_fps=float(args.movie_fps),
        crop_margin_px=int(args.crop_margin_px),
        arrow_length_px=float(args.arrow_length_px),
    )
    annotated.to_csv(output_csv, index=False)
    save_json(
        output_json,
        {
            "output_avi": str(output_avi),
            "output_csv": str(output_csv),
            "tracks_csv": str(args.tracks_csv),
            "config": str(args.config),
            "cfd_library": str(args.cfd_library),
            "video_path": str(video_path),
            "track_id": track_id_value,
            "frame_start": int(selected["frame"].min()),
            "frame_end": int(selected["frame"].max()),
            "n_frames": int(len(selected)),
            "mean_angle_error_deg": float(annotated["angle_error_deg"].dropna().mean()),
            "median_angle_error_deg": float(annotated["angle_error_deg"].dropna().median()),
            "cfd_original_valid_fraction": float(annotated["cfd_original_valid"].mean()),
            "movie": metadata,
            "notes": (
                "Observed velocity is drawn in image coordinates by flipping device-y. "
                "Centroid CFD is sampled at the tracked centroid using the row left_flow_fraction; "
                "CFD-y is also flipped for image-coordinate display."
            ),
        },
    )
    save_json(output_dir / "latest_summary.json", json.loads(output_json.read_text(encoding="utf-8")))
    print("Junction approach vector movie complete")
    print(f"  output_avi: {output_avi}")
    print(f"  track_id: {track_id_value}")
    print(f"  frames: {int(selected['frame'].min())}..{int(selected['frame'].max())} ({len(selected)} frames)")
    print(f"  median angle error: {annotated['angle_error_deg'].dropna().median():.2f} deg")
    print(f"  CFD original-valid fraction: {annotated['cfd_original_valid'].mean():.3f}")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an annotated AVI for one droplet approaching the junction and entering a branch."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tracks-csv", type=Path, default=DEFAULT_TRACKS)
    parser.add_argument("--cfd-library", type=Path, default=DEFAULT_CFD_LIBRARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--track-id", type=int, default=None)
    parser.add_argument("--pre-junction-frames", type=int, default=35)
    parser.add_argument("--post-branch-frames", type=int, default=25)
    parser.add_argument("--min-frames", type=int, default=25)
    parser.add_argument("--movie-fps", type=float, default=20.0)
    parser.add_argument("--crop-margin-px", type=int, default=90)
    parser.add_argument("--arrow-length-px", type=float, default=34.0)
    return parser.parse_args(argv)


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML config is empty or malformed: {path}")
    return data


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
    table = table.dropna(subset=["frame", "track_id", "centroid_x", "centroid_y", "dominant_region"])
    table["frame"] = table["frame"].astype(int)
    table["track_id"] = table["track_id"].astype(int)
    table["dominant_region"] = table["dominant_region"].astype(str)
    return table.sort_values(["track_id", "frame"]).reset_index(drop=True)


def select_sequence(
    tracks: pd.DataFrame,
    *,
    track_id: int | None,
    pre_junction_frames: int,
    post_branch_frames: int,
    min_frames: int,
) -> pd.DataFrame:
    candidates: list[tuple[float, pd.DataFrame]] = []
    groups = [(int(track_id), tracks.loc[tracks["track_id"] == int(track_id)])] if track_id is not None else tracks.groupby("track_id")
    for candidate_track_id, group in groups:
        group = group.sort_values("frame").reset_index(drop=True)
        sequence = candidate_sequence_for_track(group, pre_junction_frames, post_branch_frames)
        if sequence is None or len(sequence) < min_frames:
            continue
        score = sequence_score(sequence)
        candidates.append((score, sequence))
    if not candidates:
        if track_id is None:
            raise RuntimeError("Could not find an inlet -> upper_junction -> branch trajectory.")
        raise RuntimeError(f"Could not find a valid junction sequence for track_id={track_id}.")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1].copy()


def candidate_sequence_for_track(group: pd.DataFrame, pre_junction_frames: int, post_branch_frames: int) -> pd.DataFrame | None:
    group = group.sort_values("frame").reset_index(drop=True)
    regions = group["dominant_region"].to_numpy(str)
    junction_indices = np.flatnonzero(regions == JUNCTION_REGION)
    if len(junction_indices) == 0:
        return None
    for first_junction_index in junction_indices:
        preceding = regions[:first_junction_index]
        following = regions[first_junction_index + 1 :]
        if not np.any(preceding == INLET_REGION):
            continue
        branch_after = np.flatnonzero(np.isin(following, list(BRANCH_REGIONS)))
        if len(branch_after) == 0:
            continue
        first_branch_index = int(first_junction_index + 1 + branch_after[0])
        start_frame = int(group.loc[first_junction_index, "frame"]) - int(pre_junction_frames)
        end_frame = int(group.loc[first_branch_index, "frame"]) + int(post_branch_frames)
        sequence = group[(group["frame"] >= start_frame) & (group["frame"] <= end_frame)].copy()
        if len(sequence) == 0:
            continue
        return sequence
    return None


def sequence_score(sequence: pd.DataFrame) -> float:
    regions = sequence["dominant_region"].astype(str)
    branch_count = float(regions.isin(BRANCH_REGIONS).sum())
    junction_count = float((regions == JUNCTION_REGION).sum())
    inlet_count = float((regions == INLET_REGION).sum())
    finite_speed = float(np.isfinite(sequence["observed_speed_mm_per_s"]).sum())
    return branch_count * 3.0 + junction_count * 2.0 + inlet_count + finite_speed * 0.1


def annotate_cfd(rows: pd.DataFrame, context) -> pd.DataFrame:
    rows = rows.copy()
    cfd_vx: list[float] = []
    cfd_vy_image: list[float] = []
    cfd_speed: list[float] = []
    original_valid: list[bool] = []
    projection_um: list[float] = []
    angle_errors: list[float] = []
    for row in rows.itertuples(index=False):
        left_fraction = sanitize_left_fraction(float(row.left_flow_fraction), context)
        field = context.cfd_library.interpolate(left_fraction)
        points_px = np.asarray([[float(row.centroid_x), float(row.centroid_y)]], dtype=float)
        sample = field.sample_cfd(_image_points_to_library_frame(points_px, context))
        vx = float(sample.u_x_m_per_s[0] * 1000.0)
        vy_image = float(-sample.u_y_m_per_s[0] * 1000.0)
        cfd_vx.append(vx)
        cfd_vy_image.append(vy_image)
        cfd_speed.append(float(np.hypot(vx, vy_image)))
        original_valid.append(bool(sample.original_valid[0]))
        projection_um.append(float(sample.projection_distance_um[0]))
        obs_vx = float(row.observed_v_x_device_mm_per_s)
        obs_vy_image = -float(row.observed_v_y_device_mm_per_s)
        angle_errors.append(angle_between_deg(vx, vy_image, obs_vx, obs_vy_image))
    rows["cfd_centroid_vx_image_mm_s"] = cfd_vx
    rows["cfd_centroid_vy_image_mm_s"] = cfd_vy_image
    rows["cfd_centroid_speed_mm_s"] = cfd_speed
    rows["cfd_original_valid"] = original_valid
    rows["cfd_projection_distance_um"] = projection_um
    rows["observed_vx_image_mm_s"] = rows["observed_v_x_device_mm_per_s"].astype(float)
    rows["observed_vy_image_mm_s"] = -rows["observed_v_y_device_mm_per_s"].astype(float)
    rows["angle_error_deg"] = angle_errors
    return rows


def sanitize_left_fraction(left_fraction: float, context) -> float:
    split_min, split_max = context.cfd_split_bounds
    if not np.isfinite(left_fraction):
        return float((split_min + split_max) * 0.5)
    return float(np.clip(left_fraction, split_min, split_max))


def write_movie(
    *,
    rows: pd.DataFrame,
    video_path: Path,
    output_avi: Path,
    movie_fps: float,
    crop_margin_px: int,
    arrow_length_px: float,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop = crop_bounds(rows, source_width, source_height, crop_margin_px)
    x0, y0, x1, y1 = crop
    width = x1 - x0
    height = y1 - y0
    writer = cv2.VideoWriter(str(output_avi), cv2.VideoWriter_fourcc(*"MJPG"), movie_fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create AVI writer: {output_avi}")
    previous_points: list[tuple[int, int]] = []
    for row in rows.itertuples(index=False):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(row.frame))
        ok, frame = capture.read()
        if not ok or frame is None:
            writer.release()
            capture.release()
            raise RuntimeError(f"Could not read frame {int(row.frame)} from {video_path}")
        frame = frame[y0:y1, x0:x1].copy()
        local_x = int(round(float(row.centroid_x) - x0))
        local_y = int(round(float(row.centroid_y) - y0))
        previous_points.append((local_x, local_y))
        draw_annotation(frame, row, local_x, local_y, previous_points, arrow_length_px)
        writer.write(frame)
    writer.release()
    capture.release()
    return {
        "codec": "MJPG",
        "fps": float(movie_fps),
        "source_width": int(source_width),
        "source_height": int(source_height),
        "crop_x0": int(x0),
        "crop_y0": int(y0),
        "crop_x1": int(x1),
        "crop_y1": int(y1),
        "output_width": int(width),
        "output_height": int(height),
    }


def crop_bounds(rows: pd.DataFrame, source_width: int, source_height: int, margin_px: int) -> tuple[int, int, int, int]:
    x0 = max(0, int(math.floor(rows["centroid_x"].min() - margin_px)))
    y0 = max(0, int(math.floor(rows["centroid_y"].min() - margin_px)))
    x1 = min(source_width, int(math.ceil(rows["centroid_x"].max() + margin_px)))
    y1 = min(source_height, int(math.ceil(rows["centroid_y"].max() + margin_px)))
    if (x1 - x0) % 2:
        x1 = min(source_width, x1 + 1)
    if (y1 - y0) % 2:
        y1 = min(source_height, y1 + 1)
    return x0, y0, x1, y1


def draw_annotation(frame: np.ndarray, row, local_x: int, local_y: int, trail: list[tuple[int, int]], arrow_length_px: float) -> None:
    green = (80, 220, 70)
    blue = (240, 125, 45)
    white = (245, 245, 245)
    yellow = (40, 220, 250)
    red = (60, 60, 240)
    black = (20, 20, 20)
    for point_a, point_b in zip(trail[-28:-1], trail[-27:]):
        cv2.line(frame, point_a, point_b, yellow, 1, lineType=cv2.LINE_AA)
    cv2.ellipse(
        frame,
        (local_x, local_y),
        (max(2, int(round(float(row.bbox_w) * 0.5))), max(2, int(round(float(row.bbox_h) * 0.5)))),
        0.0,
        0.0,
        360.0,
        yellow,
        1,
        lineType=cv2.LINE_AA,
    )
    cv2.circle(frame, (local_x, local_y), 3, white, -1, lineType=cv2.LINE_AA)
    draw_arrow(frame, local_x, local_y, float(row.cfd_centroid_vx_image_mm_s), float(row.cfd_centroid_vy_image_mm_s), blue, arrow_length_px)
    draw_arrow(frame, local_x, local_y, float(row.observed_vx_image_mm_s), float(row.observed_vy_image_mm_s), green, arrow_length_px)
    if not bool(row.cfd_original_valid):
        cv2.circle(frame, (local_x, local_y), 8, red, 1, lineType=cv2.LINE_AA)
    annotate_text(frame, row, green, blue, yellow, white, black)


def draw_arrow(frame: np.ndarray, x: int, y: int, vx: float, vy: float, color: tuple[int, int, int], length_px: float) -> None:
    norm = float(np.hypot(vx, vy))
    if not np.isfinite(norm) or norm <= 1.0e-9:
        return
    end = (int(round(x + vx / norm * length_px)), int(round(y + vy / norm * length_px)))
    cv2.arrowedLine(frame, (x, y), end, color, 2, line_type=cv2.LINE_AA, tipLength=0.28)


def annotate_text(
    frame: np.ndarray,
    row,
    green: tuple[int, int, int],
    blue: tuple[int, int, int],
    yellow: tuple[int, int, int],
    white: tuple[int, int, int],
    black: tuple[int, int, int],
) -> None:
    lines = [
        f"frame {int(row.frame)} | track {int(row.track_id)} | {row.dominant_region}",
        f"angle CFD vs droplet: {format_float(row.angle_error_deg, 'deg')} | L split {float(row.left_flow_fraction):.3f}",
        f"green droplet dir {format_float(row.observed_speed_mm_per_s, 'mm/s')} | blue centroid CFD {format_float(row.cfd_centroid_speed_mm_s, 'mm/s')}",
        f"yellow trail/ellipse | CFD original-valid={bool(row.cfd_original_valid)} proj={float(row.cfd_projection_distance_um):.1f} um",
    ]
    x, y0 = 8, 20
    for index, text in enumerate(lines):
        y = y0 + index * 20
        cv2.putText(frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.48, black, 3, cv2.LINE_AA)
        color = white if index != 2 else green
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    cv2.putText(frame, "centroid CFD", (8, frame.shape[0] - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue, 2, cv2.LINE_AA)
    cv2.putText(frame, "droplet direction", (8, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 2, cv2.LINE_AA)


def format_float(value: float, suffix: str) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.1f} {suffix}"


def angle_between_deg(ax: float, ay: float, bx: float, by: float) -> float:
    a_norm = float(np.hypot(ax, ay))
    b_norm = float(np.hypot(bx, by))
    if a_norm <= 1.0e-12 or b_norm <= 1.0e-12:
        return float("nan")
    cos_value = np.clip((ax * bx + ay * by) / (a_norm * b_norm), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_value)))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
