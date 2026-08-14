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
from torch.utils.data import default_collate

from src.datasets.canonical_window_dataset import create_train_val_test_datasets
from src.evaluation import single_droplet_runtime_rollout as runtime_rollout
from src.evaluation.plot_cfd_field_on_frame import resolve_video_path
from src.physics.runtime import load_physics_runtime_context
from scripts.training.train_physics_markovian import move_batch_to_device


DEFAULT_CONFIG = Path("configs/experiments/physics_markovian_v1.yml")
DEFAULT_EXPERIMENT = Path("configs/experiments/video_2.yml")
DEFAULT_DATASET = Path("outputs/processed/2/canonical_dataset_v2/canonical_dataset_v2.npz")
DEFAULT_CHECKPOINT = Path("outputs/models/temp/best_stage_horizon_015.pt")
DEFAULT_LIBRARY = Path("outputs/physics/full_device_cfd/library")
DEFAULT_OUTPUT = Path("outputs/evaluation/rollout_prediction_overlay_movies")

RED = (60, 60, 240)  # predicted (BGR)
GREEN = (80, 220, 70)  # ground truth (BGR)
DIM_RED = (60, 60, 140)  # non-target predicted trail (BGR)
WHITE = (245, 245, 245)
BLACK = (20, 20, 20)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch = runtime_rollout.import_torch()
    device = runtime_rollout.select_device(torch, args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = runtime_rollout.load_yaml(args.config)
    experiment_config = runtime_rollout.load_yaml(args.experiment_config)
    dataset = runtime_rollout.load_dataset(args.dataset)
    feature_names = [str(name) for name in dataset["feature_names"]]
    runtime_rollout.validate_current_feature_contract(feature_names, config)
    feature_index = {name: index for index, name in enumerate(feature_names)}

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = runtime_rollout.load_model(torch, checkpoint, device)
    normalization = checkpoint["normalization_stats"]
    target_features = tuple(str(name) for name in checkpoint.get("target_features", config["model"]["target_features"]))
    prediction_mode = runtime_rollout.prediction_mode_from_targets(target_features)
    target_parameterization = {"mode": prediction_mode}
    runtime_context = load_physics_runtime_context(
        experiment_config_path=args.experiment_config,
        cfd_library_path=args.cfd_library,
        feature_names=feature_names,
    )
    video_path = resolve_video_path(experiment_config)

    # Real multi-droplet window dataset (same construction as training/validation), so the target
    # droplet's rollout includes whatever other real droplets actually share its window -- NOT an
    # artificially isolated single droplet, which the model never saw during training.
    _, val_dataset, _, _ = create_train_val_test_datasets(
        npz_path=args.dataset,
        stride=int(config["dataset"]["stride"]),
        T_history=int(checkpoint["model_config"]["T_history"]),
        T_future=int(args.rollout_length),
        max_droplets=int(checkpoint["model_config"]["max_droplets"]),
        target_features=target_features,
        experiment_config=args.experiment_config,
    )

    scenarios = select_scenarios(
        raw_dataset=dataset,
        val_dataset=val_dataset,
        feature_index=feature_index,
        region_labels=runtime_context.region_labels,
        count=int(args.scenario_count),
        preferred_region=str(args.preferred_region),
        track_id=args.track_id,
        frame=args.frame,
    )
    if not scenarios:
        raise RuntimeError("No single-droplet scenarios were found.")

    summaries = []
    for scenario in scenarios:
        sample = val_dataset[scenario["window_index"]]
        batch = move_batch_to_device(default_collate([sample]), device)
        rows, all_droplets_frames = runtime_rollout.rollout_droplet_with_context(
            torch=torch,
            model=model,
            batch=batch,
            dataset=val_dataset,
            target_slot=scenario["target_slot"],
            device=device,
            normalization_stats=normalization,
            rollout_length=int(args.rollout_length),
            runtime_context=runtime_context,
            target_parameterization=target_parameterization,
            scenario=scenario,
        )
        table = attach_ground_truth(pd.DataFrame(rows), dataset, feature_index, scenario)

        output_stem = f"{scenario['scenario']}_track{scenario['track_id']}_frame{scenario['frame']}"
        output_avi = output_dir / f"{output_stem}.avi"
        movie_metadata = write_overlay_movie(
            table=table,
            all_droplets_frames=all_droplets_frames,
            video_path=video_path,
            output_avi=output_avi,
            movie_fps=float(args.movie_fps),
            crop_margin_px=int(args.crop_margin_px),
            full_frame=bool(args.full_frame),
        )
        table.to_csv(output_dir / f"{output_stem}_trajectory.csv", index=False)
        summary = summarize_overlay(table)
        summary["output_avi"] = str(output_avi)
        summary.update(movie_metadata)
        summaries.append(summary)

    save_json(
        output_dir / "summary.json",
        {
            "checkpoint": runtime_rollout.checkpoint_metadata(args.checkpoint, checkpoint),
            "target_features": target_features,
            "prediction_mode": prediction_mode,
            "video_path": str(video_path),
            "rollout_length": int(args.rollout_length),
            "scenario_count": len(summaries),
            "summaries": summaries,
        },
    )
    print_summary(output_dir, summaries)


def select_scenarios(
    *,
    raw_dataset: dict[str, Any],
    val_dataset,
    feature_index: dict[str, int],
    region_labels: np.ndarray,
    count: int,
    preferred_region: str,
    track_id: int | None,
    frame: int | None,
) -> list[dict[str, Any]]:
    """Pick (window_index, target_slot) scenarios against the real windowed dataset, so the
    rollout can include whatever other real droplets share that window -- candidate scoring
    (region preference, validity) still runs against the raw arrays for speed, same as before;
    only the window/slot lookup needed for the actual multi-droplet rollout is new."""
    if track_id is not None or frame is not None:
        if track_id is None or frame is None:
            raise ValueError("--track-id and --frame must be provided together.")
        return [scenario_from_track_frame_in_dataset(raw_dataset, val_dataset, feature_index, int(track_id), int(frame), "explicit")]

    starts = np.asarray(val_dataset.start_frames, dtype=np.int64)
    candidates = []
    for start_frame in starts:
        active_tracks = np.flatnonzero(raw_dataset["mask"][:, start_frame].astype(bool))
        for track_index in active_tracks:
            state = raw_dataset["Z"][track_index, start_frame, :].astype(np.float32)
            if not runtime_rollout.initial_state_is_valid(state, feature_index):
                continue
            region = runtime_rollout.region_at_pixel(
                float(state[feature_index["x"]]), float(state[feature_index["y"]]), region_labels
            )
            if region == "outside":
                continue
            preferred = region == preferred_region
            candidates.append((not preferred, int(start_frame), int(track_index), region))

    scenarios = []
    for rank, (_, frame_value, track_index, region) in enumerate(sorted(candidates)[: max(int(count), 0)]):
        track_id_value = int(raw_dataset["track_ids"][track_index])
        scenario = scenario_from_track_frame_in_dataset(
            raw_dataset, val_dataset, feature_index, track_id_value, frame_value, f"validation_{rank:02d}"
        )
        scenario["initial_region"] = region
        scenarios.append(scenario)
    return scenarios


def scenario_from_track_frame_in_dataset(
    raw_dataset: dict[str, Any],
    val_dataset,
    feature_index: dict[str, int],
    track_id: int,
    frame: int,
    name: str,
) -> dict[str, Any]:
    scenario = runtime_rollout.scenario_from_track_frame(raw_dataset, feature_index, track_id, frame, name)
    expected_start_frame = int(frame) - (int(val_dataset.T_history) - 1)
    starts = np.asarray(val_dataset.start_frames, dtype=np.int64)
    window_positions = np.flatnonzero(starts == expected_start_frame)
    if window_positions.size == 0:
        raise ValueError(
            f"frame={frame} (expected window start {expected_start_frame}) is not a valid validation-split "
            f"window start for this dataset/stride."
        )
    window_index = int(window_positions[0])
    sample = val_dataset[window_index]
    droplet_ids = sample["droplet_ids"]
    droplet_ids = droplet_ids.numpy() if hasattr(droplet_ids, "numpy") else np.asarray(droplet_ids)
    slot_matches = np.flatnonzero(droplet_ids == track_id)
    if slot_matches.size == 0:
        raise ValueError(f"track_id={track_id} is not present in the dataset window starting at frame={expected_start_frame}.")
    scenario["window_index"] = window_index
    scenario["target_slot"] = int(slot_matches[0])
    return scenario


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay closed-loop rollout predictions against ground-truth droplet positions on the raw video."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cfd-library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rollout-length", type=int, default=40)
    parser.add_argument("--scenario-count", type=int, default=4)
    parser.add_argument("--preferred-region", default="inlet channel")
    parser.add_argument("--track-id", type=int, default=None)
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--movie-fps", type=float, default=10.0)
    parser.add_argument("--crop-margin-px", type=int, default=90)
    parser.add_argument(
        "--full-frame", dest="full_frame", action="store_true", default=True,
        help="Show the whole device/loop instead of cropping to the target droplet's own trajectory (default: on).",
    )
    parser.add_argument(
        "--crop-to-target", dest="full_frame", action="store_false",
        help="Crop tightly to the target droplet's trajectory (plus --crop-margin-px) instead of showing the full frame.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def attach_ground_truth(
    table: pd.DataFrame,
    dataset: dict[str, Any],
    feature_index: dict[str, int],
    scenario: dict[str, Any],
) -> pd.DataFrame:
    track_index = int(scenario["track_index"])
    frame_index = int(scenario["frame_index"])
    mask = dataset["mask"]
    Z = dataset["Z"]
    frames = dataset["frames"]
    n_frames = Z.shape[1]

    true_x, true_y, true_w, true_h, source_frame, has_truth = [], [], [], [], [], []
    for step in table["rollout_step"]:
        target_index = frame_index + int(step)
        available = target_index < n_frames and bool(mask[track_index, target_index])
        if available:
            state = Z[track_index, target_index, :]
            true_x.append(float(state[feature_index["x"]]))
            true_y.append(float(state[feature_index["y"]]))
            true_w.append(float(state[feature_index["bbox_w"]]))
            true_h.append(float(state[feature_index["bbox_h"]]))
        else:
            true_x.append(np.nan)
            true_y.append(np.nan)
            true_w.append(np.nan)
            true_h.append(np.nan)
        source_frame.append(int(frames[min(target_index, n_frames - 1)]))
        has_truth.append(available)

    table = table.copy()
    table["true_x"] = true_x
    table["true_y"] = true_y
    table["true_bbox_w"] = true_w
    table["true_bbox_h"] = true_h
    table["source_frame"] = source_frame
    table["has_ground_truth"] = has_truth
    finite = table["has_ground_truth"]
    table["position_error_px"] = np.where(
        finite,
        np.hypot(table["x"] - table["true_x"], table["y"] - table["true_y"]),
        np.nan,
    )
    return table


MIN_CROP_WIDTH = 360
MIN_CROP_HEIGHT = 280
TEXT_BAND_HEIGHT = 66


def crop_bounds_dual(table: pd.DataFrame, source_width: int, source_height: int, margin_px: int) -> tuple[int, int, int, int]:
    x_values = [table["x"]]
    y_values = [table["y"]]
    if bool(table["has_ground_truth"].any()):
        truthy = table.loc[table["has_ground_truth"]]
        x_values.append(truthy["true_x"])
        y_values.append(truthy["true_y"])
    all_x = pd.concat(x_values)
    all_y = pd.concat(y_values)
    x0 = max(0, int(math.floor(all_x.min() - margin_px)))
    y0 = max(0, int(math.floor(all_y.min() - margin_px)))
    x1 = min(source_width, int(math.ceil(all_x.max() + margin_px)))
    y1 = min(source_height, int(math.ceil(all_y.max() + margin_px)))
    x0, x1 = widen_to_minimum(x0, x1, source_width, MIN_CROP_WIDTH)
    y0, y1 = widen_to_minimum(y0, y1, source_height, MIN_CROP_HEIGHT)
    if (x1 - x0) % 2:
        x1 = min(source_width, x1 + 1)
    if (y1 - y0) % 2:
        y1 = min(source_height, y1 + 1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Computed crop bounds are empty")
    return x0, y0, x1, y1


def widen_to_minimum(low: int, high: int, source_extent: int, minimum: int) -> tuple[int, int]:
    span = high - low
    if span >= minimum:
        return low, high
    deficit = minimum - span
    low = max(0, low - deficit // 2)
    high = min(source_extent, low + minimum)
    low = max(0, high - minimum)
    return low, high


def write_overlay_movie(
    *,
    table: pd.DataFrame,
    all_droplets_frames: list[list[dict[str, Any]]] | None,
    video_path: Path,
    output_avi: Path,
    movie_fps: float,
    crop_margin_px: int,
    full_frame: bool = True,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if full_frame:
        x0, y0, x1, y1 = 0, 0, source_width, source_height
    else:
        x0, y0, x1, y1 = crop_bounds_dual(table, source_width, source_height, crop_margin_px)
    width, height = x1 - x0, y1 - y0 + TEXT_BAND_HEIGHT
    writer = cv2.VideoWriter(str(output_avi), cv2.VideoWriter_fourcc(*"MJPG"), movie_fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create AVI writer: {output_avi}")

    pred_trail: list[tuple[int, int]] = []
    true_trail: list[tuple[int, int]] = []
    other_trails: dict[int, list[tuple[int, int]]] = {}
    try:
        for row_index, row in enumerate(table.itertuples(index=False)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(row.source_frame))
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"Could not read frame {int(row.source_frame)} from {video_path}")
            frame = frame[y0:y1, x0:x1].copy()
            frame = cv2.copyMakeBorder(frame, 0, TEXT_BAND_HEIGHT, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)

            # Every other real droplet in the window, drawn as a thin per-frame marker only (no
            # persistent trail, to keep the target's own trail the readable "story" of the video).
            other_count = 0
            if all_droplets_frames is not None and row_index < len(all_droplets_frames):
                for droplet in all_droplets_frames[row_index]:
                    if droplet["is_target"]:
                        continue
                    other_count += 1
                    other_pred_local = (int(round(droplet["pred_x"] - x0)), int(round(droplet["pred_y"] - y0)))
                    draw_ellipse(frame, other_pred_local, droplet["pred_bbox_w"], droplet["pred_bbox_h"], RED)
                    track_trail = other_trails.setdefault(droplet["track_id"], [])
                    track_trail.append(other_pred_local)
                    draw_trail(frame, track_trail, DIM_RED)
                    if droplet["has_ground_truth"]:
                        other_true_local = (int(round(droplet["true_x"] - x0)), int(round(droplet["true_y"] - y0)))
                        draw_ellipse(frame, other_true_local, droplet["true_bbox_w"], droplet["true_bbox_h"], GREEN)

            pred_local = (int(round(float(row.x) - x0)), int(round(float(row.y) - y0)))
            pred_trail.append(pred_local)
            draw_trail(frame, pred_trail, RED)
            draw_ellipse(frame, pred_local, float(row.bbox_w), float(row.bbox_h), RED, thickness=2)

            if bool(row.has_ground_truth):
                true_local = (int(round(float(row.true_x) - x0)), int(round(float(row.true_y) - y0)))
                true_trail.append(true_local)
                draw_trail(frame, true_trail, GREEN)
                draw_ellipse(frame, true_local, float(row.true_bbox_w), float(row.true_bbox_h), GREEN, thickness=2)

            annotate_overlay_text(frame, row, other_droplet_count=other_count)
            writer.write(frame)
    finally:
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


def draw_trail(frame: np.ndarray, trail: list[tuple[int, int]], color: tuple[int, int, int]) -> None:
    for point_a, point_b in zip(trail[:-1], trail[1:]):
        cv2.line(frame, point_a, point_b, color, 1, lineType=cv2.LINE_AA)


def draw_ellipse(
    frame: np.ndarray, center: tuple[int, int], bbox_w: float, bbox_h: float, color: tuple[int, int, int], thickness: int = 1
) -> None:
    if not (np.isfinite(bbox_w) and np.isfinite(bbox_h)):
        return
    cv2.ellipse(
        frame,
        center,
        (max(2, int(round(bbox_w * 0.5))), max(2, int(round(bbox_h * 0.5)))),
        0.0,
        0.0,
        360.0,
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )
    cv2.circle(frame, center, 2, color, -1, lineType=cv2.LINE_AA)


def annotate_overlay_text(frame: np.ndarray, row, other_droplet_count: int = 0) -> None:
    error_text = f"{row.position_error_px:.1f}px" if bool(row.has_ground_truth) else "no ground truth"
    lines = [
        f"frame {int(row.source_frame)} | track {int(row.template_track_id)} | rollout step {int(row.rollout_step)}",
        f"red = predicted | green = true | error {error_text} | other droplets shown: {other_droplet_count}",
        f"region={row.region} runtime_success={bool(row.runtime_success)}",
    ]
    band_top = frame.shape[0] - TEXT_BAND_HEIGHT
    x, y0 = 8, band_top + 18
    for index, text in enumerate(lines):
        y = y0 + index * 18
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1, cv2.LINE_AA)


def summarize_overlay(table: pd.DataFrame) -> dict[str, Any]:
    valid = table.loc[table["has_ground_truth"], "position_error_px"]
    return {
        "scenario": str(table["scenario"].iloc[0]),
        "track_id": int(table["template_track_id"].iloc[0]),
        "frame": int(table["template_frame"].iloc[0]),
        "rollout_steps": int(table["rollout_step"].max()),
        "ground_truth_steps": int(table["has_ground_truth"].sum()),
        "mean_position_error_px": float(valid.mean()) if len(valid) else float("nan"),
        "final_position_error_px": float(valid.iloc[-1]) if len(valid) else float("nan"),
        "max_position_error_px": float(valid.max()) if len(valid) else float("nan"),
        "trajectory_ever_leaves_channel": bool((~table["inside_channel"].astype(bool)).any()),
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def print_summary(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    print("Rollout prediction-vs-truth overlay movies complete")
    print(f"  output: {output_dir}")
    for summary in summaries:
        print(
            f"  {summary['scenario']}: track={summary['track_id']} frame={summary['frame']} "
            f"steps={summary['rollout_steps']} ground_truth_steps={summary['ground_truth_steps']} "
            f"mean_error={summary['mean_position_error_px']:.2f}px "
            f"final_error={summary['final_position_error_px']:.2f}px "
            f"leaves_channel={summary['trajectory_ever_leaves_channel']} "
            f"-> {summary['output_avi']}"
        )


if __name__ == "__main__":
    main()
