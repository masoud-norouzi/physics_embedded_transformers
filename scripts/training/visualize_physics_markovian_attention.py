from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import default_collate

from scripts.training.train_physics_markovian import (
    denormalize_features,
    denormalize_targets,
    move_batch_to_device,
    normalize_features,
    refresh_observed_non_target_features,
    rollout_weights,
)
from src.datasets.canonical_window_dataset import CanonicalWindowDataset
from src.models.canonical_rollout_transformer import CanonicalRolloutTransformer


CSV_COLUMNS = [
    "sample_rank",
    "window_index",
    "frame_start",
    "absolute_frame",
    "rollout_step",
    "target_slot",
    "target_track_id",
    "attended_slot",
    "attended_track_id",
    "aggregated_attention",
    "normalized_attention",
    "current_x",
    "current_y",
    "distance_to_target",
    "rank",
]


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = CanonicalRolloutTransformer(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = build_validation_dataset(
        npz_path=args.npz_path,
        horizon=args.length,
        stride=int(checkpoint.get("stride", args.stride)),
        normalization_stats=checkpoint["normalization_stats"],
        history_length=int(checkpoint["model_config"]["T_history"]),
        max_droplets=int(checkpoint["model_config"]["max_droplets"]),
    )
    video_path = resolve_video_path(args.video_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.window_index is not None or args.target_slot is not None:
        if args.window_index is None or args.target_slot is None:
            raise ValueError("--window-index and --target-slot must be provided together.")
        selections = [(args.window_index, args.target_slot)]
    else:
        selections = select_validation_windows(dataset, args.num_videos, args.length)
    print(f"checkpoint: {args.checkpoint}")
    print(f"dataset:    {args.npz_path}")
    print(f"video:      {video_path}")
    print(f"device:     {device}")
    print(f"validation windows: {len(dataset)}")
    print(f"selected windows: {selections}")

    weights = rollout_weights(args.length, float(checkpoint.get("loss_alpha", 2.0)), device)
    attention_shape = None
    generated = []
    for rank, selection in enumerate(selections, start=1):
        window_index, target_slot = selection
        sample = dataset[window_index]
        batch = move_batch_to_device(default_collate([sample]), device)
        with torch.inference_mode():
            rows, render_payload, attention_shape = collect_attention_rollout(
                args=args,
                sample_rank=rank,
                model=model,
                batch=batch,
                dataset=dataset,
                normalization_stats=checkpoint["normalization_stats"],
                weights=weights,
                target_slot=target_slot,
                device=device,
            )
        assert_no_nan_payload(render_payload)
        frame_start = int(sample["frame_start"])
        track_id = int(sample["droplet_ids"][target_slot])
        stem = f"attention_validation_sample_{rank:02d}_window_{window_index:05d}_track_{track_id}"
        video_output = args.output_dir / f"{stem}.mp4"
        csv_output = args.output_dir / f"{stem}.csv"
        write_csv(csv_output, rows)
        write_attention_video(video_path, video_output, args, render_payload)
        generated.append(video_output)
        print(f"wrote: {video_output}")
        print(f"csv:   {csv_output}")

    print(f"attention tensor shape: {attention_shape}")
    print(f"videos generated: {len(generated)}")
    print(f"output directory: {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay physics Markovian Transformer attention on video.")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/models/physics_markovian_v1/best_checkpoint.pt"))
    parser.add_argument("--npz-path", type=Path, default=Path("outputs/processed/2/canonical_dataset_v2/canonical_dataset_v2.npz"))
    parser.add_argument("--video-path", type=Path, default=Path("D:/Microfluidic loop projct/new loop experiments/confined droplets 2/2.avi"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/models/physics_markovian_v1/attention"))
    parser.add_argument("--num-videos", type=int, default=5)
    parser.add_argument("--window-index", type=int, default=None)
    parser.add_argument("--target-slot", type=int, default=None)
    parser.add_argument("--length", type=int, default=50)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--layer", default="final")
    parser.add_argument("--head", default="mean")
    parser.add_argument("--line-start", choices=("pred", "true"), default="pred")
    parser.add_argument("--min-line-thickness", type=int, default=1)
    parser.add_argument("--max-line-thickness", type=int, default=8)
    parser.add_argument("--exclude-self", action="store_true")
    parser.add_argument("--no-labels", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def build_validation_dataset(npz_path, horizon, stride, normalization_stats, history_length, max_droplets):
    data = np.load(npz_path, allow_pickle=False)
    T = len(data["frames"])
    total_window = history_length + horizon
    all_starts = np.arange(0, T - total_window + 1, stride, dtype=np.int64)
    train_end = int(0.70 * len(all_starts))
    val_end = int(0.85 * len(all_starts))
    val_starts = all_starts[train_end:val_end]
    return CanonicalWindowDataset(
        npz_path=npz_path,
        start_frames=val_starts,
        T_history=history_length,
        T_future=horizon,
        max_droplets=max_droplets,
        normalization_stats=normalization_stats,
    )


def resolve_video_path(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        candidates = sorted([*path.glob("*.avi"), *path.glob("*.mp4"), *path.glob("*.mov"), *path.glob("*.mkv")])
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"Could not resolve video path: {path}")


def select_validation_windows(dataset, num_videos: int, horizon: int) -> list[tuple[int, int]]:
    if len(dataset) == 0:
        raise ValueError("Validation dataset is empty.")
    candidate_indices = np.linspace(0, len(dataset) - 1, max(num_videos * 8, num_videos), dtype=int)
    scored = []
    for window_index in candidate_indices:
        sample = dataset[int(window_index)]
        future_mask = sample["future_mask"].detach().cpu().numpy().astype(bool)
        history = sample["history_x"].detach().cpu().numpy()
        droplet_ids = sample["droplet_ids"].detach().cpu().numpy()
        valid_counts = future_mask.sum(axis=0)
        usable_slots = np.flatnonzero((droplet_ids >= 0) & (valid_counts >= max(3, horizon // 5)))
        if usable_slots.size == 0:
            continue
        active = int(future_mask.any(axis=0).sum())
        x_spread = float(np.nanmax(history[..., 0]) - np.nanmin(history[..., 0])) if active else 0.0
        slot = int(usable_slots[np.argmax(valid_counts[usable_slots])])
        scored.append((active, x_spread, int(window_index), slot))
    scored.sort(reverse=True)
    selected = []
    min_separation = max(len(dataset) // max(num_videos, 1) // 2, 1)
    for _, _, window_index, slot in scored:
        if all(abs(window_index - prev) >= min_separation for prev, _ in selected):
            selected.append((window_index, slot))
        if len(selected) == num_videos:
            return sorted(selected)
    for _, _, window_index, slot in scored:
        if (window_index, slot) not in selected:
            selected.append((window_index, slot))
        if len(selected) == num_videos:
            break
    return sorted(selected)


def collect_attention_rollout(args, sample_rank, model, batch, dataset, normalization_stats, weights, target_slot, device):
    history = batch["history_x"].clone()
    history_mask = batch["history_mask"].clone()
    frame_start = int(batch["frame_start"][0].detach().cpu())
    droplet_ids = batch["droplet_ids"][0].detach().cpu().numpy()
    feature_index = dataset.feature_indices
    max_droplets = int(history.shape[2])
    true_future_features = get_true_future_features(batch, dataset, device, args.length)
    query_token = (history.shape[1] - 1) * max_droplets + target_slot
    rows = []
    frame_payloads = []
    target_pred_trail = []
    target_true_trail = []
    attention_shape = None

    for step_index in range(args.length):
        history_phys = denormalize_features(history, normalization_stats, device)
        new_mask = batch["future_mask"][:, step_index, :]
        target_valid = bool(new_mask[0, target_slot].detach().cpu())
        if target_valid:
            attention_output = model(history, history_mask, return_attention=True, attention_layer=args.layer)
            pred_step_norm_raw = attention_output["prediction"]
            attention = attention_output["attention"][0].detach().cpu().numpy()
            attention_shape = tuple(attention_output["attention"].shape)
        else:
            pred_step_norm_raw = model(history, history_mask)
            attention = None
        pred_step_phys_raw = denormalize_targets(pred_step_norm_raw[:, None, :, :], normalization_stats, device)[:, 0, :, :]

        last_frame = history_phys[:, -1, :, :]
        new_frame_phys = last_frame.clone()
        new_frame_phys[:, :, feature_index["x"]] = last_frame[:, :, feature_index["x"]] + pred_step_phys_raw[:, :, 0]
        new_frame_phys[:, :, feature_index["y"]] = last_frame[:, :, feature_index["y"]] + pred_step_phys_raw[:, :, 1]
        new_frame_phys[:, :, feature_index["vx"]] = pred_step_phys_raw[:, :, 0]
        new_frame_phys[:, :, feature_index["vy"]] = pred_step_phys_raw[:, :, 1]

        previous_last_mask = history_mask[:, -1, :]
        true_step_features = true_future_features[:, step_index, :, :]
        true_step_features_finite = torch.isfinite(true_step_features).all(dim=-1)
        boundary_mask = new_mask & ~previous_last_mask & true_step_features_finite
        new_frame_phys[boundary_mask] = true_step_features[boundary_mask]
        refresh_observed_non_target_features(new_frame_phys, true_step_features, new_mask, feature_index)

        pred_position = new_frame_phys[0, :, [feature_index["x"], feature_index["y"]]].detach().cpu().numpy()
        true_position = true_step_features[0, :, [feature_index["x"], feature_index["y"]]].detach().cpu().numpy()
        future_mask = new_mask[0].detach().cpu().numpy().astype(bool)
        absolute_frame = int(dataset.frames[frame_start + dataset.T_history + step_index])
        if target_valid and np.isfinite(pred_position[target_slot]).all():
            target_pred_trail.append(tuple_float(pred_position[target_slot]))
        if target_valid and np.isfinite(true_position[target_slot]).all():
            target_true_trail.append(tuple_float(true_position[target_slot]))

        key_metadata = build_token_metadata(dataset, history_phys[0].detach().cpu().numpy(), history_mask[0].detach().cpu().numpy().astype(bool), droplet_ids, frame_start, step_index, max_droplets)
        top_droplets = []
        if target_valid and attention is not None:
            top_droplets = aggregate_top_attention_droplets(attention, query_token, key_metadata, true_position, future_mask, args, target_slot, int(droplet_ids[target_slot]))
        step_rows = []
        target_line_start = pred_position[target_slot] if args.line_start == "pred" else true_position[target_slot]
        for rank, droplet in enumerate(top_droplets, start=1):
            attended_point = np.asarray([droplet["current_x"], droplet["current_y"]], dtype=float)
            distance = float("nan")
            if np.isfinite(target_line_start).all() and np.isfinite(attended_point).all():
                distance = float(np.linalg.norm(target_line_start - attended_point))
            row = {
                "sample_rank": sample_rank,
                "window_index": int(np.where(dataset.start_frames == frame_start)[0][0]) if frame_start in set(dataset.start_frames.tolist()) else -1,
                "frame_start": frame_start,
                "absolute_frame": absolute_frame,
                "rollout_step": step_index + 1,
                "target_slot": target_slot,
                "target_track_id": int(droplet_ids[target_slot]),
                "attended_slot": droplet["slot"],
                "attended_track_id": droplet["track_id"],
                "aggregated_attention": droplet["aggregated_attention"],
                "normalized_attention": droplet["normalized_attention"],
                "current_x": droplet["current_x"],
                "current_y": droplet["current_y"],
                "distance_to_target": distance,
                "rank": rank,
            }
            rows.append(row)
            step_rows.append(row)
        frame_payloads.append(
            {
                "absolute_frame": absolute_frame,
                "rollout_step": step_index + 1,
                "pred_position": pred_position,
                "true_position": true_position,
                "future_mask": future_mask,
                "target_valid": target_valid,
                "target_pred_trail": list(target_pred_trail),
                "target_true_trail": list(target_true_trail),
                "top_rows": step_rows,
                "target_track_id": int(droplet_ids[target_slot]),
                "target_slot": target_slot,
            }
        )

        new_frame_norm = normalize_features(new_frame_phys, normalization_stats, device)
        new_frame_norm = torch.where(new_mask[:, :, None], new_frame_norm, torch.zeros_like(new_frame_norm))
        history = torch.cat([history[:, 1:, :, :], new_frame_norm[:, None, :, :]], dim=1)
        history_mask = torch.cat([history_mask[:, 1:, :], new_mask[:, None, :]], dim=1)
    return rows, {"frame_payloads": frame_payloads, "droplet_ids": droplet_ids}, attention_shape


def get_true_future_features(batch, dataset, device, horizon: int) -> torch.Tensor:
    droplet_ids = batch["droplet_ids"].detach().cpu().numpy()
    frame_starts = batch["frame_start"].detach().cpu().numpy()
    track_id_to_index = {int(track_id): index for index, track_id in enumerate(dataset.track_ids)}
    B, M = droplet_ids.shape
    true_features = np.full((B, horizon, M, len(dataset.feature_names)), np.nan, dtype=np.float32)
    for batch_index in range(B):
        start = int(frame_starts[batch_index]) + dataset.T_history
        end = start + horizon
        for slot_index in range(M):
            track_id = int(droplet_ids[batch_index, slot_index])
            if track_id < 0:
                continue
            droplet_index = track_id_to_index.get(track_id)
            if droplet_index is not None:
                true_features[batch_index, :, slot_index, :] = dataset.Z[droplet_index, start:end, :]
    return torch.as_tensor(true_features, dtype=torch.float32, device=device)


def build_token_metadata(dataset, history_phys, history_mask, droplet_ids, frame_start, step_index, max_droplets):
    metadata = []
    x_index = dataset.feature_indices["x"]
    y_index = dataset.feature_indices["y"]
    for time_index in range(history_phys.shape[0]):
        absolute_frame = int(dataset.frames[frame_start + step_index + time_index])
        for slot in range(max_droplets):
            x = float_or_nan(history_phys[time_index, slot, x_index])
            y = float_or_nan(history_phys[time_index, slot, y_index])
            metadata.append(
                {
                    "token": time_index * max_droplets + slot,
                    "slot": slot,
                    "track_id": int(droplet_ids[slot]),
                    "absolute_frame": absolute_frame,
                    "x": x,
                    "y": y,
                    "valid": bool(history_mask[time_index, slot] and np.isfinite([x, y]).all()),
                }
            )
    return metadata


def aggregate_top_attention_droplets(attention, query_token, key_metadata, current_positions, current_mask, args, target_slot, target_track_id):
    if args.head == "mean":
        scores = attention[:, query_token, :].mean(axis=0)
    else:
        head_index = int(args.head)
        scores = attention[head_index, query_token, :]
    grouped = {}
    for token in key_metadata:
        if not token["valid"]:
            continue
        if args.exclude_self and (token["slot"] == target_slot or token["track_id"] == target_track_id):
            continue
        key = ("track", token["track_id"]) if token["track_id"] >= 0 else ("slot", token["slot"])
        grouped.setdefault(key, {"slot": token["slot"], "track_id": token["track_id"], "aggregated_attention": 0.0})
        grouped[key]["aggregated_attention"] += float(scores[token["token"]])
    visible = []
    for item in grouped.values():
        slot = int(item["slot"])
        if slot >= len(current_mask) or not current_mask[slot]:
            continue
        xy = current_positions[slot]
        if not np.isfinite(xy).all():
            continue
        item = dict(item)
        item["current_x"] = float(xy[0])
        item["current_y"] = float(xy[1])
        visible.append(item)
    total = sum(item["aggregated_attention"] for item in visible)
    for item in visible:
        item["normalized_attention"] = item["aggregated_attention"] / total if total > 0 else 0.0
    visible.sort(key=lambda item: item["normalized_attention"], reverse=True)
    return visible[: args.top_k]


def write_attention_video(video_path: Path, output_path: Path, args, render_payload) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to create video: {output_path}")
    try:
        for payload in render_payload["frame_payloads"]:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(payload["absolute_frame"]))
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Unable to read video frame {payload['absolute_frame']}")
            annotate_frame(frame, args, payload)
            writer.write(frame)
    finally:
        writer.release()
        capture.release()


def annotate_frame(frame, args, payload) -> None:
    pred = payload["pred_position"]
    true = payload["true_position"]
    mask = payload["future_mask"]
    target_slot = payload["target_slot"]
    line_start = pred[target_slot] if args.line_start == "pred" else true[target_slot]
    for slot in np.flatnonzero(mask):
        if np.isfinite(true[slot]).all():
            cv2.circle(frame, point(true[slot]), 4, (30, 30, 30), -1)
        if np.isfinite(pred[slot]).all():
            cv2.drawMarker(frame, point(pred[slot]), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 9, 1)
    if payload["target_valid"]:
        draw_polyline(frame, payload["target_true_trail"], (40, 40, 40), 1)
        draw_polyline(frame, payload["target_pred_trail"], (0, 0, 255), 2)
        if np.isfinite(true[target_slot]).all():
            cv2.circle(frame, point(true[target_slot]), 8, (0, 255, 255), 2)
        if np.isfinite(pred[target_slot]).all():
            cv2.circle(frame, point(pred[target_slot]), 7, (0, 0, 255), 2)
    max_weight = max([row["normalized_attention"] for row in payload["top_rows"]] or [1.0])
    if payload["target_valid"] and np.isfinite(line_start).all():
        for row in reversed(payload["top_rows"]):
            end = np.asarray([row["current_x"], row["current_y"]], dtype=float)
            if not np.isfinite(end).all():
                continue
            relative = row["normalized_attention"] / max(max_weight, 1e-12)
            brightness = int(round(80 + 175 * relative))
            radius = 4 + int(round(10 * relative))
            cv2.line(frame, point(line_start), point(end), (0, brightness, brightness), args.min_line_thickness, cv2.LINE_AA)
            cv2.circle(frame, point(end), radius, (brightness, brightness, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, point(end), radius + 2, (255, 255, 255), 1, cv2.LINE_AA)
            if not args.no_labels:
                cv2.putText(frame, f"t{row['attended_track_id']} a={row['normalized_attention']:.2f}", (point(end)[0] + radius + 4, point(end)[1] - radius - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    top_text = ", ".join(f"{row['attended_track_id']}:{row['normalized_attention']:.2f}" for row in payload["top_rows"][:5])
    lines = [
        f"step {payload['rollout_step']} frame {payload['absolute_frame']}",
        f"target slot {target_slot} track {payload['target_track_id']}",
        f"attention layer {args.layer} head {args.head}",
        f"top droplets: {top_text}",
    ]
    if not payload["target_valid"]:
        lines.append("target exited FOV")
    draw_text_box(frame, lines, (10, 22))


def assert_no_nan_payload(render_payload) -> None:
    for payload in render_payload["frame_payloads"]:
        if np.isnan(payload["pred_position"][payload["future_mask"]]).any():
            raise ValueError("NaN detected in predicted positions for visible droplets.")


def write_csv(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def draw_text_box(frame, lines, origin) -> None:
    x, y = origin
    line_height = 18
    width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0] for line in lines) + 12
    height = line_height * len(lines) + 8
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 6, y - 16), (x - 6 + width, y - 16 + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + index * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)


def draw_polyline(frame, points, color, thickness) -> None:
    clean = [point(np.asarray(item, dtype=float)) for item in points if np.isfinite(item).all()]
    if len(clean) >= 2:
        cv2.polylines(frame, [np.asarray(clean, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)


def point(values) -> tuple[int, int]:
    return int(round(float(values[0]))), int(round(float(values[1])))


def tuple_float(values) -> tuple[float, float]:
    return float(values[0]), float(values[1])


def float_or_nan(value) -> float:
    value = float(value)
    return value if np.isfinite(value) else float("nan")


if __name__ == "__main__":
    main()
