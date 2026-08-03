from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib-cache").resolve()))

import numpy as np
import pandas as pd


DEFAULT_DATASET = Path("outputs/processed/2/canonical_dataset_v2/canonical_dataset_v2.npz")
DEFAULT_REGION_LABELS = Path("data/geometry/asymmetric_loop_h100/region_labels.npy")
DEFAULT_OUTPUT = Path("outputs/evaluation/bbox_nearest_neighbor_baseline")
REGION_NAMES = {
    0: "outside",
    1: "inlet channel",
    2: "outlet channel",
    3: "left branch",
    4: "right branch",
    5: "inlet junction",
    6: "outlet junction",
}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rng = np.random.default_rng(int(args.random_seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset)
    region_labels = np.load(args.region_labels)
    feature_names = [str(name) for name in dataset["feature_names"]]
    idx = {name: index for index, name in enumerate(feature_names)}
    samples = build_samples(
        dataset,
        idx,
        region_labels,
        stride=int(args.stride),
        t_history=int(args.t_history),
        t_future=int(args.t_future),
    )
    train = subsample(samples["train"], int(args.max_train_rows), rng)
    val = subsample(samples["val"], int(args.max_val_rows), rng)
    if len(train["features"]) == 0 or len(val["features"]) == 0:
        raise RuntimeError("Need non-empty train and validation samples.")

    scaler = fit_standardizer(train["features"])
    train_features = transform(train["features"], scaler)
    val_features = transform(val["features"], scaler)
    predictions = predict_knn(
        train_features=train_features,
        train_targets=train["targets"],
        val_features=val_features,
        k=int(args.k),
        chunk_size=int(args.chunk_size),
    )

    mean_prediction = np.repeat(train["targets"].mean(axis=0, keepdims=True), len(val["targets"]), axis=0)
    median_prediction = np.repeat(np.median(train["targets"], axis=0, keepdims=True), len(val["targets"]), axis=0)
    metrics = {
        "dataset": str(args.dataset),
        "region_labels": str(args.region_labels),
        "feature_names": ["x", "y", "speed"],
        "target_names": ["bbox_w", "bbox_h"],
        "train_rows": int(len(train["features"])),
        "val_rows": int(len(val["features"])),
        "k": int(args.k),
        "standardization": scaler,
        "target_distribution_train": describe_targets(train["targets"]),
        "target_distribution_val": describe_targets(val["targets"]),
        "knn": regression_metrics(predictions, val["targets"]),
        "mean_baseline": regression_metrics(mean_prediction, val["targets"]),
        "median_baseline": regression_metrics(median_prediction, val["targets"]),
        "knn_by_region": metrics_by_region(predictions, val["targets"], val["regions"]),
    }
    save_json(output_dir / "metrics.json", metrics)
    save_predictions_csv(output_dir / "predictions_sample.csv", val, predictions, max_rows=int(args.prediction_rows))
    save_scatter_plot(output_dir / "bbox_w_pred_vs_true.png", val["targets"][:, 0], predictions[:, 0], "bbox_w")
    save_scatter_plot(output_dir / "bbox_h_pred_vs_true.png", val["targets"][:, 1], predictions[:, 1], "bbox_h")
    save_region_scatter_plot(output_dir / "bbox_w_pred_vs_true_by_region.png", val["targets"][:, 0], predictions[:, 0], val["regions"], "bbox_w")
    save_region_scatter_plot(output_dir / "bbox_h_pred_vs_true_by_region.png", val["targets"][:, 1], predictions[:, 1], val["regions"], "bbox_h")
    print_summary(metrics, output_dir)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nearest-neighbor bbox baseline from position and speed.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--region-labels", type=Path, default=DEFAULT_REGION_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--t-history", type=int, default=1)
    parser.add_argument("--t-future", type=int, default=50)
    parser.add_argument("--k", type=int, default=25)
    parser.add_argument("--max-train-rows", type=int, default=120000)
    parser.add_argument("--max-val-rows", type=int, default=30000)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--prediction-rows", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=123)
    return parser.parse_args(argv)


def load_dataset(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key].copy() for key in loaded.files}


def build_samples(
    dataset: dict[str, Any],
    idx: dict[str, int],
    region_labels: np.ndarray,
    stride: int,
    t_history: int,
    t_future: int,
) -> dict[str, dict[str, np.ndarray]]:
    starts = np.arange(0, dataset["Z"].shape[1] - (int(t_history) + int(t_future)) + 1, int(stride), dtype=np.int64)
    train_end = int(0.70 * len(starts))
    val_end = int(0.85 * len(starts))
    return {
        "train": rows_from_frame_indices(dataset, idx, region_labels, starts[:train_end]),
        "val": rows_from_frame_indices(dataset, idx, region_labels, starts[train_end:val_end]),
    }


def rows_from_frame_indices(
    dataset: dict[str, Any],
    idx: dict[str, int],
    region_labels: np.ndarray,
    frame_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    mask = dataset["mask"][:, frame_indices].astype(bool)
    track_local, frame_local = np.nonzero(mask)
    frame_indices_for_rows = frame_indices[frame_local]
    rows = dataset["Z"][track_local, frame_indices_for_rows, :].astype(np.float32)
    vx = rows[:, idx["vx"]]
    vy = rows[:, idx["vy"]]
    features = np.column_stack([rows[:, idx["x"]], rows[:, idx["y"]], np.hypot(vx, vy)]).astype(np.float32)
    targets = rows[:, [idx["bbox_w"], idx["bbox_h"]]].astype(np.float32)
    finite = np.isfinite(features).all(axis=1) & np.isfinite(targets).all(axis=1) & (targets[:, 0] > 0.0) & (targets[:, 1] > 0.0)
    return {
        "features": features[finite],
        "targets": targets[finite],
        "track_ids": dataset["track_ids"][track_local[finite]].astype(np.int64),
        "frames": dataset["frames"][frame_indices_for_rows[finite]].astype(np.int64),
        "regions": regions_for_xy(features[finite, 0], features[finite, 1], region_labels),
    }


def subsample(sample: dict[str, np.ndarray], max_rows: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    if max_rows <= 0 or len(sample["features"]) <= max_rows:
        return sample
    selected = np.sort(rng.choice(len(sample["features"]), size=int(max_rows), replace=False))
    return {key: value[selected] for key, value in sample.items()}


def fit_standardizer(features: np.ndarray) -> dict[str, list[float]]:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std <= 1.0e-12, 1.0, std)
    return {"mean": mean.tolist(), "std": std.tolist()}


def transform(features: np.ndarray, scaler: dict[str, list[float]]) -> np.ndarray:
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    return (features - mean.reshape(1, -1)) / std.reshape(1, -1)


def predict_knn(
    *,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    val_features: np.ndarray,
    k: int,
    chunk_size: int,
) -> np.ndarray:
    k = min(max(int(k), 1), len(train_features))
    predictions = np.empty((len(val_features), train_targets.shape[1]), dtype=np.float32)
    train_sq = np.sum(train_features**2, axis=1).reshape(1, -1)
    for start in range(0, len(val_features), int(chunk_size)):
        stop = min(start + int(chunk_size), len(val_features))
        chunk = val_features[start:stop]
        distances = np.sum(chunk**2, axis=1, keepdims=True) + train_sq - 2.0 * chunk @ train_features.T
        nearest = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        predictions[start:stop] = train_targets[nearest].mean(axis=1)
    return predictions


def regression_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    error = prediction - truth
    abs_error = np.abs(error)
    squared = error**2
    return {
        "rmse_bbox_w": float(np.sqrt(np.mean(squared[:, 0]))),
        "rmse_bbox_h": float(np.sqrt(np.mean(squared[:, 1]))),
        "rmse_joint": float(np.sqrt(np.mean(np.sum(squared, axis=1)))),
        "mae_bbox_w": float(np.mean(abs_error[:, 0])),
        "mae_bbox_h": float(np.mean(abs_error[:, 1])),
        "median_abs_error_bbox_w": float(np.median(abs_error[:, 0])),
        "median_abs_error_bbox_h": float(np.median(abs_error[:, 1])),
        "p95_abs_error_bbox_w": float(np.percentile(abs_error[:, 0], 95)),
        "p95_abs_error_bbox_h": float(np.percentile(abs_error[:, 1], 95)),
        "bias_bbox_w": float(np.mean(error[:, 0])),
        "bias_bbox_h": float(np.mean(error[:, 1])),
    }


def metrics_by_region(prediction: np.ndarray, truth: np.ndarray, regions: np.ndarray) -> dict[str, Any]:
    result = {}
    region_strings = regions.astype(str)
    for region in sorted(set(region_strings.tolist())):
        mask = region_strings == region
        item = regression_metrics(prediction[mask], truth[mask])
        item["count"] = int(np.count_nonzero(mask))
        result[region] = item
    return result


def describe_targets(targets: np.ndarray) -> dict[str, Any]:
    return {
        "bbox_w": describe_array(targets[:, 0]),
        "bbox_h": describe_array(targets[:, 1]),
    }


def describe_array(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }


def save_predictions_csv(path: Path, val: dict[str, np.ndarray], prediction: np.ndarray, max_rows: int) -> None:
    n = min(max(int(max_rows), 0), len(prediction))
    table = pd.DataFrame(
        {
            "track_id": val["track_ids"][:n],
            "frame": val["frames"][:n],
            "x": val["features"][:n, 0],
            "y": val["features"][:n, 1],
            "speed": val["features"][:n, 2],
            "region": val["regions"][:n],
            "true_bbox_w": val["targets"][:n, 0],
            "true_bbox_h": val["targets"][:n, 1],
            "pred_bbox_w": prediction[:n, 0],
            "pred_bbox_h": prediction[:n, 1],
        }
    )
    table.to_csv(path, index=False)


def regions_for_xy(x: np.ndarray, y: np.ndarray, region_labels: np.ndarray) -> np.ndarray:
    labels = []
    height, width = region_labels.shape
    for x_value, y_value in zip(x, y):
        col = int(round(float(x_value)))
        row = int(round(float(y_value)))
        if row < 0 or row >= height or col < 0 or col >= width:
            labels.append("outside")
        else:
            labels.append(REGION_NAMES.get(int(region_labels[row, col]), "unknown"))
    return np.asarray(labels, dtype=object)


def save_scatter_plot(path: Path, truth: np.ndarray, prediction: np.ndarray, label: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    ax.scatter(truth, prediction, s=3, alpha=0.15)
    low = float(min(np.min(truth), np.min(prediction)))
    high = float(max(np.max(truth), np.max(prediction)))
    ax.plot([low, high], [low, high], color="black", linewidth=1)
    ax.set_xlabel(f"true {label}")
    ax.set_ylabel(f"predicted {label}")
    ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_region_scatter_plot(path: Path, truth: np.ndarray, prediction: np.ndarray, regions: np.ndarray, label: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return
    region_strings = regions.astype(str)
    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    low = float(min(np.min(truth), np.min(prediction)))
    high = float(max(np.max(truth), np.max(prediction)))
    for region in sorted(set(region_strings.tolist())):
        mask = region_strings == region
        ax.scatter(truth[mask], prediction[mask], s=5, alpha=0.35, label=f"{region} (n={int(mask.sum())})")
    ax.plot([low, high], [low, high], color="black", linewidth=1)
    ax.set_xlabel(f"true {label}")
    ax.set_ylabel(f"predicted {label}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, markerscale=2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_summary(metrics: dict[str, Any], output_dir: Path) -> None:
    print("Nearest-neighbor bbox baseline complete")
    print(f"  output: {output_dir}")
    print(f"  train_rows={metrics['train_rows']} val_rows={metrics['val_rows']} k={metrics['k']}")
    for name in ("knn", "mean_baseline", "median_baseline"):
        item = metrics[name]
        print(
            f"  {name}: "
            f"rmse_w={item['rmse_bbox_w']:.4f} rmse_h={item['rmse_bbox_h']:.4f} "
            f"mae_w={item['mae_bbox_w']:.4f} mae_h={item['mae_bbox_h']:.4f}"
        )


if __name__ == "__main__":
    main()
