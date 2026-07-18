from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .ellipse import rasterize_bbox_ellipse

PHYSICAL_LABELS = {
    1: "inlet",
    2: "outlet",
    3: "left",
    4: "right",
    5: "upper_junction",
    6: "lower_junction",
}
REGION_COLUMNS = list(PHYSICAL_LABELS.values())
RAW_COLUMNS = [f"w_{name}_raw" for name in REGION_COLUMNS]
NORM_COLUMNS = [f"w_{name}" for name in REGION_COLUMNS]
DOMINANT_REGION_NAMES = {
    "inlet": "inlet",
    "outlet": "outlet",
    "left": "left_branch",
    "right": "right_branch",
    "upper_junction": "upper_junction",
    "lower_junction": "lower_junction",
}
REQUIRED_TRACK_COLUMNS = {
    "frame",
    "track_id",
    "centroid_x",
    "centroid_y",
    "bbox_w",
    "bbox_h",
}
OCCUPANCY_EPSILON = 1e-12


def validate_label_map(region_labels: np.ndarray) -> np.ndarray:
    """Validate and return an integer region-label map."""
    labels = np.asarray(region_labels)
    if labels.ndim != 2:
        raise ValueError(f"Region label map must be 2D, got shape {labels.shape}")
    valid_ids = set(PHYSICAL_LABELS).union({0})
    unknown = set(int(value) for value in np.unique(labels)).difference(valid_ids)
    if unknown:
        raise ValueError(f"Region label map contains unknown IDs: {sorted(unknown)}")
    return labels


def calculate_ellipse_occupancy(
    region_labels: np.ndarray,
    center_x: float,
    center_y: float,
    bbox_width: float,
    bbox_height: float,
    minimum_physical_coverage: float = 0.95,
) -> dict[str, Any]:
    """Calculate raw and normalized physical-region occupancy for one ellipse."""
    labels = validate_label_map(region_labels)
    if not 0 <= minimum_physical_coverage <= 1:
        raise ValueError("minimum_physical_coverage must be in [0, 1]")
    raster = rasterize_bbox_ellipse(center_x, center_y, bbox_width, bbox_height, labels.shape)
    local_labels = labels[raster.y0 : raster.y1, raster.x0 : raster.x1][raster.mask]
    total = int(local_labels.size)
    if total == 0:
        raise ValueError("Ellipse contains zero raster pixels")

    result: dict[str, Any] = {
        "ellipse_pixel_count": total,
        "image_boundary_clipped": bool(raster.image_boundary_clipped),
    }
    raw_values = []
    for label_id, name in PHYSICAL_LABELS.items():
        value = float(np.count_nonzero(local_labels == label_id) / total)
        if value < -1e-12 or value > 1 + 1e-12:
            raise ValueError(f"Raw occupancy for {name} is outside [0, 1]: {value}")
        result[f"w_{name}_raw"] = value
        raw_values.append(value)
    coverage = float(sum(raw_values))
    if coverage < -1e-12 or coverage > 1 + 1e-12:
        raise ValueError(f"physical_coverage_raw is outside [0, 1]: {coverage}")
    coverage = min(1.0, max(0.0, coverage))
    result["physical_coverage_raw"] = coverage
    computable = coverage > OCCUPANCY_EPSILON
    result["low_physical_coverage"] = bool(coverage < minimum_physical_coverage)
    result["occupancy_computable"] = bool(computable)
    if computable:
        normalized = [value / coverage for value in raw_values]
        normalized_sum = float(sum(normalized))
        if abs(normalized_sum - 1.0) > 1e-10:
            raise ValueError(f"Computable normalized occupancy does not sum to 1: {normalized_sum}")
        for name, value in zip(REGION_COLUMNS, normalized):
            result[f"w_{name}"] = float(value)
        result["normalized_occupancy_sum"] = normalized_sum
        dominant_idx = int(np.argmax(normalized))
        result["dominant_region"] = DOMINANT_REGION_NAMES[REGION_COLUMNS[dominant_idx]]
        result["number_active_regions"] = int(np.count_nonzero(np.asarray(normalized) > 0.01))
    else:
        for name in REGION_COLUMNS:
            result[f"w_{name}"] = np.nan
        result["normalized_occupancy_sum"] = np.nan
        result["dominant_region"] = ""
        result["number_active_regions"] = 0
    return result


def _validate_tracking_columns(tracks: pd.DataFrame) -> None:
    missing = REQUIRED_TRACK_COLUMNS.difference(tracks.columns)
    if missing:
        raise ValueError(f"Tracked features are missing required columns: {sorted(missing)}")
    numeric = ["centroid_x", "centroid_y", "bbox_w", "bbox_h"]
    values = tracks[numeric].to_numpy(float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Tracked centroid and bounding-box columns must be finite")
    if (tracks["bbox_w"] <= 0).any() or (tracks["bbox_h"] <= 0).any():
        raise ValueError("Tracked bounding-box width and height must be positive")


def calculate_dataset_occupancy(
    tracks: pd.DataFrame,
    region_labels: np.ndarray,
    um_per_px: float,
    minimum_physical_coverage: float = 0.95,
) -> pd.DataFrame:
    """Calculate occupancy rows for all tracked droplet-frame samples."""
    if um_per_px <= 0 or not np.isfinite(um_per_px):
        raise ValueError("Pixel scale um_per_px must be positive and finite")
    labels = validate_label_map(region_labels)
    _validate_tracking_columns(tracks)
    n_rows = len(tracks)
    frames = tracks["frame"].to_numpy(dtype=np.int64)
    track_ids = tracks["track_id"].to_numpy(dtype=np.int64)
    center_x = tracks["centroid_x"].to_numpy(dtype=float)
    center_y = tracks["centroid_y"].to_numpy(dtype=float)
    bbox_w = tracks["bbox_w"].to_numpy(dtype=float)
    bbox_h = tracks["bbox_h"].to_numpy(dtype=float)

    raw = np.zeros((n_rows, 6), dtype=float)
    norm = np.full((n_rows, 6), np.nan, dtype=float)
    coverage = np.zeros(n_rows, dtype=float)
    valid = np.zeros(n_rows, dtype=bool)
    normalized_sum = np.full(n_rows, np.nan, dtype=float)
    ellipse_pixel_count = np.zeros(n_rows, dtype=np.int32)
    clipped = np.zeros(n_rows, dtype=bool)
    dominant = np.full(n_rows, "", dtype=object)
    active = np.zeros(n_rows, dtype=np.int16)
    image_shape = labels.shape

    for idx in range(n_rows):
        counts, total, was_clipped = _ellipse_label_counts_fast(
            labels, center_x[idx], center_y[idx], bbox_w[idx], bbox_h[idx], image_shape
        )
        ellipse_pixel_count[idx] = total
        clipped[idx] = was_clipped
        raw_values = counts[1:7] / total
        raw[idx] = raw_values
        sample_coverage = float(np.sum(raw_values))
        if sample_coverage < -1e-12 or sample_coverage > 1 + 1e-12:
            raise ValueError(f"physical_coverage_raw is outside [0, 1]: {sample_coverage}")
        sample_coverage = min(1.0, max(0.0, sample_coverage))
        coverage[idx] = sample_coverage
        if sample_coverage > OCCUPANCY_EPSILON:
            valid[idx] = True
            norm_values = raw_values / sample_coverage
            norm[idx] = norm_values
            sample_sum = float(np.sum(norm_values))
            if abs(sample_sum - 1.0) > 1e-10:
                raise ValueError(f"Computable normalized occupancy does not sum to 1: {sample_sum}")
            normalized_sum[idx] = sample_sum
            dominant[idx] = DOMINANT_REGION_NAMES[REGION_COLUMNS[int(np.argmax(norm_values))]]
            active[idx] = int(np.count_nonzero(norm_values > 0.01))

    data: dict[str, Any] = {
        "frame": frames,
        "track_id": track_ids,
        "center_x_um": center_x * um_per_px,
        "center_y_um": center_y * um_per_px,
        "bbox_width_um": bbox_w * um_per_px,
        "bbox_height_um": bbox_h * um_per_px,
        "ellipse_area_um2": np.pi * (bbox_w * um_per_px / 2.0) * (bbox_h * um_per_px / 2.0),
        "ellipse_pixel_count": ellipse_pixel_count,
    }
    for col_idx, name in enumerate(REGION_COLUMNS):
        data[f"w_{name}_raw"] = raw[:, col_idx]
    data["physical_coverage_raw"] = coverage
    data["low_physical_coverage"] = coverage < minimum_physical_coverage
    data["occupancy_computable"] = valid
    for col_idx, name in enumerate(REGION_COLUMNS):
        data[f"w_{name}"] = norm[:, col_idx]
    data["normalized_occupancy_sum"] = normalized_sum
    data["image_boundary_clipped"] = clipped
    data["dominant_region"] = dominant
    data["number_active_regions"] = active
    return pd.DataFrame(data)


def _ellipse_label_counts_fast(
    labels: np.ndarray,
    center_x: float,
    center_y: float,
    bbox_width: float,
    bbox_height: float,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, int, bool]:
    rx = bbox_width / 2.0
    ry = bbox_height / 2.0
    raw_x0 = int(np.floor(center_x - rx))
    raw_x1 = int(np.ceil(center_x + rx)) + 1
    raw_y0 = int(np.floor(center_y - ry))
    raw_y1 = int(np.ceil(center_y + ry)) + 1
    height, width = image_shape
    x0 = max(0, raw_x0)
    y0 = max(0, raw_y0)
    x1 = min(width, raw_x1)
    y1 = min(height, raw_y1)
    clipped = x0 != raw_x0 or y0 != raw_y0 or x1 != raw_x1 or y1 != raw_y1
    if x0 >= x1 or y0 >= y1:
        raise ValueError("Ellipse bounding window does not intersect the image")
    yy, xx = np.ogrid[y0:y1, x0:x1]
    ellipse = ((xx - center_x) / rx) ** 2 + ((yy - center_y) / ry) ** 2 <= 1.0
    total = int(np.count_nonzero(ellipse))
    if total == 0:
        raise ValueError("Ellipse contains zero raster pixels")
    local_labels = labels[y0:y1, x0:x1][ellipse]
    counts = np.bincount(local_labels.astype(np.int16), minlength=7)
    return counts, total, clipped


def summarize_occupancy(
    occupancy: pd.DataFrame,
    experiment_id: str,
    device_id: str,
    tracked_feature_path: str,
    region_label_path: str,
    um_per_px: float,
    minimum_physical_coverage: float,
) -> dict[str, Any]:
    """Build a JSON-serializable summary for an occupancy table."""
    total = int(len(occupancy))
    computable = occupancy["occupancy_computable"].astype(bool)
    computable_count = int(computable.sum())
    noncomputable_count = total - computable_count
    low_coverage = occupancy["low_physical_coverage"].astype(bool)
    low_coverage_count = int(low_coverage.sum())
    coverage = occupancy["physical_coverage_raw"].to_numpy(float)
    percentiles = {str(p): float(np.percentile(coverage, p)) for p in [1, 5, 25, 50, 75, 95, 99]}
    if computable_count:
        norm_error = float(np.nanmax(np.abs(occupancy.loc[computable, "normalized_occupancy_sum"].to_numpy(float) - 1.0)))
        dominant_counts = {
            key: int(value)
            for key, value in occupancy.loc[computable, "dominant_region"].value_counts().sort_index().to_dict().items()
        }
        active_counts = {
            str(int(key)): int(value)
            for key, value in occupancy.loc[computable, "number_active_regions"].value_counts().sort_index().to_dict().items()
        }
    else:
        norm_error = float("nan")
        dominant_counts = {}
        active_counts = {}
    summary = {
        "experiment_id": experiment_id,
        "device_id": device_id,
        "input_tracked_feature_path": tracked_feature_path,
        "region_label_path": region_label_path,
        "pixel_scale_um": float(um_per_px),
        "configured_low_coverage_threshold": float(minimum_physical_coverage),
        "total_droplet_frame_samples": total,
        "occupancy_computable_count": computable_count,
        "occupancy_noncomputable_count": noncomputable_count,
        "occupancy_computable_fraction": computable_count / total if total else 0.0,
        "low_physical_coverage_count": low_coverage_count,
        "low_physical_coverage_fraction": low_coverage_count / total if total else 0.0,
        "physical_coverage_minimum": float(np.min(coverage)) if total else float("nan"),
        "physical_coverage_maximum": float(np.max(coverage)) if total else float("nan"),
        "physical_coverage_mean": float(np.mean(coverage)) if total else float("nan"),
        "physical_coverage_median": float(np.median(coverage)) if total else float("nan"),
        "physical_coverage_percentiles": percentiles,
        "coverage_ge_0.95_count": int(np.count_nonzero(coverage >= 0.95)),
        "coverage_ge_0.95_fraction": float(np.count_nonzero(coverage >= 0.95) / total) if total else 0.0,
        "coverage_ge_0.98_count": int(np.count_nonzero(coverage >= 0.98)),
        "coverage_ge_0.98_fraction": float(np.count_nonzero(coverage >= 0.98) / total) if total else 0.0,
        "coverage_ge_0.99_count": int(np.count_nonzero(coverage >= 0.99)),
        "coverage_ge_0.99_fraction": float(np.count_nonzero(coverage >= 0.99) / total) if total else 0.0,
        "maximum_normalized_sum_error": norm_error,
        "dominant_region_counts": dominant_counts,
        "number_active_regions_counts": active_counts,
        "image_boundary_clipped_count": int(occupancy["image_boundary_clipped"].sum()),
        "image_boundary_clipped_fraction": float(occupancy["image_boundary_clipped"].sum() / total) if total else 0.0,
    }
    return summary
