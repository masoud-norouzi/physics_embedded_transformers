import numpy as np
import pandas as pd
import pytest

from scripts.compute_droplet_occupancy import _save_diagnostics
from src.occupancy.calculator import (
    NORM_COLUMNS,
    calculate_dataset_occupancy,
    calculate_ellipse_occupancy,
    summarize_occupancy,
    validate_label_map,
)


def _label_first_ellipse_pixels(
    shape: tuple[int, int],
    center_x: float,
    center_y: float,
    bbox_width: float,
    bbox_height: float,
    n_physical: int,
    label: int = 1,
) -> np.ndarray:
    labels = np.zeros(shape, dtype=np.uint8)
    yy, xx = np.indices(shape, dtype=float)
    ellipse = ((xx - center_x) / (bbox_width / 2)) ** 2 + ((yy - center_y) / (bbox_height / 2)) ** 2 <= 1
    coords = np.argwhere(ellipse)
    labels[coords[:n_physical, 0], coords[:n_physical, 1]] = label
    return labels


def test_ellipse_inside_one_region_is_approximately_one() -> None:
    labels = np.ones((20, 20), dtype=np.uint8)
    result = calculate_ellipse_occupancy(labels, 10, 10, 6, 6)
    assert result["occupancy_computable"]
    assert not result["low_physical_coverage"]
    assert result["w_inlet"] == pytest.approx(1.0)
    assert result["physical_coverage_raw"] == pytest.approx(1.0)


def test_ellipse_crossing_two_regions_splits_fractionally() -> None:
    labels = np.zeros((20, 20), dtype=np.uint8)
    labels[:, :10] = 1
    labels[:, 10:] = 2
    result = calculate_ellipse_occupancy(labels, 9.5, 10, 8, 8)
    assert result["occupancy_computable"]
    assert result["w_inlet"] == pytest.approx(result["w_outlet"], abs=0.08)
    assert result["w_inlet"] + result["w_outlet"] == pytest.approx(1.0)


def test_raw_occupancies_sum_to_physical_coverage() -> None:
    labels = np.zeros((20, 20), dtype=np.uint8)
    labels[5:15, 5:10] = 1
    labels[5:15, 10:15] = 2
    result = calculate_ellipse_occupancy(labels, 10, 10, 10, 10, minimum_physical_coverage=0.1)
    raw_sum = sum(
        result[f"w_{name}_raw"]
        for name in ["inlet", "outlet", "left", "right", "upper_junction", "lower_junction"]
    )
    assert raw_sum == pytest.approx(result["physical_coverage_raw"])


def test_above_threshold_normalizes_to_one() -> None:
    labels = np.ones((20, 20), dtype=np.uint8)
    result = calculate_ellipse_occupancy(labels, 10, 10, 6, 6, minimum_physical_coverage=0.95)
    assert result["occupancy_computable"]
    assert result["normalized_occupancy_sum"] == pytest.approx(1.0)


def test_coverage_exactly_at_threshold_is_not_low() -> None:
    labels = _label_first_ellipse_pixels((80, 80), 20, 20, 2.5, 58.0, 114)
    result = calculate_ellipse_occupancy(labels, 20, 20, 2.5, 58.0, minimum_physical_coverage=0.95)
    assert result["physical_coverage_raw"] == pytest.approx(0.95)
    assert not result["low_physical_coverage"]
    assert result["occupancy_computable"]


def test_below_threshold_is_low_but_normalized() -> None:
    labels = np.zeros((20, 20), dtype=np.uint8)
    labels[8:12, 8:12] = 1
    result = calculate_ellipse_occupancy(labels, 10, 10, 12, 12, minimum_physical_coverage=0.95)
    assert result["low_physical_coverage"]
    assert result["occupancy_computable"]
    assert all(np.isfinite(result[column]) for column in NORM_COLUMNS)
    assert result["normalized_occupancy_sum"] == pytest.approx(1.0)


def test_zero_coverage_noncomputable_and_normalized_nan() -> None:
    labels = np.zeros((20, 20), dtype=np.uint8)
    result = calculate_ellipse_occupancy(labels, 10, 10, 6, 6, minimum_physical_coverage=0.95)
    assert result["low_physical_coverage"]
    assert not result["occupancy_computable"]
    assert all(np.isnan(result[column]) for column in NORM_COLUMNS)
    assert np.isnan(result["normalized_occupancy_sum"])


def test_no_unassigned_physical_state_column_and_no_occupancy_valid() -> None:
    labels = np.ones((20, 20), dtype=np.uint8)
    result = calculate_ellipse_occupancy(labels, 10, 10, 6, 6)
    assert "w_unassigned" not in result
    assert "w_unassigned_raw" not in result
    assert "occupancy_valid" not in result


def test_invalid_label_ids_are_rejected() -> None:
    labels = np.full((5, 5), 9, dtype=np.uint8)
    with pytest.raises(ValueError, match="unknown IDs"):
        validate_label_map(labels)


def test_zero_or_negative_bbox_dimensions_rejected() -> None:
    labels = np.ones((20, 20), dtype=np.uint8)
    with pytest.raises(ValueError, match="positive"):
        calculate_ellipse_occupancy(labels, 10, 10, 0, 5)


def test_local_window_rasterization_matches_full_frame_reference() -> None:
    labels = np.zeros((30, 30), dtype=np.uint8)
    labels[:, :15] = 1
    labels[:, 15:] = 2
    cx, cy, bw, bh = 14.5, 12.0, 9.0, 7.0
    result = calculate_ellipse_occupancy(labels, cx, cy, bw, bh)
    yy, xx = np.indices(labels.shape, dtype=float)
    full = ((xx - cx) / (bw / 2)) ** 2 + ((yy - cy) / (bh / 2)) ** 2 <= 1
    total = np.count_nonzero(full)
    assert result["ellipse_pixel_count"] == total
    assert result["w_inlet_raw"] == pytest.approx(np.count_nonzero(labels[full] == 1) / total)


def test_xy_coordinate_convention_and_boundary_clipping() -> None:
    labels = np.zeros((10, 20), dtype=np.uint8)
    labels[:, 0:5] = 1
    labels[:, 5:20] = 2
    result = calculate_ellipse_occupancy(labels, 1, 5, 6, 6)
    assert result["image_boundary_clipped"]
    assert result["w_inlet_raw"] > result["w_outlet_raw"]


def test_dataset_output_and_summary_counts_match() -> None:
    labels = np.ones((30, 30), dtype=np.uint8)
    tracks = pd.DataFrame(
        {
            "frame": [0, 0, 1],
            "track_id": [1, 2, 1],
            "centroid_x": [10.0, 15.0, 20.0],
            "centroid_y": [10.0, 15.0, 20.0],
            "bbox_w": [6.0, 6.0, 6.0],
            "bbox_h": [6.0, 6.0, 6.0],
        }
    )
    output = calculate_dataset_occupancy(tracks, labels, um_per_px=4.0)
    assert len(output) == len(tracks)
    summary = summarize_occupancy(output, "exp", "dev", "tracks.csv", "labels.npy", 4.0, 0.95)
    assert summary["total_droplet_frame_samples"] == 3
    assert summary["occupancy_computable_count"] == int(output["occupancy_computable"].sum())
    assert summary["low_physical_coverage_count"] == int(output["low_physical_coverage"].sum())
    assert "occupancy_valid" not in output.columns


def test_diagnostic_plot_generation_with_low_coverage(tmp_path) -> None:
    labels = np.ones((30, 30), dtype=np.uint8)
    tracks = pd.DataFrame(
        {
            "frame": [0, 1],
            "track_id": [1, 2],
            "centroid_x": [10.0, 10.0],
            "centroid_y": [10.0, 10.0],
            "bbox_w": [6.0, 12.0],
            "bbox_h": [6.0, 12.0],
        }
    )
    output = calculate_dataset_occupancy(tracks, labels, um_per_px=4.0)
    output.loc[1, "physical_coverage_raw"] = 0.5
    output.loc[1, "low_physical_coverage"] = True
    summary = summarize_occupancy(output, "exp", "dev", "tracks.csv", "labels.npy", 4.0, 0.95)
    path = tmp_path / "diagnostics.png"
    _save_diagnostics(summary, output, path)
    assert path.exists()
