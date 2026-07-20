import numpy as np
import pandas as pd

from src.physics.geometry.regions import RegionLabel, build_region_label_map, validate_region_label_map


def _synthetic_device() -> dict:
    return {
        "device": {
            "id": "synthetic",
            "calibration": {"um_per_px": 1.0},
            "geometry": {
                "junctions": {
                    "upper": {"center_px": [5.0, 5.0], "size_um": [3.0, 3.0]},
                    "lower": {"center_px": [5.0, 12.0], "size_um": [3.0, 3.0]},
                }
            },
        }
    }


def _synthetic_inputs() -> tuple[np.ndarray, pd.DataFrame]:
    mask = np.zeros((20, 12), dtype=bool)
    mask[0:5, 5] = True
    mask[4:7, 4:7] = True
    mask[7:11, 3] = True
    mask[7:11, 7] = True
    mask[11:14, 4:7] = True
    mask[14:19, 5] = True
    rows = []
    rows.extend({"x": 5, "y": y, "channel": "inlet"} for y in range(0, 6))
    rows.extend({"x": 5, "y": y, "channel": "outlet"} for y in range(12, 19))
    rows.extend({"x": 3, "y": y, "channel": "left"} for y in range(5, 13))
    rows.extend({"x": 7, "y": y, "channel": "right"} for y in range(5, 13))
    return mask, pd.DataFrame(rows)


def test_region_label_ids_shape_and_mask_containment() -> None:
    mask, centerlines = _synthetic_inputs()
    labels, _ = build_region_label_map(mask, centerlines, _synthetic_device())
    assert labels.shape == mask.shape
    assert set(np.unique(labels)).issubset({int(label) for label in RegionLabel})
    assert not np.any((labels != 0) & ~mask)


def test_junction_precedence_and_mutual_exclusivity() -> None:
    mask, centerlines = _synthetic_inputs()
    labels, _ = build_region_label_map(mask, centerlines, _synthetic_device())
    assert labels[5, 5] == int(RegionLabel.UPPER_JUNCTION)
    assert labels[12, 5] == int(RegionLabel.LOWER_JUNCTION)
    assert labels.ndim == 2


def test_physical_regions_are_nonempty() -> None:
    mask, centerlines = _synthetic_inputs()
    labels, _ = build_region_label_map(mask, centerlines, _synthetic_device())
    for label in RegionLabel:
        if label != RegionLabel.UNASSIGNED:
            assert np.count_nonzero(labels == int(label)) > 0


def test_connectivity_checks_pass() -> None:
    mask, centerlines = _synthetic_inputs()
    _, diagnostics = build_region_label_map(mask, centerlines, _synthetic_device())
    assert diagnostics["connectivity"] == {
        "inlet_connects_upper_junction": True,
        "outlet_connects_lower_junction": True,
        "left_branch_connects_upper_and_lower_junctions": True,
        "right_branch_connects_upper_and_lower_junctions": True,
    }


def test_metadata_counts_match_label_map() -> None:
    mask, centerlines = _synthetic_inputs()
    labels, diagnostics = build_region_label_map(mask, centerlines, _synthetic_device())
    validated = validate_region_label_map(labels, mask)
    assert diagnostics["label_counts"] == validated["label_counts"]
    for label in RegionLabel:
        name = validated["label_counts"]
        assert name
    assert validated["total_channel_pixels"] == int(np.count_nonzero(mask))
    assert validated["background_pixel_count"] == int(np.count_nonzero(~mask))
