from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.single_droplet_rollout import (
    FEATURE_NAMES,
    HistoricalLookup,
    PhysicsContext,
    build_historical_lookup,
    compute_single_droplet_hydraulics,
    estimate_shape_and_occupancy,
    normalize_enriched_regions,
    reconstruct_nonpredicted_state,
    sample_cfd_at_current_split,
    validate_feature_contract,
    validation_start_frame_values,
)


class _Convention:
    def image_points_to_device(self, points):
        return np.asarray(points, dtype=float)


class _Geometry:
    convention = _Convention()


class _Sample:
    def __init__(self, split: float):
        self.cfd_u = np.asarray([split], dtype=float)
        self.cfd_v = np.asarray([-split], dtype=float)
        self.cfd_valid = np.asarray([True], dtype=bool)


class _Field:
    def __init__(self, split: float):
        self.split = split

    def sample_device(self, points, convention):
        return _Sample(self.split)

    def sample_cfd(self, points):
        return _Sample(self.split)


class _Library:
    def __init__(self):
        self.fractions = (0.0, 1.0)
        self.calls = []

    def interpolate(self, split):
        self.calls.append(float(split))
        return _Field(float(split))


def _feature_index():
    return {name: idx for idx, name in enumerate(FEATURE_NAMES)}


def _config_with_features(names):
    return {"model": {"input_feature_names": list(names)}}


def _context() -> PhysicsContext:
    return PhysicsContext(
        geometry=_Geometry(),
        cfd_library=_Library(),
        region_labels=np.ones((100, 100), dtype=np.uint8),
        hydraulic_constants={
            "left_length_um": 1791.0,
            "right_length_um": 1491.0,
            "droplet_equivalent_length_um": 223.65,
            "total_mixture_flow_ul_hr": 1960.0,
            "channel_width_um": 100.0,
            "channel_height_um": 100.0,
            "continuous_flow_ul_hr": 1950.0,
            "dispersed_flow_ul_hr": 10.0,
        },
        cfd_min_split=0.0,
        cfd_max_split=1.0,
    )


def test_feature_contract_requires_exact_ordered_dataset_match():
    validate_feature_contract(list(FEATURE_NAMES), _config_with_features(FEATURE_NAMES))


def test_feature_contract_rejects_reordered_dataset_features():
    reordered = list(FEATURE_NAMES)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(ValueError, match="feature order does not match"):
        validate_feature_contract(reordered, _config_with_features(FEATURE_NAMES))


def _lookup(occupancy: list[float], circularity: float = 0.9) -> HistoricalLookup:
    idx = _feature_index()
    values = np.zeros((10, len(FEATURE_NAMES)), dtype=np.float32)
    positions = []
    for i in range(10):
        values[i, idx["x"]] = 10.0 + 0.01 * i
        values[i, idx["y"]] = 20.0 + 0.01 * i
        values[i, idx["circularity"]] = circularity
        values[i, idx["cfd_valid"]] = 1.0
        for name, value in zip(FEATURE_NAMES[8:14], occupancy):
            values[i, idx[name]] = value
        positions.append([values[i, idx["x"]], values[i, idx["y"]]])
    return HistoricalLookup(np.asarray(positions, dtype=np.float32), values, idx)


def test_initial_state_reconstruction_overwrites_historical_physics_values():
    idx = _feature_index()
    state = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    state[idx["x"]] = 10.0
    state[idx["y"]] = 20.0
    state[idx["vx"]] = 1.5
    state[idx["vy"]] = -0.5
    state[idx["left_flow_fraction"]] = 0.123
    state[idx["cfd_u"]] = 99.0
    state[idx["cfd_v"]] = -99.0
    state[idx["cfd_valid"]] = 0.0

    reconstruct_nonpredicted_state(state, idx, _context(), _lookup([0, 0, 1, 0, 0, 0]), 0, "case")

    assert state[idx["x"]] == pytest.approx(10.0)
    assert state[idx["y"]] == pytest.approx(20.0)
    assert state[idx["vx"]] == pytest.approx(1.5)
    assert state[idx["vy"]] == pytest.approx(-0.5)
    assert state[idx["left_flow_fraction"]] != pytest.approx(0.123)
    assert state[idx["cfd_u"]] != pytest.approx(99.0)
    assert state[idx["cfd_v"]] != pytest.approx(-99.0)
    assert state[idx["cfd_valid"]] == 1.0


def test_hydraulic_visibility_zero_left_and_right_occupancy_matches_empty_baseline():
    idx = _feature_index()
    state = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    result = compute_single_droplet_hydraulics(state, idx, _context())
    split = result["left_flow_ul_hr"] / (result["left_flow_ul_hr"] + result["right_flow_ul_hr"])
    assert split == pytest.approx(1491.0 / (1791.0 + 1491.0))


def test_hydraulic_visibility_left_and_right_branch_occupancy_change_split():
    idx = _feature_index()
    context = _context()
    baseline = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    left = baseline.copy()
    right = baseline.copy()
    left[idx["occupancy_left_branch"]] = 1.0
    right[idx["occupancy_right_branch"]] = 1.0

    base_h = compute_single_droplet_hydraulics(baseline, idx, context)
    left_h = compute_single_droplet_hydraulics(left, idx, context)
    right_h = compute_single_droplet_hydraulics(right, idx, context)
    base_split = base_h["left_flow_ul_hr"] / (base_h["left_flow_ul_hr"] + base_h["right_flow_ul_hr"])
    left_split = left_h["left_flow_ul_hr"] / (left_h["left_flow_ul_hr"] + left_h["right_flow_ul_hr"])
    right_split = right_h["left_flow_ul_hr"] / (right_h["left_flow_ul_hr"] + right_h["right_flow_ul_hr"])

    assert left_h["left_effective_length_um"] > base_h["left_effective_length_um"]
    assert right_h["right_effective_length_um"] > base_h["right_effective_length_um"]
    assert left_split < base_split
    assert right_split > base_split


def test_cfd_refresh_uses_updated_left_flow_fraction_each_call():
    idx = _feature_index()
    context = _context()
    state = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    state[idx["left_flow_fraction"]] = 0.25
    sample_cfd_at_current_split(state, idx, context, 1, "case")
    state[idx["left_flow_fraction"]] = 0.75
    sample_cfd_at_current_split(state, idx, context, 2, "case")

    assert context.cfd_library.calls == [0.25, 0.75]
    assert state[idx["cfd_u"]] == pytest.approx(0.75)
    assert state[idx["cfd_v"]] == pytest.approx(-0.75)


def test_inverse_distance_occupancy_estimates_are_finite_nonnegative_and_normalized():
    estimate = estimate_shape_and_occupancy(np.asarray([10.0, 20.0]), _lookup([0.2, 0.3, 0.5, 0.0, 0.0, 0.0]))
    occ = np.asarray(list(estimate["occupancy"].values()), dtype=float)
    assert np.isfinite(occ).all()
    assert np.all(occ >= 0.0)
    assert occ.sum() == pytest.approx(1.0)


def test_validation_split_maps_indices_to_noncontiguous_frame_values():
    frames = np.arange(100, 220, 2)
    info = validation_start_frame_values(frames, stride=5, t_history=1, t_future=10)
    starts = np.arange(0, len(frames) - 11 + 1, 5, dtype=np.int64)
    train_end = int(0.70 * len(starts))
    val_end = int(0.85 * len(starts))
    expected = {int(frames[idx]) for idx in starts[train_end:val_end]}
    assert info["validation_start_frame_values"] == expected
    assert info["frame_values_contiguous"] is False


def test_region_normalization_maps_raw_enriched_labels_to_canonical_names():
    table = pd.DataFrame(
        {
            "dominant_region": ["inlet", "upper_junction", "left", "right", "lower_junction", "outlet"],
            "cfd_valid": [True] * 6,
        }
    )
    info = normalize_enriched_regions(table)
    assert info["normalization_map"]["inlet"] == "inlet channel"
    assert info["normalization_map"]["upper_junction"] == "inlet junction"
    assert table[info["normalized_column"]].tolist() == [
        "inlet channel",
        "inlet junction",
        "left branch",
        "right branch",
        "outlet junction",
        "outlet channel",
    ]
