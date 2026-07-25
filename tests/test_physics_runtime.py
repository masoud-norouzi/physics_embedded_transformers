from types import SimpleNamespace

import numpy as np
import pytest

from src.datasets.canonical_dataset_builder import CANONICAL_V2_FEATURE_NAMES
from src.physics.geometry.coordinates import CoordinateConvention
from src.physics.runtime import (
    CANONICAL_RUNTIME_FEATURE_NAMES,
    MODEL_PREDICTION_FEATURE_NAMES,
    PhysicsRuntimeContext,
    PhysicsRuntimeDiagnostics,
    compute_occupancy,
    construct_ellipses,
    sample_cfd,
    step,
    update_hydraulics,
    update_positions,
)


class _FakeField:
    def __init__(self, left_fraction: float) -> None:
        self.left_fraction = float(left_fraction)

    def sample_cfd(self, points_um):
        points = np.asarray(points_um, dtype=float)
        return SimpleNamespace(
            cfd_u_norm=np.full(len(points), self.left_fraction),
            cfd_v_norm=np.full(len(points), -self.left_fraction),
            original_valid=np.ones(len(points), dtype=bool),
            projection_distance_um=np.zeros(len(points), dtype=float),
        )


class _FakeLibrary:
    fractions = (0.0, 1.0)

    def __init__(self) -> None:
        geometry = SimpleNamespace(coordinate_frame="device_cartesian_y_up")
        self.cases = [SimpleNamespace(mesh=SimpleNamespace(geometry=geometry))]
        self.requested = []

    def interpolate(self, left_fraction: float):
        self.requested.append(float(left_fraction))
        return _FakeField(left_fraction)


def _context(region_labels: np.ndarray | None = None) -> PhysicsRuntimeContext:
    if region_labels is None:
        region_labels = np.full((80, 80), 3, dtype=np.uint8)
    return PhysicsRuntimeContext(
        feature_names=tuple(CANONICAL_RUNTIME_FEATURE_NAMES),
        region_labels=region_labels,
        velocity_mm_s_per_px_frame=10.416,
        hydraulic_constants={
            "left_length_um": 1000.0,
            "right_length_um": 1000.0,
            "droplet_equivalent_length_um": 100.0,
            "total_mixture_flow_ul_hr": 100.0,
            "channel_width_um": 100.0,
            "channel_height_um": 100.0,
            "continuous_flow_ul_hr": 90.0,
            "dispersed_flow_ul_hr": 10.0,
        },
        cfd_library=_FakeLibrary(),
        coordinate_convention=CoordinateConvention(pixel_scale_um_per_px=4.0, y_reference_px=79.0),
        superficial_velocity_mm_s=2.7777777777777777,
    )


def _state(context: PhysicsRuntimeContext, rows: int = 2) -> np.ndarray:
    state = np.zeros((rows, len(context.feature_names)), dtype=np.float32)
    idx = context.feature_index
    state[0, idx["x"]] = 20.0
    state[0, idx["y"]] = 20.0
    state[0, idx["bbox_w"]] = 10.0
    state[0, idx["bbox_h"]] = 8.0
    return state


def test_runtime_feature_order_matches_canonical_v2() -> None:
    assert CANONICAL_RUNTIME_FEATURE_NAMES == CANONICAL_V2_FEATURE_NAMES
    assert MODEL_PREDICTION_FEATURE_NAMES == ["vx", "vy", "bbox_w", "bbox_h"]


def test_update_positions_uses_mm_s_to_px_frame_conversion() -> None:
    context = _context()
    state = _state(context)
    prediction = np.array([[10.416, -20.832, 12.0, 9.0], [999.0, 999.0, 1.0, 1.0]], dtype=np.float32)
    update_positions(state, prediction, context, np.array([True, False]))
    idx = context.feature_index
    assert state[0, idx["x"]] == pytest.approx(21.0)
    assert state[0, idx["y"]] == pytest.approx(18.0)
    assert state[0, idx["vx"]] == pytest.approx(10.416)
    assert state[0, idx["vy"]] == pytest.approx(-20.832)
    assert state[0, idx["bbox_w"]] == pytest.approx(12.0)
    assert state[1].sum() == pytest.approx(0.0)


def test_construct_ellipses_and_compute_occupancy_reuse_region_labels() -> None:
    labels = np.zeros((80, 80), dtype=np.uint8)
    labels[:, :40] = 3
    labels[:, 40:] = 4
    context = _context(labels)
    state = _state(context)
    active = np.array([True, False])
    ellipses = construct_ellipses(state, context, active)
    occupancy = compute_occupancy(ellipses, context, active)
    assert ellipses[0] is not None
    assert ellipses[1] is None
    left_index = CANONICAL_RUNTIME_FEATURE_NAMES.index("occupancy_left_branch") - CANONICAL_RUNTIME_FEATURE_NAMES.index("occupancy_inlet_channel")
    right_index = CANONICAL_RUNTIME_FEATURE_NAMES.index("occupancy_right_branch") - CANONICAL_RUNTIME_FEATURE_NAMES.index("occupancy_inlet_channel")
    assert occupancy[0, left_index] == pytest.approx(1.0)
    assert occupancy[0, right_index] == pytest.approx(0.0)
    assert occupancy[1].sum() == pytest.approx(0.0)


def test_out_of_image_predicted_ellipse_has_zero_occupancy_instead_of_crashing() -> None:
    context = _context()
    state = _state(context)
    idx = context.feature_index
    state[0, idx["x"]] = 10000.0
    state[0, idx["y"]] = 10000.0
    active = np.array([True, False])
    ellipses = construct_ellipses(state, context, active)
    occupancy = compute_occupancy(ellipses, context, active)
    assert ellipses[0] is None
    assert occupancy[0].sum() == pytest.approx(0.0)
    assert context.diagnostics.snapshot()["ellipse_outside_image"] == 1


def test_zero_pixel_predicted_ellipse_has_zero_occupancy_instead_of_crashing() -> None:
    context = _context()
    state = _state(context)
    idx = context.feature_index
    state[0, idx["x"]] = 20.5
    state[0, idx["y"]] = 20.5
    state[0, idx["bbox_w"]] = 1.0e-6
    state[0, idx["bbox_h"]] = 1.0e-6
    active = np.array([True, False])
    ellipses = construct_ellipses(state, context, active)
    occupancy = compute_occupancy(ellipses, context, active)
    assert ellipses[0] is None
    assert occupancy[0].sum() == pytest.approx(0.0)
    assert context.diagnostics.snapshot()["ellipse_zero_raster_pixels"] == 1


def test_update_hydraulics_recomputes_flow_split_from_occupancy() -> None:
    context = _context()
    occupancy = np.zeros((2, 6), dtype=float)
    occupancy[0, 2] = 1.0
    hydraulic = update_hydraulics(occupancy, np.array([True, False]), context)
    assert hydraulic["left_effective_length_um"] == pytest.approx(1100.0)
    assert hydraulic["right_effective_length_um"] == pytest.approx(1000.0)
    assert hydraulic["left_flow_fraction"] == pytest.approx(1000.0 / 2100.0)
    assert hydraulic["superficial_velocity_mm_s"] == pytest.approx(context.superficial_velocity_mm_s)


def test_sample_cfd_uses_updated_split_and_returns_normalized_components() -> None:
    context = _context()
    state = _state(context)
    active = np.array([True, False])
    sampled = sample_cfd(state, 0.42, context, active)
    assert context.cfd_library.requested == [pytest.approx(0.42)]
    assert sampled["cfd_u_norm"][0] == pytest.approx(0.42)
    assert sampled["cfd_v_norm"][0] == pytest.approx(-0.42)
    assert sampled["cfd_u_norm"][1] == pytest.approx(0.0)


def test_sample_cfd_clamps_lookup_split_to_library_bounds_without_changing_hydraulic_state() -> None:
    context = _context()
    state = _state(context)
    active = np.array([True, False])
    sampled = sample_cfd(state, -0.01, context, active)
    assert context.cfd_library.requested == [pytest.approx(0.0)]
    assert sampled["cfd_u_norm"][0] == pytest.approx(0.0)
    assert sampled["cfd_v_norm"][0] == pytest.approx(0.0)
    assert context.diagnostics.snapshot()["cfd_split_clamped_low"] == 1


def test_runtime_diagnostics_reset_and_snapshot() -> None:
    diagnostics = PhysicsRuntimeDiagnostics()
    diagnostics.increment("example")
    diagnostics.increment("example", 2)
    assert diagnostics.snapshot() == {"example": 3}
    diagnostics.reset()
    assert diagnostics.snapshot() == {}


def test_step_executes_full_prediction_to_next_state_pipeline() -> None:
    context = _context()
    state = _state(context)
    prediction = np.array([[10.416, 0.0, 10.0, 8.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    next_state = step(state, prediction, context)
    idx = context.feature_index
    expected_split = 1000.0 / 2100.0
    assert next_state.shape == state.shape
    assert next_state[0, idx["x"]] == pytest.approx(21.0)
    assert next_state[0, idx["vx"]] == pytest.approx(10.416)
    assert next_state[0, idx["occupancy_left_branch"]] == pytest.approx(1.0)
    assert next_state[0, idx["left_flow_fraction"]] == pytest.approx(expected_split)
    assert next_state[0, idx["cfd_u_norm"]] == pytest.approx(expected_split)
    assert next_state[0, idx["cfd_v_norm"]] == pytest.approx(-expected_split)
    assert next_state[1].sum() == pytest.approx(0.0)
    diagnostics = context.diagnostics.snapshot()
    assert diagnostics["runtime_calls"] == 1
    assert diagnostics["active_droplet_updates"] == 1


def test_step_is_functionally_pure_and_does_not_mutate_inputs() -> None:
    context = _context()
    state = _state(context)
    prediction = np.array([[10.416, 0.0, 10.0, 8.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    state_before = state.copy()
    prediction_before = prediction.copy()

    next_state = step(state, prediction, context)

    assert np.array_equal(state, state_before)
    assert np.array_equal(prediction, prediction_before)
    assert not np.shares_memory(next_state, state)
    assert not np.array_equal(next_state, state)
