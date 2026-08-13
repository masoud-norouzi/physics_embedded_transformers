from __future__ import annotations

import numpy as np
import pytest

from src.physics.geometry.wall_sdf import (
    build_wall_sdf,
    clamp_to_channel_numpy,
    ellipse_support_radius,
    sample_wall_sdf_numpy,
)
from src.physics.runtime import CANONICAL_RUNTIME_FEATURE_NAMES, PhysicsRuntimeContext, update_positions


def _band_mask(shape: tuple[int, int] = (40, 40), band: tuple[int, int] = (10, 30)) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[band[0] : band[1], :] = True
    return mask


def test_build_wall_sdf_is_positive_inside_negative_outside_and_zero_at_the_wall() -> None:
    wall_sdf = build_wall_sdf(_band_mask())
    assert wall_sdf.sdf[10, 20] == pytest.approx(1.0)
    assert wall_sdf.sdf[20, 20] == pytest.approx(10.0)
    assert wall_sdf.sdf[9, 20] == pytest.approx(-1.0)
    assert wall_sdf.sdf[0, 20] == pytest.approx(-10.0)


def test_clamp_pushes_only_along_the_wall_normal_direction() -> None:
    wall_sdf = build_wall_sdf(_band_mask())
    candidate = np.array([[20.0, 5.0]])
    bbox = np.array([[4.0, 4.0]])
    result = clamp_to_channel_numpy(candidate, bbox, wall_sdf)
    assert result[0, 0] == pytest.approx(20.0)
    assert result[0, 1] == pytest.approx(12.0)


def test_clamp_is_a_no_op_when_already_contained() -> None:
    wall_sdf = build_wall_sdf(_band_mask())
    candidate = np.array([[20.0, 20.0]])
    bbox = np.array([[4.0, 4.0]])
    result = clamp_to_channel_numpy(candidate, bbox, wall_sdf)
    assert result[0] == pytest.approx([20.0, 20.0])


def test_clamp_push_magnitude_scales_with_the_axis_aligned_toward_the_wall() -> None:
    wall_sdf = build_wall_sdf(_band_mask())
    candidate = np.array([[20.0, 5.0]])
    wide_short = clamp_to_channel_numpy(candidate, np.array([[40.0, 4.0]]), wall_sdf)
    narrow_tall = clamp_to_channel_numpy(candidate, np.array([[4.0, 40.0]]), wall_sdf)
    assert wide_short[0, 1] == pytest.approx(12.0)
    assert narrow_tall[0, 1] == pytest.approx(30.0)
    assert wide_short[0, 1] != narrow_tall[0, 1]


def test_ellipse_support_radius_depends_on_push_direction() -> None:
    half_width, half_height = 10.0, 2.0
    along_x = ellipse_support_radius(half_width, half_height, 1.0, 0.0)
    along_y = ellipse_support_radius(half_width, half_height, 0.0, 1.0)
    diagonal = ellipse_support_radius(half_width, half_height, np.sqrt(0.5), np.sqrt(0.5))
    assert along_x == pytest.approx(10.0)
    assert along_y == pytest.approx(2.0)
    assert along_y < diagonal < along_x


def test_clamp_snaps_candidates_predicted_far_outside_the_array_before_pushing_inward() -> None:
    wall_sdf = build_wall_sdf(_band_mask())
    candidate = np.array([[20.0, -5000.0]])
    bbox = np.array([[4.0, 4.0]])
    result = clamp_to_channel_numpy(candidate, bbox, wall_sdf)
    assert result[0, 0] == pytest.approx(20.0)
    sdf_value, _, _ = sample_wall_sdf_numpy(wall_sdf, result)
    assert sdf_value[0] >= 2.0 - 1.0e-6


def test_update_positions_never_leaves_the_channel_under_adversarial_velocity() -> None:
    mask = _band_mask(shape=(60, 60), band=(20, 40))
    wall_sdf = build_wall_sdf(mask)
    context = PhysicsRuntimeContext(
        feature_names=tuple(CANONICAL_RUNTIME_FEATURE_NAMES),
        region_labels=mask.astype(np.uint8),
        velocity_mm_s_per_px_frame=1.0,
        hydraulic_constants={},
        cfd_library=None,
        coordinate_convention=None,
        superficial_velocity_mm_s=0.0,
        wall_sdf=wall_sdf,
    )
    idx = context.feature_index
    state = np.zeros((1, len(context.feature_names)), dtype=np.float32)
    state[0, idx["x"]] = 30.0
    state[0, idx["y"]] = 30.0
    state[0, idx["bbox_w"]] = 4.0
    state[0, idx["bbox_h"]] = 4.0
    active = np.array([True])

    rng = np.random.default_rng(0)
    for _ in range(25):
        velocity = rng.uniform(-500.0, 500.0, size=2).astype(np.float32)
        prediction = np.array([[velocity[0], velocity[1], 4.0, 4.0]], dtype=np.float32)
        update_positions(state, prediction, context, active, prediction_mode="velocity")
        point = state[0, [idx["x"], idx["y"]]].reshape(1, 2)
        sdf_value, _, _ = sample_wall_sdf_numpy(wall_sdf, point)
        assert sdf_value[0] >= 2.0 - 1.0e-6


def test_update_positions_without_wall_sdf_is_unclamped_for_backward_compatibility() -> None:
    mask = _band_mask(shape=(60, 60), band=(20, 40))
    context = PhysicsRuntimeContext(
        feature_names=tuple(CANONICAL_RUNTIME_FEATURE_NAMES),
        region_labels=mask.astype(np.uint8),
        velocity_mm_s_per_px_frame=1.0,
        hydraulic_constants={},
        cfd_library=None,
        coordinate_convention=None,
        superficial_velocity_mm_s=0.0,
    )
    idx = context.feature_index
    state = np.zeros((1, len(context.feature_names)), dtype=np.float32)
    state[0, idx["x"]] = 30.0
    state[0, idx["y"]] = 30.0
    state[0, idx["bbox_w"]] = 4.0
    state[0, idx["bbox_h"]] = 4.0
    prediction = np.array([[0.0, -100.0, 4.0, 4.0]], dtype=np.float32)
    update_positions(state, prediction, context, np.array([True]), prediction_mode="velocity")
    assert state[0, idx["y"]] == pytest.approx(-70.0)


def test_clamp_to_channel_torch_matches_numpy_and_stays_contained() -> None:
    torch = pytest.importorskip("torch")
    from src.physics.constraints import clamp_to_channel_torch, wall_sdf_to_torch

    wall_sdf = build_wall_sdf(_band_mask(shape=(60, 60), band=(20, 40)))
    sdf_t, grad_x_t, grad_y_t = wall_sdf_to_torch(wall_sdf)

    candidate_np = np.array([[25.0, 15.0]])
    bbox_np = np.array([[4.0, 4.0]])
    expected = clamp_to_channel_numpy(candidate_np, bbox_np, wall_sdf)

    candidate = torch.tensor(candidate_np, dtype=torch.float32, requires_grad=True)
    bbox = torch.tensor(bbox_np, dtype=torch.float32)
    result = clamp_to_channel_torch(candidate, bbox, sdf_t, grad_x_t, grad_y_t)
    assert result.detach().numpy() == pytest.approx(expected, abs=1.0e-4)

    result.sum().backward()
    assert torch.isfinite(candidate.grad).all()


def test_clamp_to_channel_torch_never_leaves_the_channel_under_adversarial_pushes() -> None:
    torch = pytest.importorskip("torch")
    from src.physics.constraints import clamp_to_channel_torch, wall_sdf_to_torch

    wall_sdf = build_wall_sdf(_band_mask(shape=(60, 60), band=(20, 40)))
    sdf_t, grad_x_t, grad_y_t = wall_sdf_to_torch(wall_sdf)
    bbox = torch.tensor([[4.0, 4.0]], dtype=torch.float32)

    rng = np.random.default_rng(1)
    position = torch.tensor([[30.0, 30.0]], dtype=torch.float32)
    for _ in range(25):
        push = torch.as_tensor(rng.uniform(-500.0, 500.0, size=(1, 2)), dtype=torch.float32)
        position = clamp_to_channel_torch(position + push, bbox, sdf_t, grad_x_t, grad_y_t)
        sdf_value, _, _ = sample_wall_sdf_numpy(wall_sdf, position.numpy())
        assert sdf_value[0] >= 2.0 - 1.0e-3


def test_clamp_to_channel_torch_gradient_is_finite_at_a_medial_axis_point() -> None:
    # Regression test: torch.sqrt(x) has an infinite/undefined derivative at x == 0, and a
    # real device's wall SDF has genuine (0, 0)-gradient points (medial axis / channel
    # centerline). Masking the *forward* value with torch.where/clamp_min is not enough --
    # PyTorch still computes gradients through the unused branch, and 0 * inf/nan is nan in
    # IEEE arithmetic, so it silently poisons every model weight the next time .backward()
    # runs. This previously produced permanent NaN losses partway through real training,
    # not caught by the adversarial-push test above because it never happened to land
    # exactly on a zero-gradient point.
    torch = pytest.importorskip("torch")
    from src.physics.constraints import clamp_to_channel_torch, wall_sdf_to_torch

    wall_sdf = build_wall_sdf(_band_mask(shape=(40, 40), band=(10, 30)))
    sdf_t, grad_x_t, grad_y_t = wall_sdf_to_torch(wall_sdf)
    grad_x_t[20, 20] = 0.0
    grad_y_t[20, 20] = 0.0

    candidate = torch.tensor([[20.0, 20.0]], dtype=torch.float32, requires_grad=True)
    bbox = torch.tensor([[4.0, 4.0]], dtype=torch.float32, requires_grad=True)
    result = clamp_to_channel_torch(candidate, bbox, sdf_t, grad_x_t, grad_y_t)
    result.sum().backward()

    assert torch.isfinite(candidate.grad).all()
    assert torch.isfinite(bbox.grad).all()
