from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.physics.targets import (
    derive_speed_angle_targets_np,
    reconstruct_velocity_from_speed_angle_torch,
    wrap_angle_np,
    wrap_angle_torch,
)


def test_wrap_angle_handles_pi_boundary() -> None:
    values = np.asarray([-3.5 * np.pi, -np.pi, 0.0, np.pi, 3.5 * np.pi])
    wrapped = wrap_angle_np(values)
    assert np.all(wrapped >= -np.pi)
    assert np.all(wrapped < np.pi)
    assert wrapped[2] == pytest.approx(0.0)
    torch_wrapped = wrap_angle_torch(torch.as_tensor(values, dtype=torch.float32))
    assert torch.all(torch_wrapped >= -np.pi)
    assert torch.all(torch_wrapped < np.pi)


def test_speed_angle_target_reconstructs_velocity_with_flipped_cfd_y() -> None:
    idx = {"vx": 0, "vy": 1, "bbox_w": 2, "bbox_h": 3, "cfd_u_norm": 4, "cfd_v_norm": 5}
    future = np.asarray([[[0.0, 2.0, 20.0, 10.0, 0.0, -1.0]]], dtype=np.float32)
    previous = np.asarray([[[1.0, 0.0, 20.0, 10.0, 0.0, -1.0]]], dtype=np.float32)

    target = derive_speed_angle_targets_np(future, previous, idx)
    assert target[0, 0, 0] == pytest.approx(2.0)
    assert target[0, 0, 1] == pytest.approx(0.0)

    velocity, fallback = reconstruct_velocity_from_speed_angle_torch(
        torch.as_tensor(target),
        torch.as_tensor(previous),
        idx,
        cfd_flip_y=True,
    )
    assert fallback[0, 0].item() is False
    assert velocity[0, 0, 0].item() == pytest.approx(0.0, abs=1.0e-6)
    assert velocity[0, 0, 1].item() == pytest.approx(2.0, abs=1.0e-6)


def test_speed_angle_falls_back_to_previous_velocity_when_cfd_zero() -> None:
    idx = {"vx": 0, "vy": 1, "bbox_w": 2, "bbox_h": 3, "cfd_u_norm": 4, "cfd_v_norm": 5}
    target = torch.tensor([[[2.0, 0.0, 20.0, 10.0]]])
    previous = torch.tensor([[[0.0, 3.0, 20.0, 10.0, 0.0, 0.0]]])
    velocity, fallback = reconstruct_velocity_from_speed_angle_torch(target, previous, idx)
    assert fallback[0, 0].item() is True
    assert velocity[0, 0, 0].item() == pytest.approx(0.0, abs=1.0e-6)
    assert velocity[0, 0, 1].item() == pytest.approx(2.0, abs=1.0e-6)
