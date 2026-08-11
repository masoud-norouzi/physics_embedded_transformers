from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import matplotlib.tri as mtri
import numpy as np


@dataclass(frozen=True)
class TriangleVelocityField:
    """Piecewise-linear velocity lookup on a triangular CFD mesh."""

    nodes_um: np.ndarray
    elements: np.ndarray
    velocity_um_per_s: np.ndarray

    def __post_init__(self) -> None:
        nodes = np.asarray(self.nodes_um, dtype=float)
        elements = np.asarray(self.elements, dtype=np.int64)
        velocity = np.asarray(self.velocity_um_per_s, dtype=float)
        if nodes.ndim != 2 or nodes.shape[1] != 2:
            raise ValueError(f"nodes_um must have shape (N, 2), got {nodes.shape}")
        if elements.ndim != 2 or elements.shape[1] != 3:
            raise ValueError(f"elements must have shape (M, 3), got {elements.shape}")
        if velocity.shape != nodes.shape:
            raise ValueError(f"velocity_um_per_s must have shape {nodes.shape}, got {velocity.shape}")
        object.__setattr__(self, "nodes_um", nodes)
        object.__setattr__(self, "elements", elements)
        object.__setattr__(self, "velocity_um_per_s", velocity)
        tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
        object.__setattr__(self, "_finder", tri.get_trifinder())

    def evaluate(self, point_um: np.ndarray) -> tuple[np.ndarray, str]:
        point = np.asarray(point_um, dtype=float)
        elem_id = int(self._finder(float(point[0]), float(point[1])))
        if elem_id < 0:
            return np.full(2, np.nan), "outside_mesh"
        vertices = self.nodes_um[self.elements[elem_id]]
        weights = barycentric_weights(point, vertices)
        if weights is None or not np.isfinite(weights).all():
            return np.full(2, np.nan), "interpolation_failure"
        velocity = weights @ self.velocity_um_per_s[self.elements[elem_id]]
        if not np.isfinite(velocity).all():
            return np.full(2, np.nan), "interpolation_failure"
        return velocity, "ok"


@dataclass(frozen=True)
class StreamlineTrace:
    seed_id: int
    seed_um: np.ndarray
    points_um: np.ndarray
    elapsed_s: np.ndarray
    arc_length_um: np.ndarray
    termination_reason: str

    @property
    def steps(self) -> int:
        return int(max(0, len(self.points_um) - 1))


def barycentric_weights(point: np.ndarray, vertices: np.ndarray) -> np.ndarray | None:
    a, b, c = np.asarray(vertices, dtype=float)
    matrix = np.column_stack([a - c, b - c])
    rhs = np.asarray(point, dtype=float) - c
    det = float(np.linalg.det(matrix))
    if abs(det) <= 1.0e-12:
        return None
    l1, l2 = np.linalg.solve(matrix, rhs)
    return np.array([l1, l2, 1.0 - l1 - l2], dtype=float)


def trace_streamline_time(
    seed_id: int,
    seed_um: np.ndarray,
    field: TriangleVelocityField,
    *,
    inside: Callable[[np.ndarray], np.ndarray],
    max_step_um: float = 4.0,
    max_time_s: float = 10.0,
    max_steps: int = 5000,
    low_speed_um_per_s: float = 1.0e-6,
) -> StreamlineTrace:
    """Trace dx/dt = v(x) with RK4 and accumulate advective time."""
    if max_step_um <= 0.0:
        raise ValueError("max_step_um must be positive")
    if max_time_s <= 0.0:
        raise ValueError("max_time_s must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    point = np.asarray(seed_um, dtype=float)
    points = [point.copy()]
    elapsed = [0.0]
    arc_length = [0.0]
    reason = "max_steps"
    for _ in range(int(max_steps)):
        if not bool(inside(point.reshape(1, 2))[0]):
            reason = "left_domain"
            break
        velocity, status = field.evaluate(point)
        if status != "ok":
            reason = status
            break
        speed = float(np.linalg.norm(velocity))
        if speed <= low_speed_um_per_s:
            reason = "low_speed"
            break
        dt = min(float(max_step_um) / speed, float(max_time_s) - elapsed[-1])
        if dt <= 0.0:
            reason = "max_time"
            break
        candidate, status = rk4_step(point, dt, field)
        if status != "ok":
            reason = status
            break
        if not bool(inside(candidate.reshape(1, 2))[0]):
            candidate = _adaptive_inside_step(point, candidate - point, inside)
            if candidate is None:
                reason = "left_domain"
                break
        ds = float(np.linalg.norm(candidate - point))
        point = candidate
        points.append(point.copy())
        elapsed.append(float(elapsed[-1] + dt))
        arc_length.append(float(arc_length[-1] + ds))
        if elapsed[-1] >= max_time_s:
            reason = "max_time"
            break
    return StreamlineTrace(
        seed_id=int(seed_id),
        seed_um=np.asarray(seed_um, dtype=float),
        points_um=np.asarray(points, dtype=float),
        elapsed_s=np.asarray(elapsed, dtype=float),
        arc_length_um=np.asarray(arc_length, dtype=float),
        termination_reason=reason,
    )


def rk4_step(point_um: np.ndarray, dt_s: float, field: TriangleVelocityField) -> tuple[np.ndarray, str]:
    point = np.asarray(point_um, dtype=float)
    k1, status = field.evaluate(point)
    if status != "ok":
        return point.copy(), status
    k2, status = field.evaluate(point + 0.5 * dt_s * k1)
    if status != "ok":
        return point + dt_s * k1, "ok"
    k3, status = field.evaluate(point + 0.5 * dt_s * k2)
    if status != "ok":
        return point + dt_s * k2, "ok"
    k4, status = field.evaluate(point + dt_s * k3)
    if status != "ok":
        return point + dt_s * k3, "ok"
    return point + dt_s * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0, "ok"


def _adaptive_inside_step(
    current_um: np.ndarray,
    delta_um: np.ndarray,
    inside: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray | None:
    scale = 0.5
    for _ in range(12):
        candidate = current_um + scale * delta_um
        if bool(inside(candidate.reshape(1, 2))[0]):
            return candidate
        scale *= 0.5
    return None
