from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .types import InterpolatedVelocityField, VelocityFieldCase


EXACT_FRACTION_ATOL = 1.0e-12


def validate_left_fraction(left_fraction: float) -> float:
    try:
        alpha = float(left_fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"left_fraction must be numeric, got {left_fraction!r}") from exc
    if not math.isfinite(alpha):
        raise ValueError(f"left_fraction must be finite, got {left_fraction!r}")
    return alpha


def interpolate_split(
    cases: Sequence[VelocityFieldCase],
    left_fraction: float,
    exact_atol: float = EXACT_FRACTION_ATOL,
) -> InterpolatedVelocityField:
    """Interpolate P2 FEM velocity coefficients between neighboring split cases."""
    if len(cases) < 2:
        raise ValueError("At least two CFD cases are required for split interpolation")
    alpha = validate_left_fraction(left_fraction)
    fractions = np.asarray([case.left_fraction for case in cases], dtype=float)
    if np.any(np.diff(fractions) <= 0.0):
        raise ValueError("CFD split cases must be strictly ordered by left fraction")
    if alpha < fractions[0] - exact_atol or alpha > fractions[-1] + exact_atol:
        raise ValueError(
            f"left_fraction={alpha:.12g} is outside the supported Version 1 interpolation range "
            f"[{fractions[0]:.2f}, {fractions[-1]:.2f}]"
        )

    exact = np.flatnonzero(np.isclose(fractions, alpha, rtol=0.0, atol=exact_atol))
    if len(exact):
        case = cases[int(exact[0])]
        return InterpolatedVelocityField(
            requested_left_fraction=float(case.left_fraction),
            requested_right_fraction=float(case.right_fraction),
            lower_library_fraction=float(case.left_fraction),
            upper_library_fraction=float(case.left_fraction),
            interpolation_weight=0.0,
            velocity_dof_m_per_s=case.velocity_dof_m_per_s.copy(),
            velocity_dof_coordinates_um=case.velocity_dof_coordinates_um.copy(),
            velocity_node_m_per_s=case.velocity_node_m_per_s.copy(),
            nodes_um=case.nodes_um.copy(),
            elements=case.elements.copy(),
            mesh=case.mesh,
            velocity_basis_metadata=_basis_metadata(case),
            units=dict(case.units),
            cfd_version=case.cfd_version,
            mesh_version=case.mesh_version,
            exact_match=True,
            lower_case_id=case.case_id,
            upper_case_id=case.case_id,
        )

    upper_idx = int(np.searchsorted(fractions, alpha, side="right"))
    lower_idx = upper_idx - 1
    if lower_idx < 0 or upper_idx >= len(cases):
        raise ValueError(
            f"left_fraction={alpha:.12g} is outside the supported Version 1 interpolation range "
            f"[{fractions[0]:.2f}, {fractions[-1]:.2f}]"
        )
    low = cases[lower_idx]
    high = cases[upper_idx]
    denom = high.left_fraction - low.left_fraction
    if denom <= 0.0:
        raise ValueError("Neighboring CFD split fractions are not strictly increasing")
    weight = float((alpha - low.left_fraction) / denom)
    velocity_dof = (1.0 - weight) * low.velocity_dof_m_per_s + weight * high.velocity_dof_m_per_s
    velocity_node = (1.0 - weight) * low.velocity_node_m_per_s + weight * high.velocity_node_m_per_s
    return InterpolatedVelocityField(
        requested_left_fraction=float(alpha),
        requested_right_fraction=float(1.0 - alpha),
        lower_library_fraction=float(low.left_fraction),
        upper_library_fraction=float(high.left_fraction),
        interpolation_weight=weight,
        velocity_dof_m_per_s=velocity_dof,
        velocity_dof_coordinates_um=low.velocity_dof_coordinates_um.copy(),
        velocity_node_m_per_s=velocity_node,
        nodes_um=low.nodes_um.copy(),
        elements=low.elements.copy(),
        mesh=low.mesh,
        velocity_basis_metadata=_basis_metadata(low),
        units=dict(low.units),
        cfd_version=low.cfd_version,
        mesh_version=low.mesh_version,
        exact_match=False,
        lower_case_id=low.case_id,
        upper_case_id=high.case_id,
    )


def _basis_metadata(case: VelocityFieldCase) -> dict[str, object]:
    return {
        "element_pair": "P2 velocity / P1 pressure",
        "velocity_dof_count": int(len(case.velocity_dof_m_per_s)),
        "velocity_components": ["u_x", "u_y"],
        "coefficient_storage": "paired vector P2 velocity DOFs from frozen CFD output",
    }
