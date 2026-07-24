from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.physics.full_device_cfd.domain import inside_full_device_domain

from .library import DEFAULT_LIBRARY_PATH, VelocityFieldLibrary


def validate_cfd_velocity_normalization(
    library_path: str | Path = DEFAULT_LIBRARY_PATH,
    left_fraction: float | None = None,
    output_root: str | Path = "outputs/physics/interpolation/cfd_normalization_validation",
    random_seed: int = 20260724,
    sample_count: int = 8,
) -> dict[str, Any]:
    library = VelocityFieldLibrary.from_directory(library_path)
    fraction = float(left_fraction if left_fraction is not None else library.fractions[len(library.fractions) // 2])
    field = library.interpolate(fraction)
    reference = float(field.inlet_reference_velocity_m_per_s)
    if reference <= 0.0 or not np.isfinite(reference):
        raise ValueError(f"Invalid inlet reference velocity: {reference}")

    rng = np.random.default_rng(random_seed)
    points = _random_inside_points(field, sample_count, rng)
    samples = field.sample_cfd(points)
    raw = np.column_stack([samples.u_x_m_per_s, samples.u_y_m_per_s])
    normalized = np.column_stack([samples.cfd_u_norm, samples.cfd_v_norm])
    raw_speed = samples.speed_m_per_s
    normalized_speed = samples.cfd_speed_norm
    direction_from_normalized = normalized / normalized_speed[:, None]
    direction_match = np.allclose(
        direction_from_normalized,
        np.column_stack([samples.cfd_dir_x, samples.cfd_dir_y]),
        rtol=1.0e-12,
        atol=1.0e-12,
        equal_nan=True,
    )
    magnitude_match = np.allclose(normalized_speed, raw_speed / reference, rtol=1.0e-12, atol=1.0e-14, equal_nan=True)

    out = Path(output_root)
    out.mkdir(parents=True, exist_ok=True)
    rows = np.column_stack([points, raw, raw_speed, normalized, normalized_speed])
    np.savetxt(
        out / "sampled_normalization_comparison.csv",
        rows,
        delimiter=",",
        header="x_um,y_um,u_x_m_per_s,u_y_m_per_s,speed_m_per_s,cfd_u_norm,cfd_v_norm,cfd_speed_norm",
        comments="",
    )
    summary = {
        "library_path": str(Path(library_path)),
        "left_fraction": fraction,
        "inlet_reference_velocity_m_per_s": reference,
        "library_inlet_reference_velocity_m_per_s": float(library.inlet_reference_velocity_m_per_s),
        "library_reference_invariant": bool(
            np.allclose(
                [case.inlet_reference_velocity_m_per_s for case in library.cases],
                library.inlet_reference_velocity_m_per_s,
                rtol=1.0e-12,
                atol=1.0e-14,
            )
        ),
        "reference_definition": "Analytical maximum of prescribed parabolic inlet profile: 1.5 * |inlet_flux| / channel_width.",
        "inlet_centerline_normalization": "The analytical inlet centerline velocity normalizes to 1.0 by definition; no arbitrary mesh sample is required to attain exactly 1.0.",
        "maximum_normalized_inlet_velocity": 1.0,
        "minimum_normalized_inlet_velocity": 0.0,
        "sample_count": int(len(points)),
        "directions_unchanged": bool(direction_match),
        "normalized_magnitudes_equal_raw_divided_by_inlet_reference": bool(magnitude_match),
        "sample_csv": str(out / "sampled_normalization_comparison.csv"),
        "sample_preview": [
            {
                "x_um": float(points[i, 0]),
                "y_um": float(points[i, 1]),
                "u_x_m_per_s": float(raw[i, 0]),
                "u_y_m_per_s": float(raw[i, 1]),
                "speed_m_per_s": float(raw_speed[i]),
                "cfd_u_norm": float(normalized[i, 0]),
                "cfd_v_norm": float(normalized[i, 1]),
                "cfd_speed_norm": float(normalized_speed[i]),
            }
            for i in range(min(5, len(points)))
        ],
    }
    (out / "cfd_normalization_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "cfd_normalization_validation_summary.md").write_text(_markdown(summary), encoding="utf-8")
    return summary


def _random_inside_points(field, sample_count: int, rng: np.random.Generator) -> np.ndarray:
    nodes = field.nodes_um
    lo = nodes.min(axis=0)
    hi = nodes.max(axis=0)
    points = []
    attempts = 0
    while len(points) < sample_count and attempts < sample_count * 1000:
        attempts += 1
        candidate = rng.uniform(lo, hi)
        if inside_full_device_domain(candidate.reshape(1, 2), field.mesh.geometry)[0]:
            points.append(candidate)
    if len(points) < sample_count:
        raise RuntimeError(f"Could only find {len(points)} inside-domain sample points")
    return np.asarray(points, dtype=float)


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# CFD Velocity Normalization Validation",
        "",
        f"- Library: `{summary['library_path']}`",
        f"- Split: `{summary['left_fraction']:.6f}`",
        f"- Inlet maximum velocity: `{summary['inlet_reference_velocity_m_per_s']:.12g}` m/s",
        "- Analytical inlet centerline normalized velocity: `1.0` by definition",
        "- Analytical inlet wall normalized velocity: `0.0` by definition",
        f"- Library reference invariant: `{summary['library_reference_invariant']}`",
        f"- Directions unchanged: `{summary['directions_unchanged']}`",
        f"- Normalized magnitude check: `{summary['normalized_magnitudes_equal_raw_divided_by_inlet_reference']}`",
        "",
        "The CFD solution files remain dimensional. Normalization is applied only by the interpolation/sampling layer.",
    ]
    return "\n".join(lines)
