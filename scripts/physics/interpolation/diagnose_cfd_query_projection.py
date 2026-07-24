from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd

from src.physics.full_device_cfd.domain import build_full_device_cfd_geometry
from src.physics.interpolation import VelocityFieldLibrary


DEFAULT_OUTPUT = Path("outputs/physics/interpolation/cfd_query_projection")


def main() -> None:
    output_dir = DEFAULT_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)
    library = VelocityFieldLibrary.from_directory()
    field = library.interpolate(0.5000000004936045)
    geometry = build_full_device_cfd_geometry()
    points, labels = representative_points(geometry)
    samples = field.sample_cfd(points)

    table = pd.DataFrame(
        {
            "label": labels,
            "original_x_um": points[:, 0],
            "original_y_um": points[:, 1],
            "original_valid": samples.original_valid,
            "sample_x_um": samples.sample_x,
            "sample_y_um": samples.sample_y,
            "projection_distance_um": samples.projection_distance,
            "cfd_valid": samples.cfd_valid,
            "cfd_u_m_per_s": samples.cfd_u,
            "cfd_v_m_per_s": samples.cfd_v,
        }
    )
    table.to_csv(output_dir / "cfd_query_projection_examples.csv", index=False)
    save_figure(output_dir / "cfd_query_projection_examples.png", field, points, samples, labels)
    print(f"wrote: {(output_dir / 'cfd_query_projection_examples.png').resolve()}")
    print(f"wrote: {(output_dir / 'cfd_query_projection_examples.csv').resolve()}")
    print(table.to_string(index=False))


def representative_points(geometry) -> tuple[np.ndarray, list[str]]:
    left = geometry.centerlines["left"]
    left_idx = len(left.points_um) // 2
    right = geometry.centerlines["right"]
    right_idx = len(right.points_um) // 2
    inlet = geometry.centerlines["inlet"]
    outlet = geometry.centerlines["outlet"]
    points = np.vstack(
        [
            left.points_um[left_idx],
            left.points_um[left_idx] + left.normals[left_idx] * 85.0,
            right.points_um[right_idx] - right.normals[right_idx] * 85.0,
            inlet.points_um[0] + np.array([0.0, 35.0]),
            outlet.points_um[-1] + np.array([0.0, -35.0]),
        ]
    )
    labels = [
        "inside_left_branch",
        "outside_left_wall",
        "outside_right_wall",
        "above_inlet_boundary",
        "below_truncated_outlet",
    ]
    return points, labels


def save_figure(path: Path, field, original: np.ndarray, samples, labels: list[str]) -> None:
    mesh = field.mesh
    tri = mtri.Triangulation(mesh.nodes_um[:, 0], mesh.nodes_um[:, 1], mesh.elements)
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.triplot(tri, color="0.82", linewidth=0.25, alpha=0.65)
    sample_points = samples.sample_points_um
    invalid = ~samples.original_valid
    ax.scatter(original[invalid, 0], original[invalid, 1], s=42, color="#dc2626", label="original outside query", zorder=4)
    ax.scatter(sample_points[invalid, 0], sample_points[invalid, 1], s=42, color="#2563eb", label="projected sample point", zorder=5)
    ax.scatter(original[~invalid, 0], original[~invalid, 1], s=36, color="0.25", label="original valid query", zorder=4)
    for i, label in enumerate(labels):
        if invalid[i]:
            ax.plot([original[i, 0], sample_points[i, 0]], [original[i, 1], sample_points[i, 1]], color="#7c3aed", linewidth=1.0, zorder=3)
        ax.annotate(label, xy=original[i], xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlabel("x_device_um")
    ax.set_ylabel("y_device_um")
    ax.set_title("CFD query projection diagnostics")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.18)
    fig.savefig(path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
