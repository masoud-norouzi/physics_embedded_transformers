from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.physics.cfd.flow_coordinates import (
    FlowCoordinateBuildConfig,
    FlowCoordinateMap,
    build_flow_coordinate_map,
    stream_function_boundary_values,
)
from src.physics.cfd.streamlines import TriangleVelocityField, trace_streamline_time
from src.physics.full_device_cfd.domain import build_full_device_cfd_geometry
from src.physics.interpolation import VelocityFieldLibrary


LIBRARY_PATH = Path("outputs/physics/full_device_cfd/library")


def test_trace_streamline_time_accumulates_advective_time_for_uniform_flow() -> None:
    nodes = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    elements = np.array([[0, 1, 2], [1, 3, 2]])
    velocity = np.tile(np.array([2.0, 0.0]), (4, 1))
    field = TriangleVelocityField(nodes, elements, velocity)
    trace = trace_streamline_time(
        0,
        np.array([1.0, 5.0]),
        field,
        inside=lambda pts: (pts[:, 0] >= 0.0) & (pts[:, 0] <= 10.0) & (pts[:, 1] >= 0.0) & (pts[:, 1] <= 10.0),
        max_step_um=2.0,
        max_time_s=3.0,
        max_steps=10,
    )

    assert trace.termination_reason == "max_time"
    assert trace.elapsed_s[-1] == pytest.approx(3.0)
    assert trace.points_um[-1, 0] == pytest.approx(7.0)
    assert np.allclose(trace.points_um[:, 1], 5.0)
    assert trace.arc_length_um[-1] == pytest.approx(6.0)


@pytest.fixture(scope="module")
def full_device_library() -> VelocityFieldLibrary:
    if not (LIBRARY_PATH / "production_split_library.csv").exists():
        pytest.skip("Full-device production CFD library is not available")
    return VelocityFieldLibrary.from_directory(LIBRARY_PATH)


def test_build_flow_coordinate_map_for_full_device_case_smoke(full_device_library: VelocityFieldLibrary, tmp_path: Path) -> None:
    geometry = build_full_device_cfd_geometry()
    case = full_device_library.case_for_fraction(0.5000000004936045)
    coord_map, traces = build_flow_coordinate_map(
        case,
        config=FlowCoordinateBuildConfig(seed_count=15, max_step_um=12.0, max_time_s=0.6, max_steps=400),
    )

    assert coord_map.metadata["version"] == "flow_coordinates_v1"
    assert coord_map.psi_nodes.shape == (len(case.nodes_um),)
    assert np.isfinite(coord_map.psi_nodes).all()
    assert len(traces) == 15
    assert len(coord_map.sample_points_um) > 100

    points = np.vstack(
        [
            geometry.centerlines["inlet"].points_um[len(geometry.centerlines["inlet"].points_um) // 2],
            geometry.upper_junction_um,
            geometry.centerlines["left"].points_um[len(geometry.centerlines["left"].points_um) // 2],
        ]
    )
    sampled = coord_map.sample(points, geometry)
    assert sampled.valid.tolist() == [True, True, True]
    assert np.isfinite(sampled.psi).all()
    assert np.isfinite(sampled.n).all()
    assert np.isfinite(sampled.time_s).all()

    path = tmp_path / "flow_coordinates.npz"
    coord_map.save(path)
    reloaded = FlowCoordinateMap.load(path)
    reloaded_sampled = reloaded.sample(points, geometry)
    assert np.allclose(reloaded_sampled.psi, sampled.psi)
    assert np.allclose(reloaded_sampled.time_s, sampled.time_s)


def test_stream_function_wall_boundary_values_are_constant_per_wall_component(full_device_library: VelocityFieldLibrary) -> None:
    case = full_device_library.case_for_fraction(0.5000000004936045)
    values = stream_function_boundary_values(
        case.nodes_um,
        case.mesh.boundary_facets,
        case.mesh.geometry,
        {key: float(value) for key, value in case.metadata["fluxes_m2_per_s"].items()},
        case.left_fraction,
    )
    components = _wall_components(case.mesh.boundary_facets["wall"])
    spreads = []
    means = []
    for component in components:
        component_values = values[component]
        spreads.append(float(np.nanmax(component_values) - np.nanmin(component_values)))
        means.append(float(np.nanmean(component_values)))

    assert len(components) == 3
    total = abs(float(case.metadata["fluxes_m2_per_s"]["left_branch"])) + abs(float(case.metadata["fluxes_m2_per_s"]["right_branch"]))
    assert max(spreads) / total < 1.0e-15
    assert sorted(means) == pytest.approx([0.0, 0.5 * total, total])


def _wall_components(edges: np.ndarray) -> list[np.ndarray]:
    adjacency: dict[int, set[int]] = {}
    for a, b in np.asarray(edges, dtype=np.int64):
        adjacency.setdefault(int(a), set()).add(int(b))
        adjacency.setdefault(int(b), set()).add(int(a))
    seen: set[int] = set()
    components = []
    for start in sorted(adjacency):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(np.asarray(component, dtype=np.int64))
    return components
