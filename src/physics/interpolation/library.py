from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.physics.cfd.domain import build_junction_geometry
from src.physics.cfd.mesh import TriangularMesh, label_mesh_boundaries, label_mesh_boundary_facets
from src.physics.full_device_cfd.domain import build_full_device_cfd_geometry
from src.physics.full_device_cfd.mesh import FullDeviceMesh, label_boundary_facets as label_full_device_boundary_facets

from .split_interpolator import interpolate_split
from .types import InterpolatedVelocityField, VelocityFieldCase


CFD_VERSION = "1.0"
MESH_VERSION = "production_v1"
DEFAULT_CONFIG_PATH = Path("configs/physics/junction_cfd.yml")
DEFAULT_LIBRARY_PATH = Path("outputs/physics/full_device_cfd/library")
DEFAULT_JUNCTION_LIBRARY_PATH = Path("outputs/physics/junction_cfd/solutions")
EXPECTED_FRACTIONS = tuple(round(0.05 * i, 2) for i in range(1, 20))
FULL_DEVICE_MANIFEST = "production_split_library.csv"


class VelocityFieldLibrary:
    """Frozen CFD Version 1 split library with continuous split interpolation."""

    def __init__(self, root: Path, cases: list[VelocityFieldCase], index: dict) -> None:
        self.root = root
        self.cases = tuple(cases)
        self.index = index
        self.fractions = tuple(case.left_fraction for case in self.cases)
        self.cases_by_fraction = {case.left_fraction: case for case in self.cases}

    @classmethod
    def from_directory(
        cls,
        path: str | Path = DEFAULT_LIBRARY_PATH,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ) -> "VelocityFieldLibrary":
        root = Path(path)
        if not root.exists():
            raise FileNotFoundError(f"CFD Version 1 solution library does not exist: {root}")
        manifest_path = root / FULL_DEVICE_MANIFEST
        if manifest_path.exists():
            return cls._from_full_device_manifest(root, manifest_path, config_path)
        index_path = root / "library_index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"CFD library is missing {FULL_DEVICE_MANIFEST} or library_index.json under: {root}")
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"CFD Version 1 library index is malformed: {index_path}") from exc
        _validate_index(index, index_path)

        geometry = build_junction_geometry(config_path)
        records = _ordered_records(index)
        cases = [_load_case(root, records[0], geometry)]
        for record in records[1:]:
            cases.append(_load_case(root, record, geometry, reference_mesh=cases[0].mesh))
        _validate_library_cases(cases)
        return cls(root=root, cases=cases, index=index)

    @classmethod
    def _from_full_device_manifest(cls, root: Path, manifest_path: Path, config_path: str | Path) -> "VelocityFieldLibrary":
        try:
            records = pd.read_csv(manifest_path).to_dict("records")
        except Exception as exc:
            raise ValueError(f"Full-device production manifest is malformed: {manifest_path}") from exc
        if not records:
            raise ValueError(f"Full-device production manifest has no rows: {manifest_path}")
        geometry = build_full_device_cfd_geometry(config_path)
        ordered = sorted(records, key=lambda item: float(item["achieved_split"]))
        cases = [_load_full_device_case(root, ordered[0], geometry)]
        for record in ordered[1:]:
            cases.append(_load_full_device_case(root, record, geometry, reference_mesh=cases[0].mesh))
        _validate_library_cases(cases)
        index = {
            "library_type": "full_device_production",
            "manifest_path": str(manifest_path),
            "coordinate": "achieved_split",
            "records": ordered,
        }
        return cls(root=root, cases=cases, index=index)

    def interpolate(self, left_fraction: float) -> InterpolatedVelocityField:
        return interpolate_split(self.cases, left_fraction)

    def case_for_fraction(self, left_fraction: float, atol: float = 1.0e-12) -> VelocityFieldCase:
        alpha = float(left_fraction)
        for case in self.cases:
            if abs(case.left_fraction - alpha) <= atol:
                return case
        raise KeyError(f"No exact CFD case is stored for left_fraction={left_fraction!r}")


def _validate_index(index: dict, path: Path) -> None:
    if index.get("cfd_version") != CFD_VERSION:
        raise ValueError(f"{path} has cfd_version={index.get('cfd_version')!r}; expected {CFD_VERSION!r}")
    if index.get("mesh_version") != MESH_VERSION:
        raise ValueError(f"{path} has mesh_version={index.get('mesh_version')!r}; expected {MESH_VERSION!r}")
    records = index.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path} must contain a non-empty records list")
    fractions = [round(float(record["left_fraction"]), 2) for record in records]
    if fractions != list(EXPECTED_FRACTIONS):
        raise ValueError(f"Expected Version 1 split grid {list(EXPECTED_FRACTIONS)}, got {fractions}")


def _ordered_records(index: dict) -> list[dict]:
    return sorted(index["records"], key=lambda item: float(item["left_fraction"]))


def _load_case(root: Path, record: dict, geometry, reference_mesh: TriangularMesh | None = None) -> VelocityFieldCase:
    left_fraction = float(record["left_fraction"])
    case_id = str(record.get("split_name") or _case_id(left_fraction))
    case_path = root / case_id
    metadata_path = case_path / "reports" / "solution_metadata.json"
    flux_path = case_path / "reports" / "flux_report.json"
    field_path = case_path / "fields" / "stokes_solution.npz"
    for required in (metadata_path, flux_path, field_path):
        if not required.exists():
            raise FileNotFoundError(f"Missing frozen CFD case artifact: {required}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        flux_report = json.loads(flux_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed CFD metadata for case {case_id}") from exc
    _validate_case_metadata(case_id, left_fraction, metadata, flux_report)

    with np.load(field_path) as loaded:
        arrays = {key: loaded[key].copy() for key in loaded.files}
    required_keys = {
        "nodes_um",
        "elements",
        "velocity_nodes_m_per_s",
        "velocity_dof_coordinates_um",
        "velocity_dofs_m_per_s",
    }
    missing = sorted(required_keys - set(arrays))
    if missing:
        raise ValueError(f"Frozen CFD case {case_id} is missing field arrays: {missing}")
    nodes = _as_float_array(arrays["nodes_um"], (None, 2), f"{case_id}: nodes_um")
    elements = _as_int_array(arrays["elements"], (None, 3), f"{case_id}: elements")
    velocity_nodes = _as_float_array(arrays["velocity_nodes_m_per_s"], (len(nodes), 2), f"{case_id}: velocity_nodes_m_per_s")
    velocity_dofs = _as_float_array(arrays["velocity_dofs_m_per_s"], (None, 2), f"{case_id}: velocity_dofs_m_per_s")
    dof_coords = _as_float_array(arrays["velocity_dof_coordinates_um"], (len(velocity_dofs), 2), f"{case_id}: velocity_dof_coordinates_um")
    if not np.isfinite(dof_coords).all() or not np.isfinite(velocity_dofs).all() or not np.isfinite(velocity_nodes).all():
        raise ValueError(f"Frozen CFD case {case_id} contains non-finite velocity or coordinate arrays")

    if reference_mesh is not None and np.array_equal(elements, reference_mesh.elements) and np.allclose(
        nodes, reference_mesh.nodes_um, rtol=0.0, atol=1.0e-12
    ):
        mesh = reference_mesh
    else:
        boundary_nodes = _boundary_nodes_from_facets(elements)
        mesh = TriangularMesh(
            nodes_um=nodes,
            elements=elements,
            geometry=geometry,
            boundary_node_indices=boundary_nodes,
            boundary_labels=label_mesh_boundaries(nodes, boundary_nodes, geometry),
            boundary_facets=label_mesh_boundary_facets(nodes, elements, geometry),
        )
    return VelocityFieldCase(
        case_id=case_id,
        path=case_path,
        left_fraction=left_fraction,
        right_fraction=float(record["right_fraction"]),
        velocity_dof_m_per_s=velocity_dofs,
        velocity_dof_coordinates_um=dof_coords,
        velocity_node_m_per_s=velocity_nodes,
        nodes_um=nodes,
        elements=elements,
        mesh=mesh,
        metadata=metadata,
        flux_report=flux_report,
        cfd_version=CFD_VERSION,
        mesh_version=MESH_VERSION,
        units={
            "position": "um",
            "velocity": "m/s",
            "split_parameter": "left branch flow fraction",
        },
    )


def _load_full_device_case(root: Path, record: dict, geometry, reference_mesh: FullDeviceMesh | None = None) -> VelocityFieldCase:
    left_fraction = float(record["achieved_split"])
    case_path = Path(record["solution_path"])
    if not case_path.is_absolute():
        case_path = root / case_path
    case_id = case_path.name
    metadata_path = case_path / "metadata.json"
    field_path = case_path / "stokes_solution.npz"
    for required in (metadata_path, field_path):
        if not required.exists():
            raise FileNotFoundError(f"Missing full-device CFD library artifact: {required}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed full-device CFD metadata for case {case_id}") from exc

    with np.load(field_path) as loaded:
        arrays = {key: loaded[key].copy() for key in loaded.files}
    required_keys = {
        "nodes_um",
        "elements",
        "velocity_nodes_m_per_s",
        "velocity_dof_coordinates_um",
        "velocity_dofs_m_per_s",
    }
    missing = sorted(required_keys - set(arrays))
    if missing:
        raise ValueError(f"Full-device CFD case {case_id} is missing field arrays: {missing}")
    nodes = _as_float_array(arrays["nodes_um"], (None, 2), f"{case_id}: nodes_um")
    elements = _as_int_array(arrays["elements"], (None, 3), f"{case_id}: elements")
    velocity_nodes = _as_float_array(arrays["velocity_nodes_m_per_s"], (len(nodes), 2), f"{case_id}: velocity_nodes_m_per_s")
    velocity_dofs = _as_float_array(arrays["velocity_dofs_m_per_s"], (None, 2), f"{case_id}: velocity_dofs_m_per_s")
    dof_coords = _as_float_array(arrays["velocity_dof_coordinates_um"], (len(velocity_dofs), 2), f"{case_id}: velocity_dof_coordinates_um")
    if not np.isfinite(dof_coords).all() or not np.isfinite(velocity_dofs).all() or not np.isfinite(velocity_nodes).all():
        raise ValueError(f"Full-device CFD case {case_id} contains non-finite velocity or coordinate arrays")

    if reference_mesh is not None and np.array_equal(elements, reference_mesh.elements) and np.allclose(
        nodes, reference_mesh.nodes_um, rtol=0.0, atol=1.0e-12
    ):
        mesh = reference_mesh
    else:
        boundary_facets = label_full_device_boundary_facets(nodes, elements, geometry)
        boundary_nodes = np.unique(np.concatenate([edges.ravel() for edges in boundary_facets.values() if len(edges)])).astype(np.int64)
        labels = {name: np.unique(edges.ravel()).astype(np.int64) if len(edges) else np.array([], dtype=np.int64) for name, edges in boundary_facets.items()}
        mesh = FullDeviceMesh(
            nodes_um=nodes,
            elements=elements,
            geometry=geometry,
            boundary_node_indices=boundary_nodes,
            boundary_labels=labels,
            boundary_facets=boundary_facets,
            generation_runtime_s=0.0,
        )
    fluxes = metadata.get("fluxes_m2_per_s", {})
    return VelocityFieldCase(
        case_id=case_id,
        path=case_path,
        left_fraction=left_fraction,
        right_fraction=1.0 - left_fraction,
        velocity_dof_m_per_s=velocity_dofs,
        velocity_dof_coordinates_um=dof_coords,
        velocity_node_m_per_s=velocity_nodes,
        nodes_um=nodes,
        elements=elements,
        mesh=mesh,
        metadata=metadata,
        flux_report={"validation_status": "passed", "fluxes_m2_per_s": fluxes},
        cfd_version="full_device_production",
        mesh_version="full_device_24um",
        units={
            "position": "um",
            "velocity": "m/s",
            "split_parameter": "achieved left branch flow fraction",
        },
    )


def _validate_case_metadata(case_id: str, left_fraction: float, metadata: dict, flux_report: dict) -> None:
    if metadata.get("cfd_version") != CFD_VERSION:
        raise ValueError(f"{case_id} has cfd_version={metadata.get('cfd_version')!r}; expected {CFD_VERSION!r}")
    if metadata.get("mesh_version") != MESH_VERSION:
        raise ValueError(f"{case_id} has mesh_version={metadata.get('mesh_version')!r}; expected {MESH_VERSION!r}")
    if abs(float(metadata.get("requested_left_fraction", np.nan)) - left_fraction) > 1.0e-12:
        raise ValueError(f"{case_id} metadata requested_left_fraction does not match the library index")
    if flux_report.get("validation_status") != "passed":
        raise ValueError(f"{case_id} flux report is not a passed validated CFD case")


def _validate_library_cases(cases: Iterable[VelocityFieldCase]) -> None:
    ordered = list(cases)
    fractions = [case.left_fraction for case in ordered]
    if fractions != sorted(set(fractions)):
        raise ValueError("CFD split fractions must be unique and strictly ordered")
    reference = ordered[0]
    for case in ordered[1:]:
        if not np.array_equal(case.elements, reference.elements):
            raise ValueError(f"Mesh connectivity mismatch between {reference.case_id} and {case.case_id}")
        if not np.allclose(case.nodes_um, reference.nodes_um, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"Mesh coordinates mismatch between {reference.case_id} and {case.case_id}")
        if case.velocity_dof_m_per_s.shape != reference.velocity_dof_m_per_s.shape:
            raise ValueError(f"Velocity coefficient shape mismatch between {reference.case_id} and {case.case_id}")
        if case.velocity_node_m_per_s.shape != reference.velocity_node_m_per_s.shape:
            raise ValueError(f"Velocity node shape mismatch between {reference.case_id} and {case.case_id}")


def _as_float_array(array: np.ndarray, shape: tuple[int | None, ...], name: str) -> np.ndarray:
    out = np.asarray(array, dtype=float)
    _validate_shape(out, shape, name)
    return out


def _as_int_array(array: np.ndarray, shape: tuple[int | None, ...], name: str) -> np.ndarray:
    out = np.asarray(array, dtype=np.int64)
    _validate_shape(out, shape, name)
    return out


def _validate_shape(array: np.ndarray, shape: tuple[int | None, ...], name: str) -> None:
    if array.ndim != len(shape):
        raise ValueError(f"{name} must have {len(shape)} dimensions, got shape {array.shape}")
    for actual, expected in zip(array.shape, shape):
        if expected is not None and actual != expected:
            raise ValueError(f"{name} has shape {array.shape}; expected {shape}")


def _boundary_nodes_from_facets(elements: np.ndarray) -> np.ndarray:
    edges = np.vstack([elements[:, [0, 1]], elements[:, [1, 2]], elements[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique[counts == 1]
    return np.unique(boundary_edges.ravel()).astype(np.int64)


def _case_id(left_fraction: float) -> str:
    return f"split_0p{int(round(left_fraction * 100)):02d}"
