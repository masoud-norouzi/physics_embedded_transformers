from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.config.loader import load_experiment_config
from src.config.velocity import load_velocity_conversion_from_experiment
from src.physics.enrichment.coordinate_mapping import build_coordinate_transform
from src.physics.enrichment.tracking_enricher import compute_inlet_superficial_velocity_mm_s
from src.physics.geometry.coordinates import CoordinateConvention
from src.physics.hydraulics import (
    compute_frame_baseline_hydraulics_from_occupancies,
    compute_isolated_droplet_equivalent_length_um,
)
from src.physics.interpolation import VelocityFieldLibrary
from src.physics.occupancy.calculator import calculate_raster_occupancy_validated, validate_label_map
from src.physics.occupancy.ellipse import EllipseRaster, rasterize_bbox_ellipse


CANONICAL_RUNTIME_FEATURE_NAMES = [
    "x",
    "y",
    "vx",
    "vy",
    "bbox_w",
    "bbox_h",
    "cfd_u_norm",
    "cfd_v_norm",
    "superficial_velocity",
    "left_flow_fraction",
    "occupancy_inlet_channel",
    "occupancy_inlet_junction",
    "occupancy_left_branch",
    "occupancy_right_branch",
    "occupancy_outlet_junction",
    "occupancy_outlet_channel",
]
MODEL_PREDICTION_FEATURE_NAMES = ["vx", "vy", "bbox_w", "bbox_h"]
_OCCUPANCY_FEATURE_TO_RAW = {
    "occupancy_inlet_channel": "w_inlet",
    "occupancy_inlet_junction": "w_upper_junction",
    "occupancy_left_branch": "w_left",
    "occupancy_right_branch": "w_right",
    "occupancy_outlet_junction": "w_lower_junction",
    "occupancy_outlet_channel": "w_outlet",
}
_OCCUPANCY_FEATURES = tuple(_OCCUPANCY_FEATURE_TO_RAW)


@dataclass(frozen=True)
class PhysicsRuntimeContext:
    """Dependencies required for deterministic closed-loop physics updates."""

    feature_names: tuple[str, ...]
    region_labels: np.ndarray
    velocity_mm_s_per_px_frame: float
    hydraulic_constants: dict[str, float]
    cfd_library: Any
    coordinate_convention: CoordinateConvention
    superficial_velocity_mm_s: float
    minimum_physical_coverage: float = 0.95

    @property
    def feature_index(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.feature_names)}

    @property
    def cfd_split_bounds(self) -> tuple[float, float]:
        fractions = tuple(float(value) for value in self.cfd_library.fractions)
        return min(fractions), max(fractions)


def load_physics_runtime_context(
    *,
    experiment_config_path: str | Path = "configs/experiments/video_2.yml",
    region_labels_path: str | Path | None = None,
    cfd_library_path: str | Path = "outputs/physics/full_device_cfd/library",
    feature_names: list[str] | tuple[str, ...] = CANONICAL_RUNTIME_FEATURE_NAMES,
    minimum_physical_coverage: float = 0.95,
) -> PhysicsRuntimeContext:
    """Load runtime dependencies from the existing production configuration."""
    experiment_path = Path(experiment_config_path)
    loaded = load_experiment_config(experiment_path)
    device = loaded["device"]["device"]
    labels_path = Path(region_labels_path or device["geometry"]["region_labels_path"])
    if not labels_path.exists():
        raise FileNotFoundError(f"Region label map does not exist: {labels_path}")
    region_labels = validate_label_map(np.load(labels_path))
    velocity_conversion = load_velocity_conversion_from_experiment(experiment_path)
    transform = build_coordinate_transform(str(experiment_path))
    return PhysicsRuntimeContext(
        feature_names=tuple(feature_names),
        region_labels=region_labels,
        velocity_mm_s_per_px_frame=float(velocity_conversion["velocity_mm_s_per_px_frame"]),
        hydraulic_constants=_hydraulic_constants_from_config(loaded),
        cfd_library=VelocityFieldLibrary.from_directory(cfd_library_path),
        coordinate_convention=transform.convention,
        superficial_velocity_mm_s=compute_inlet_superficial_velocity_mm_s(experiment_path),
        minimum_physical_coverage=minimum_physical_coverage,
    )


def step(
    current_state: np.ndarray,
    model_prediction: np.ndarray,
    context: PhysicsRuntimeContext,
    active_mask: np.ndarray | None = None,
    profile: dict[str, float] | None = None,
) -> np.ndarray:
    """Advance one deterministic physics state from learned kinematic predictions."""
    if profile is None:
        state = _state_matrix(current_state, context)
        prediction = _prediction_matrix(model_prediction, len(state))
        active = infer_active_mask(state, context) if active_mask is None else _mask(active_mask, len(state))

        updated = state.copy()
        update_positions(updated, prediction, context, active)
        ellipses = construct_ellipses(updated, context, active)
        occupancy = compute_occupancy(ellipses, context, active)
        hydraulics = update_hydraulics(occupancy, active, context)
        cfd = sample_cfd(updated, hydraulics["left_flow_fraction"], context, active)
        assemble_state(updated, prediction, occupancy, hydraulics, cfd, context, active)
        return _restore_shape(current_state, updated)

    total_start = time.perf_counter()
    section_start = time.perf_counter()
    state = _state_matrix(current_state, context)
    prediction = _prediction_matrix(model_prediction, len(state))
    active = infer_active_mask(state, context) if active_mask is None else _mask(active_mask, len(state))
    _profile_add(profile, "runtime_prepare_inputs_seconds", time.perf_counter() - section_start)
    _profile_add(profile, "runtime_active_droplets", float(np.count_nonzero(active)))

    section_start = time.perf_counter()
    updated = state.copy()
    _profile_add(profile, "runtime_state_copy_seconds", time.perf_counter() - section_start)

    section_start = time.perf_counter()
    update_positions(updated, prediction, context, active)
    _profile_add(profile, "runtime_update_positions_seconds", time.perf_counter() - section_start)

    section_start = time.perf_counter()
    ellipses = construct_ellipses(updated, context, active)
    _profile_add(profile, "runtime_construct_ellipses_seconds", time.perf_counter() - section_start)

    section_start = time.perf_counter()
    occupancy = compute_occupancy(ellipses, context, active)
    _profile_add(profile, "runtime_compute_occupancy_seconds", time.perf_counter() - section_start)

    section_start = time.perf_counter()
    hydraulics = update_hydraulics(occupancy, active, context)
    _profile_add(profile, "runtime_update_hydraulics_seconds", time.perf_counter() - section_start)

    section_start = time.perf_counter()
    cfd = sample_cfd(updated, hydraulics["left_flow_fraction"], context, active, profile=profile)
    _profile_add(profile, "runtime_sample_cfd_seconds", time.perf_counter() - section_start)

    section_start = time.perf_counter()
    assemble_state(updated, prediction, occupancy, hydraulics, cfd, context, active)
    _profile_add(profile, "runtime_assemble_state_seconds", time.perf_counter() - section_start)

    section_start = time.perf_counter()
    restored = _restore_shape(current_state, updated)
    _profile_add(profile, "runtime_restore_shape_seconds", time.perf_counter() - section_start)
    _profile_add(profile, "runtime_step_total_seconds", time.perf_counter() - total_start)
    return restored


def update_positions(
    state: np.ndarray,
    prediction: np.ndarray,
    context: PhysicsRuntimeContext,
    active_mask: np.ndarray,
) -> np.ndarray:
    """Update centroids in image pixels using predicted mm/s velocities."""
    idx = context.feature_index
    scale = float(context.velocity_mm_s_per_px_frame)
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("velocity_mm_s_per_px_frame must be positive and finite")
    if np.any(active_mask):
        if np.any(prediction[active_mask, 2:] <= 0.0):
            raise ValueError("Predicted bbox_w and bbox_h must be positive for active droplets")
        state[active_mask, idx["x"]] += prediction[active_mask, 0] / scale
        state[active_mask, idx["y"]] += prediction[active_mask, 1] / scale
        state[active_mask, idx["vx"]] = prediction[active_mask, 0]
        state[active_mask, idx["vy"]] = prediction[active_mask, 1]
        state[active_mask, idx["bbox_w"]] = prediction[active_mask, 2]
        state[active_mask, idx["bbox_h"]] = prediction[active_mask, 3]
    return state


def construct_ellipses(
    state: np.ndarray,
    context: PhysicsRuntimeContext,
    active_mask: np.ndarray,
) -> list[EllipseRaster | None]:
    """Construct the same bbox ellipse geometry used by enrichment occupancy."""
    idx = context.feature_index
    ellipses: list[EllipseRaster | None] = [None] * len(state)
    for row in np.flatnonzero(active_mask):
        ellipses[row] = rasterize_bbox_ellipse(
            float(state[row, idx["x"]]),
            float(state[row, idx["y"]]),
            float(state[row, idx["bbox_w"]]),
            float(state[row, idx["bbox_h"]]),
            context.region_labels.shape,
        )
    return ellipses


def compute_occupancy(
    ellipses: list[EllipseRaster | None],
    context: PhysicsRuntimeContext,
    active_mask: np.ndarray,
) -> np.ndarray:
    """Compute normalized regional occupancy for active predicted ellipses."""
    occupancy = np.zeros((len(ellipses), len(_OCCUPANCY_FEATURES)), dtype=float)
    for row in np.flatnonzero(active_mask):
        raster = ellipses[row]
        if raster is None:
            raise ValueError(f"Missing ellipse raster for active row {row}")
        result = calculate_raster_occupancy_validated(
            context.region_labels,
            raster,
            context.minimum_physical_coverage,
        )
        if not bool(result["occupancy_computable"]):
            continue
        occupancy[row] = [float(result[raw_name]) for raw_name in _OCCUPANCY_FEATURE_TO_RAW.values()]
    return occupancy


def update_hydraulics(
    occupancy: np.ndarray,
    active_mask: np.ndarray,
    context: PhysicsRuntimeContext,
) -> dict[str, Any]:
    """Recompute branch hydraulics from newly computed droplet occupancies."""
    active_occupancy = occupancy[active_mask]
    left = float(active_occupancy[:, _occupancy_index("occupancy_left_branch")].sum()) if len(active_occupancy) else 0.0
    right = float(active_occupancy[:, _occupancy_index("occupancy_right_branch")].sum()) if len(active_occupancy) else 0.0
    constants = context.hydraulic_constants
    result = compute_frame_baseline_hydraulics_from_occupancies(
        left,
        right,
        frame=0,
        n_droplets_total=int(np.count_nonzero(active_mask)),
        left_length_um=constants["left_length_um"],
        right_length_um=constants["right_length_um"],
        droplet_equivalent_length_um=constants["droplet_equivalent_length_um"],
        total_mixture_flow_ul_hr=constants["total_mixture_flow_ul_hr"],
        channel_width_um=constants["channel_width_um"],
        channel_height_um=constants["channel_height_um"],
        continuous_flow_ul_hr=constants.get("continuous_flow_ul_hr"),
        dispersed_flow_ul_hr=constants.get("dispersed_flow_ul_hr"),
    )
    total_flow = float(result["left_flow_ul_hr"] + result["right_flow_ul_hr"])
    result["left_flow_fraction"] = float(result["left_flow_ul_hr"] / total_flow)
    result["superficial_velocity_mm_s"] = float(context.superficial_velocity_mm_s)
    return result


def sample_cfd(
    state: np.ndarray,
    left_flow_fraction: float,
    context: PhysicsRuntimeContext,
    active_mask: np.ndarray,
    profile: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Sample normalized full-device CFD at updated active centroids."""
    section_start = time.perf_counter() if profile is not None else None
    cfd_u = np.zeros(len(state), dtype=float)
    cfd_v = np.zeros(len(state), dtype=float)
    if profile is not None:
        _profile_add(profile, "runtime_cfd_output_allocation_seconds", time.perf_counter() - section_start)

    section_start = time.perf_counter() if profile is not None else None
    if not np.any(active_mask):
        if profile is not None:
            _profile_add(profile, "runtime_cfd_active_check_seconds", time.perf_counter() - section_start)
        return {"cfd_u_norm": cfd_u, "cfd_v_norm": cfd_v}
    if profile is not None:
        _profile_add(profile, "runtime_cfd_active_check_seconds", time.perf_counter() - section_start)

    section_start = time.perf_counter() if profile is not None else None
    split_min, split_max = context.cfd_split_bounds
    if left_flow_fraction < split_min - 1e-12 or left_flow_fraction > split_max + 1e-12:
        raise ValueError(
            f"left_flow_fraction={left_flow_fraction:.12g} is outside CFD library range "
            f"[{split_min:.12g}, {split_max:.12g}]"
        )
    sample_split = float(np.clip(left_flow_fraction, split_min, split_max))
    if profile is not None:
        _profile_add(profile, "runtime_cfd_split_prepare_seconds", time.perf_counter() - section_start)

    section_start = time.perf_counter() if profile is not None else None
    idx = context.feature_index
    points_px = state[active_mask][:, [idx["x"], idx["y"]]]
    if profile is not None:
        _profile_add(profile, "runtime_cfd_point_extract_seconds", time.perf_counter() - section_start)

    section_start = time.perf_counter() if profile is not None else None
    points_um = _image_points_to_library_frame(points_px, context)
    if profile is not None:
        _profile_add(profile, "runtime_cfd_coordinate_transform_seconds", time.perf_counter() - section_start)

    section_start = time.perf_counter() if profile is not None else None
    interpolated = context.cfd_library.interpolate(sample_split)
    if profile is not None:
        _profile_add(profile, "runtime_cfd_interpolate_seconds", time.perf_counter() - section_start)

    section_start = time.perf_counter() if profile is not None else None
    samples = interpolated.sample_cfd(points_um)
    if profile is not None:
        _profile_add(profile, "runtime_cfd_sample_points_seconds", time.perf_counter() - section_start)

    section_start = time.perf_counter() if profile is not None else None
    finite = np.isfinite(samples.cfd_u_norm) & np.isfinite(samples.cfd_v_norm)
    if not np.all(finite):
        raise ValueError("CFD sampling returned non-finite normalized velocity for an active droplet")
    active_rows = np.flatnonzero(active_mask)
    cfd_u[active_rows] = samples.cfd_u_norm
    cfd_v[active_rows] = samples.cfd_v_norm
    if profile is not None:
        _profile_add(profile, "runtime_cfd_validate_assign_seconds", time.perf_counter() - section_start)
    return {"cfd_u_norm": cfd_u, "cfd_v_norm": cfd_v}


def assemble_state(
    state: np.ndarray,
    prediction: np.ndarray,
    occupancy: np.ndarray,
    hydraulics: dict[str, Any],
    cfd: dict[str, np.ndarray],
    context: PhysicsRuntimeContext,
    active_mask: np.ndarray,
) -> np.ndarray:
    """Assemble the canonical next state in model feature order."""
    idx = context.feature_index
    for occ_idx, name in enumerate(_OCCUPANCY_FEATURES):
        state[active_mask, idx[name]] = occupancy[active_mask, occ_idx]
    state[active_mask, idx["superficial_velocity"]] = float(hydraulics["superficial_velocity_mm_s"])
    state[active_mask, idx["left_flow_fraction"]] = float(hydraulics["left_flow_fraction"])
    state[active_mask, idx["cfd_u_norm"]] = cfd["cfd_u_norm"][active_mask]
    state[active_mask, idx["cfd_v_norm"]] = cfd["cfd_v_norm"][active_mask]
    state[~active_mask] = 0.0
    _validate_next_state(state, context, active_mask)
    return state


def infer_active_mask(state: np.ndarray, context: PhysicsRuntimeContext) -> np.ndarray:
    idx = context.feature_index
    return (
        np.isfinite(state[:, idx["x"]])
        & np.isfinite(state[:, idx["y"]])
        & np.isfinite(state[:, idx["bbox_w"]])
        & np.isfinite(state[:, idx["bbox_h"]])
        & (state[:, idx["bbox_w"]] > 0.0)
        & (state[:, idx["bbox_h"]] > 0.0)
    )


def _hydraulic_constants_from_config(loaded: dict[str, Any]) -> dict[str, float]:
    experiment = loaded["experiment"]["experiment"]
    device = loaded["device"]["device"]
    branches = device["loop"]["branches"]
    left_length = float(branches["left"]["length_um"])
    right_length = float(branches["right"]["length_um"])
    short_length = min(left_length, right_length)
    ratio = float(device.get("hydraulics", {}).get("isolated_droplet_resistance", {}).get("ratio_to_short_branch", 0.15))
    continuous = _phase_flow(experiment, "continuous")
    dispersed = _phase_flow(experiment, "dispersed")
    channel = device["channel"]
    return {
        "left_length_um": left_length,
        "right_length_um": right_length,
        "droplet_equivalent_length_um": compute_isolated_droplet_equivalent_length_um(short_length, ratio),
        "total_mixture_flow_ul_hr": continuous + dispersed,
        "continuous_flow_ul_hr": continuous,
        "dispersed_flow_ul_hr": dispersed,
        "channel_width_um": float(channel["width_um"]),
        "channel_height_um": float(channel["height_um"]),
    }


def _phase_flow(experiment: dict[str, Any], phase: str) -> float:
    value = float(experiment["phases"][phase]["flow_rate_ul_per_hr"])
    if value < 0 or not np.isfinite(value):
        raise ValueError(f"{phase} flow_rate_ul_per_hr must be finite and nonnegative")
    return value


def _image_points_to_library_frame(points_px: np.ndarray, context: PhysicsRuntimeContext) -> np.ndarray:
    geometry = context.cfd_library.cases[0].mesh.geometry
    if getattr(geometry, "coordinate_frame", "") == "device_cartesian_y_up":
        return context.coordinate_convention.image_points_to_device(points_px)
    return context.coordinate_convention.image_points_to_cfd(points_px)


def _occupancy_index(name: str) -> int:
    return list(_OCCUPANCY_FEATURES).index(name)


def _state_matrix(current_state: np.ndarray, context: PhysicsRuntimeContext) -> np.ndarray:
    state = np.asarray(current_state, dtype=np.float32)
    if state.ndim == 1:
        state = state.reshape(1, -1)
    if state.ndim != 2 or state.shape[1] != len(context.feature_names):
        raise ValueError(
            f"current_state must have shape (N, {len(context.feature_names)}) or ({len(context.feature_names)},), "
            f"got {state.shape}"
        )
    return state


def _prediction_matrix(model_prediction: np.ndarray, n_rows: int) -> np.ndarray:
    prediction = np.asarray(model_prediction, dtype=np.float32)
    if prediction.ndim == 1:
        prediction = prediction.reshape(1, -1)
    if prediction.shape != (n_rows, len(MODEL_PREDICTION_FEATURE_NAMES)):
        raise ValueError(
            f"model_prediction must have shape ({n_rows}, {len(MODEL_PREDICTION_FEATURE_NAMES)}), "
            f"got {prediction.shape}"
        )
    if not np.isfinite(prediction).all():
        raise ValueError("model_prediction must be finite")
    return prediction


def _mask(active_mask: np.ndarray, n_rows: int) -> np.ndarray:
    mask = np.asarray(active_mask, dtype=bool)
    if mask.shape != (n_rows,):
        raise ValueError(f"active_mask must have shape ({n_rows},), got {mask.shape}")
    return mask


def _restore_shape(original: np.ndarray, updated: np.ndarray) -> np.ndarray:
    return updated[0].copy() if np.asarray(original).ndim == 1 else updated


def _validate_next_state(state: np.ndarray, context: PhysicsRuntimeContext, active_mask: np.ndarray) -> None:
    if not np.isfinite(state).all():
        raise ValueError("Next state contains non-finite values")
    if not np.any(active_mask):
        return
    idx = context.feature_index
    occupancy = np.column_stack([state[:, idx[name]] for name in _OCCUPANCY_FEATURES])
    totals = occupancy[active_mask].sum(axis=1)
    nonzero = totals > 0.0
    if np.any(nonzero) and not np.allclose(totals[nonzero], 1.0, atol=1e-6, rtol=0.0):
        raise ValueError(f"Active occupancy fractions must sum to one, got {totals.tolist()}")
    split = state[active_mask, idx["left_flow_fraction"]]
    if np.any((split < -1e-12) | (split > 1.0 + 1e-12)):
        raise ValueError("left_flow_fraction must remain in [0, 1]")


def _profile_add(profile: dict[str, float], key: str, value: float) -> None:
    profile[key] = float(profile.get(key, 0.0) + value)
