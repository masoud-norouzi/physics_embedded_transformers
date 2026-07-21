from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


ENRICHMENT_VERSION = "1.0"
COORDINATE_TRANSFORM_VERSION = "device_pixels_to_cfd_um_v1"


@dataclass(frozen=True)
class EnrichmentConfig:
    experiment_id: str = "video_2"
    experiment_config_path: Path = Path("configs/experiments/video_2.yml")
    tracking_path: Path = Path("data/processed/2/tracked_features.csv")
    hydraulic_state_path: Path = Path("outputs/physics/video_2/baseline_hydraulic_state.csv")
    cfd_library_path: Path = Path("outputs/physics/full_device_cfd/library")
    occupancy_path: Path = Path("outputs/physics/video_2/droplet_occupancy.csv")
    output_root: Path = Path("outputs/physics/video_2/enrichment")


@dataclass(frozen=True)
class EnrichmentSummary:
    experiment_id: str
    source_tracking_path: str
    source_tracking_sha256: str
    hydraulic_input_path: str
    hydraulic_input_sha256: str
    cfd_library_path: str
    cfd_version: str
    mesh_version: str
    interpolation_module_version: str
    coordinate_transform_version: str
    coordinate_transform_description: str
    output_path: str
    row_count: int
    column_count: int
    original_column_count: int
    inside_cfd_domain_rows: int
    inside_cfd_domain_fraction: float
    unique_tracks_inside_cfd_domain: int
    inside_domain_by_region: dict[str, int]
    flow_direction_alignment: dict[str, float]
    missing_value_counts: dict[str, int]
    validation: dict[str, Any]
    generation_timestamp_utc: str
