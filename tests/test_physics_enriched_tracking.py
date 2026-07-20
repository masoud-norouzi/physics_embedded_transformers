from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.physics.enrichment.coordinate_mapping import CoordinateTransform, map_tracking_coordinates
from src.physics.enrichment.tracking_enricher import (
    _prepare_hydraulic_features,
    _sample_cfd_background,
    build_physics_enriched_tracking,
)
from src.physics.enrichment.types import EnrichmentConfig
from src.physics.enrichment.validation import validate_hydraulic_state, validate_tracking_hydraulic_join


def test_coordinate_mapping_gives_expected_values_for_reference_points() -> None:
    tracking = pd.DataFrame({"centroid_x": [10.0, 330.0], "centroid_y": [20.0, 215.0]})
    mapped = map_tracking_coordinates(
        tracking,
        CoordinateTransform(um_per_px=4.0, y_reference_px=596.0, tracking_x_column="centroid_x", tracking_y_column="centroid_y"),
    )

    assert mapped["x_device_um"].tolist() == [40.0, 1320.0]
    assert mapped["y_device_um"].tolist() == [2304.0, 1524.0]
    assert mapped["x_cfd_um"].tolist() == [40.0, 1320.0]
    assert mapped["y_cfd_um"].tolist() == [80.0, 860.0]


def test_hydraulic_join_is_one_to_one_by_frame_and_missing_frames_fail() -> None:
    tracking = pd.DataFrame({"frame": [0, 1], "track_id": [10, 11]})
    hydraulic = _hydraulic_table([0, 1])
    validate_hydraulic_state(hydraulic)
    features = _prepare_hydraulic_features(hydraulic)
    joined = tracking.merge(features, on="frame", how="left", validate="many_to_one", sort=False)

    validate_tracking_hydraulic_join(tracking, features, joined)

    missing_features = features[features["frame"] == 0]
    missing_joined = tracking.merge(missing_features, on="frame", how="left", validate="many_to_one", sort=False)
    with pytest.raises(ValueError, match="Missing hydraulic state"):
        validate_tracking_hydraulic_join(tracking, missing_features, missing_joined)


def test_duplicate_hydraulic_rows_fail_clearly() -> None:
    hydraulic = pd.concat([_hydraulic_table([0]), _hydraulic_table([0])], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate frame"):
        validate_hydraulic_state(hydraulic)


def test_vectorized_framewise_cfd_sampling_and_nan_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    library = _FakeLibrary()
    table = pd.DataFrame(
        {
            "x_cfd_um": [1.0, -1.0, 2.0],
            "y_cfd_um": [0.0, 0.0, 0.0],
            "left_flow_fraction": [0.25, 0.25, 0.75],
            "right_flow_fraction": [0.75, 0.75, 0.25],
        }
    )
    monkeypatch.setattr("src.physics.enrichment.tracking_enricher.inside_junction_domain", lambda points, geometry, tolerance_um=0.0: points[:, 0] >= 0)

    sampled = _sample_cfd_background(table, library)

    assert sampled["inside_cfd_domain"].tolist() == [True, False, True]
    assert np.isfinite(sampled.loc[[0, 2], "background_speed_m_per_s"]).all()
    assert sampled.loc[1, ["background_u_x_device_m_per_s", "background_u_y_device_m_per_s", "background_speed_m_per_s"]].isna().all()
    assert np.allclose(
        sampled.loc[[0, 2], "background_speed_m_per_s"],
        np.sqrt(sampled.loc[[0, 2], "background_u_x_device_m_per_s"] ** 2 + sampled.loc[[0, 2], "background_u_y_device_m_per_s"] ** 2),
    )
    dirs = sampled.loc[[0, 2], ["background_direction_x", "background_direction_y"]].to_numpy(float)
    assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0)
    assert sampled.loc[0, "cfd_alpha_low"] == 0.0
    assert sampled.loc[0, "cfd_alpha_high"] == 0.5
    assert sampled.loc[0, "cfd_interpolation_weight"] == 0.5
    assert sampled.loc[2, "cfd_alpha_low"] == 0.5
    assert sampled.loc[2, "cfd_alpha_high"] == 1.0


def test_build_enriched_tracking_preserves_inputs_and_is_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tracking_path = tmp_path / "tracked_features.csv"
    hydraulic_path = tmp_path / "baseline_hydraulic_state.csv"
    experiment_path = tmp_path / "video_2.yml"
    device_path = tmp_path / "device.yml"
    metadata_path = tmp_path / "region_metadata.json"
    tracking = pd.DataFrame(
        {
            "frame": [0, 1, 1],
            "track_id": [1, 1, 2],
            "centroid_x": [1.0, 2.0, -1.0],
            "centroid_y": [0.0, 0.0, 0.0],
            "label": [1, 1, 2],
        }
    )
    tracking.to_csv(tracking_path, index=False)
    _hydraulic_table([0, 1]).to_csv(hydraulic_path, index=False)
    metadata_path.write_text('{"image_shape": [10, 10]}', encoding="utf-8")
    device_path.write_text("device:\n  id: asymmetric_loop_h100\n  calibration:\n    um_per_px: 4.0\n", encoding="utf-8")
    experiment_path.write_text(
        f"experiment:\n  id: video_2\n  device_id: asymmetric_loop_h100\n",
        encoding="utf-8",
    )
    import src.config.loader as loader
    import src.physics.enrichment.coordinate_mapping as coordinate_mapping

    def fake_load_experiment_config(experiment_path_arg, configs_root="configs"):
            return {
                "experiment": {"experiment": {"id": "video_2", "device_id": "asymmetric_loop_h100"}},
                "device": {
                    "device": {
                        "id": "asymmetric_loop_h100",
                        "calibration": {"um_per_px": 4.0},
                        "geometry": {"region_metadata_path": str(metadata_path)},
                    }
                },
            }

    monkeypatch.setattr(loader, "load_experiment_config", fake_load_experiment_config)
    monkeypatch.setattr(coordinate_mapping, "load_experiment_config", fake_load_experiment_config)
    monkeypatch.setattr("src.physics.enrichment.tracking_enricher.VelocityFieldLibrary", SimpleNamespace(from_directory=lambda path: _FakeLibrary()))
    monkeypatch.setattr("src.physics.enrichment.tracking_enricher.inside_junction_domain", lambda points, geometry, tolerance_um=0.0: points[:, 0] >= 0)

    config = EnrichmentConfig(
        experiment_id="video_2",
        experiment_config_path=experiment_path,
        tracking_path=tracking_path,
        hydraulic_state_path=hydraulic_path,
        cfd_library_path=tmp_path / "library",
        occupancy_path=tmp_path / "missing_occupancy.csv",
        output_root=tmp_path / "enrichment",
    )
    before_bytes = tracking_path.read_bytes()
    first, first_summary = build_physics_enriched_tracking(config, overwrite=True)
    second, second_summary = build_physics_enriched_tracking(config, overwrite=True)

    assert tracking_path.read_bytes() == before_bytes
    assert len(first) == len(tracking)
    assert list(first.columns[: len(tracking.columns)]) == list(tracking.columns)
    pd.testing.assert_frame_equal(first, second)
    assert first_summary.row_count == second_summary.row_count == 3
    assert first_summary.column_count == second_summary.column_count
    assert not first[["frame", "track_id"]].duplicated().any()
    assert first.loc[~first["inside_cfd_domain"], "background_speed_m_per_s"].isna().all()


def _hydraulic_table(frames: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame": frames,
            "left_flow_ul_hr": [980.0 for _ in frames],
            "right_flow_ul_hr": [980.0 for _ in frames],
            "total_mixture_input_flow_ul_hr": [1960.0 for _ in frames],
            "left_velocity_um_s": [27000.0 for _ in frames],
            "right_velocity_um_s": [27000.0 for _ in frames],
        }
    )


class _FakeSample:
    def __init__(self, points: np.ndarray, alpha: float) -> None:
        self.inside_domain = points[:, 0] >= 0
        self.u_x_m_per_s = np.where(self.inside_domain, alpha + points[:, 0] * 0.01, np.nan)
        self.u_y_m_per_s = np.where(self.inside_domain, 1.0 - alpha, np.nan)
        self.speed_m_per_s = np.sqrt(self.u_x_m_per_s**2 + self.u_y_m_per_s**2)


class _FakeField:
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def sample(self, points: np.ndarray) -> _FakeSample:
        return _FakeSample(points, self.alpha)

    def sample_cfd(self, points: np.ndarray) -> _FakeSample:
        return self.sample(points)


class _FakeLibrary:
    def __init__(self) -> None:
        mesh = SimpleNamespace(
            geometry=object(),
            nodes_um=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            elements=np.array([[0, 1, 2]]),
        )
        self.fractions = (0.0, 0.5, 1.0)
        self.cases = [
            SimpleNamespace(left_fraction=0.0, cfd_version="1.0", mesh_version="production_v1", mesh=mesh),
            SimpleNamespace(left_fraction=0.5, cfd_version="1.0", mesh_version="production_v1", mesh=mesh),
            SimpleNamespace(left_fraction=1.0, cfd_version="1.0", mesh_version="production_v1", mesh=mesh),
        ]

    def interpolate(self, left_fraction: float) -> _FakeField:
        return _FakeField(float(left_fraction))
