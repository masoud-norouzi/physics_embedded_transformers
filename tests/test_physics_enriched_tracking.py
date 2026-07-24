from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.physics.enrichment.coordinate_mapping import CoordinateTransform, map_tracking_coordinates
from src.physics.enrichment.tracking_enricher import (
    compute_inlet_superficial_velocity_mm_s,
    _prepare_hydraulic_features,
    _sample_cfd_background,
    build_physics_enriched_tracking,
    trim_acquisition_domain_by_cutoffs,
)
from src.physics.enrichment.types import EnrichmentConfig
from src.physics.enrichment.validation import validate_hydraulic_state, validate_tracking_hydraulic_join


def test_coordinate_mapping_gives_expected_values_for_reference_points() -> None:
    tracking = pd.DataFrame({"centroid_x": [10.0, 330.0], "centroid_y": [20.0, 215.0]})
    mapped = map_tracking_coordinates(
        tracking,
        CoordinateTransform(um_per_px=4.0, frame_rate_fps=2604.0, y_reference_px=596.0, tracking_x_column="centroid_x", tracking_y_column="centroid_y"),
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
    assert features["superficial_velocity"].iloc[0] == pytest.approx(54.44444444444444)

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
                "x_device_um": [1.0, -1.0, 2.0],
                "y_device_um": [0.0, 0.0, 0.0],
                "left_flow_fraction": [0.25, 0.25, 0.75],
            "right_flow_fraction": [0.75, 0.75, 0.25],
        }
    )
    sampled = _sample_cfd_background(table, library)

    assert sampled["inside_cfd_domain"].tolist() == [True, False, True]
    assert sampled["cfd_valid"].tolist() == [True, False, True]
    assert np.isfinite(sampled.loc[[0, 2], "cfd_speed_norm"]).all()
    assert np.isfinite(sampled.loc[[0, 2], "background_speed_m_per_s"]).all()
    assert sampled.loc[1, ["cfd_u_norm", "cfd_v_norm", "cfd_speed_norm", "background_u_x_device_m_per_s", "background_u_y_device_m_per_s", "background_speed_m_per_s"]].isna().all()
    assert np.allclose(
        sampled.loc[[0, 2], "cfd_speed_norm"],
        np.sqrt(sampled.loc[[0, 2], "cfd_u_norm"] ** 2 + sampled.loc[[0, 2], "cfd_v_norm"] ** 2),
    )
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
    occupancy_path = tmp_path / "occupancy.csv"
    tracking = pd.DataFrame(
        {
            "frame": [0, 1, 1],
            "track_id": [1, 1, 2],
            "centroid_x": [1.0, 2.0, -1.0],
            "centroid_y": [0.0, 0.0, 6.0],
            "label": [1, 1, 2],
        }
    )
    tracking.to_csv(tracking_path, index=False)
    pd.DataFrame(
        {
            "frame": [0, 1, 1],
            "track_id": [1, 1, 2],
            "dominant_region": ["left", "left", "outlet"],
        }
    ).to_csv(occupancy_path, index=False)
    _hydraulic_table([0, 1]).to_csv(hydraulic_path, index=False)
    metadata_path.write_text('{"image_shape": [10, 10]}', encoding="utf-8")
    device_path.write_text("device:\n  id: asymmetric_loop_h100\n  calibration:\n    um_per_px: 4.0\n", encoding="utf-8")
    experiment_path.write_text(
        f"experiment:\n  id: video_2\n  device_id: asymmetric_loop_h100\n  frame_rate_fps: 2604.0\n",
        encoding="utf-8",
    )
    import src.config.loader as loader
    import src.physics.enrichment.coordinate_mapping as coordinate_mapping

    def fake_load_experiment_config(experiment_path_arg, configs_root="configs"):
            return {
                "experiment": {
                    "experiment": {
                        "id": "video_2",
                        "device_id": "asymmetric_loop_h100",
                        "frame_rate_fps": 2604.0,
                        "phases": {
                            "continuous": {"flow_rate_ul_per_hr": 1950.0},
                            "dispersed": {"flow_rate_ul_per_hr": 100.0},
                        },
                        "acquisition": {
                            "domain_trim": {
                                "enabled": True,
                                "inlet_margin_px": 5.0,
                                "outlet_margin_px": 5.0,
                            }
                        },
                    }
                },
                "device": {
                    "device": {
                        "id": "asymmetric_loop_h100",
                        "calibration": {"um_per_px": 4.0},
                        "channel": {"width_um": 100.0, "height_um": 100.0},
                        "geometry": {"region_metadata_path": str(metadata_path)},
                    }
                },
            }

    monkeypatch.setattr(loader, "load_experiment_config", fake_load_experiment_config)
    monkeypatch.setattr(coordinate_mapping, "load_experiment_config", fake_load_experiment_config)
    monkeypatch.setattr("src.physics.enrichment.tracking_enricher.load_experiment_config", fake_load_experiment_config)
    monkeypatch.setattr("src.physics.enrichment.tracking_enricher.VelocityFieldLibrary", SimpleNamespace(from_directory=lambda path: _FakeLibrary()))

    config = EnrichmentConfig(
        experiment_id="video_2",
        experiment_config_path=experiment_path,
        tracking_path=tracking_path,
        hydraulic_state_path=hydraulic_path,
        cfd_library_path=tmp_path / "library",
        occupancy_path=occupancy_path,
        output_root=tmp_path / "enrichment",
    )
    before_bytes = tracking_path.read_bytes()
    first, first_summary = build_physics_enriched_tracking(config, overwrite=True)
    second, second_summary = build_physics_enriched_tracking(config, overwrite=True)

    assert tracking_path.read_bytes() == before_bytes
    assert len(first) == len(tracking) - 1
    assert list(first.columns[: len(tracking.columns)]) == list(tracking.columns)
    pd.testing.assert_frame_equal(first, second)
    assert first_summary.row_count == second_summary.row_count == 2
    assert first_summary.column_count == second_summary.column_count
    assert first_summary.acquisition_domain_trim["method"] == "leading_inlet_and_terminal_outlet_geometric_cutoff"
    assert first_summary.acquisition_domain_trim["outlet"]["rows_removed"] == 1
    assert first_summary.acquisition_domain_trim["outlet"]["affected_tracks"] == 1
    assert not first[["frame", "track_id"]].duplicated().any()
    assert first["superficial_velocity"].nunique() == 1
    assert first["superficial_velocity"].iloc[0] == pytest.approx(56.94444444444444)
    assert first["cfd_valid"].all()
    assert first["cfd_valid"].equals(first["inside_cfd_domain"])
    assert first.loc[~first["inside_cfd_domain"], "background_speed_m_per_s"].isna().all()


def test_superficial_velocity_comes_from_experiment_and_device_config(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_load_experiment_config(experiment_path_arg, configs_root="configs"):
        return {
            "experiment": {
                "experiment": {
                    "id": "video_2",
                    "device_id": "asymmetric_loop_h100",
                    "phases": {
                        "continuous": {"flow_rate_ul_per_hr": 1950.0},
                        "dispersed": {"flow_rate_ul_per_hr": 100.0},
                    },
                }
            },
            "device": {"device": {"id": "asymmetric_loop_h100", "channel": {"width_um": 100.0, "height_um": 100.0}}},
        }

    monkeypatch.setattr("src.physics.enrichment.tracking_enricher.load_experiment_config", fake_load_experiment_config)
    assert compute_inlet_superficial_velocity_mm_s("unused.yml") == pytest.approx(56.94444444444444)


def test_acquisition_trim_removes_inlet_prefix_and_outlet_suffix_without_affecting_other_tracks() -> None:
    table = _trim_table()
    trimmed, report = _trim_for_test(table)

    assert report["inlet"]["rows_removed"] == 2
    assert report["outlet"]["rows_removed"] == 2
    assert report["outlet"]["removed_suffix_length"] == {"min": 2, "median": 2.0, "max": 2, "mean": 2.0}
    assert trimmed["cfd_valid"].all()
    assert trimmed.loc[trimmed["track_id"] == 1, "frame"].tolist() == [2]
    assert trimmed.loc[trimmed["track_id"] == 2, "frame"].tolist() == [1, 2]
    assert trimmed.loc[trimmed["track_id"] == 3, "frame"].tolist() == [0]


def test_outlet_cutoff_then_upstream_jitter_removes_terminal_suffix() -> None:
    table = _trim_table()
    table.loc[(table["track_id"] == 1) & (table["frame"] == 4), ["cfd_valid", "inside_cfd_domain", "y_device_um"]] = [True, True, 55.0]

    trimmed, report = _trim_for_test(table)

    assert report["outlet"]["rows_removed"] == 2
    assert trimmed.loc[trimmed["track_id"] == 1, "frame"].tolist() == [2]


def test_inlet_jitter_after_entering_interior_does_not_cut_middle() -> None:
    table = _trim_table()
    table.loc[(table["track_id"] == 1) & (table["frame"] == 3), ["cfd_valid", "inside_cfd_domain", "dominant_region", "y_device_um"]] = [True, True, "inlet", 2368.0]

    trimmed, _ = _trim_for_test(table)

    assert trimmed.loc[trimmed["track_id"] == 1, "frame"].tolist() == [2, 3]


def test_region_safeguards_prevent_non_inlet_or_non_outlet_removal() -> None:
    table = _trim_table()
    table.loc[(table["track_id"] == 2) & (table["frame"] == 1), ["dominant_region", "y_device_um"]] = ["left", 2380.0]
    table.loc[(table["track_id"] == 2) & (table["frame"] == 2), ["dominant_region", "y_device_um"]] = ["right", 10.0]

    trimmed, _ = _trim_for_test(table)

    assert trimmed.loc[trimmed["track_id"] == 2, "frame"].tolist() == [1, 2]


def test_tracks_entirely_within_excluded_margins_are_removed() -> None:
    table = _trim_table()
    extra = pd.DataFrame(
        {
            "track_id": [4, 4],
            "frame": [0, 1],
            "cfd_valid": [True, True],
            "inside_cfd_domain": [True, True],
            "x_device_um": [30.0, 31.0],
            "y_device_um": [2380.0, 2370.0],
            "x_cfd_um": [30.0, 31.0],
            "y_cfd_um": [4.0, 14.0],
            "dominant_region": ["inlet", "inlet"],
        }
    )
    table = pd.concat([table, extra], ignore_index=True)

    trimmed, report = _trim_for_test(table)

    assert 4 not in set(trimmed["track_id"])
    assert report["tracks_removed_entirely"]["total"] == 1
    assert report["tracks_removed_entirely"]["inlet_only"] == 1


def _trim_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "track_id": [1, 1, 1, 1, 1, 2, 2, 3],
            "frame": [0, 1, 2, 3, 4, 1, 2, 0],
            "cfd_valid": [True, True, True, False, False, True, True, True],
            "inside_cfd_domain": [True, True, True, False, False, True, True, True],
            "x_device_um": [10.0, 11.0, 12.0, 13.0, 14.0, 20.0, 21.0, 30.0],
            "y_device_um": [2380.0, 2370.0, 100.0, 44.0, 30.0, 80.0, 70.0, 2368.0],
            "x_cfd_um": [10.0, 11.0, 12.0, 13.0, 14.0, 20.0, 21.0, 30.0],
            "y_cfd_um": [4.0, 14.0, 2284.0, 2340.0, 2354.0, 2020.0, 2030.0, 16.0],
            "dominant_region": ["inlet", "inlet", "outlet", "outlet", "outlet", "left", "left", "left"],
        }
    )


def _trim_for_test(table: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    return trim_acquisition_domain_by_cutoffs(
        table,
        inlet_end_y_device_um=2384.0,
        outlet_end_y_device_um=24.0,
        inlet_margin_px=5.0,
        outlet_margin_px=5.0,
        um_per_px=4.0,
        inlet_cutoff_y_device_um=2364.0,
        outlet_cutoff_y_device_um=44.0,
    )


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
        self.direction_x = self.u_x_m_per_s / self.speed_m_per_s
        self.direction_y = self.u_y_m_per_s / self.speed_m_per_s
        self.inlet_reference_velocity_m_per_s = 2.0

    @property
    def cfd_u(self) -> np.ndarray:
        return self.u_x_m_per_s

    @property
    def cfd_v(self) -> np.ndarray:
        return self.u_y_m_per_s

    @property
    def cfd_speed(self) -> np.ndarray:
        return self.speed_m_per_s

    @property
    def cfd_u_norm(self) -> np.ndarray:
        return self.u_x_m_per_s / self.inlet_reference_velocity_m_per_s

    @property
    def cfd_v_norm(self) -> np.ndarray:
        return self.u_y_m_per_s / self.inlet_reference_velocity_m_per_s

    @property
    def cfd_speed_norm(self) -> np.ndarray:
        return self.speed_m_per_s / self.inlet_reference_velocity_m_per_s

    @property
    def cfd_dir_x(self) -> np.ndarray:
        return self.direction_x

    @property
    def cfd_dir_y(self) -> np.ndarray:
        return self.direction_y

    @property
    def cfd_valid(self) -> np.ndarray:
        return self.inside_domain


class _FakeField:
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.inlet_reference_velocity_m_per_s = 2.0

    def sample(self, points: np.ndarray) -> _FakeSample:
        return _FakeSample(points, self.alpha)

    def sample_cfd(self, points: np.ndarray) -> _FakeSample:
        return self.sample(points)


class _FakeLibrary:
    def __init__(self) -> None:
        mesh = SimpleNamespace(
            geometry=SimpleNamespace(coordinate_frame="device_cartesian_y_up"),
            nodes_um=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            elements=np.array([[0, 1, 2]]),
        )
        self.fractions = (0.0, 0.5, 1.0)
        self.inlet_reference_velocity_m_per_s = 2.0
        self.cases = [
            SimpleNamespace(left_fraction=0.0, cfd_version="1.0", mesh_version="production_v1", mesh=mesh, inlet_reference_velocity_m_per_s=2.0),
            SimpleNamespace(left_fraction=0.5, cfd_version="1.0", mesh_version="production_v1", mesh=mesh, inlet_reference_velocity_m_per_s=2.0),
            SimpleNamespace(left_fraction=1.0, cfd_version="1.0", mesh_version="production_v1", mesh=mesh, inlet_reference_velocity_m_per_s=2.0),
        ]

    def interpolate(self, left_fraction: float) -> _FakeField:
        return _FakeField(float(left_fraction))
