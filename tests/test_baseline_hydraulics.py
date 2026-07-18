import numpy as np
import pandas as pd
import pytest

from src.hydraulics import CANONICAL_ML_FEATURES
from scripts.compute_baseline_hydraulics import _load_physical_constants
from src.hydraulics.baseline import (
    compute_baseline_hydraulic_state,
    compute_branch_flow_rates_ul_hr,
    compute_effective_branch_lengths_um,
    compute_effective_branch_occupancies,
    compute_frame_baseline_hydraulics,
    compute_isolated_droplet_equivalent_length_um,
    flow_rate_ul_hr_to_superficial_velocity_um_s,
)


def _frame(rows: list[dict]) -> pd.DataFrame:
    base = {
        "frame": 0,
        "track_id": 1,
        "occupancy_computable": True,
        "w_inlet": 0.0,
        "w_outlet": 0.0,
        "w_left": 0.0,
        "w_right": 0.0,
        "w_upper_junction": 0.0,
        "w_lower_junction": 0.0,
    }
    return pd.DataFrame([{**base, **row} for row in rows])


def test_isolated_droplet_equivalent_length() -> None:
    assert compute_isolated_droplet_equivalent_length_um(1491.0, 0.15) == pytest.approx(223.65)


def test_empty_channel_flow_split_favors_shorter_branch_and_conserves_flow() -> None:
    left, right = compute_branch_flow_rates_ul_hr(1960.0, 1791.0, 1491.0)
    assert right > left
    assert left + right == pytest.approx(1960.0)


def test_empty_channel_velocity_conversion() -> None:
    flow = 36.0
    assert flow_rate_ul_hr_to_superficial_velocity_um_s(flow, 100.0, 100.0) == pytest.approx(1000.0)


def test_increasing_left_occupancy_increases_left_resistance_and_decreases_left_flow() -> None:
    droplet_length = 223.65
    empty_left, empty_right = compute_effective_branch_lengths_um(1791.0, 1491.0, droplet_length, 0.0, 0.0)
    loaded_left, loaded_right = compute_effective_branch_lengths_um(1791.0, 1491.0, droplet_length, 2.0, 0.0)
    assert loaded_left > empty_left
    empty_flow_left, _ = compute_branch_flow_rates_ul_hr(1950.0, empty_left, empty_right)
    loaded_flow_left, _ = compute_branch_flow_rates_ul_hr(1950.0, loaded_left, loaded_right)
    assert loaded_flow_left < empty_flow_left


def test_increasing_right_occupancy_decreases_right_flow_and_velocity() -> None:
    droplet_length = 223.65
    empty_left, empty_right = compute_effective_branch_lengths_um(1791.0, 1491.0, droplet_length, 0.0, 0.0)
    loaded_left, loaded_right = compute_effective_branch_lengths_um(1791.0, 1491.0, droplet_length, 0.0, 2.0)
    _, empty_right_flow = compute_branch_flow_rates_ul_hr(1950.0, empty_left, empty_right)
    _, loaded_right_flow = compute_branch_flow_rates_ul_hr(1950.0, loaded_left, loaded_right)
    assert loaded_right > empty_right
    assert loaded_right_flow < empty_right_flow
    assert flow_rate_ul_hr_to_superficial_velocity_um_s(loaded_right_flow, 100, 100) < flow_rate_ul_hr_to_superficial_velocity_um_s(empty_right_flow, 100, 100)


def test_equal_effective_lengths_produce_equal_flow_and_velocity() -> None:
    left, right = compute_branch_flow_rates_ul_hr(100.0, 10.0, 10.0)
    assert left == pytest.approx(right)
    assert flow_rate_ul_hr_to_superficial_velocity_um_s(left, 10, 10) == pytest.approx(
        flow_rate_ul_hr_to_superficial_velocity_um_s(right, 10, 10)
    )


def test_fractional_occupancy_contributes_linearly() -> None:
    left, right = compute_effective_branch_lengths_um(100.0, 100.0, 20.0, 0.25, 0.5)
    assert left == pytest.approx(105.0)
    assert right == pytest.approx(110.0)


def test_junction_inlet_outlet_occupancy_does_not_alter_baseline_resistance() -> None:
    frame = _frame(
        [
            {"w_inlet": 1.0, "track_id": 1},
            {"w_outlet": 1.0, "track_id": 2},
            {"w_upper_junction": 1.0, "track_id": 3},
            {"w_lower_junction": 1.0, "track_id": 4},
        ]
    )
    n_left, n_right = compute_effective_branch_occupancies(frame)
    assert n_left == 0.0
    assert n_right == 0.0


def test_flow_conservation_holds_in_frame_state() -> None:
    result = compute_frame_baseline_hydraulics(
        _frame([{"w_left": 0.5, "w_right": 0.25}]),
        frame=0,
        left_length_um=1791.0,
        right_length_um=1491.0,
        droplet_equivalent_length_um=223.65,
        total_mixture_flow_ul_hr=1960.0,
        channel_width_um=100.0,
        channel_height_um=100.0,
        continuous_flow_ul_hr=1950.0,
        dispersed_flow_ul_hr=10.0,
    )
    assert result["flow_conservation_error_ul_hr"] == pytest.approx(0.0, abs=1e-9)


def test_frame_aggregation_sums_branch_occupancy() -> None:
    frame = _frame([{"w_left": 0.25, "w_right": 0.1}, {"w_left": 0.75, "w_right": 0.2, "track_id": 2}])
    assert compute_effective_branch_occupancies(frame) == pytest.approx((1.0, 0.3))


def test_output_has_one_row_per_unique_frame_and_diagnostic_columns() -> None:
    occupancy = pd.concat(
        [
            _frame([{"frame": 0, "w_left": 1.0}]),
            _frame([{"frame": 1, "w_right": 1.0}]),
            _frame([{"frame": 1, "w_left": 0.5, "track_id": 2}]),
        ],
        ignore_index=True,
    )
    state = compute_baseline_hydraulic_state(
        occupancy,
        left_length_um=100.0,
        right_length_um=80.0,
        droplet_equivalent_length_um=10.0,
        total_mixture_flow_ul_hr=100.0,
        channel_width_um=10.0,
        channel_height_um=10.0,
        continuous_flow_ul_hr=90.0,
        dispersed_flow_ul_hr=10.0,
    )
    assert len(state) == occupancy["frame"].nunique()
    for column in ["left_flow_ul_hr", "right_flow_ul_hr", "left_velocity_um_s", "right_velocity_um_s"]:
        assert column in state.columns
    for column in ["continuous_input_flow_ul_hr", "dispersed_input_flow_ul_hr", "total_mixture_input_flow_ul_hr"]:
        assert column in state.columns


def test_canonical_ml_feature_names_only_velocities() -> None:
    assert CANONICAL_ML_FEATURES == ["left_velocity_um_s", "right_velocity_um_s"]


def test_invalid_dimensions_negative_flow_and_noncomputable_rows_rejected() -> None:
    with pytest.raises(ValueError):
        flow_rate_ul_hr_to_superficial_velocity_um_s(1.0, 0.0, 10.0)
    with pytest.raises(ValueError):
        compute_branch_flow_rates_ul_hr(-1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        compute_effective_branch_lengths_um(1.0, 1.0, 1.0, -0.1, 0.0)
    bad = _frame([{"occupancy_computable": False}])
    with pytest.raises(ValueError):
        compute_effective_branch_occupancies(bad)


def _config_with_flows(continuous: float | None, dispersed: float | None) -> dict:
    phases = {}
    if continuous is not None:
        phases["continuous"] = {"flow_rate_ul_per_hr": continuous}
    if dispersed is not None:
        phases["dispersed"] = {"flow_rate_ul_per_hr": dispersed}
    return {
        "experiment": {"experiment": {"id": "exp", "phases": phases}},
        "device": {
            "device": {
                "loop": {
                    "branches": {
                        "left": {"length_um": 1791.0},
                        "right": {"length_um": 1491.0},
                    }
                },
                "hydraulics": {
                    "isolated_droplet_resistance": {"ratio_to_short_branch": 0.15}
                },
                "channel": {"width_um": 100.0, "height_um": 100.0},
            }
        },
    }


def test_total_mixture_flow_equals_continuous_plus_dispersed_and_video2_values() -> None:
    constants = _load_physical_constants(_config_with_flows(1950.0, 10.0))
    assert constants["continuous_flow_ul_hr"] == pytest.approx(1950.0)
    assert constants["dispersed_flow_ul_hr"] == pytest.approx(10.0)
    assert constants["total_mixture_flow_ul_hr"] == pytest.approx(1960.0)
    assert constants["dispersed_flow_fraction"] == pytest.approx(10.0 / 1960.0)


def test_nonzero_dispersed_flow_scales_flows_and_velocities_without_changing_fractions() -> None:
    left_1950, right_1950 = compute_branch_flow_rates_ul_hr(1950.0, 1791.0, 1491.0)
    left_1960, right_1960 = compute_branch_flow_rates_ul_hr(1960.0, 1791.0, 1491.0)
    assert left_1960 / left_1950 == pytest.approx(1960.0 / 1950.0)
    assert right_1960 / right_1950 == pytest.approx(1960.0 / 1950.0)
    assert left_1960 / 1960.0 == pytest.approx(left_1950 / 1950.0)
    assert right_1960 / 1960.0 == pytest.approx(right_1950 / 1950.0)
    assert flow_rate_ul_hr_to_superficial_velocity_um_s(left_1960, 100, 100) / flow_rate_ul_hr_to_superficial_velocity_um_s(left_1950, 100, 100) == pytest.approx(1960.0 / 1950.0)


def test_phase_flow_validation_allows_one_zero_but_rejects_bad_configs() -> None:
    assert _load_physical_constants(_config_with_flows(0.0, 10.0))["total_mixture_flow_ul_hr"] == pytest.approx(10.0)
    assert _load_physical_constants(_config_with_flows(10.0, 0.0))["total_mixture_flow_ul_hr"] == pytest.approx(10.0)
    with pytest.raises(ValueError, match="Total mixture flow"):
        _load_physical_constants(_config_with_flows(0.0, 0.0))
    with pytest.raises(ValueError, match="continuous"):
        _load_physical_constants(_config_with_flows(-1.0, 1.0))
    with pytest.raises(ValueError, match="dispersed"):
        _load_physical_constants(_config_with_flows(1.0, -1.0))
    with pytest.raises(ValueError, match="continuous"):
        _load_physical_constants(_config_with_flows(None, 1.0))
    with pytest.raises(ValueError, match="dispersed"):
        _load_physical_constants(_config_with_flows(1.0, None))


def test_current_empty_channel_values_reflect_1960_ul_hr() -> None:
    left, right = compute_branch_flow_rates_ul_hr(1960.0, 1791.0, 1491.0)
    assert left == pytest.approx(890.4204753199268)
    assert right == pytest.approx(1069.5795246800732)
    assert flow_rate_ul_hr_to_superficial_velocity_um_s(left, 100, 100) == pytest.approx(24733.90209222019)
    assert flow_rate_ul_hr_to_superficial_velocity_um_s(right, 100, 100) == pytest.approx(29710.542352224255)
