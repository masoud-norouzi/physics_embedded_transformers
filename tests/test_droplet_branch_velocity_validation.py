import json

import numpy as np
import pandas as pd
import pytest

from scripts.validate_droplet_branch_velocity import (
    build_branch_interior_samples,
    build_candidate_table,
    build_pooled_samples,
    build_summary,
    calculate_speeds,
    prepare_tracked_velocity,
    pearson_r,
    save_scatter_plot,
    save_track_plot,
    select_candidates,
    spearman_r,
    validate_normalized_samples,
    validate_tracked_velocity,
)


def _tracks(rows: list[dict]) -> pd.DataFrame:
    base = {"frame": 0, "track_id": 1, "vx": 3.0, "vy": 4.0, "tracked_velocity_units": "px/frame"}
    return pd.DataFrame([{**base, **row} for row in rows])


def _position_tracks(track_id: int = 1, frames=None, xs=None, ys=None) -> pd.DataFrame:
    frames = [0, 1, 2] if frames is None else frames
    xs = [0.0, 1.0, 2.0] if xs is None else xs
    ys = [0.0, 0.0, 0.0] if ys is None else ys
    return pd.DataFrame({"frame": frames, "track_id": track_id, "centroid_x": xs, "centroid_y": ys})


def _occupancy(rows: list[dict]) -> pd.DataFrame:
    base = {"frame": 0, "track_id": 1, "occupancy_computable": True, "w_left": 1.0, "w_right": 0.0}
    return pd.DataFrame([{**base, **row} for row in rows])


def _hydraulic(rows: list[dict]) -> pd.DataFrame:
    base = {"frame": 0, "left_velocity_um_s": 10.0, "right_velocity_um_s": 20.0}
    return pd.DataFrame([{**base, **row} for row in rows])


def _candidate(track_id: int, branch: str, std: float, corr: float = 0.0, n: int = 120) -> dict:
    return {
        "track_id": track_id,
        "branch": branch,
        "frame_start": 0,
        "frame_end": n - 1,
        "n_samples": n,
        "mean_droplet_speed": 1.0,
        "std_droplet_speed": 1.0,
        "mean_branch_velocity_um_s": 2.0,
        "std_branch_velocity_um_s": std,
        "pearson_r": corr,
        "spearman_r": corr,
        "tracked_velocity_units": "px/frame",
    }


def test_speed_is_calculated_from_vx_vy() -> None:
    result = calculate_speeds(_tracks([{"vx": 3.0, "vy": 4.0}]))
    assert result.loc[0, "droplet_speed"] == pytest.approx(5.0)


def test_constant_x_motion_derives_expected_vx_and_speed() -> None:
    prepared, metadata = prepare_tracked_velocity(_position_tracks(xs=[0.0, 2.0, 4.0], ys=[5.0, 5.0, 5.0]))
    middle = prepared.loc[prepared["frame"] == 1].iloc[0]
    assert middle["vx"] == pytest.approx(2.0)
    assert middle["vy"] == pytest.approx(0.0)
    assert calculate_speeds(prepared).loc[prepared["frame"] == 1, "droplet_speed"].iloc[0] == pytest.approx(2.0)
    assert metadata["tracked_velocity_source"] == "centered_position_difference"
    assert metadata["tracked_velocity_units"] == "px/frame"


def test_constant_y_motion_derives_expected_vy_and_speed() -> None:
    prepared, _ = prepare_tracked_velocity(_position_tracks(xs=[1.0, 1.0, 1.0], ys=[0.0, 3.0, 6.0]))
    middle = calculate_speeds(prepared).loc[prepared["frame"] == 1].iloc[0]
    assert middle["vx"] == pytest.approx(0.0)
    assert middle["vy"] == pytest.approx(3.0)
    assert middle["droplet_speed"] == pytest.approx(3.0)


def test_diagonal_motion_derives_speed_from_components() -> None:
    prepared, _ = prepare_tracked_velocity(_position_tracks(xs=[0.0, 3.0, 6.0], ys=[0.0, 4.0, 8.0]))
    middle = calculate_speeds(prepared).loc[prepared["frame"] == 1].iloc[0]
    assert middle["vx"] == pytest.approx(3.0)
    assert middle["vy"] == pytest.approx(4.0)
    assert middle["droplet_speed"] == pytest.approx(5.0)


def test_centered_difference_aligns_velocity_with_middle_frame_and_excludes_endpoints() -> None:
    prepared, _ = prepare_tracked_velocity(_position_tracks(frames=[10, 11, 12, 13], xs=[0, 2, 4, 6], ys=[0, 0, 0, 0]))
    assert prepared.loc[prepared["frame"].isin([10, 13]), ["vx", "vy"]].isna().all().all()
    assert prepared.loc[prepared["frame"].isin([11, 12]), "vx"].tolist() == [2.0, 2.0]


def test_samples_adjacent_to_frame_gap_are_excluded_from_derived_velocity() -> None:
    prepared, _ = prepare_tracked_velocity(_position_tracks(frames=[0, 1, 3, 4], xs=[0, 1, 3, 4], ys=[0, 0, 0, 0]))
    assert prepared[["vx", "vy"]].isna().all().all()


def test_track_boundaries_are_never_mixed() -> None:
    tracks = pd.concat(
        [
            _position_tracks(track_id=1, frames=[0, 1], xs=[0, 1], ys=[0, 0]),
            _position_tracks(track_id=2, frames=[2, 3, 4], xs=[100, 103, 106], ys=[0, 0, 0]),
        ],
        ignore_index=True,
    )
    prepared, _ = prepare_tracked_velocity(tracks)
    assert prepared.loc[prepared["track_id"] == 1, "vx"].isna().all()
    assert prepared.loc[(prepared["track_id"] == 2) & (prepared["frame"] == 3), "vx"].iloc[0] == pytest.approx(3.0)


def test_existing_vx_vy_columns_are_preferred_when_present() -> None:
    prepared, metadata = prepare_tracked_velocity(
        pd.DataFrame(
            {
                "frame": [0, 1, 2],
                "track_id": [1, 1, 1],
                "centroid_x": [0.0, 100.0, 200.0],
                "centroid_y": [0.0, 0.0, 0.0],
                "vx": [7.0, 7.0, 7.0],
                "vy": [8.0, 8.0, 8.0],
                "tracked_velocity_units": ["custom"] * 3,
            }
        )
    )
    assert prepared["vx"].tolist() == [7.0, 7.0, 7.0]
    assert metadata["tracked_velocity_source"] == "existing_velocity_columns"
    assert metadata["tracked_velocity_units"] == "custom"


def test_position_derived_velocity_is_used_when_vx_vy_absent() -> None:
    prepared, metadata = prepare_tracked_velocity(_position_tracks())
    assert metadata["tracked_velocity_source"] == "centered_position_difference"
    assert prepared.loc[prepared["frame"] == 1, "vx"].iloc[0] == pytest.approx(1.0)


def test_left_interior_samples_receive_left_velocity() -> None:
    samples = build_branch_interior_samples(_tracks([{}]), _occupancy([{}]), _hydraulic([{}]))
    assert samples.loc[0, "branch"] == "left"
    assert samples.loc[0, "calculated_branch_velocity_um_s"] == pytest.approx(10.0)


def test_right_interior_samples_receive_right_velocity() -> None:
    samples = build_branch_interior_samples(
        _tracks([{}]),
        _occupancy([{"w_left": 0.0, "w_right": 1.0}]),
        _hydraulic([{}]),
    )
    assert samples.loc[0, "branch"] == "right"
    assert samples.loc[0, "calculated_branch_velocity_um_s"] == pytest.approx(20.0)


def test_junction_and_transition_samples_are_excluded() -> None:
    samples = build_branch_interior_samples(
        _tracks([{"frame": 0}, {"frame": 1}, {"frame": 2}]),
        _occupancy(
            [
                {"frame": 0, "w_left": 0.94, "w_right": 0.0},
                {"frame": 1, "w_left": 0.5, "w_right": 0.5},
                {"frame": 2, "w_left": 0.0, "w_right": 0.94},
            ]
        ),
        _hydraulic([{"frame": 0}, {"frame": 1}, {"frame": 2}]),
    )
    assert samples.empty


def test_ambiguous_dual_branch_samples_are_rejected() -> None:
    samples = build_branch_interior_samples(
        _tracks([{}]),
        _occupancy([{"w_left": 0.96, "w_right": 0.96}]),
        _hydraulic([{}]),
    )
    assert samples.empty


def test_per_track_z_score_has_zero_mean_and_unit_sample_std() -> None:
    branch_samples = pd.DataFrame(
        {
            "frame": np.arange(5),
            "track_id": 1,
            "branch": "left",
            "droplet_speed": [1, 2, 3, 4, 5],
            "calculated_branch_velocity_um_s": [2, 4, 6, 8, 10],
            "tracked_velocity_units": "px/frame",
            "w_left": 1.0,
            "w_right": 0.0,
        }
    )
    candidates, normalized = build_candidate_table(branch_samples, minimum_branch_interior_samples=5)
    pooled = build_pooled_samples(candidates.assign(selection_rank=[1]), normalized)
    assert pooled["droplet_speed_z"].mean() == pytest.approx(0.0)
    assert pooled["branch_velocity_z"].mean() == pytest.approx(0.0)
    assert pooled["droplet_speed_z"].std(ddof=1) == pytest.approx(1.0)
    assert pooled["branch_velocity_z"].std(ddof=1) == pytest.approx(1.0)


def test_zero_variance_tracks_are_rejected() -> None:
    branch_samples = pd.DataFrame(
        {
            "frame": np.arange(5),
            "track_id": 1,
            "branch": "left",
            "droplet_speed": [1, 1, 1, 1, 1],
            "calculated_branch_velocity_um_s": [2, 3, 4, 5, 6],
            "tracked_velocity_units": "px/frame",
            "w_left": 1.0,
            "w_right": 0.0,
        }
    )
    candidates, _ = build_candidate_table(branch_samples, minimum_branch_interior_samples=5)
    assert candidates.empty


def test_candidate_ranking_is_deterministic_and_not_by_correlation() -> None:
    candidates = pd.DataFrame(
        [_candidate(i, "left", std=100 - i, corr=-0.9) for i in range(1, 6)]
        + [_candidate(i, "right", std=100 - i, corr=0.99) for i in range(6, 11)]
    )
    selected = select_candidates(candidates)
    assert selected["track_id"].tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert selected.loc[selected["branch"] == "left", "pearson_r"].tolist() == [-0.9] * 5


def test_five_left_five_right_balancing_when_available() -> None:
    candidates = pd.DataFrame(
        [_candidate(i, "left", std=20 - i) for i in range(1, 8)]
        + [_candidate(i, "right", std=20 - i) for i in range(8, 15)]
    )
    selected = select_candidates(candidates)
    assert (selected["branch"] == "left").sum() == 5
    assert (selected["branch"] == "right").sum() == 5


def test_selection_fallback_when_one_branch_has_fewer_than_five() -> None:
    candidates = pd.DataFrame(
        [_candidate(i, "left", std=20 - i) for i in range(1, 4)]
        + [_candidate(i, "right", std=20 - i) for i in range(4, 14)]
    )
    selected = select_candidates(candidates)
    assert (selected["branch"] == "left").sum() == 3
    assert (selected["branch"] == "right").sum() == 7


def test_at_most_one_segment_per_track_when_unique_tracks_available() -> None:
    candidates = pd.DataFrame(
        [_candidate(1, "left", std=100), _candidate(1, "right", std=99)]
        + [_candidate(i, "left" if i % 2 else "right", std=90 - i) for i in range(2, 12)]
    )
    selected = select_candidates(candidates)
    assert selected["track_id"].nunique() == 10


def test_pearson_and_spearman_are_calculated_correctly() -> None:
    x = pd.Series([1, 2, 3, 4])
    y = pd.Series([2, 4, 6, 8])
    assert pearson_r(x, y) == pytest.approx(1.0)
    assert spearman_r(x, y) == pytest.approx(1.0)
    assert pearson_r(x, -y) == pytest.approx(-1.0)
    assert spearman_r(x, -y) == pytest.approx(-1.0)


def test_pooled_sample_output_contains_only_selected_tracks() -> None:
    selected = pd.DataFrame([_candidate(1, "left", 1.0), _candidate(2, "right", 1.0)])
    selected.insert(0, "selection_rank", [1, 2])
    normalized = {
        (1, "left"): pd.DataFrame(
            {"track_id": [1], "branch": ["left"], "frame": [0], "droplet_speed": [1], "calculated_branch_velocity_um_s": [2], "droplet_speed_z": [0], "branch_velocity_z": [0]}
        ),
        (2, "right"): pd.DataFrame(
            {"track_id": [2], "branch": ["right"], "frame": [0], "droplet_speed": [1], "calculated_branch_velocity_um_s": [2], "droplet_speed_z": [0], "branch_velocity_z": [0]}
        ),
    }
    pooled = build_pooled_samples(selected, normalized)
    assert set(pooled["track_id"]) == {1, 2}


def test_exactly_ten_individual_plot_paths_are_produced(tmp_path) -> None:
    rows = []
    normalized = {}
    for idx in range(10):
        branch = "left" if idx < 5 else "right"
        row = _candidate(idx + 1, branch, 10 - idx)
        rows.append(row)
        normalized[(idx + 1, branch)] = pd.DataFrame(
            {
                "frame": np.arange(3),
                "droplet_speed_z": [-1.0, 0.0, 1.0],
                "branch_velocity_z": [-1.0, 0.0, 1.0],
            }
        )
    selected = select_candidates(pd.DataFrame(rows))
    paths = [
        save_track_plot(row, normalized[(int(row.track_id), str(row.branch))], tmp_path)
        for row in selected.itertuples(index=False)
    ]
    assert len(paths) == 10
    assert len(list(tmp_path.glob("track_*_normalized_velocity.png"))) == 10


def test_summary_metadata_records_normalization_and_units(tmp_path) -> None:
    candidates = pd.DataFrame([_candidate(i, "left" if i < 6 else "right", 10 - i) for i in range(1, 11)])
    selected = select_candidates(candidates)
    pooled = pd.DataFrame(
        {"branch_velocity_z": [-1.0, 0.0, 1.0], "droplet_speed_z": [-1.0, 0.0, 1.0]}
    )
    stats = save_scatter_plot(pooled, selected, tmp_path / "scatter.png")
    config = {"experiment": {"experiment": {"id": "exp", "device_id": "dev"}}, "device": {"device": {"id": "dev"}}}
    summary = build_summary(
        config,
        tmp_path / "tracks.csv",
        "px/frame",
        "centered_position_difference",
        ["centroid_x", "centroid_y"],
        tmp_path / "occ.csv",
        tmp_path / "hyd.csv",
        0.95,
        100,
        candidates,
        selected,
        pooled,
        stats,
    )
    assert summary["normalization_method"] == "per selected (track_id, branch) z-score"
    assert summary["normalization_ddof"] == 1
    assert summary["tracked_velocity_units"] == "px/frame"
    assert summary["tracked_velocity_source"] == "centered_position_difference"
    assert "normalized temporal variation" in summary["unit_interpretation"]
    assert json.dumps(summary)


def test_duplicate_join_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        validate_tracked_velocity(_tracks([{}, {}]))


def test_missing_velocity_and_position_columns_fail_loudly() -> None:
    with pytest.raises(ValueError, match="supported centroid"):
        validate_tracked_velocity(pd.DataFrame({"frame": [0], "track_id": [1]}))


def test_normalized_comparison_proceeds_with_different_physical_units() -> None:
    tracks = _position_tracks(frames=[0, 1, 2, 3, 4], xs=[0, 1, 2, 4, 7], ys=[0, 0, 0, 0, 0])
    occ = pd.DataFrame(
        {
            "frame": [0, 1, 2, 3, 4],
            "track_id": [1] * 5,
            "occupancy_computable": [True] * 5,
            "w_left": [1.0] * 5,
            "w_right": [0.0] * 5,
        }
    )
    hyd = pd.DataFrame(
        {
            "frame": [0, 1, 2, 3, 4],
            "left_velocity_um_s": [10, 11, 12, 14, 17],
            "right_velocity_um_s": [20] * 5,
        }
    )
    samples = build_branch_interior_samples(tracks, occ, hyd)
    candidates, _ = build_candidate_table(samples, minimum_branch_interior_samples=3)
    assert len(candidates) == 1
    assert candidates.loc[0, "tracked_velocity_units"] == "px/frame"


def test_validate_normalized_samples_rejects_nonselected_track() -> None:
    selected = pd.DataFrame([_candidate(1, "left", 1.0)])
    pooled = pd.DataFrame(
        {
            "track_id": [2] * 100,
            "branch": ["left"] * 100,
            "droplet_speed_z": np.linspace(-1, 1, 100),
            "branch_velocity_z": np.linspace(-1, 1, 100),
        }
    )
    with pytest.raises(ValueError, match="non-selected"):
        validate_normalized_samples(pooled, selected)
