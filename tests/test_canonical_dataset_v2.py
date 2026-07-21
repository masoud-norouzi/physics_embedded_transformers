from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.datasets.canonical_dataset_builder import (
    CANONICAL_V2_FEATURE_NAMES,
    CanonicalDatasetBuilder,
    CanonicalDatasetV2Builder,
)


def test_v2_preserves_track_continuity_frame_ordering_masks_and_padding(tmp_path: Path) -> None:
    v1_csv, v2_csv, occ_csv = _write_synthetic_inputs(tmp_path)
    v1_npz = tmp_path / "canonical_v1.npz"
    v2_npz = tmp_path / "canonical_dataset_v2" / "canonical_dataset_v2.npz"

    CanonicalDatasetBuilder(v1_csv, v1_npz).run()
    CanonicalDatasetV2Builder(v2_csv, v2_npz, occupancy_csv=occ_csv).run()

    v1 = np.load(v1_npz)
    v2 = np.load(v2_npz)
    assert np.array_equal(v2["track_ids"], v1["track_ids"])
    assert np.array_equal(v2["frames"], v1["frames"])
    assert np.array_equal(v2["mask"], v1["mask"])
    assert v2["Z"].shape[:2] == v1["Z"].shape[:2]
    assert v2["Z"].shape[2] == len(CANONICAL_V2_FEATURE_NAMES)
    assert [str(name) for name in v2["feature_names"]] == CANONICAL_V2_FEATURE_NAMES
    assert np.isnan(v2["Z"][1, 0]).all()


def test_v2_feature_alignment_and_invalid_cfd_neutralization(tmp_path: Path) -> None:
    _, v2_csv, occ_csv = _write_synthetic_inputs(tmp_path)
    output = tmp_path / "canonical_dataset_v2" / "canonical_dataset_v2.npz"
    metadata = output.with_suffix(".metadata.json")

    CanonicalDatasetV2Builder(v2_csv, output, occupancy_csv=occ_csv, metadata_json=metadata).run()

    data = np.load(output)
    features = [str(name) for name in data["feature_names"]]
    idx = {name: i for i, name in enumerate(features)}
    Z = data["Z"]
    track_ids = data["track_ids"]
    frames = data["frames"]
    track_index = {track_id: i for i, track_id in enumerate(track_ids)}
    frame_index = {frame: i for i, frame in enumerate(frames)}

    row = Z[track_index[2], frame_index[2]]
    assert row[idx["cfd_valid"]] == 0.0
    assert row[idx["cfd_u"]] == 0.0
    assert row[idx["cfd_v"]] == 0.0

    occupancy = row[
        [
            idx["occupancy_inlet_channel"],
            idx["occupancy_inlet_junction"],
            idx["occupancy_left_branch"],
            idx["occupancy_right_branch"],
            idx["occupancy_outlet_junction"],
            idx["occupancy_outlet_channel"],
        ]
    ]
    assert np.isclose(occupancy.sum(), 1.0)
    assert row[idx["left_flow_fraction"]] == np.float32(0.52)
    assert json.loads(metadata.read_text(encoding="utf-8"))["dataset_version"] == "canonical_dataset_v2"


def test_v2_output_shapes_match_previous_builder_contract(tmp_path: Path) -> None:
    v1_csv, v2_csv, occ_csv = _write_synthetic_inputs(tmp_path)
    v1_npz = tmp_path / "canonical_v1.npz"
    v2_npz = tmp_path / "canonical_dataset_v2" / "canonical_dataset_v2.npz"

    CanonicalDatasetBuilder(v1_csv, v1_npz).run()
    summary = CanonicalDatasetV2Builder(v2_csv, v2_npz, occupancy_csv=occ_csv).run()

    v1 = np.load(v1_npz)
    v2 = np.load(v2_npz)
    assert v2["mask"].shape == v1["mask"].shape == (2, 4)
    assert v2["track_ids"].shape == v1["track_ids"].shape == (2,)
    assert v2["frames"].shape == v1["frames"].shape == (4,)
    assert summary["num_tracks"] == 2
    assert summary["num_frames"] == 4
    assert summary["feature_count"] == 15


def _write_synthetic_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    rows = [
        {"frame": 0, "track_id": 1, "centroid_x": 10.0, "centroid_y": 50.0, "circularity": 0.91, "cfd_u": 1.0, "cfd_v": 0.1, "cfd_valid": True, "left_flow_fraction": 0.50},
        {"frame": 1, "track_id": 1, "centroid_x": 11.0, "centroid_y": 51.0, "circularity": 0.92, "cfd_u": 1.1, "cfd_v": 0.1, "cfd_valid": True, "left_flow_fraction": 0.51},
        {"frame": 3, "track_id": 1, "centroid_x": 13.0, "centroid_y": 53.0, "circularity": 0.94, "cfd_u": 1.3, "cfd_v": 0.1, "cfd_valid": True, "left_flow_fraction": 0.53},
        {"frame": 1, "track_id": 2, "centroid_x": 20.0, "centroid_y": 70.0, "circularity": 0.81, "cfd_u": 2.0, "cfd_v": 0.2, "cfd_valid": True, "left_flow_fraction": 0.51},
        {"frame": 2, "track_id": 2, "centroid_x": 21.0, "centroid_y": 71.0, "circularity": 0.82, "cfd_u": np.nan, "cfd_v": np.nan, "cfd_valid": False, "left_flow_fraction": 0.52},
        {"frame": 3, "track_id": 2, "centroid_x": 22.0, "centroid_y": 72.0, "circularity": 0.83, "cfd_u": 2.2, "cfd_v": 0.2, "cfd_valid": True, "left_flow_fraction": 0.53},
    ]
    enriched = pd.DataFrame(rows)
    v2_csv = tmp_path / "physics_enriched_tracked_features.csv"
    enriched.to_csv(v2_csv, index=False)

    v1_csv = tmp_path / "tracked_features.csv"
    enriched[["frame", "track_id", "centroid_x", "centroid_y", "circularity"]].to_csv(v1_csv, index=False)

    occ = enriched[["frame", "track_id"]].copy()
    occ["w_inlet"] = 0.2
    occ["w_upper_junction"] = 0.1
    occ["w_left"] = 0.2
    occ["w_right"] = 0.2
    occ["w_lower_junction"] = 0.1
    occ["w_outlet"] = 0.2
    occ_csv = tmp_path / "droplet_occupancy.csv"
    occ.to_csv(occ_csv, index=False)
    return v1_csv, v2_csv, occ_csv
