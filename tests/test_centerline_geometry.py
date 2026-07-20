from pathlib import Path

import numpy as np
import pytest
import yaml

from src.config import load_experiment_config
from src.physics.geometry.centerlines import (
    build_device_geometry,
    cumulative_arc_length,
    order_branch_points,
    reconstruct_path_order,
    standardize_orientation,
    tangent_normal_vectors,
)


def test_cumulative_arc_length_simple_polyline() -> None:
    xy = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 4.0]])
    np.testing.assert_allclose(cumulative_arc_length(xy), [0.0, 5.0, 8.0])


def test_tangent_normal_properties() -> None:
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    tangents, normals = tangent_normal_vectors(xy)
    np.testing.assert_allclose(np.linalg.norm(tangents, axis=1), 1.0)
    np.testing.assert_allclose(np.sum(tangents * normals, axis=1), 0.0)
    np.testing.assert_allclose(normals, [[-0.0, 1.0], [-0.0, 1.0], [-0.0, 1.0]])


def test_branch_orientation_reversal() -> None:
    branches = {
        "inlet": np.array([[0.0, 0.0], [0.0, 1.0]]),
        "left": np.array([[-1.0, 3.0], [-1.0, 1.0]]),
        "right": np.array([[1.0, 3.0], [1.0, 1.0]]),
        "outlet": np.array([[0.0, 4.0], [0.0, 3.0]]),
    }
    oriented, upper, lower, _ = standardize_orientation(branches)
    np.testing.assert_allclose(upper, [0.0, 1.0])
    np.testing.assert_allclose(lower, [0.0, 3.0])
    assert oriented["left"][0, 1] == 1.0
    assert oriented["right"][0, 1] == 1.0
    assert oriented["outlet"][0, 1] == 3.0


def test_adjacency_ordering_unsorted_path() -> None:
    xy = np.array([[1.0, 0.0], [0.0, 0.0], [3.0, 0.0], [2.0, 0.0]])
    ordered = reconstruct_path_order(xy)
    np.testing.assert_allclose(cumulative_arc_length(ordered), [0.0, 1.0, 2.0, 3.0])


def test_reject_disconnected_branch() -> None:
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]])
    with pytest.raises(ValueError, match="single non-branching path"):
        reconstruct_path_order(xy)


def test_reject_forked_branch() -> None:
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="forked"):
        reconstruct_path_order(xy)


def test_experiment_to_device_config_resolution(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    (configs / "devices").mkdir(parents=True)
    (configs / "experiments").mkdir()
    (configs / "devices" / "dev.yml").write_text("device:\n  id: dev\n", encoding="utf-8")
    experiment = configs / "experiments" / "exp.yml"
    experiment.write_text("experiment:\n  id: exp\n  device_id: dev\n", encoding="utf-8")
    loaded = load_experiment_config(experiment, configs_root=configs)
    assert loaded["device"]["device"]["id"] == "dev"


def test_configured_branch_length_tolerance(tmp_path: Path) -> None:
    csv_path = tmp_path / "centerlines.csv"
    csv_path.write_text(
        "x,y,channel\n"
        "0,0,inlet\n0,1,inlet\n"
        "-1,1,left\n-1,2,left\n-1,3,left\n"
        "1,1,right\n1,2,right\n"
        "0,2,outlet\n0,3,outlet\n",
        encoding="utf-8",
    )
    device = yaml.safe_load(
        """
device:
  id: dev
  calibration:
    um_per_px: 1.0
  channel:
    width_px: 0.5
  loop:
    branches:
      left:
        length_px: 2.0
        length_um: 2.0
      right:
        length_px: 1.0
        length_um: 1.0
"""
    )
    build_device_geometry(csv_path, device, length_tolerance_px=0.01)
    device["device"]["loop"]["branches"]["left"]["length_px"] = 10.0
    with pytest.raises(ValueError, match="exceeding tolerance"):
        build_device_geometry(csv_path, device, length_tolerance_px=0.01)


def test_explicit_order_column() -> None:
    xy = np.array([[2.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    ordered = order_branch_points(xy, np.array([2, 0, 1]))
    np.testing.assert_allclose(ordered[:, 0], [0.0, 1.0, 2.0])
