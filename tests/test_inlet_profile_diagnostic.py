from pathlib import Path

import numpy as np

from src.physics.cfd.inlet_profile_diagnostic import run_inlet_profile_diagnostic


CONFIG = Path("configs/physics/junction_cfd.yml")


def test_raw_fem_inlet_profile_diagnostic_writes_outputs(tmp_path: Path) -> None:
    diagnostics = run_inlet_profile_diagnostic(CONFIG, output_root=tmp_path, sample_count=51)

    assert len(diagnostics) == 3
    assert (tmp_path / "reports" / "inlet_profile_diagnostics.json").exists()
    assert (tmp_path / "reports" / "inlet_profile_diagnostics.md").exists()
    for item in diagnostics:
        assert (tmp_path / "samples" / f"{item.name}.csv").exists()
        assert (tmp_path / "figures" / f"{item.name}.png").exists()


def test_raw_fem_near_inlet_matches_prescribed_poiseuille_profile(tmp_path: Path) -> None:
    diagnostics = run_inlet_profile_diagnostic(CONFIG, output_root=tmp_path, sample_count=51)
    near_inlet = next(item for item in diagnostics if item.name == "near_inlet")

    assert near_inlet.downstream_sign_ok
    assert np.isfinite(near_inlet.mean_axial_velocity_m_per_s)
    assert np.isfinite(near_inlet.rmse_m_per_s)
    assert near_inlet.maximum_axial_velocity_m_per_s > 0
