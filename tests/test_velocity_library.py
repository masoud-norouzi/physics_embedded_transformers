import json
from pathlib import Path

from src.physics.cfd.velocity_library import (
    _existing_case_is_valid,
    _monotonicity_checks,
    split_case_id,
    split_grid,
)


def test_velocity_library_split_grid_is_version_1_grid() -> None:
    fractions = split_grid()

    assert len(fractions) == 19
    assert fractions[0] == 0.05
    assert fractions[-1] == 0.95
    assert 0.50 in fractions
    assert 0.0 not in fractions
    assert 1.0 not in fractions


def test_velocity_library_output_directory_naming() -> None:
    assert split_case_id(0.05) == "split_0p05"
    assert split_case_id(0.50) == "split_0p50"
    assert split_case_id(0.95) == "split_0p95"


def test_existing_case_skip_requires_valid_metadata_and_flux_report(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "solution_metadata.json").write_text(
        json.dumps({"cfd_version": "1.0", "mesh_version": "production_v1", "requested_left_fraction": 0.25}),
        encoding="utf-8",
    )
    (reports / "flux_report.json").write_text(json.dumps({"validation_status": "passed"}), encoding="utf-8")

    assert _existing_case_is_valid(tmp_path, 0.25)
    assert not _existing_case_is_valid(tmp_path, 0.30)


def test_velocity_library_monotonicity_checks_pass_for_ordered_records() -> None:
    records = []
    for left in split_grid():
        records.append(
            {
                "left_fraction": left,
                "right_fraction": 1.0 - left,
                "left_outlet_flux": left,
                "right_outlet_flux": 1.0 - left,
                "measured_left_fraction": left,
                "measured_right_fraction": 1.0 - left,
                "maximum_velocity": 1.0,
                "pressure_range": 1.0 + left,
                "separatrix_seed_index": int((1.0 - left) * 100),
            }
        )

    checks = _monotonicity_checks(records)

    assert all(checks.values())


def test_generated_velocity_library_index_contains_valid_records_when_present() -> None:
    index_path = Path("outputs/physics/junction_cfd/solutions/library_index.json")
    if not index_path.exists():
        return

    index = json.loads(index_path.read_text(encoding="utf-8"))

    assert len(index["records"]) == 19
    assert all(record["validation_status"] == "passed" for record in index["records"])
    assert max(abs(record["measured_left_fraction"] - record["left_fraction"]) for record in index["records"]) < 1.0e-6
    assert all(index["monotonicity_checks"].values())
