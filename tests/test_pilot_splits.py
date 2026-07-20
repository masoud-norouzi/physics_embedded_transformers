from pathlib import Path

from src.physics.cfd.pilot_splits import PILOT_LEFT_FRACTIONS


def test_pilot_split_set_is_limited_to_requested_cases() -> None:
    assert PILOT_LEFT_FRACTIONS == (0.10, 0.30, 0.90)


def test_pilot_split_outputs_are_separate_from_validated_50_50() -> None:
    roots = [Path("outputs/physics/junction_cfd/solutions") / f"split_0p{int(left * 100):02d}" for left in PILOT_LEFT_FRACTIONS]

    assert Path("outputs/physics/junction_cfd/solutions/split_0p50") not in roots
    assert len(set(roots)) == 3
