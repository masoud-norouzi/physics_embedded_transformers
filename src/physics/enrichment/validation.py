from __future__ import annotations

import numpy as np
import pandas as pd


def validate_hydraulic_state(hydraulic: pd.DataFrame) -> None:
    required = {
        "frame",
        "left_flow_ul_hr",
        "right_flow_ul_hr",
        "total_mixture_input_flow_ul_hr",
        "left_velocity_um_s",
        "right_velocity_um_s",
    }
    missing = required.difference(hydraulic.columns)
    if missing:
        raise ValueError(f"Hydraulic state is missing required columns: {sorted(missing)}")
    if hydraulic["frame"].duplicated().any():
        duplicated = hydraulic.loc[hydraulic["frame"].duplicated(), "frame"].head().tolist()
        raise ValueError(f"Hydraulic state has duplicate frame rows: {duplicated}")
    finite_columns = sorted(required - {"frame"})
    if not np.isfinite(hydraulic[finite_columns].to_numpy(float)).all():
        raise ValueError("Hydraulic state contains non-finite required values")


def validate_tracking_hydraulic_join(tracking: pd.DataFrame, hydraulic: pd.DataFrame, joined: pd.DataFrame) -> None:
    if "frame" not in tracking.columns:
        raise ValueError("Tracking table is missing frame column")
    missing_frames = sorted(set(tracking["frame"].unique()).difference(set(hydraulic["frame"].unique())))
    if missing_frames:
        preview = missing_frames[:10]
        raise ValueError(f"Missing hydraulic state for tracked frames: {preview}")
    if len(joined) != len(tracking):
        raise ValueError(f"Hydraulic join changed row count from {len(tracking)} to {len(joined)}")
    if not joined["frame"].equals(tracking["frame"]):
        raise ValueError("Hydraulic join changed frame ordering")
    fractions = joined[["left_flow_fraction", "right_flow_fraction"]].to_numpy(float)
    if not np.isfinite(fractions).all():
        raise ValueError("Joined flow fractions must be finite")
    if not np.allclose(fractions.sum(axis=1), 1.0, atol=1.0e-10, rtol=0.0):
        raise ValueError("left_flow_fraction + right_flow_fraction must equal 1")


def validate_row_preservation(original: pd.DataFrame, enriched: pd.DataFrame) -> None:
    if len(original) != len(enriched):
        raise ValueError("Enriched output row count does not match tracking input")
    for column in original.columns:
        if column not in enriched.columns:
            raise ValueError(f"Original tracking column was dropped: {column}")
        if pd.api.types.is_numeric_dtype(original[column]):
            left = original[column].to_numpy()
            right = enriched[column].to_numpy()
            if not np.array_equal(left, right):
                raise ValueError(f"Original numeric tracking column changed: {column}")
        elif not original[column].equals(enriched[column]):
            raise ValueError(f"Original tracking column changed: {column}")
    for key in ("frame", "track_id"):
        if key in original.columns and not original[key].equals(enriched[key]):
            raise ValueError(f"{key} values changed during enrichment")


def validate_sampled_fields(enriched: pd.DataFrame, alpha_tolerance: float = 1.0e-12) -> None:
    if "cfd_valid" in enriched.columns and not enriched["cfd_valid"].equals(enriched["inside_cfd_domain"]):
        raise ValueError("cfd_valid and inside_cfd_domain must match")
    inside = enriched["inside_cfd_domain"].to_numpy(bool)
    cfd_cols = ["cfd_u", "cfd_v", "cfd_speed", "cfd_dir_x", "cfd_dir_y"]
    if all(column in enriched.columns for column in cfd_cols):
        cfd_inside = enriched.loc[inside, cfd_cols[:3]].to_numpy(float)
        if len(cfd_inside) and not np.isfinite(cfd_inside).all():
            raise ValueError("Valid CFD rows must have finite cfd_u, cfd_v, and cfd_speed")
        cfd_outside = enriched.loc[~inside, cfd_cols].to_numpy(float)
        if len(cfd_outside) and not np.isnan(cfd_outside).all():
            raise ValueError("Invalid CFD rows must have NaN CFD values")
        speed = enriched["cfd_speed"].to_numpy(float)
        ux = enriched["cfd_u"].to_numpy(float)
        uy = enriched["cfd_v"].to_numpy(float)
        if np.any(inside):
            expected = np.sqrt(ux[inside] ** 2 + uy[inside] ** 2)
            if not np.allclose(speed[inside], expected, rtol=1.0e-12, atol=1.0e-14):
                raise ValueError("cfd_speed does not match cfd_u/cfd_v")
            nonzero = speed[inside] > 1.0e-14
            dirs = enriched.loc[inside, ["cfd_dir_x", "cfd_dir_y"]].to_numpy(float)
            if np.any(nonzero):
                norms = np.linalg.norm(dirs[nonzero], axis=1)
                if not np.allclose(norms, 1.0, atol=1.0e-10, rtol=0.0):
                    raise ValueError("CFD direction vectors are not normalized")
    velocity_cols = [
        "background_u_x_device_m_per_s",
        "background_u_y_device_m_per_s",
        "background_speed_m_per_s",
        "background_direction_x",
        "background_direction_y",
    ]
    inside_values = enriched.loc[inside, velocity_cols[:3]].to_numpy(float)
    if len(inside_values) and not np.isfinite(inside_values).all():
        raise ValueError("Inside-domain rows must have finite sampled velocity and speed")
    outside_values = enriched.loc[~inside, velocity_cols].to_numpy(float)
    if len(outside_values) and not np.isnan(outside_values).all():
        raise ValueError("Outside-domain background velocity fields must be NaN")
    speed = enriched["background_speed_m_per_s"].to_numpy(float)
    ux = enriched["background_u_x_device_m_per_s"].to_numpy(float)
    uy = enriched["background_u_y_device_m_per_s"].to_numpy(float)
    if np.any(inside):
        expected = np.sqrt(ux[inside] ** 2 + uy[inside] ** 2)
        if not np.allclose(speed[inside], expected, rtol=1.0e-12, atol=1.0e-14):
            raise ValueError("background_speed_m_per_s does not match velocity components")
        nonzero = speed[inside] > 1.0e-14
        dirs = enriched.loc[inside, ["background_direction_x", "background_direction_y"]].to_numpy(float)
        if np.any(nonzero):
            norms = np.linalg.norm(dirs[nonzero], axis=1)
            if not np.allclose(norms, 1.0, atol=1.0e-10, rtol=0.0):
                raise ValueError("Background direction vectors are not normalized")
    if not np.allclose(
        enriched["left_flow_fraction"].to_numpy(float),
        enriched["cfd_left_fraction"].to_numpy(float),
        atol=alpha_tolerance,
        rtol=0.0,
    ):
        raise ValueError("CFD requested alpha does not match hydraulic left_flow_fraction")


def validation_summary(original: pd.DataFrame, enriched: pd.DataFrame, hydraulic: pd.DataFrame) -> dict[str, object]:
    return {
        "row_count_preserved": len(original) == len(enriched),
        "original_columns_preserved": all(column in enriched.columns for column in original.columns),
        "hydraulic_rows_unique_by_frame": not hydraulic["frame"].duplicated().any(),
        "hydraulic_join_coverage": float(enriched[["left_flow_fraction", "right_flow_fraction"]].notna().all(axis=1).mean()),
        "duplicate_rows_introduced": len(enriched) != len(original),
        "sampled_field_semantics_validated": True,
    }
