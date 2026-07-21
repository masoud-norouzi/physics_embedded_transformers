from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CANONICAL_V2_FEATURE_NAMES = [
    "x",
    "y",
    "vx",
    "vy",
    "circularity",
    "cfd_u",
    "cfd_v",
    "left_flow_fraction",
    "occupancy_inlet_channel",
    "occupancy_inlet_junction",
    "occupancy_left_branch",
    "occupancy_right_branch",
    "occupancy_outlet_junction",
    "occupancy_outlet_channel",
    "cfd_valid",
]


OCCUPANCY_COLUMNS = {
    "w_inlet": "occupancy_inlet_channel",
    "w_upper_junction": "occupancy_inlet_junction",
    "w_left": "occupancy_left_branch",
    "w_right": "occupancy_right_branch",
    "w_lower_junction": "occupancy_outlet_junction",
    "w_outlet": "occupancy_outlet_channel",
}


class CanonicalDatasetBuilder:
    """Builder copied from the previous canonical dataset format."""

    def __init__(self, input_csv, output_npz, feature_names=None, inlet_y_max_px=100.0):
        self.input_csv = Path(input_csv)
        self.output_npz = Path(output_npz)
        self.feature_names = feature_names or ["x", "y", "vx", "vy", "circularity"]
        self.inlet_y_max_px = float(inlet_y_max_px)
        self.inlet_velocity_diagnostics: dict[str, Any] = {}

    def run(self) -> dict[str, Any]:
        tracks = pd.read_csv(self.input_csv)
        prepared, diagnostics = self._prepare_tracks(tracks)

        interpolated = self._interpolate_tracks(prepared)
        interpolated = self._add_velocities(interpolated)
        interpolated = self._patch_new_inlet_velocities(interpolated)
        interpolated, post_diagnostics = self._postprocess_interpolated(interpolated)
        diagnostics.update(post_diagnostics)

        Z, mask, track_ids, frames = self._build_tensor(interpolated)
        self.output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.output_npz,
            Z=Z,
            mask=mask,
            track_ids=track_ids,
            frames=frames,
            feature_names=np.asarray(self.feature_names, dtype=str),
        )

        summary = self._summary(Z, mask, track_ids, frames, diagnostics)
        self._write_metadata(summary, interpolated)
        self._print_summary(summary)
        return summary

    def _prepare_tracks(self, tracks: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        tracks = self._standardize_columns(tracks)
        tracks = tracks[["frame", "track_id", "x", "y", "circularity"]]
        tracks = tracks.sort_values(["track_id", "frame"]).reset_index(drop=True)
        return tracks, {"input_rows": int(len(tracks)), "dropped_rows": 0, "dropped_tracks": 0, "drop_reasons": []}

    def _standardize_columns(self, tracks: pd.DataFrame) -> pd.DataFrame:
        rename_map = {}
        if "centroid_x" in tracks.columns and "x" not in tracks.columns:
            rename_map["centroid_x"] = "x"
        if "centroid_y" in tracks.columns and "y" not in tracks.columns:
            rename_map["centroid_y"] = "y"
        return tracks.rename(columns=rename_map)

    def _interpolate_tracks(self, tracks: pd.DataFrame) -> pd.DataFrame:
        filled_tracks = []
        for track_id, track in tracks.groupby("track_id", sort=True):
            track = track.sort_values("frame").set_index("frame")
            frame_index = np.arange(int(track.index.min()), int(track.index.max()) + 1)
            filled = track.reindex(frame_index)
            filled.index.name = "frame"
            filled["track_id"] = track_id
            value_columns = [column for column in tracks.columns if column not in {"frame", "track_id"}]
            filled[value_columns] = filled[value_columns].interpolate(method="linear")
            filled_tracks.append(filled.reset_index())
        if not filled_tracks:
            return pd.DataFrame(columns=tracks.columns)
        return pd.concat(filled_tracks, ignore_index=True)

    def _add_velocities(self, tracks: pd.DataFrame) -> pd.DataFrame:
        tracks = tracks.sort_values(["track_id", "frame"]).reset_index(drop=True)
        tracks["vx"] = tracks.groupby("track_id")["x"].diff()
        tracks["vy"] = tracks.groupby("track_id")["y"].diff()
        return tracks

    def _patch_new_inlet_velocities(self, tracks: pd.DataFrame) -> pd.DataFrame:
        tracks = tracks.sort_values(["track_id", "frame"]).reset_index(drop=True).copy()
        inlet_region = tracks["y"].le(self.inlet_y_max_px)
        finite_velocity = np.isfinite(tracks["vx"]) & np.isfinite(tracks["vy"])
        inlet_velocity_samples = tracks.loc[inlet_region & finite_velocity, ["vx", "vy"]]
        if inlet_velocity_samples.empty:
            raise ValueError(f"No finite inlet velocity observations found for y <= {self.inlet_y_max_px:g} px.")

        mean_inlet_vx = float(inlet_velocity_samples["vx"].mean())
        mean_inlet_vy = float(inlet_velocity_samples["vy"].mean())
        first_visible = tracks.groupby("track_id")["frame"].transform("min").eq(tracks["frame"])
        missing_vx = ~np.isfinite(tracks["vx"])
        missing_vy = ~np.isfinite(tracks["vy"])
        patch_rows = first_visible & inlet_region & (missing_vx | missing_vy)
        patch_vx = patch_rows & missing_vx
        patch_vy = patch_rows & missing_vy
        tracks.loc[patch_vx, "vx"] = mean_inlet_vx
        tracks.loc[patch_vy, "vy"] = mean_inlet_vy
        affected_tracks = tracks.loc[patch_rows, "track_id"].dropna().unique()
        self.inlet_velocity_diagnostics = {
            "inlet_y_max_px": self.inlet_y_max_px,
            "mean_inlet_vx": mean_inlet_vx,
            "mean_inlet_vy": mean_inlet_vy,
            "finite_inlet_velocity_observations": int(len(inlet_velocity_samples)),
            "patched_rows": int(patch_rows.sum()),
            "patched_velocity_values": int(patch_vx.sum() + patch_vy.sum()),
            "affected_unique_tracks": int(len(affected_tracks)),
        }
        return tracks

    def _postprocess_interpolated(self, tracks: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        return tracks, {}

    def _build_tensor(self, tracks: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        missing_features = [name for name in self.feature_names if name not in tracks.columns]
        if missing_features:
            raise KeyError(f"Missing requested feature columns: {missing_features}")

        track_ids = np.asarray(sorted(tracks["track_id"].dropna().unique()))
        frames = np.asarray([], dtype=int) if tracks.empty else np.arange(int(tracks["frame"].min()), int(tracks["frame"].max()) + 1)
        track_index = {track_id: i for i, track_id in enumerate(track_ids)}
        frame_index = {frame: i for i, frame in enumerate(frames)}
        Z = np.full((len(track_ids), len(frames), len(self.feature_names)), np.nan, dtype=np.float32)
        mask = np.zeros((len(track_ids), len(frames)), dtype=bool)

        for _, row in tracks.iterrows():
            i = track_index[row["track_id"]]
            t = frame_index[int(row["frame"])]
            Z[i, t, :] = row[self.feature_names].to_numpy(dtype=np.float32)
            mask[i, t] = True
        return Z, mask, track_ids, frames

    def _summary(self, Z, mask, track_ids, frames, diagnostics: dict[str, Any]) -> dict[str, Any]:
        return {
            "dataset_version": "canonical_dataset_v1",
            "source_dataset": str(self.input_csv),
            "output_npz": str(self.output_npz),
            "num_tracks": int(len(track_ids)),
            "num_frames": int(len(frames)),
            "feature_count": int(len(self.feature_names)),
            "feature_names": list(self.feature_names),
            "Z_shape": list(Z.shape),
            "mask_shape": list(mask.shape),
            "mask_coverage": float(mask.mean()) if mask.size else 0.0,
            "inlet_velocity_patch": self.inlet_velocity_diagnostics,
            **diagnostics,
        }

    def _write_metadata(self, summary: dict[str, Any], tracks: pd.DataFrame) -> None:
        metadata_path = self.output_npz.with_suffix(".metadata.json")
        metadata_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def _print_summary(self, summary: dict[str, Any]) -> None:
        print(f"N: {summary['num_tracks']}")
        print(f"T: {summary['num_frames']}")
        print(f"F: {summary['feature_count']}")
        print(f"Z shape: {tuple(summary['Z_shape'])}")
        print(f"mask coverage: {summary['mask_coverage']:.4f}")
        print(f"output path: {summary['output_npz']}")


class CanonicalDatasetV2Builder(CanonicalDatasetBuilder):
    """Physics-enabled canonical dataset v2 using enriched tracking features."""

    def __init__(
        self,
        input_csv,
        output_npz,
        *,
        occupancy_csv=None,
        metadata_json=None,
        feature_names=None,
        inlet_y_max_px=100.0,
        neutral_invalid_cfd_value=0.0,
    ):
        super().__init__(input_csv, output_npz, feature_names or CANONICAL_V2_FEATURE_NAMES, inlet_y_max_px)
        self.occupancy_csv = Path(occupancy_csv) if occupancy_csv is not None else None
        self.metadata_json = Path(metadata_json) if metadata_json is not None else Path(output_npz).with_suffix(".metadata.json")
        self.neutral_invalid_cfd_value = float(neutral_invalid_cfd_value)

    def _prepare_tracks(self, tracks: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        original_rows = int(len(tracks))
        tracks = self._standardize_columns(tracks)
        tracks = self._attach_occupancy(tracks)
        required = [
            "frame",
            "track_id",
            "x",
            "y",
            "circularity",
            "cfd_u",
            "cfd_v",
            "left_flow_fraction",
            "cfd_valid",
            *OCCUPANCY_COLUMNS.values(),
        ]
        missing = [column for column in required if column not in tracks.columns]
        if missing:
            raise KeyError(f"Physics-enriched input is missing required columns: {missing}")
        before = len(tracks)
        finite_required = ["frame", "track_id", "x", "y", "circularity", "left_flow_fraction", *OCCUPANCY_COLUMNS.values()]
        tracks = tracks[np.isfinite(tracks[finite_required].to_numpy(float)).all(axis=1)].copy()
        dropped_rows = before - len(tracks)
        tracks["cfd_valid"] = tracks["cfd_valid"].astype(float)
        self._validate_occupancy(tracks)
        columns = ["frame", "track_id", "x", "y", "circularity", "cfd_u", "cfd_v", "left_flow_fraction", *OCCUPANCY_COLUMNS.values(), "cfd_valid"]
        tracks = tracks[columns].sort_values(["track_id", "frame"]).reset_index(drop=True)
        return tracks, {
            "dataset_version": "canonical_dataset_v2",
            "input_rows": original_rows,
            "dropped_rows": int(dropped_rows),
            "dropped_tracks": 0,
            "drop_reasons": ["non-finite required non-CFD state values"] if dropped_rows else [],
        }

    def _attach_occupancy(self, tracks: pd.DataFrame) -> pd.DataFrame:
        if all(target in tracks.columns for target in OCCUPANCY_COLUMNS.values()):
            return tracks
        if all(source in tracks.columns for source in OCCUPANCY_COLUMNS):
            return tracks.rename(columns=OCCUPANCY_COLUMNS)
        if self.occupancy_csv is None:
            raise KeyError("Occupancy fractions are missing and no occupancy_csv was provided")
        occupancy = pd.read_csv(self.occupancy_csv, usecols=["frame", "track_id", *OCCUPANCY_COLUMNS.keys()])
        occupancy = occupancy.rename(columns=OCCUPANCY_COLUMNS)
        if occupancy[["frame", "track_id"]].duplicated().any():
            raise ValueError("Occupancy CSV has duplicate frame/track_id rows")
        return tracks.merge(occupancy, on=["frame", "track_id"], how="left", validate="one_to_one")

    def _validate_occupancy(self, tracks: pd.DataFrame) -> None:
        cols = list(OCCUPANCY_COLUMNS.values())
        occupancy = tracks[cols].to_numpy(float)
        if np.any(occupancy < -1.0e-9):
            raise ValueError("Normalized occupancy fractions contain negative values")
        sums = occupancy.sum(axis=1)
        if not np.allclose(sums, 1.0, atol=1.0e-6, rtol=0.0):
            bad = int(np.sum(~np.isclose(sums, 1.0, atol=1.0e-6, rtol=0.0)))
            raise ValueError(f"Normalized occupancy fractions must sum to 1; bad rows: {bad}")

    def _postprocess_interpolated(self, tracks: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        tracks = tracks.copy()
        cfd_valid = tracks["cfd_valid"].to_numpy(float) >= 0.5
        invalid = ~cfd_valid
        invalid_nonfinite_before = int((~np.isfinite(tracks.loc[invalid, ["cfd_u", "cfd_v"]].to_numpy(float))).sum()) if invalid.any() else 0
        tracks.loc[invalid, ["cfd_u", "cfd_v"]] = self.neutral_invalid_cfd_value
        tracks["cfd_valid"] = cfd_valid.astype(np.float32)
        return tracks, {
            "valid_cfd_fraction": float(cfd_valid.mean()) if len(cfd_valid) else 0.0,
            "invalid_cfd_rows": int(invalid.sum()),
            "invalid_cfd_values_replaced_after_retaining_cfd_valid": invalid_nonfinite_before,
        }

    def _write_metadata(self, summary: dict[str, Any], tracks: pd.DataFrame) -> None:
        summary = dict(summary)
        summary.update(
            {
                "dataset_version": "canonical_dataset_v2",
                "source_dataset": str(self.input_csv),
                "occupancy_source": str(self.occupancy_csv) if self.occupancy_csv else "embedded in source dataset",
                "normalization_rules": {
                    "occupancy": "six regional occupancy fractions are already normalized per droplet and verified to sum to 1",
                    "numeric_state": "no feature standardization is applied by the canonical builder",
                    "velocity": "vx/vy are frame-to-frame first differences in image-pixel coordinates, preserving v1 behavior",
                },
                "invalid_cfd_handling": {
                    "cfd_valid": "retained as a separate numeric validity feature before replacement",
                    "invalid_cfd_u_v": f"invalid or outside-domain cfd_u/cfd_v are replaced with neutral value {self.neutral_invalid_cfd_value:g}",
                },
                "feature_names_and_order": list(self.feature_names),
            }
        )
        self.metadata_json.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def _summary(self, Z, mask, track_ids, frames, diagnostics: dict[str, Any]) -> dict[str, Any]:
        summary = super()._summary(Z, mask, track_ids, frames, diagnostics)
        summary["dataset_version"] = "canonical_dataset_v2"
        summary["metadata_json"] = str(self.metadata_json)
        return summary

    def _print_summary(self, summary: dict[str, Any]) -> None:
        super()._print_summary(summary)
        print(f"valid CFD fraction: {summary.get('valid_cfd_fraction', float('nan')):.4f}")
        print(f"dropped rows: {summary.get('dropped_rows', 0)}")
        print(f"dropped tracks: {summary.get('dropped_tracks', 0)}")
