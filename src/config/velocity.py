from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config.loader import load_experiment_config


DEFAULT_EXPERIMENT_CONFIG = Path("configs/experiments/video_2.yml")


def velocity_scale_mm_s_per_px_frame(pixel_scale_um_per_px: float, frame_rate_fps: float) -> float:
    values = np.asarray([pixel_scale_um_per_px, frame_rate_fps], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Pixel scale and frame rate must be finite")
    if pixel_scale_um_per_px <= 0:
        raise ValueError("pixel_scale_um_per_px must be positive")
    if frame_rate_fps <= 0:
        raise ValueError("frame_rate_fps must be positive")
    return float(pixel_scale_um_per_px / 1000.0 * frame_rate_fps)


def load_velocity_conversion_from_experiment(
    experiment_config: str | Path = DEFAULT_EXPERIMENT_CONFIG,
) -> dict[str, float]:
    loaded = load_experiment_config(experiment_config)
    experiment = loaded["experiment"]["experiment"]
    device = loaded["device"]["device"]
    pixel_scale_um_per_px = float(device["calibration"]["um_per_px"])
    frame_rate_fps = float(experiment["frame_rate_fps"])
    return {
        "pixel_scale_um_per_px": pixel_scale_um_per_px,
        "frame_rate_fps": frame_rate_fps,
        "velocity_mm_s_per_px_frame": velocity_scale_mm_s_per_px_frame(pixel_scale_um_per_px, frame_rate_fps),
    }
