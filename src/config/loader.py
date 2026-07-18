from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path, label: str) -> dict[str, Any]:
    """Load a non-empty YAML mapping with clear errors."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"{label} YAML is malformed: {path}") from exc

    if loaded is None:
        raise ValueError(f"{label} YAML is empty: {path}")
    if not isinstance(loaded, dict):
        raise ValueError(f"{label} YAML must contain a top-level mapping: {path}")
    return loaded


def load_experiment_config(
    experiment_path: str | Path,
    configs_root: str | Path = "configs",
) -> dict[str, dict[str, Any]]:
    """Load an experiment config and its linked device config."""
    experiment_path = Path(experiment_path)
    configs_root = Path(configs_root)

    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment config does not exist: {experiment_path}")

    experiment_config = _load_yaml(experiment_path, "Experiment config")
    experiment = experiment_config.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError(f"Experiment config is missing top-level 'experiment' mapping: {experiment_path}")

    device_id = experiment.get("device_id")
    if not device_id:
        raise ValueError(f"Experiment config is missing experiment.device_id: {experiment_path}")

    device_path = configs_root / "devices" / f"{device_id}.yml"
    if not device_path.exists():
        raise FileNotFoundError(f"Device config does not exist for device_id '{device_id}': {device_path}")

    device_config = _load_yaml(device_path, "Device config")
    device = device_config.get("device")
    if not isinstance(device, dict):
        raise ValueError(f"Device config is missing top-level 'device' mapping: {device_path}")

    loaded_device_id = device.get("id")
    if loaded_device_id != device_id:
        raise ValueError(
            f"Device ID mismatch: experiment references '{device_id}', "
            f"but {device_path} contains '{loaded_device_id}'"
        )

    return {"experiment": experiment_config, "device": device_config}
