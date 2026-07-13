"""Configuration loading and normalization for the XRD strain pipeline."""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when the experiment configuration is incomplete or inconsistent."""


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load an experiment YAML file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ConfigError(f"Expected a mapping in {config_path}")
    return config


def scan_by_id(config: dict[str, Any], scan_id: str) -> dict[str, Any]:
    """Return one scan entry by scan_id."""

    for scan in config.get("scans", []):
        if scan.get("scan_id") == scan_id:
            return scan
    raise ConfigError(f"Scan id not found: {scan_id}")


def reference_scan(config: dict[str, Any]) -> dict[str, Any]:
    """Return the configured reference scan."""

    return scan_by_id(config, config["reference_scan_id"])


def scan_ids(config: dict[str, Any]) -> list[str]:
    """Return scan ids in configured order."""

    scans = config.get("scans", [])
    ids = [scan.get("scan_id") for scan in scans]
    missing = [index for index, scan_id in enumerate(ids) if scan_id is None]
    if missing:
        raise ConfigError(f"Missing scan_id in scans at index/indices: {missing}")
    return ids


def merged_scan_settings(config: dict[str, Any], scan_id: str) -> dict[str, Any]:
    """Merge global defaults with one scan entry.

    The returned mapping is intentionally close to the parameter names used by
    nxprocess_lab.LabCreate so notebook cells can inspect it directly.
    """

    scan = deepcopy(scan_by_id(config, scan_id))
    project = config.get("project", {})
    unit_cell = config.get("unit_cell_initial", config.get("unit_cell", {}))
    refined_cell = config.get("unit_cell_refined", unit_cell)
    theta_scan = config.get("theta_scan", {})
    instrument = config.get("instrument", {})
    raw_data_root = project.get("raw_data_root")
    output_root = project.get("output_root", "Processed_Data")
    calibration_root = project.get("calibration_root", "Calibration")

    if scan.get("datafile_path") is None and raw_data_root and scan.get("DATA_directory"):
        scan["datafile_path"] = str((Path("..") / ".." / raw_data_root / scan["DATA_directory"])) + "/"
    if scan.get("output_directory") is None and scan.get("scan_id"):
        scan["output_directory"] = str(Path(output_root) / scan["scan_id"])
    if (
        instrument.get("calibration_path_without_extension") is None
        and instrument.get("calibration_file_base")
        and scan.get("output_directory")
    ):
        calibration_path = Path(calibration_root) / instrument["calibration_file_base"]
        scan_output_path = Path(scan["output_directory"])
        instrument = deepcopy(instrument)
        instrument["calibration_path_without_extension"] = os.path.relpath(
            calibration_path,
            start=scan_output_path,
        )

    merged = {
        **theta_scan,
        **instrument,
        **scan,
        "unit_cell_group": unit_cell.get("unit_cell_group"),
        "lattice_centring": unit_cell.get("lattice_centring"),
        "unitcell_a": unit_cell.get("a"),
        "unitcell_b": unit_cell.get("b"),
        "unitcell_c": unit_cell.get("c"),
        "unitcell_alpha": unit_cell.get("alpha"),
        "unitcell_beta": unit_cell.get("beta"),
        "unitcell_gamma": unit_cell.get("gamma"),
        "refined_unit_cell_group": refined_cell.get("unit_cell_group"),
        "refined_lattice_centring": refined_cell.get("lattice_centring"),
        "refined_unitcell_a": refined_cell.get("a"),
        "refined_unitcell_b": refined_cell.get("b"),
        "refined_unitcell_c": refined_cell.get("c"),
        "refined_unitcell_alpha": refined_cell.get("alpha"),
        "refined_unitcell_beta": refined_cell.get("beta"),
        "refined_unitcell_gamma": refined_cell.get("gamma"),
    }
    return merged


def all_scan_settings(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return merged settings for all scans in configured order."""

    return [merged_scan_settings(config, scan_id) for scan_id in scan_ids(config)]


def grouped_scan_settings(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Return merged scan settings grouped by logical scan/strain point.

    A single full scan and a collection of cross-section scans can both be
    analyzed as one logical item. Scans without an explicit scan_group_id use
    their own scan_id as the group id.
    """

    groups: dict[str, list[dict[str, Any]]] = {}
    for settings in all_scan_settings(config):
        group_id = settings.get("scan_group_id") or settings["scan_id"]
        settings["scan_group_id"] = group_id
        groups.setdefault(group_id, []).append(settings)
    return groups


def logical_scan_groups(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return compact metadata for each logical scan group."""

    summaries = []
    for group_id, scans in grouped_scan_settings(config).items():
        first = scans[0]
        summaries.append(
            {
                "scan_group_id": group_id,
                "role": first.get("role"),
                "strain": first.get("strain"),
                "sample_temperature": first.get("sample_temperature"),
                "scan_ids": [scan["scan_id"] for scan in scans],
                "cross_section_ids": [
                    scan.get("cross_section_id", scan["scan_id"])
                    for scan in scans
                ],
            }
        )
    return summaries


def require_values(settings: dict[str, Any], keys: list[str]) -> None:
    """Raise ConfigError if required settings are missing or null."""

    missing = [key for key in keys if settings.get(key) is None]
    if missing:
        raise ConfigError("Missing required setting(s): " + ", ".join(missing))
