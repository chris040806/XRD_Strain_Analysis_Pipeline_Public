"""Compact workflow helpers for notebook-level analysis steps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import csv

import numpy as np

from .lab_wrappers import combine_local_transform
from .lab_wrappers import estimate_momentum_bin_sizes
from .lab_wrappers import labcreate_output_paths
from .lab_wrappers import momentum_line_scans
from .lab_wrappers import transform_local_window
from .plotting import plot_detector_window
from .plotting import plot_momentum_line_scans
from .plotting import resolve_overlay_png
from .plotting import select_detector_window_interactive
from .window_utils import normalize_detector_window
from .window_utils import load_window_definition
from .window_utils import save_window_definition


CDW_WINDOW_ROLES = {"signal", "background"}


def cdw_window_definition_path(
    repo_root: str | Path,
    cdw_id: str,
    window_role: str,
) -> Path:
    """Return the role-specific saved detector-window path for one CDW."""

    window_role = _normalize_window_role(window_role)
    cdw_id = _validate_cdw_id(cdw_id)
    return Path(repo_root) / "configs" / "peak_windows" / f"{cdw_id}_{window_role}.json"


def load_cdw_window_definition(
    repo_root: str | Path,
    cdw_id: str,
    window_role: str,
) -> dict[str, Any]:
    """Load one role-specific CDW detector-window definition."""

    path = cdw_window_definition_path(repo_root, cdw_id, window_role)
    document = load_window_definition(path)
    bounds = _bounds_from_window_document(document)
    center = _center_from_window_document(document)
    return {
        "path": path,
        "document": document,
        "bounds": bounds,
        "center": center,
        "scan_id": document["reference_scan_id"],
        "peak_id": document["peak_id"],
        "cdw_id": document.get("cdw_id", cdw_id),
        "window_role": document.get("window_role", window_role),
    }


def save_cdw_window_definition(
    repo_root: str | Path,
    cdw_id: str,
    window_role: str,
    reference_scan_id: str,
    bounds: tuple[int, int, int, int, int, int],
    center: tuple[float, float, float],
    target_hkl=None,
    parent_bragg_hkl=None,
) -> Path:
    """Save one CDW signal/background detector window without clobbering other roles."""

    window_role = _normalize_window_role(window_role)
    cdw_id = _validate_cdw_id(cdw_id)
    path = cdw_window_definition_path(repo_root, cdw_id, window_role)
    return save_window_definition(
        path,
        peak_id=f"{cdw_id}_{window_role}",
        reference_scan_id=reference_scan_id,
        bounds=bounds,
        center=center,
        target_hkl=target_hkl,
        parent_bragg_hkl=parent_bragg_hkl,
        extra_metadata={
            "cdw_id": cdw_id,
            "window_role": window_role,
        },
    )


def select_or_load_cdw_window(
    settings: dict[str, Any],
    repo_root: str | Path,
    config: dict[str, Any],
    cdw_id: str,
    window_role: str,
    initial_center: tuple[float, float, float],
    initial_bounds: tuple[int, int, int, int, int, int],
    overlay_file: str | None = None,
    use_saved: bool = True,
    run_interactive_selector: bool = False,
    require_saved_or_interactive: bool = False,
    save: bool = True,
    target_hkl=None,
    parent_bragg_hkl=None,
    crop_margin: int = 40,
) -> dict[str, Any]:
    """Load, optionally reselect, preview, and save a CDW detector window.

    This is the notebook convenience layer: each CDW gets a persistent signal
    JSON and background JSON, so switching from CDW1 to CDW2 no longer erases
    the accepted CDW1 bounds.
    """

    repo_root = Path(repo_root)
    cdw_id = _validate_cdw_id(cdw_id)
    window_role = _normalize_window_role(window_role)
    window_path = cdw_window_definition_path(repo_root, cdw_id, window_role)

    source = "initial"
    center = tuple(initial_center)
    bounds = normalize_detector_window(*initial_bounds)
    if use_saved and window_path.exists():
        loaded = load_cdw_window_definition(repo_root, cdw_id, window_role)
        center = loaded["center"]
        bounds = loaded["bounds"]
        source = "saved"
    elif require_saved_or_interactive and not run_interactive_selector:
        raise FileNotFoundError(
            f"Missing saved {cdw_id} {window_role} window: {window_path}. "
            "Turn on the interactive selector once to choose and save it."
        )

    xstart, xend, ystart, yend, zstart, zend = bounds
    overlay_root = repo_root / config["project"]["overlay_root"]
    overlay_png = resolve_overlay_png(overlay_root, settings["scan_id"], overlay_file)

    if run_interactive_selector:
        print(f"Opening {cdw_id} {window_role} detector selector...")
        xstart, xend, ystart, yend = select_detector_window_interactive(
            overlay_png,
            initial_bounds=(xstart, xend, ystart, yend),
            center=(center[0], center[1]),
        )
        center = (center[0], center[1], center[2])
        source = "interactive"

    bounds = normalize_detector_window(xstart, xend, ystart, yend, zstart, zend)
    graphs_dir = repo_root / config["project"]["output_root"] / "Graphs" / settings["scan_id"]
    preview_path = graphs_dir / f"{cdw_id}_{window_role}_window.png"
    plot_detector_window(
        overlay_png,
        bounds[0],
        bounds[1],
        bounds[2],
        bounds[3],
        center=(center[0], center[1]),
        crop_margin=crop_margin,
        save_path=preview_path,
    )

    if save:
        save_cdw_window_definition(
            repo_root,
            cdw_id=cdw_id,
            window_role=window_role,
            reference_scan_id=settings["scan_id"],
            bounds=bounds,
            center=center,
            target_hkl=target_hkl,
            parent_bragg_hkl=parent_bragg_hkl,
        )

    return {
        "cdw_id": cdw_id,
        "window_role": window_role,
        "source": source,
        "path": window_path,
        "preview_path": preview_path,
        "overlay_png": overlay_png,
        "center": center,
        "bounds": bounds,
        "xstart": bounds[0],
        "xend": bounds[1],
        "ystart": bounds[2],
        "yend": bounds[3],
        "zstart": bounds[4],
        "zend": bounds[5],
    }


def cdw_series_inputs_from_saved_windows(
    repo_root: str | Path,
    cdw_id: str,
) -> dict[str, Any]:
    """Return signal/background centers and bounds for Stage 7 from saved JSON."""

    signal = load_cdw_window_definition(repo_root, cdw_id, "signal")
    background = load_cdw_window_definition(repo_root, cdw_id, "background")
    return {
        "cdw_id": cdw_id,
        "signal_center": signal["center"],
        "signal_bounds": signal["bounds"],
        "background_center": background["center"],
        "background_bounds": background["bounds"],
        "signal_window": signal,
        "background_window": background,
    }


def _normalize_window_role(window_role: str) -> str:
    role = str(window_role).strip().lower()
    if role not in CDW_WINDOW_ROLES:
        raise ValueError(f"window_role must be one of {sorted(CDW_WINDOW_ROLES)}")
    return role


def _validate_cdw_id(cdw_id: str) -> str:
    cdw_id = str(cdw_id).strip()
    if not cdw_id or cdw_id in {".", ".."} or "/" in cdw_id or "\\" in cdw_id:
        raise ValueError("cdw_id must be a non-empty filename-safe identifier")
    return cdw_id


def _bounds_from_window_document(document: dict[str, Any]) -> tuple[int, int, int, int, int, int]:
    bounds = document["bounds"]
    return normalize_detector_window(
        bounds["xstart"],
        bounds["xend"],
        bounds["ystart"],
        bounds["yend"],
        bounds["zstart"],
        bounds["zend"],
    )


def _center_from_window_document(document: dict[str, Any]) -> tuple[float, float, float]:
    center = document["center"]
    return (
        float(center["x"]),
        float(center["y"]),
        float(center["z_frame"]),
    )


def prepare_reference_momentum_binning(
    prepared: dict[str, dict[str, Any]],
    repo_root: str | Path,
    scan_id: str,
    peak_id: str,
    deltah: float | None,
    deltak: float | None,
    deltal: float | None,
    run_estimate: bool = False,
    percentile: float = 90.0,
    coverage_factor: float = 1.25,
    verbose: bool = True,
) -> dict[str, Any]:
    """Resolve reference momentum settings, transform path, and H/K/L bin sizes."""

    repo_root = Path(repo_root)
    settings = prepared[scan_id]["settings"]
    transform_path = (
        repo_root
        / settings["output_directory"]
        / "Transforms"
        / f"{peak_id}_transform_local.npz"
    )

    bin_estimate = None
    if run_estimate:
        bin_estimate = estimate_momentum_bin_sizes(
            transform_path,
            percentile=percentile,
            coverage_factor=coverage_factor,
        )
        deltah = bin_estimate["deltah"] if deltah is None else deltah
        deltak = bin_estimate["deltak"] if deltak is None else deltak
        deltal = bin_estimate["deltal"] if deltal is None else deltal

        if verbose:
            print("Estimated bin sizes from adjacent local-transform pixel spacings:")
            for axis in ("h", "k", "l"):
                summary = bin_estimate[axis]
                print(
                    f"  {axis.upper()}: median={summary['median']:.6g}, "
                    f"p{bin_estimate['percentile']:.0f}={summary['percentile']:.6g}, "
                    f"suggested={summary['suggested']:.6g}, "
                    f"range=[{summary['range_min']:.6f}, {summary['range_max']:.6f}]"
                )
    else:
        missing = [
            name
            for name, value in (
                ("MOMENTUM_DELTAH", deltah),
                ("MOMENTUM_DELTAK", deltak),
                ("MOMENTUM_DELTAL", deltal),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "Set fixed bin sizes before disabling bin estimation: "
                + ", ".join(missing)
            )
        if verbose:
            print("Skipped bin-size estimation; using fixed momentum bin sizes.")

    if verbose:
        print(
            f"Using deltah={deltah:.6g}, "
            f"deltak={deltak:.6g}, deltal={deltal:.6g}"
        )

    return {
        "settings": settings,
        "local_transform_path": transform_path,
        "deltah": float(deltah),
        "deltak": float(deltak),
        "deltal": float(deltal),
        "bin_estimate": bin_estimate,
    }


def select_transform_bin_background_patch(
    settings: dict[str, Any],
    repo_root: str | Path,
    config: dict[str, Any],
    nxprocess_lab,
    background_id: str,
    center: tuple[float, float, float],
    bounds: tuple[int, int, int, int, int, int],
    deltah: float,
    deltak: float,
    deltal: float,
    overlay_file: str | None = None,
    run_interactive_selector: bool = False,
    crop_margin: int = 40,
) -> dict[str, Any]:
    """Select, transform, bin, and summarize one detector background patch."""

    repo_root = Path(repo_root)
    output_dir = repo_root / settings["output_directory"]
    transforms_dir = output_dir / "Transforms"
    graphs_dir = repo_root / config["project"]["output_root"] / "Graphs" / settings["scan_id"] / background_id

    xstart, xend, ystart, yend, zstart, zend = normalize_detector_window(*bounds)
    overlay_root = repo_root / config["project"]["overlay_root"]
    overlay_png = resolve_overlay_png(overlay_root, settings["scan_id"], overlay_file)

    if run_interactive_selector:
        print(f"Opening background detector selector for {background_id}...")
        xstart, xend, ystart, yend = select_detector_window_interactive(
            overlay_png,
            initial_bounds=(xstart, xend, ystart, yend),
            center=(center[0], center[1]),
        )

    preview_path = graphs_dir / f"{background_id}_detector_window.png"
    plot_detector_window(
        overlay_png,
        xstart,
        xend,
        ystart,
        yend,
        center=(center[0], center[1]),
        crop_margin=crop_margin,
        save_path=preview_path,
    )

    window_path = repo_root / "configs" / "peak_windows" / f"{background_id}.json"
    save_window_definition(
        window_path,
        peak_id=background_id,
        reference_scan_id=settings["scan_id"],
        bounds=(xstart, xend, ystart, yend, zstart, zend),
        center=center,
        target_hkl=None,
        parent_bragg_hkl=None,
    )

    local_transform = transform_local_window(
        settings,
        repo_root,
        nxprocess_lab,
        peak_id=background_id,
        xstart=xstart,
        xend=xend,
        ystart=ystart,
        yend=yend,
        zstart=zstart,
        zend=zend,
        center=center,
    )
    momentum_grid = combine_local_transform(
        settings,
        repo_root,
        nxprocess_lab,
        peak_id=background_id,
        deltah=deltah,
        deltak=deltak,
        deltal=deltal,
        local_transform_path=local_transform["save_path"],
        save_path=transforms_dir / f"{background_id}_background_momentum_grid.npz",
    )
    background = background_per_voxel(
        local_transform["save_path"],
        deltah=deltah,
        deltak=deltak,
        deltal=deltal,
        save_path=transforms_dir / f"{background_id}_background_summary.json",
    )
    return {
        "background_id": background_id,
        "window_path": window_path,
        "preview_path": preview_path,
        "local_transform": local_transform,
        "momentum_grid": momentum_grid,
        "background": background,
    }


def background_per_voxel(
    local_transform_path: str | Path,
    deltah: float,
    deltak: float,
    deltal: float,
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    """Calculate background intensity diagnostics from one transformed patch."""

    local_transform_path = Path(local_transform_path)
    with np.load(local_transform_path, allow_pickle=False) as local:
        data_t = local["data_t"].astype(np.float64)
        h = local["h"]
        k = local["k"]
        l = local["l"]

    if data_t.size == 0:
        raise ValueError(f"Empty background transform: {local_transform_path}")

    h_range = _axis_range(h, deltah)
    k_range = _axis_range(k, deltak)
    l_range = _axis_range(l, deltal)
    h_idx = _closest_indices(h_range, h)
    k_idx = _closest_indices(k_range, k)
    l_idx = _closest_indices(l_range, l)
    hkl_idx = np.column_stack([h_idx, k_idx, l_idx])

    unique_bins, inverse, samples_per_bin = np.unique(
        hkl_idx,
        return_inverse=True,
        return_counts=True,
        axis=0,
    )
    counts_per_bin = np.bincount(inverse, weights=data_t)
    intensity_per_occupied_voxel = counts_per_bin / samples_per_bin

    summary = {
        "local_transform_path": str(local_transform_path),
        "deltah": float(deltah),
        "deltak": float(deltak),
        "deltal": float(deltal),
        "detector_pixels": int(data_t.size),
        "occupied_momentum_voxels": int(unique_bins.shape[0]),
        "total_counts": float(np.sum(data_t)),
        "mean_counts_per_detector_pixel": float(np.mean(data_t)),
        "median_counts_per_detector_pixel": float(np.median(data_t)),
        "mean_counts_per_occupied_momentum_voxel": float(np.mean(intensity_per_occupied_voxel)),
        "median_counts_per_occupied_momentum_voxel": float(np.median(intensity_per_occupied_voxel)),
        "std_counts_per_occupied_momentum_voxel": float(np.std(intensity_per_occupied_voxel)),
    }

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        summary["save_path"] = str(save_path)

    return summary


def momentum_integration_bounds_from_background(
    line_scans: dict[str, Any],
    background_level: float | dict[str, float] | None,
    verbose: bool = True,
) -> dict[str, Any] | None:
    """Estimate HKL integration bounds from line-scan intersections with background."""

    if background_level is None:
        if verbose:
            print("Skipped background-crossing integration defaults because no background line is available.")
        return None

    background_levels = _background_levels_by_axis(background_level)
    center_h, center_k, center_l = line_scans["center_hkl"]
    h_min, h_max, h_crossings = _scan_background_crossings(
        line_scans["H"]["axis"],
        line_scans["H"]["intensity"],
        background_levels["H"],
        center_h,
    )
    k_min, k_max, k_crossings = _scan_background_crossings(
        line_scans["K"]["axis"],
        line_scans["K"]["intensity"],
        background_levels["K"],
        center_k,
    )
    l_min, l_max, l_crossings = _scan_background_crossings(
        line_scans["L"]["axis"],
        line_scans["L"]["intensity"],
        background_levels["L"],
        center_l,
    )

    result = {
        "background_level": background_levels,
        "center_hkl": (float(center_h), float(center_k), float(center_l)),
        "bounds": {
            "hmin": h_min,
            "hmax": h_max,
            "kmin": k_min,
            "kmax": k_max,
            "lmin": l_min,
            "lmax": l_max,
        },
        "crossings": {
            "h": h_crossings,
            "k": k_crossings,
            "l": l_crossings,
        },
    }

    if verbose:
        print("Line-scan background levels:")
        print(f"  H scan: {background_levels['H']:.6g}")
        print(f"  K scan: {background_levels['K']:.6g}")
        print(f"  L scan: {background_levels['L']:.6g}")
        print("Background crossings:")
        print(f"  H crossings: {np.array2string(h_crossings, precision=6, separator=', ')}")
        print(f"  K crossings: {np.array2string(k_crossings, precision=6, separator=', ')}")
        print(f"  L crossings: {np.array2string(l_crossings, precision=6, separator=', ')}")
        print("Suggested default HKL integration bounds:")
        print(f"  MOMENTUM_INTEGRATION_HMIN = {h_min:.6f}")
        print(f"  MOMENTUM_INTEGRATION_HMAX = {h_max:.6f}")
        print(f"  MOMENTUM_INTEGRATION_KMIN = {k_min:.6f}")
        print(f"  MOMENTUM_INTEGRATION_KMAX = {k_max:.6f}")
        print(f"  MOMENTUM_INTEGRATION_LMIN = {l_min:.6f}")
        print(f"  MOMENTUM_INTEGRATION_LMAX = {l_max:.6f}")

    return result


def momentum_line_scan_background_levels(
    line_scans: dict[str, Any],
    background_per_voxel: float | None,
) -> dict[str, float] | None:
    """Scale per-voxel background to the summed voxel count of each line scan."""

    if background_per_voxel is None:
        return None

    windows = line_scans.get("windows", {})
    h_width = _window_width(windows.get("h"))
    k_width = _window_width(windows.get("k"))
    l_width = _window_width(windows.get("l"))
    background_per_voxel = float(background_per_voxel)
    return {
        "H": background_per_voxel * k_width * l_width,
        "K": background_per_voxel * h_width * l_width,
        "L": background_per_voxel * h_width * k_width,
    }


def integrate_momentum_roi(
    transform_data: np.ndarray,
    Hrange: np.ndarray,
    Krange: np.ndarray,
    Lrange: np.ndarray,
    bounds: dict[str, float],
    background_level: float | None = None,
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    """Integrate a rectangular HKL ROI and calculate centroid diagnostics."""

    h_slice = _range_slice(Hrange, bounds["hmin"], bounds["hmax"])
    k_slice = _range_slice(Krange, bounds["kmin"], bounds["kmax"])
    l_slice = _range_slice(Lrange, bounds["lmin"], bounds["lmax"])
    roi = np.asarray(transform_data[h_slice, k_slice, l_slice], dtype=np.float64)
    occupied = roi != 0
    occupied_count = int(np.count_nonzero(occupied))
    voxel_count = int(roi.size)
    raw_intensity = float(np.sum(roi[occupied]))

    if background_level is None:
        background_per_voxel = 0.0
    else:
        background_per_voxel = float(background_level)

    background_total = float(background_per_voxel * occupied_count)
    net = roi - background_per_voxel
    positive = np.where(occupied & (net > 0), net, 0.0)
    positive_intensity = float(np.sum(positive))
    net_intensity = float(raw_intensity - background_total)

    h_values = np.asarray(Hrange[h_slice], dtype=np.float64)
    k_values = np.asarray(Krange[k_slice], dtype=np.float64)
    l_values = np.asarray(Lrange[l_slice], dtype=np.float64)
    centroid = _centroid_hkl(h_values, k_values, l_values, positive)
    max_index = np.unravel_index(np.nanargmax(roi), roi.shape)
    max_hkl = (
        float(h_values[max_index[0]]),
        float(k_values[max_index[1]]),
        float(l_values[max_index[2]]),
    )

    result = {
        "bounds": {
            "hmin": float(h_values[0]),
            "hmax": float(h_values[-1]),
            "kmin": float(k_values[0]),
            "kmax": float(k_values[-1]),
            "lmin": float(l_values[0]),
            "lmax": float(l_values[-1]),
        },
        "requested_bounds": {key: float(value) for key, value in bounds.items()},
        "index_bounds": {
            "h": (int(h_slice.start), int(h_slice.stop)),
            "k": (int(k_slice.start), int(k_slice.stop)),
            "l": (int(l_slice.start), int(l_slice.stop)),
        },
        "voxel_count": voxel_count,
        "occupied_voxel_count": occupied_count,
        "raw_intensity": raw_intensity,
        "background_per_occupied_voxel": background_per_voxel,
        "background_total": background_total,
        "net_intensity": net_intensity,
        "positive_net_intensity": positive_intensity,
        "centroid_hkl": centroid,
        "max_hkl": max_hkl,
    }

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        result["save_path"] = str(save_path)

    return result


def _background_levels_by_axis(background_level: float | dict[str, float]) -> dict[str, float]:
    if isinstance(background_level, dict):
        return {
            "H": float(background_level["H"]),
            "K": float(background_level["K"]),
            "L": float(background_level["L"]),
        }
    value = float(background_level)
    return {"H": value, "K": value, "L": value}


def _window_width(window) -> int:
    if window is None:
        return 1
    start, stop = window
    return max(1, int(stop) - int(start))


def stitch_momentum_cross_sections(
    momentum_grid_paths: list[str | Path],
    save_path: str | Path | None = None,
    scan_group_id: str | None = None,
) -> dict[str, Any]:
    """Sum multiple binned cross-section momentum grids onto one HKL grid.

    The input files should be NPZ files written by combine_local_transform.
    All grids must use the same deltah/deltak/deltal. Empty/unsampled voxels
    remain zero in transform_data and have zero in coverage.
    """

    paths = [Path(path) for path in momentum_grid_paths]
    if not paths:
        raise ValueError("At least one momentum grid path is required")

    grids = []
    deltas = None
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing momentum grid: {path}")
        with np.load(path, allow_pickle=False) as grid:
            transform_data = grid["transform_data"]
            h_range = grid["Hrange"]
            k_range = grid["Krange"]
            l_range = grid["Lrange"]
            current_deltas = (
                _axis_step(h_range),
                _axis_step(k_range),
                _axis_step(l_range),
            )
            if deltas is None:
                deltas = current_deltas
            elif not np.allclose(deltas, current_deltas, rtol=1e-6, atol=1e-12):
                raise ValueError(
                    "Momentum grids must use matching bin sizes. "
                    f"Expected {deltas}, got {current_deltas} for {path}"
                )
            grids.append(
                {
                    "path": path,
                    "transform_data": transform_data.astype(np.float64, copy=True),
                    "Hrange": h_range.astype(np.float64, copy=True),
                    "Krange": k_range.astype(np.float64, copy=True),
                    "Lrange": l_range.astype(np.float64, copy=True),
                }
            )

    deltah, deltak, deltal = deltas
    combined_h = _shared_axis([grid["Hrange"] for grid in grids], deltah)
    combined_k = _shared_axis([grid["Krange"] for grid in grids], deltak)
    combined_l = _shared_axis([grid["Lrange"] for grid in grids], deltal)
    combined = np.zeros((len(combined_h), len(combined_k), len(combined_l)), dtype=np.float64)
    coverage = np.zeros_like(combined, dtype=np.uint16)

    for grid in grids:
        h_idx = _axis_indices(combined_h, grid["Hrange"], deltah)
        k_idx = _axis_indices(combined_k, grid["Krange"], deltak)
        l_idx = _axis_indices(combined_l, grid["Lrange"], deltal)
        target = np.ix_(h_idx, k_idx, l_idx)
        data = grid["transform_data"]
        combined[target] += data
        coverage[target] += (data != 0)

    max_index = np.unravel_index(np.nanargmax(combined), combined.shape)
    max_hkl = (
        float(combined_h[max_index[0]]),
        float(combined_k[max_index[1]]),
        float(combined_l[max_index[2]]),
    )
    result = {
        "transform_data": combined,
        "coverage": coverage,
        "Hrange": combined_h,
        "Krange": combined_k,
        "Lrange": combined_l,
        "max_hkl": max_hkl,
        "scan_group_id": scan_group_id,
        "source_paths": paths,
        "deltah": float(deltah),
        "deltak": float(deltak),
        "deltal": float(deltal),
    }

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_path,
            transform_data=combined,
            coverage=coverage,
            Hrange=combined_h,
            Krange=combined_k,
            Lrange=combined_l,
            max_hkl=np.asarray(max_hkl),
            scan_group_id="" if scan_group_id is None else scan_group_id,
            source_paths=np.asarray([str(path) for path in paths]),
            deltah=float(deltah),
            deltak=float(deltak),
            deltal=float(deltal),
        )
        result["save_path"] = save_path

    return result


def stitch_scan_group_momentum(
    scan_settings: list[dict[str, Any]],
    repo_root: str | Path,
    config: dict[str, Any],
    peak_id: str,
    scan_group_id: str | None = None,
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    """Stitch all cross-section momentum grids for one logical scan group."""

    if not scan_settings:
        raise ValueError("scan_settings must contain at least one scan")

    repo_root = Path(repo_root)
    if scan_group_id is None:
        scan_group_id = scan_settings[0].get("scan_group_id", scan_settings[0]["scan_id"])

    grid_paths = [
        repo_root
        / settings["output_directory"]
        / "Transforms"
        / f"{peak_id}_momentum_grid.npz"
        for settings in scan_settings
    ]
    if save_path is None:
        save_path = (
            repo_root
            / config["project"]["output_root"]
            / scan_group_id
            / "Transforms"
            / f"{peak_id}_stitched_momentum_grid.npz"
        )

    result = stitch_momentum_cross_sections(
        grid_paths,
        save_path=save_path,
        scan_group_id=scan_group_id,
    )
    manifest_path = Path(save_path).parent.parent / "source_scan_files.json"
    result["source_manifest_path"] = write_scan_group_source_manifest(
        scan_settings,
        repo_root=repo_root,
        save_path=manifest_path,
    )
    return result


def process_cdw_strain_series(
    cdw_id: str,
    scan_groups: dict[str, list[dict[str, Any]]],
    group_ids: list[str],
    repo_root: str | Path,
    config: dict[str, Any],
    nxprocess_lab,
    signal_center: tuple[float, float, float],
    signal_bounds: tuple[int, int, int, int, int, int],
    background_center: tuple[float, float, float],
    background_bounds: tuple[int, int, int, int, int, int],
    deltah: float,
    deltak: float,
    deltal: float,
    integration_bounds: dict[str, float],
    strain_values: dict[str, float] | None = None,
    background_stat: str = "mean",
    run_transforms: bool = True,
    run_background: bool = True,
    run_stitching: bool = True,
    overwrite_existing: bool = False,
    auto_integration_bounds: bool = False,
    hscan_min: float | None = None,
    hscan_max: float | None = None,
    kscan_min: float | None = None,
    kscan_max: float | None = None,
    lscan_min: float | None = None,
    lscan_max: float | None = None,
    hscan_step: int = 2,
    kscan_step: int = 2,
    lscan_step: int = 2,
) -> dict[str, Any]:
    """Process one CDW detector ROI across logical strain/cross-section groups.

    Each physical scan is transformed and binned with the same detector-space
    signal/background windows and the same H/K/L bin sizes. Cross sections are
    stitched per logical scan group before integration.
    """

    repo_root = Path(repo_root)
    analysis_dir = (
        repo_root
        / config["project"]["output_root"]
        / "CDW_Analysis"
        / cdw_id
    )
    analysis_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    group_results = {}

    for group_id in group_ids:
        if group_id not in scan_groups:
            raise KeyError(f"Unknown scan group: {group_id}")

        group_scan_settings = scan_groups[group_id]
        signal_grid_paths = []
        background_summaries = []

        for settings in group_scan_settings:
            transforms_dir = repo_root / settings["output_directory"] / "Transforms"
            signal_local_path = transforms_dir / f"{cdw_id}_transform_local.npz"
            signal_grid_path = transforms_dir / f"{cdw_id}_momentum_grid.npz"
            background_id = f"{cdw_id}_background"
            background_local_path = transforms_dir / f"{background_id}_transform_local.npz"
            background_grid_path = transforms_dir / f"{background_id}_background_momentum_grid.npz"
            background_summary_path = transforms_dir / f"{background_id}_background_summary.json"

            if run_transforms and (overwrite_existing or not signal_grid_path.exists()):
                transform_local_window(
                    settings,
                    repo_root,
                    nxprocess_lab,
                    peak_id=cdw_id,
                    xstart=signal_bounds[0],
                    xend=signal_bounds[1],
                    ystart=signal_bounds[2],
                    yend=signal_bounds[3],
                    zstart=signal_bounds[4],
                    zend=signal_bounds[5],
                    center=signal_center,
                    save_path=signal_local_path,
                )
                combine_local_transform(
                    settings,
                    repo_root,
                    nxprocess_lab,
                    peak_id=cdw_id,
                    deltah=deltah,
                    deltak=deltak,
                    deltal=deltal,
                    local_transform_path=signal_local_path,
                    save_path=signal_grid_path,
                )
            signal_grid_paths.append(signal_grid_path)

            if run_background and (
                overwrite_existing or not background_summary_path.exists()
            ):
                transform_local_window(
                    settings,
                    repo_root,
                    nxprocess_lab,
                    peak_id=background_id,
                    xstart=background_bounds[0],
                    xend=background_bounds[1],
                    ystart=background_bounds[2],
                    yend=background_bounds[3],
                    zstart=background_bounds[4],
                    zend=background_bounds[5],
                    center=background_center,
                    save_path=background_local_path,
                )
                combine_local_transform(
                    settings,
                    repo_root,
                    nxprocess_lab,
                    peak_id=background_id,
                    deltah=deltah,
                    deltak=deltak,
                    deltal=deltal,
                    local_transform_path=background_local_path,
                    save_path=background_grid_path,
                )
                background_per_voxel(
                    background_local_path,
                    deltah=deltah,
                    deltak=deltak,
                    deltal=deltal,
                    save_path=background_summary_path,
                )

            if background_summary_path.exists():
                background_summaries.append(
                    json.loads(background_summary_path.read_text(encoding="utf-8"))
                )

        group_output_dir = repo_root / config["project"]["output_root"] / group_id
        group_transform_dir = group_output_dir / "Transforms"
        group_graph_dir = group_output_dir / "Graphs" / cdw_id
        group_transform_dir.mkdir(parents=True, exist_ok=True)
        group_graph_dir.mkdir(parents=True, exist_ok=True)
        source_manifest_path = write_scan_group_source_manifest(
            group_scan_settings,
            repo_root=repo_root,
            save_path=group_output_dir / "source_scan_files.json",
        )

        if len(signal_grid_paths) > 1 and run_stitching:
            momentum = stitch_momentum_cross_sections(
                signal_grid_paths,
                save_path=group_transform_dir / f"{cdw_id}_stitched_momentum_grid.npz",
                scan_group_id=group_id,
            )
            momentum_grid_path = momentum["save_path"]
        else:
            with np.load(signal_grid_paths[0], allow_pickle=False) as grid:
                momentum = {
                    "transform_data": grid["transform_data"],
                    "Hrange": grid["Hrange"],
                    "Krange": grid["Krange"],
                    "Lrange": grid["Lrange"],
                    "max_hkl": tuple(float(value) for value in grid["max_hkl"]),
                    "save_path": signal_grid_paths[0],
                }
            momentum_grid_path = signal_grid_paths[0]

        background_level = _combine_background_level(background_summaries, background_stat)
        group_integration_bounds = integration_bounds
        integration_bound_source = "manual"
        line_scans = None
        integration_defaults = None
        if auto_integration_bounds:
            line_scans = momentum_line_scans(
                transform_data=momentum["transform_data"],
                Hrange=momentum["Hrange"],
                Krange=momentum["Krange"],
                Lrange=momentum["Lrange"],
                hmin=hscan_min,
                hmax=hscan_max,
                kmin=kscan_min,
                kmax=kscan_max,
                lmin=lscan_min,
                lmax=lscan_max,
                hstep=hscan_step,
                kstep=kscan_step,
                lstep=lscan_step,
                center_hkl=momentum.get("max_hkl"),
            )
            background_line_levels = momentum_line_scan_background_levels(
                line_scans,
                background_level,
            )
            integration_defaults = momentum_integration_bounds_from_background(
                line_scans,
                background_line_levels,
                verbose=False,
            )
            if integration_defaults is not None:
                group_integration_bounds = integration_defaults["bounds"]
                integration_bound_source = "background_crossing"

        integration_path = group_transform_dir / f"{cdw_id}_momentum_integration.json"
        integration = integrate_momentum_roi(
            transform_data=momentum["transform_data"],
            Hrange=momentum["Hrange"],
            Krange=momentum["Krange"],
            Lrange=momentum["Lrange"],
            bounds=group_integration_bounds,
            background_level=background_level,
            save_path=integration_path,
        )

        strain = (
            strain_values[group_id]
            if strain_values and group_id in strain_values
            else group_scan_settings[0].get("strain")
        )
        sweep_direction = group_scan_settings[0].get("sweep_direction", "up")
        strain_float = float(strain) if strain is not None else np.nan
        poisson_sigma = float(np.sqrt(max(integration["raw_intensity"], 0.0)))
        row = {
            "cdw_id": cdw_id,
            "scan_group_id": group_id,
            "strain": strain_float,
            "sweep_direction": sweep_direction,
            "raw_intensity": integration["raw_intensity"],
            "background_total": integration["background_total"],
            "net_intensity": integration["net_intensity"],
            "positive_net_intensity": integration["positive_net_intensity"],
            "poisson_sigma_raw": poisson_sigma,
            "background_per_occupied_voxel": integration["background_per_occupied_voxel"],
            "occupied_voxel_count": integration["occupied_voxel_count"],
            "integration_bound_source": integration_bound_source,
            "centroid_h": None if integration["centroid_hkl"] is None else integration["centroid_hkl"][0],
            "centroid_k": None if integration["centroid_hkl"] is None else integration["centroid_hkl"][1],
            "centroid_l": None if integration["centroid_hkl"] is None else integration["centroid_hkl"][2],
            "max_h": integration["max_hkl"][0],
            "max_k": integration["max_hkl"][1],
            "max_l": integration["max_hkl"][2],
            "momentum_grid_path": str(momentum_grid_path),
            "integration_path": str(integration_path),
            "source_manifest_path": str(source_manifest_path),
        }
        rows.append(row)
        group_results[group_id] = {
            "momentum": momentum,
            "background_summaries": background_summaries,
            "integration": integration,
            "integration_bound_source": integration_bound_source,
            "integration_defaults": integration_defaults,
            "line_scans": line_scans,
            "row": row,
        }
        print(
            f"{cdw_id} {group_id}: net={integration['net_intensity']:.6g}, "
            f"positive_net={integration['positive_net_intensity']:.6g}, "
            f"strain={strain_float:.6g}, bounds={integration_bound_source}"
        )

    summary_csv = analysis_dir / f"{cdw_id}_strain_intensity_summary.csv"
    summary_json = analysis_dir / f"{cdw_id}_strain_intensity_summary.json"
    plot_path = analysis_dir / f"{cdw_id}_intensity_vs_strain.png"
    _write_rows_csv(summary_csv, rows)
    summary_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    plot_cdw_strain_intensity(rows, save_path=plot_path)

    return {
        "cdw_id": cdw_id,
        "rows": rows,
        "group_results": group_results,
        "summary_csv": summary_csv,
        "summary_json": summary_json,
        "plot_path": plot_path,
    }


def _default_overlay_file(settings: dict[str, Any]) -> str:
    data_directory = settings.get("DATA_directory")
    if data_directory:
        return f"{Path(str(data_directory)).parts[0]}.png"
    return f"{settings['scan_id']}.png"


def _settings_for_group(
    scan_groups: dict[str, list[dict[str, Any]]],
    group_id: str,
) -> dict[str, Any]:
    if group_id not in scan_groups:
        raise KeyError(f"Unknown scan id/group id: {group_id}")
    scans = scan_groups[group_id]
    if not scans:
        raise ValueError(f"No scan settings for {group_id}")
    return scans[0]


def plot_cdw_series_detector_windows(
    cdw_series_result: dict[str, Any],
    scan_groups: dict[str, list[dict[str, Any]]],
    repo_root: str | Path,
    config: dict[str, Any],
    signal_bounds: tuple[int, int, int, int, int, int],
    signal_center: tuple[float, float, float],
    background_bounds: tuple[int, int, int, int, int, int] | None = None,
    background_center: tuple[float, float, float] | None = None,
    overlay_files: dict[str, str] | None = None,
    crop_margin: int = 40,
) -> dict[str, dict[str, Path]]:
    """Plot the selected detector windows on each scan's Max-of-All PNG."""

    repo_root = Path(repo_root)
    overlay_root = repo_root / config["project"]["overlay_root"]
    cdw_id = cdw_series_result["cdw_id"]
    diagnostics_dir = (
        repo_root
        / config["project"]["output_root"]
        / "CDW_Analysis"
        / cdw_id
        / "Diagnostics"
        / "detector_windows"
    )
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    xstart, xend, ystart, yend, _zstart, _zend = signal_bounds
    signal_xy = (signal_center[0], signal_center[1])
    paths: dict[str, dict[str, Path]] = {}
    overlay_files = overlay_files or {}

    for row in cdw_series_result["rows"]:
        group_id = row["scan_group_id"]
        settings = _settings_for_group(scan_groups, group_id)
        overlay_file = (
            overlay_files.get(group_id)
            or overlay_files.get(settings["scan_id"])
            or _default_overlay_file(settings)
        )
        overlay_png = resolve_overlay_png(overlay_root, settings["scan_id"], overlay_file)
        scan_paths: dict[str, Path] = {}

        signal_path = diagnostics_dir / f"{group_id}_{cdw_id}_signal_window.png"
        plot_detector_window(
            overlay_png,
            xstart,
            xend,
            ystart,
            yend,
            center=signal_xy,
            crop_margin=crop_margin,
            save_path=signal_path,
        )
        scan_paths["signal"] = signal_path

        if background_bounds is not None:
            bxstart, bxend, bystart, byend, _bzstart, _bzend = background_bounds
            background_path = diagnostics_dir / f"{group_id}_{cdw_id}_background_window.png"
            plot_detector_window(
                overlay_png,
                bxstart,
                bxend,
                bystart,
                byend,
                center=None if background_center is None else (background_center[0], background_center[1]),
                crop_margin=crop_margin,
                save_path=background_path,
            )
            scan_paths["background"] = background_path

        paths[group_id] = scan_paths
        print(f"{group_id}: saved detector window diagnostic(s) using {overlay_png.name}")

    return paths


def plot_cdw_series_integration_diagnostics(
    cdw_series_result: dict[str, Any],
    repo_root: str | Path,
    config: dict[str, Any],
    hmin: float,
    hmax: float,
    kmin: float,
    kmax: float,
    lmin: float,
    lmax: float,
    hstep: int = 2,
    kstep: int = 2,
    lstep: int = 2,
    show_background: bool = False,
) -> dict[str, dict[str, Path]]:
    """Plot per-scan H/K/L line scans with integration bounds."""

    repo_root = Path(repo_root)
    cdw_id = cdw_series_result["cdw_id"]
    diagnostics_dir = (
        repo_root
        / config["project"]["output_root"]
        / "CDW_Analysis"
        / cdw_id
        / "Diagnostics"
        / "momentum_integration"
    )
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    bounds = {
        "hmin": hmin,
        "hmax": hmax,
        "kmin": kmin,
        "kmax": kmax,
        "lmin": lmin,
        "lmax": lmax,
    }
    paths: dict[str, dict[str, Path]] = {}

    for row in cdw_series_result["rows"]:
        group_id = row["scan_group_id"]
        result = cdw_series_result["group_results"][group_id]
        momentum = result["momentum"]
        integration = result["integration"]
        center_hkl = integration.get("max_hkl")
        line_scan_path = diagnostics_dir / f"{group_id}_{cdw_id}_momentum_line_scans.npz"
        figure_path = diagnostics_dir / f"{group_id}_{cdw_id}_momentum_integration_diagnostic.png"

        line_scans = momentum_line_scans(
            transform_data=momentum["transform_data"],
            Hrange=momentum["Hrange"],
            Krange=momentum["Krange"],
            Lrange=momentum["Lrange"],
            hmin=hmin,
            hmax=hmax,
            kmin=kmin,
            kmax=kmax,
            lmin=lmin,
            lmax=lmax,
            hstep=hstep,
            kstep=kstep,
            lstep=lstep,
            center_hkl=center_hkl,
            save_path=line_scan_path,
        )
        background_levels = None
        if show_background:
            background_levels = momentum_line_scan_background_levels(
                line_scans,
                integration.get("background_per_occupied_voxel"),
            )
        plot_momentum_line_scans(
            line_scans,
            save_path=figure_path,
            title=f"{cdw_id} Momentum Integration: {group_id}",
            background_y=background_levels,
            background_label="Scaled background",
            integration_bounds=integration["bounds"],
        )
        paths[group_id] = {
            "line_scans": line_scan_path,
            "figure": figure_path,
        }
        print(f"{group_id}: saved momentum integration diagnostic")

    return paths


def write_scan_group_source_manifest(
    scan_settings: list[dict[str, Any]],
    repo_root: str | Path,
    save_path: str | Path,
) -> Path:
    """Save a manifest pointing to the physical NXS/HDF5 files in a scan group."""

    repo_root = Path(repo_root)
    save_path = Path(save_path)
    entries = []
    for settings in scan_settings:
        paths = labcreate_output_paths(settings, repo_root)
        entries.append(
            {
                "scan_id": settings["scan_id"],
                "scan_group_id": settings.get("scan_group_id", settings["scan_id"]),
                "cross_section_id": settings.get("cross_section_id"),
                "strain": settings.get("strain"),
                "sweep_direction": settings.get("sweep_direction", "up"),
                "nxs": str(paths["nxs"]),
                "hdf5": str(paths["hdf5"]),
            }
        )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    return save_path


def plot_cdw_strain_intensity(
    rows: list[dict[str, Any]],
    save_path: str | Path | None = None,
    intensity_key: str = "net_intensity",
    error_key: str = "poisson_sigma_raw",
):
    """Plot CDW integrated intensity versus strain with simple Poisson bars."""

    import matplotlib.pyplot as plt

    finite_rows = [
        row
        for row in rows
        if np.isfinite(row.get("strain", np.nan))
    ]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for direction, direction_rows in _rows_by_sweep_direction(finite_rows).items():
        direction_rows.sort(key=lambda row: row["strain"])
        x = np.asarray([row["strain"] for row in direction_rows], dtype=float)
        y = np.asarray([row[intensity_key] for row in direction_rows], dtype=float)
        yerr = np.asarray([row[error_key] for row in direction_rows], dtype=float)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt=_sweep_direction_format(direction),
            capsize=3,
            label=_sweep_direction_label(direction),
        )
    ax.set_xlabel("Strain")
    ax.set_ylabel(intensity_key.replace("_", " "))
    ax.grid(alpha=0.25)
    if finite_rows:
        ax.legend(frameon=False)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)

    return fig, ax


def load_cdw_series_summary(summary_path: str | Path) -> list[dict[str, Any]]:
    """Load a CDW strain-series summary written by process_cdw_strain_series."""

    summary_path = Path(summary_path)
    if summary_path.suffix.lower() == ".json":
        return json.loads(summary_path.read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    with summary_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            converted: dict[str, Any] = {}
            for key, value in row.items():
                if value in {"", "None", None}:
                    converted[key] = None
                    continue
                try:
                    converted[key] = float(value)
                except (TypeError, ValueError):
                    converted[key] = value
            rows.append(converted)
    return rows


def _rows_by_sweep_direction(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        direction = str(row.get("sweep_direction") or "up")
        grouped.setdefault(direction, []).append(row)
    return grouped


def _sweep_direction_label(direction: str) -> str:
    labels = {
        "up": "up sweep",
        "down": "down sweep",
        "unknown": "unspecified sweep",
    }
    return labels.get(direction, direction.replace("_", " "))


def _sweep_direction_format(direction: str) -> str:
    formats = {
        "up": "o-",
        "down": "s--",
    }
    return formats.get(direction, "^-")


def plot_final_cdw_voltage_series(
    rows: list[dict[str, Any]],
    save_path: str | Path | None = None,
    cdw_label: str | None = None,
    intensity_key: str = "net_intensity",
    error_key: str | None = "poisson_sigma_raw",
    normalize: bool = False,
    reference_voltage: float | None = 0.0,
    xlabel: str = "Voltage / strain setting",
    ylabel: str | None = None,
):
    """Plot final CDW intensity versus voltage/strain setting."""

    import matplotlib.pyplot as plt

    finite_rows = [
        row
        for row in rows
        if np.isfinite(float(row.get("strain", np.nan)))
        and np.isfinite(float(row.get(intensity_key, np.nan)))
    ]
    reference_value = None
    if normalize and finite_rows:
        reference_rows = sorted(finite_rows, key=lambda row: float(row["strain"]))
        reference_x = np.asarray([float(row["strain"]) for row in reference_rows], dtype=float)
        reference_y = np.asarray([float(row[intensity_key]) for row in reference_rows], dtype=float)
        if reference_voltage is None:
            reference_value = reference_y[0]
        else:
            reference_index = int(np.argmin(np.abs(reference_x - float(reference_voltage))))
            reference_value = reference_y[reference_index]
        if reference_value == 0:
            raise ValueError("Cannot normalize by zero reference intensity")

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    label = cdw_label or (str(finite_rows[0].get("cdw_id")) if finite_rows else None)
    for direction, direction_rows in _rows_by_sweep_direction(finite_rows).items():
        direction_rows.sort(key=lambda row: float(row["strain"]))
        x = np.asarray([float(row["strain"]) for row in direction_rows], dtype=float)
        y = np.asarray([float(row[intensity_key]) for row in direction_rows], dtype=float)
        yerr = None
        if error_key is not None and all(row.get(error_key) is not None for row in direction_rows):
            yerr = np.asarray([float(row[error_key]) for row in direction_rows], dtype=float)
        if normalize and reference_value is not None:
            y = y / reference_value
            if yerr is not None:
                yerr = yerr / abs(reference_value)
        branch_label = label
        if len(_rows_by_sweep_direction(finite_rows)) > 1:
            branch_label = f"{label} {_sweep_direction_label(direction)}" if label else _sweep_direction_label(direction)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt=_sweep_direction_format(direction),
            capsize=3,
            linewidth=1.4,
            markersize=5,
            label=branch_label,
        )
    ax.set_xlabel(xlabel)
    if ylabel is None:
        ylabel = "Normalized net intensity" if normalize else "Net intensity (a.u.)"
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    if label:
        ax.legend(frameon=False)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_final_cdw_voltage_overlay(
    series_rows: dict[str, list[dict[str, Any]]],
    save_path: str | Path | None = None,
    intensity_key: str = "net_intensity",
    error_key: str | None = "poisson_sigma_raw",
    normalize: bool = False,
    reference_voltage: float | None = 0.0,
    xlabel: str = "Voltage / strain setting",
):
    """Overlay final voltage-series plots for several CDWs."""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for cdw_label, rows in series_rows.items():
        finite_rows = [
            row
            for row in rows
            if np.isfinite(float(row.get("strain", np.nan)))
            and np.isfinite(float(row.get(intensity_key, np.nan)))
        ]
        if not finite_rows:
            continue

        reference_value = None
        if normalize:
            reference_rows = sorted(finite_rows, key=lambda row: float(row["strain"]))
            reference_x = np.asarray([float(row["strain"]) for row in reference_rows], dtype=float)
            reference_y = np.asarray([float(row[intensity_key]) for row in reference_rows], dtype=float)
            if reference_voltage is None:
                reference_value = reference_y[0]
            else:
                reference_index = int(np.argmin(np.abs(reference_x - float(reference_voltage))))
                reference_value = reference_y[reference_index]
            if reference_value == 0:
                raise ValueError(f"Cannot normalize {cdw_label} by zero reference intensity")

        direction_groups = _rows_by_sweep_direction(finite_rows)
        for direction, direction_rows in direction_groups.items():
            direction_rows.sort(key=lambda row: float(row["strain"]))
            x = np.asarray([float(row["strain"]) for row in direction_rows], dtype=float)
            y = np.asarray([float(row[intensity_key]) for row in direction_rows], dtype=float)
            yerr = None
            if error_key is not None and all(row.get(error_key) is not None for row in direction_rows):
                yerr = np.asarray([float(row[error_key]) for row in direction_rows], dtype=float)
            if normalize and reference_value is not None:
                y = y / reference_value
                if yerr is not None:
                    yerr = yerr / abs(reference_value)
            label = cdw_label
            if len(direction_groups) > 1:
                label = f"{cdw_label} {_sweep_direction_label(direction)}"
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt=_sweep_direction_format(direction),
                capsize=3,
                linewidth=1.4,
                markersize=5,
                label=label,
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized net intensity" if normalize else "Net intensity (a.u.)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def _range_slice(axis_values: np.ndarray, lower: float, upper: float) -> slice:
    axis_values = np.asarray(axis_values, dtype=np.float64)
    lower = float(lower)
    upper = float(upper)
    if lower > upper:
        lower, upper = upper, lower
    start = int(np.searchsorted(axis_values, lower, side="left"))
    stop = int(np.searchsorted(axis_values, upper, side="right"))
    start = max(0, min(start, len(axis_values) - 1))
    stop = max(start + 1, min(stop, len(axis_values)))
    return slice(start, stop)


def _centroid_hkl(
    h_values: np.ndarray,
    k_values: np.ndarray,
    l_values: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, float] | None:
    total = float(np.sum(weights))
    if total <= 0:
        return None
    h_grid, k_grid, l_grid = np.meshgrid(h_values, k_values, l_values, indexing="ij")
    return (
        float(np.sum(h_grid * weights) / total),
        float(np.sum(k_grid * weights) / total),
        float(np.sum(l_grid * weights) / total),
    )


def _scan_background_crossings(
    axis_values: np.ndarray,
    intensity: np.ndarray,
    background_level: float,
    center_value: float,
) -> tuple[float, float, np.ndarray]:
    axis_values = np.asarray(axis_values)
    intensity = np.asarray(intensity)
    finite = np.isfinite(axis_values) & np.isfinite(intensity)
    axis_values = axis_values[finite]
    intensity = intensity[finite]

    if axis_values.size == 0:
        raise ValueError("Cannot estimate integration bounds from an empty line scan")

    order = np.argsort(axis_values)
    axis_values = axis_values[order]
    intensity = intensity[order]
    relative = intensity - background_level

    crossings = []
    for index in range(len(axis_values) - 1):
        y0 = relative[index]
        y1 = relative[index + 1]
        if y0 == 0:
            crossings.append(float(axis_values[index]))
        elif y0 * y1 < 0:
            fraction = -y0 / (y1 - y0)
            crossing = axis_values[index] + fraction * (axis_values[index + 1] - axis_values[index])
            crossings.append(float(crossing))

    crossings = np.asarray(crossings, dtype=float)
    left = crossings[crossings < center_value]
    right = crossings[crossings > center_value]
    lower = float(left.max()) if len(left) else float(axis_values.min())
    upper = float(right.min()) if len(right) else float(axis_values.max())
    return lower, upper, crossings


def _axis_step(axis_values: np.ndarray) -> float:
    axis_values = np.asarray(axis_values, dtype=float)
    if axis_values.size < 2:
        raise ValueError("Momentum axes must contain at least two bins")
    return float(np.median(np.diff(axis_values)))


def _shared_axis(axes: list[np.ndarray], delta: float) -> np.ndarray:
    lower = min(float(axis[0]) for axis in axes)
    upper = max(float(axis[-1]) for axis in axes)
    count = int(round((upper - lower) / delta)) + 1
    return lower + np.arange(count, dtype=float) * delta


def _axis_indices(target_axis: np.ndarray, source_axis: np.ndarray, delta: float) -> np.ndarray:
    indices = np.rint((source_axis - target_axis[0]) / delta).astype(int)
    if np.any(indices < 0) or np.any(indices >= len(target_axis)):
        raise ValueError("Source axis falls outside target stitched axis")
    if not np.allclose(target_axis[indices], source_axis, rtol=1e-6, atol=max(abs(delta) * 1e-6, 1e-12)):
        raise ValueError("Source axis is not aligned with the stitched target axis")
    return indices


def _axis_range(values: np.ndarray, delta: float) -> np.ndarray:
    if delta <= 0:
        raise ValueError("Momentum bin sizes must be positive")
    return np.linspace(
        np.min(values),
        np.max(values),
        int((np.max(values) - np.min(values)) / delta + 1),
    )


def _closest_indices(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(axis, values, side="left")
    previous_is_closer = (idx == len(axis)) | (
        np.fabs(values - axis[np.maximum(idx - 1, 0)])
        < np.fabs(values - axis[np.minimum(idx, len(axis) - 1)])
    )
    idx[previous_is_closer] -= 1
    return idx


def _combine_background_level(
    summaries: list[dict[str, Any]],
    background_stat: str,
) -> float | None:
    if not summaries:
        return None
    stat_keys = {
        "mean": "mean_counts_per_occupied_momentum_voxel",
        "median": "median_counts_per_occupied_momentum_voxel",
    }
    if background_stat not in stat_keys:
        raise ValueError("background_stat must be 'mean' or 'median'")
    values = [float(summary[stat_keys[background_stat]]) for summary in summaries]
    return float(np.mean(values))


def _write_rows_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    columns = list(rows[0].keys())
    lines = [",".join(columns)]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if value is None:
                values.append("")
            else:
                text = str(value).replace('"', '""')
                if any(char in text for char in [",", "\n", '"']):
                    text = f'"{text}"'
                values.append(text)
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
