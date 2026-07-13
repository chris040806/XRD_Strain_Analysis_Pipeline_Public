"""Thin adapters from pipeline config to existing lab functions."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from contextlib import redirect_stdout
import io
import re
from pathlib import Path
from typing import Any

import numpy as np

from .config import require_values
from .window_utils import normalize_detector_window


LABCREATE_REQUIRED_KEYS = [
    "file",
    "calibration_path_without_extension",
    "imsize",
    "imstart",
    "imend",
    "det_bg",
    "frame_time",
    "chi",
    "omega",
    "phi_start",
    "phi_end",
    "phi_step",
    "sample_label",
    "sample_name",
    "sample_temperature",
    "unit_cell_group",
    "lattice_centring",
    "unitcell_a",
    "unitcell_b",
    "unitcell_c",
    "unitcell_alpha",
    "unitcell_beta",
    "unitcell_gamma",
    "calibrant",
    "calibration_date",
    "datafile_path",
    "data_base_name",
    "chunk_size",
    "null_image",
    "imstartdiff",
    "gapstart",
    "gapend",
]


def add_lab_module_path(repo_root: str | Path, lab_module_path: str | Path) -> Path:
    """Add the folder containing nxprocess_lab.py and nxrefine_lab.py to sys.path."""

    lab_path = (Path(repo_root) / lab_module_path).resolve()
    if str(lab_path) not in sys.path:
        sys.path.insert(0, str(lab_path))
    return lab_path


def labcreate_kwargs(settings: dict[str, Any]) -> dict[str, Any]:
    """Build keyword arguments for nxprocess_lab.LabCreate."""

    require_values(settings, LABCREATE_REQUIRED_KEYS)
    return {
        "file": settings["file"],
        "calib_path": settings["calibration_path_without_extension"],
        "imsize": int(settings["imsize"]),
        "imstart": int(settings["imstart"]),
        "imend": int(settings["imend"]),
        "det_bg": settings["det_bg"],
        "frame_time": settings["frame_time"],
        "orientation_matrix": np.array(settings["orientation_matrix"], dtype=float),
        "chi": settings["chi"],
        "omega": settings["omega"],
        "phi_start": settings["phi_start"],
        "phi_end": settings["phi_end"],
        "phi_step": settings["phi_step"],
        "sample_label": settings["sample_label"],
        "sample_name": settings["sample_name"],
        "sample_temperature": settings["sample_temperature"],
        "unit_cell_group": settings["unit_cell_group"],
        "lattice_centring": settings["lattice_centring"],
        "unitcell_a": settings["unitcell_a"],
        "unitcell_b": settings["unitcell_b"],
        "unitcell_c": settings["unitcell_c"],
        "unitcell_alpha": settings["unitcell_alpha"],
        "unitcell_beta": settings["unitcell_beta"],
        "unitcell_gamma": settings["unitcell_gamma"],
        "calibrant": settings["calibrant"],
        "calibration_date": settings["calibration_date"],
        "datafile_path": settings["datafile_path"],
        "data_base_name": settings["data_base_name"],
        "chunk_size": int(settings["chunk_size"]),
        "null_image": bool(settings["null_image"]),
        "imstartdiff": int(settings["imstartdiff"]),
        "gapstart": settings["gapstart"],
        "gapend": settings["gapend"],
    }


def labcreate_output_paths(settings: dict[str, Any], base_dir: str | Path = ".") -> dict[str, Path]:
    """Return the files LabCreate is expected to create for one scan."""

    require_values(settings, ["file", "sample_name", "sample_temperature", "output_directory"])
    base = Path(base_dir)
    output_dir = base / settings["output_directory"]
    return {
        "nxs": output_dir / f"{settings['file']}.nxs",
        "hdf5": output_dir / f"{settings['sample_name']}_{settings['sample_temperature']}K.hdf5",
    }


def confirm_existing_outputs(scan_id: str, settings: dict[str, Any], base_dir: str | Path) -> str:
    """Ask what to do if LabCreate output files already exist."""

    outputs = labcreate_output_paths(settings, base_dir)
    existing = [path for path in outputs.values() if path.exists()]
    if not existing:
        return "create"

    print(f"Existing output file(s) for {scan_id}:")
    for path in existing:
        print(f"  {path}")
    choice = input(
        "Type 'rewrite' to replace and recreate, 'skip' to skip this scan, "
        "or 'stop' to stop: "
    ).strip().lower()
    if choice == "delete":
        choice = "rewrite"
    if choice not in {"rewrite", "skip", "stop"}:
        print("Unrecognized choice; stopping to avoid overwriting data.")
        return "stop"
    if choice == "rewrite":
        for path in existing:
            path.unlink()
            print(f"Deleted {path}")
        return "create"
    return choice


def create_scan_files(prepared: dict[str, dict[str, Any]], repo_root: str | Path, nxprocess_lab) -> None:
    """Create NeXus/HDF5 files for prepared scans using nxprocess_lab.LabCreate."""

    repo_root = Path(repo_root)
    for scan_id, stage_args in prepared.items():
        action = confirm_existing_outputs(scan_id, stage_args["settings"], repo_root)
        if action == "stop":
            break
        if action == "skip":
            print(f"Skipping {scan_id}")
            continue
        output_dir = repo_root / stage_args["settings"]["output_directory"]
        print(f"Creating NeXus/HDF5 for {scan_id} in {output_dir}")
        with working_directory(output_dir):
            nxprocess_lab.LabCreate(**stage_args["create"])


@contextmanager
def working_directory(path: str | Path):
    """Temporarily run code from a specific directory."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    try:
        import os

        os.chdir(path)
        yield path
    finally:
        os.chdir(previous)


def unitcell_kwargs(settings: dict[str, Any]) -> dict[str, Any]:
    """Build keyword arguments for nxprocess_lab.LabReduce_unitcell."""

    keys = [
        "file",
        "refined_unitcell_a",
        "refined_unitcell_b",
        "refined_unitcell_c",
        "refined_unitcell_alpha",
        "refined_unitcell_beta",
        "refined_unitcell_gamma",
    ]
    require_values(settings, keys)
    return {
        "file": settings["file"],
        "unitcell_a": settings["refined_unitcell_a"],
        "unitcell_b": settings["refined_unitcell_b"],
        "unitcell_c": settings["refined_unitcell_c"],
        "unitcell_alpha": settings["refined_unitcell_alpha"],
        "unitcell_beta": settings["refined_unitcell_beta"],
        "unitcell_gamma": settings["refined_unitcell_gamma"],
    }


def labreduce_kwargs(settings: dict[str, Any], threshold: float | None = None) -> dict[str, Any]:
    """Build starter keyword arguments for nxprocess_lab.LabReduce."""

    require_values(settings, ["file", "imstart", "imend", "null_image", "imstartdiff"])
    return {
        "file": settings["file"],
        "imstart": int(settings["imstart"]),
        "imend": int(settings["imend"]),
        "threshold": threshold,
        "null_image": bool(settings["null_image"]),
        "imstartdiff": int(settings["imstartdiff"]),
    }


def labreduce_peaklist_kwargs(settings: dict[str, Any], peaklist) -> dict[str, Any]:
    """Build keyword arguments for nxprocess_lab.LabReduce_peaklist."""

    require_values(settings, ["file", "imstart", "imend", "null_image", "imstartdiff"])
    return {
        "file": settings["file"],
        "imstart": int(settings["imstart"]),
        "imend": int(settings["imend"]),
        "threshold": None,
        "null_image": bool(settings["null_image"]),
        "imstartdiff": int(settings["imstartdiff"]),
        "peaklist": peaklist,
    }


def write_manual_peaklists(prepared: dict[str, dict[str, Any]], repo_root: str | Path, nxprocess_lab, np) -> None:
    """Write /entry/postpeaks from configured curated peak-list files."""

    repo_root = Path(repo_root)
    for scan_id, stage_args in prepared.items():
        settings = stage_args["settings"]
        peaklist_file = settings.get("manual_peaklist_file")
        if not peaklist_file:
            print(f"Skipping {scan_id}: no manual_peaklist_file configured")
            continue

        peaklist_path = repo_root / peaklist_file
        if not peaklist_path.exists():
            raise FileNotFoundError(f"Manual peak list not found for {scan_id}: {peaklist_path}")

        try:
            peaklist = np.loadtxt(peaklist_path, delimiter=",")
        except ValueError:
            peaklist = np.loadtxt(peaklist_path)
        if peaklist.ndim == 1:
            peaklist = peaklist.reshape(1, -1)
        if peaklist.shape[1] != 4:
            raise ValueError(
                f"Manual peak list for {scan_id} must have 4 columns: "
                "z_frame, y, x, intensity"
            )

        output_dir = repo_root / settings["output_directory"]
        print(f"Writing manual peak list for {scan_id}: {peaklist_path}")
        with working_directory(output_dir):
            nxprocess_lab.LabReduce_peaklist(**labreduce_peaklist_kwargs(settings, peaklist))


def prepare_orientation(
    settings: dict[str, Any],
    repo_root: str | Path,
    nxprocess_lab,
    hklselect: bool = True,
    peak_tolerance: float = 5.0,
    ring_tolerance: float = 1.0,
    hkl_tolerance: float = 0.05,
    mode: int = 1,
):
    """Run automatic orientation candidate search for one scan."""

    require_values(settings, ["file", "output_directory"])
    output_dir = Path(repo_root) / settings["output_directory"]
    with working_directory(output_dir):
        return nxprocess_lab.LabRefine_prepare(
            file=settings["file"],
            hklselect=hklselect,
            peak_tolerance=peak_tolerance,
            ring_tolerance=ring_tolerance,
            hkl_tolerance=hkl_tolerance,
            mode=mode,
        )


def check_orientation(refinevars, nxprocess_lab, hkl_tolerance: float = 0.1, write_peaklist: bool = True) -> None:
    """Print automatic orientation candidate diagnostics."""

    nxprocess_lab.LabRefine_check(
        refinevars=refinevars,
        hkl_tolerance=hkl_tolerance,
        write_peaklist=write_peaklist,
    )


def select_orientation_grain(refinevars, nxprocess_lab, grain: int = 0) -> None:
    """Write the selected automatic orientation grain to the NeXus file."""

    nxprocess_lab.LabRefine_grain(refinevars=refinevars, grain=grain)


def list_manual_peak_hkls(refinevars, nxprocess_lab, i: int, j: int, ring_tolerance: int = 0) -> None:
    """Print possible HKL assignments for two selected peaks."""

    nxprocess_lab.UBpeaks(
        refinevars=refinevars,
        i=i,
        j=j,
        ring_tolerance=ring_tolerance,
    )


def check_manual_peak_orientation(
    refinevars,
    nxprocess_lab,
    i: int,
    j: int,
    i_ring: int = 0,
    j_ring: int = 0,
) -> None:
    """Check a manual pair/ring assignment using ImageD11 orientation output."""

    nxprocess_lab.UBorientcheck(
        refinevars=refinevars,
        i=i,
        j=j,
        i_ring=i_ring,
        j_ring=j_ring,
    )


def prepare_manual_ub(
    refinevars,
    nxprocess_lab,
    i: int,
    j: int,
    hkl_tolerance: float = 0.1,
    write_peaklist: bool = True,
    ring_tolerance: int | None = None,
):
    """Generate candidate manual UB matrices from two selected peaks."""

    if ring_tolerance is None:
        return nxprocess_lab.UBmanual_prepare(
            refinevars=refinevars,
            i=i,
            j=j,
            hkl_tolerance=hkl_tolerance,
            write_peaklist=write_peaklist,
        )
    return nxprocess_lab.UBmanual_ring_tolerance_prepare(
        refinevars=refinevars,
        i=i,
        j=j,
        ring_tolerance=ring_tolerance,
        hkl_tolerance=hkl_tolerance,
        write_peaklist=write_peaklist,
    )


def select_manual_ub(refinevars, nxprocess_lab, ulist, iU: int = 0) -> None:
    """Write one candidate manual UB matrix to the NeXus file."""

    nxprocess_lab.UBmanual(refinevars=refinevars, Ulist=ulist, iU=iU)


def q_vectors_from_config(config: dict[str, Any]) -> list[tuple[float, float, float]]:
    """Return CDW q vectors from the experiment config."""

    q_vectors = []
    for index, q_vector in enumerate(config.get("q_vectors", [])):
        if q_vector is None:
            raise ValueError(f"Bad q_vectors entry at index {index}: {q_vector!r}")
        try:
            q_vectors.append((float(q_vector["h"]), float(q_vector["k"]), float(q_vector["l"])))
        except KeyError as error:
            raise ValueError(f"q_vectors entry at index {index} is missing {error}") from error
    return q_vectors


def generate_cdw_satellites(bragg_list, q_vectors) -> list[tuple[float, float, float]]:
    """Generate CDW satellites at parent Bragg HKL +/- each q vector."""

    satellites = []
    seen = set()
    for h, k, l in bragg_list:
        for qh, qk, ql in q_vectors:
            for sign in (1.0, -1.0):
                satellite = (h + sign * qh, k + sign * qk, l + sign * ql)
                key = tuple(round(value, 8) for value in satellite)
                if key not in seen:
                    seen.add(key)
                    satellites.append(satellite)
    return satellites


def cdw_targets_for_parent(parent_bragg, q_vectors) -> dict[str, Any]:
    """Return one parent Bragg target group and its CDW satellites."""

    parent = tuple(parent_bragg)
    cdw_targets = generate_cdw_satellites([parent], q_vectors)
    return {
        "parent_bragg": parent,
        "targets": [parent] + cdw_targets,
        "cdw_targets": cdw_targets,
    }


def parent_cdw_target_groups(parent_bragg_list, q_vectors) -> list[dict[str, Any]]:
    """Build target groups for several candidate parent Bragg peaks."""

    return [cdw_targets_for_parent(parent_bragg, q_vectors) for parent_bragg in parent_bragg_list]


def unique_hkls(target_groups) -> list[tuple[float, float, float]]:
    """Return all HKLs from target groups without duplicates."""

    targets = []
    seen = set()
    for group in target_groups:
        for hkl in group["targets"]:
            key = tuple(round(float(value), 8) for value in hkl)
            if key not in seen:
                seen.add(key)
                targets.append(tuple(hkl))
    return targets


def _hkl_key(hkl) -> tuple[float, float, float]:
    return tuple(round(float(value), 8) for value in hkl)


def score_parent_orientation_hits(hits, target_groups) -> list[dict[str, Any]]:
    """Score each chi/omega by how many CDW targets it captures for each parent Bragg."""

    orientation_keys = sorted({(hit["chi"], hit["omega"]) for hit in hits})
    hit_hkls_by_orientation = {}
    for chi, omega in orientation_keys:
        hit_hkls_by_orientation[(chi, omega)] = {
            _hkl_key((hit["h"], hit["k"], hit["l"]))
            for hit in hits
            if hit["chi"] == chi and hit["omega"] == omega
        }

    rows = []
    for group in target_groups:
        parent = tuple(group["parent_bragg"])
        parent_key = _hkl_key(parent)
        cdw_keys = {_hkl_key(hkl) for hkl in group["cdw_targets"]}
        all_target_keys = {parent_key} | cdw_keys
        for chi, omega in orientation_keys:
            seen = hit_hkls_by_orientation[(chi, omega)] & all_target_keys
            if not seen:
                continue
            bragg_seen = parent_key in seen
            cdw_seen = len(seen & cdw_keys)
            group_hits = [
                hit
                for hit in hits
                if hit["chi"] == chi
                and hit["omega"] == omega
                and _hkl_key((hit["h"], hit["k"], hit["l"])) in all_target_keys
            ]
            rows.append(
                {
                    "parent_bragg": parent,
                    "chi": chi,
                    "omega": omega,
                    "bragg_seen": bragg_seen,
                    "cdw_seen": cdw_seen,
                    "total_targets_seen": len(seen),
                    "cdw_total": len(cdw_keys),
                    "min_phi": min(hit["phi"] for hit in group_hits),
                    "max_phi": max(hit["phi"] for hit in group_hits),
                }
            )

    return sorted(
        rows,
        key=lambda row: (
            row["bragg_seen"],
            row["cdw_seen"],
            row["total_targets_seen"],
            -abs(row["max_phi"] - row["min_phi"]),
        ),
        reverse=True,
    )


def prepare_transform(
    settings: dict[str, Any],
    repo_root: str | Path,
    nxprocess_lab,
    itn: int = 1000,
    chunkx: int = 5,
    chunky: int = 5,
    chunkz: int = 1,
) -> dict[str, Any]:
    """Set the usable frame count and prepare transform metadata in one scan file."""

    require_values(settings, ["file", "output_directory"])
    if itn <= 0:
        raise ValueError("Transform preparation itn must be positive")
    if min(chunkx, chunky, chunkz) <= 0:
        raise ValueError("Transform chunk dimensions must be positive")

    output_dir = Path(repo_root) / settings["output_directory"]
    nxs_path = output_dir / (settings["file"] + ".nxs")
    if not nxs_path.exists():
        raise FileNotFoundError(f"Missing scan .nxs file: {nxs_path}")

    nxroot = nxprocess_lab.nxload(str(nxs_path), "rw")
    try:
        with nxroot.entry.nxfile:
            frame_count = int(nxroot.entry.data.intensity.shape[0])
            if "last" not in nxroot.entry.data.attrs:
                nxroot.entry.data.attrs["last"] = frame_count
            else:
                frame_count = int(nxroot.entry.data.attrs["last"])
    finally:
        nxroot.close()

    with working_directory(output_dir):
        nxprocess_lab.LabRefine_transform_prepare(
            file=settings["file"],
            itn=int(itn),
            chunkx=int(chunkx),
            chunky=int(chunky),
            chunkz=int(chunkz),
        )

    nxroot = nxprocess_lab.nxload(str(nxs_path), "r")
    try:
        group = nxroot.entry.transform_prepare
        return {
            "nxs_path": nxs_path,
            "last": frame_count,
            "chunkx": int(group.chunkx.nxdata),
            "chunky": int(group.chunky.nxdata),
            "chunkz": int(group.chunkz.nxdata),
            "subdatalen": int(group.subdatalen.nxdata),
            "datalen": int(group.datalen.nxdata),
        }
    finally:
        nxroot.close()


def transform_local_window(
    settings: dict[str, Any],
    repo_root: str | Path,
    nxprocess_lab,
    peak_id: str,
    xstart: int,
    xend: int,
    ystart: int,
    yend: int,
    zstart: int,
    zend: int,
    center: tuple[float, float, float] | None = None,
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    """Transform one bounded detector/frame window and save its pixel HKLs."""

    require_values(settings, ["scan_id", "file", "output_directory"])
    peak_id = str(peak_id).strip()
    if not peak_id or peak_id in {".", ".."} or "/" in peak_id or "\\" in peak_id:
        raise ValueError("peak_id must be a non-empty filename-safe identifier")
    xstart, xend, ystart, yend, zstart, zend = normalize_detector_window(
        xstart, xend, ystart, yend, zstart, zend
    )

    output_dir = Path(repo_root) / settings["output_directory"]
    nxs_path = output_dir / (settings["file"] + ".nxs")
    if not nxs_path.exists():
        raise FileNotFoundError(f"Missing scan .nxs file: {nxs_path}")

    with working_directory(output_dir):
        data_t, h, k, l = nxprocess_lab.LabRefine_transform_local(
            file=settings["file"],
            xstart=xstart,
            xend=xend,
            ystart=ystart,
            yend=yend,
            zstart=zstart,
            zend=zend,
        )

    if save_path is None:
        save_path = output_dir / "Transforms" / f"{peak_id}_transform_local.npz"
    else:
        save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    center_values = center if center is not None else (np.nan, np.nan, np.nan)
    np.savez_compressed(
        save_path,
        data_t=data_t,
        h=h,
        k=k,
        l=l,
        scan_id=settings["scan_id"],
        peak_id=peak_id,
        xstart=xstart,
        xend=xend,
        ystart=ystart,
        yend=yend,
        zstart=zstart,
        zend=zend,
        x=center_values[0],
        y=center_values[1],
        z=center_values[2],
    )
    return {
        "data_t": data_t,
        "h": h,
        "k": k,
        "l": l,
        "scan_id": settings["scan_id"],
        "peak_id": peak_id,
        "save_path": save_path,
    }


def combine_local_transform(
    settings: dict[str, Any],
    repo_root: str | Path,
    nxprocess_lab,
    peak_id: str,
    deltah: float,
    deltak: float,
    deltal: float,
    local_transform_path: str | Path | None = None,
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    """Re-bin a saved local pixel transform into a regular H/K/L grid."""

    require_values(settings, ["scan_id", "file", "output_directory"])
    if min(deltah, deltak, deltal) <= 0:
        raise ValueError("deltah, deltak, and deltal must be positive")

    output_dir = Path(repo_root) / settings["output_directory"]
    if local_transform_path is None:
        local_transform_path = output_dir / "Transforms" / f"{peak_id}_transform_local.npz"
    else:
        local_transform_path = Path(local_transform_path)
    if not local_transform_path.exists():
        raise FileNotFoundError(f"Missing local transform: {local_transform_path}")

    with np.load(local_transform_path, allow_pickle=False) as local:
        data_t = local["data_t"]
        h = local["h"]
        k = local["k"]
        l = local["l"]

    with working_directory(output_dir):
        transform_data, hrange, krange, lrange = nxprocess_lab.LabRefine_combine_transform(
            file=settings["file"],
            data_t=data_t,
            _h=h,
            _k=k,
            _l=l,
            deltah=float(deltah),
            deltak=float(deltak),
            deltal=float(deltal),
        )

    max_index = np.unravel_index(np.nanargmax(transform_data), transform_data.shape)
    max_hkl = (
        float(hrange[max_index[0]]),
        float(krange[max_index[1]]),
        float(lrange[max_index[2]]),
    )
    if save_path is None:
        save_path = output_dir / "Transforms" / f"{peak_id}_momentum_grid.npz"
    else:
        save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_path,
        transform_data=transform_data,
        Hrange=hrange,
        Krange=krange,
        Lrange=lrange,
        scan_id=settings["scan_id"],
        peak_id=peak_id,
        deltah=float(deltah),
        deltak=float(deltak),
        deltal=float(deltal),
        max_hkl=np.asarray(max_hkl),
    )
    return {
        "transform_data": transform_data,
        "Hrange": hrange,
        "Krange": krange,
        "Lrange": lrange,
        "max_hkl": max_hkl,
        "save_path": save_path,
        "local_transform_path": local_transform_path,
    }


def estimate_momentum_bin_sizes(
    local_transform_path: str | Path,
    percentile: float = 90.0,
    coverage_factor: float = 1.25,
) -> dict[str, Any]:
    """Estimate H/K/L bin sizes from adjacent local-transform pixel spacings."""

    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in the range (0, 100]")
    if coverage_factor <= 0:
        raise ValueError("coverage_factor must be positive")

    local_transform_path = Path(local_transform_path)
    if not local_transform_path.exists():
        raise FileNotFoundError(f"Missing local transform: {local_transform_path}")

    with np.load(local_transform_path, allow_pickle=False) as local:
        h = local["h"]
        k = local["k"]
        l = local["l"]
        nx = int(local["xend"] - local["xstart"])
        ny = int(local["yend"] - local["ystart"])
        nz = int(local["zend"] - local["zstart"])

    expected_size = nx * ny * nz
    if min(nx, ny, nz) <= 0 or h.size != expected_size:
        raise ValueError(
            "Local transform shape metadata does not match flattened H/K/L arrays"
        )

    def spacing_summary(values: np.ndarray) -> dict[str, float]:
        cube = values.reshape(nz, ny, nx)
        diffs = []
        for axis in range(3):
            axis_diffs = np.abs(np.diff(cube, axis=axis)).ravel()
            axis_diffs = axis_diffs[np.isfinite(axis_diffs) & (axis_diffs > 0)]
            if axis_diffs.size:
                diffs.append(axis_diffs)
        if not diffs:
            raise ValueError("Could not estimate nonzero adjacent pixel spacings")
        spacings = np.concatenate(diffs)
        base = float(np.percentile(spacings, percentile))
        return {
            "min": float(np.min(spacings)),
            "median": float(np.median(spacings)),
            "percentile": base,
            "suggested": base * float(coverage_factor),
            "range_min": float(np.min(values)),
            "range_max": float(np.max(values)),
        }

    h_summary = spacing_summary(h)
    k_summary = spacing_summary(k)
    l_summary = spacing_summary(l)
    return {
        "local_transform_path": local_transform_path,
        "percentile": float(percentile),
        "coverage_factor": float(coverage_factor),
        "deltah": h_summary["suggested"],
        "deltak": k_summary["suggested"],
        "deltal": l_summary["suggested"],
        "h": h_summary,
        "k": k_summary,
        "l": l_summary,
    }


def momentum_line_scans(
    transform_data: np.ndarray,
    Hrange: np.ndarray,
    Krange: np.ndarray,
    Lrange: np.ndarray,
    hmin: float | None = None,
    hmax: float | None = None,
    kmin: float | None = None,
    kmax: float | None = None,
    lmin: float | None = None,
    lmax: float | None = None,
    hstep: int = 2,
    kstep: int = 2,
    lstep: int = 2,
    center_hkl: tuple[float, float, float] | None = None,
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return H/K/L line scans using the same summing logic as nxprocess_lab."""

    if transform_data.ndim != 3:
        raise ValueError("transform_data must be a 3D H/K/L array")
    if transform_data.shape != (len(Hrange), len(Krange), len(Lrange)):
        raise ValueError("transform_data shape must match Hrange/Krange/Lrange lengths")
    if min(hstep, kstep, lstep) < 0:
        raise ValueError("hstep, kstep, and lstep must be non-negative")

    if center_hkl is None:
        center_index = np.unravel_index(np.nanargmax(transform_data), transform_data.shape)
    else:
        center_index = (
            int(np.searchsorted(Hrange, center_hkl[0], side="left")),
            int(np.searchsorted(Krange, center_hkl[1], side="left")),
            int(np.searchsorted(Lrange, center_hkl[2], side="left")),
        )
        center_index = tuple(
            max(0, min(index, transform_data.shape[axis] - 1))
            for axis, index in enumerate(center_index)
        )

    def range_slice(axis_values: np.ndarray, lower, upper) -> slice:
        lower = float(axis_values[0]) if lower is None else float(lower)
        upper = float(axis_values[-1]) if upper is None else float(upper)
        start = int(np.searchsorted(axis_values, lower, side="left"))
        stop = int(np.searchsorted(axis_values, upper, side="left"))
        start = max(0, min(start, len(axis_values) - 1))
        stop = max(start + 1, min(stop, len(axis_values)))
        return slice(start, stop)

    def centered_slice(index: int, half_width: int, axis_size: int) -> slice:
        return slice(max(0, index - int(half_width)), min(axis_size, index + int(half_width)))

    h_slice = range_slice(Hrange, hmin, hmax)
    k_slice = range_slice(Krange, kmin, kmax)
    l_slice = range_slice(Lrange, lmin, lmax)
    h_window = centered_slice(center_index[0], hstep, transform_data.shape[0])
    k_window = centered_slice(center_index[1], kstep, transform_data.shape[1])
    l_window = centered_slice(center_index[2], lstep, transform_data.shape[2])

    h_scan = np.sum(transform_data[h_slice, k_window, l_window], axis=(1, 2))
    k_scan = np.sum(transform_data[h_window, k_slice, l_window], axis=(0, 2))
    l_scan = np.sum(transform_data[h_window, k_window, l_slice], axis=(0, 1))

    result = {
        "center_index": center_index,
        "center_hkl": (
            float(Hrange[center_index[0]]),
            float(Krange[center_index[1]]),
            float(Lrange[center_index[2]]),
        ),
        "H": {"axis": Hrange[h_slice], "intensity": h_scan},
        "K": {"axis": Krange[k_slice], "intensity": k_scan},
        "L": {"axis": Lrange[l_slice], "intensity": l_scan},
        "windows": {
            "h": (h_window.start, h_window.stop),
            "k": (k_window.start, k_window.stop),
            "l": (l_window.start, l_window.stop),
        },
        "ranges": {
            "h": (h_slice.start, h_slice.stop),
            "k": (k_slice.start, k_slice.stop),
            "l": (l_slice.start, l_slice.stop),
        },
    }

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_path,
            center_index=np.asarray(center_index),
            center_hkl=np.asarray(result["center_hkl"]),
            H_axis=result["H"]["axis"],
            H_intensity=result["H"]["intensity"],
            K_axis=result["K"]["axis"],
            K_intensity=result["K"]["intensity"],
            L_axis=result["L"]["axis"],
            L_intensity=result["L"]["intensity"],
            h_window=np.asarray(result["windows"]["h"]),
            k_window=np.asarray(result["windows"]["k"]),
            l_window=np.asarray(result["windows"]["l"]),
            h_range=np.asarray(result["ranges"]["h"]),
            k_range=np.asarray(result["ranges"]["k"]),
            l_range=np.asarray(result["ranges"]["l"]),
        )
        result["save_path"] = save_path

    return result


def predict_bragg_positions(
    settings: dict[str, Any],
    repo_root: str | Path,
    nxprocess_lab,
    bragg_list,
    detector_size: int | None = None,
    keep_off_detector: bool = False,
    frame_min: int | None = None,
    frame_max: int | None = None,
):
    """Project Bragg HKLs to detector coordinates using the current orientation matrix."""

    require_values(settings, ["file", "output_directory"])
    if detector_size is None:
        detector_size = int(settings.get("imsize", 3450))
    first_frame = int(settings.get("imstart", 1))
    last_frame = int(settings.get("imend", 400))
    accepted_frame_min = first_frame if frame_min is None else int(frame_min)
    accepted_frame_max = last_frame if frame_max is None else int(frame_max)
    predictions = []
    output_dir = Path(repo_root) / settings["output_directory"]
    with working_directory(output_dir):
        for h, k, l in bragg_list:
            peaks = nxprocess_lab.LabRefine_getxyz(file=settings["file"], h=h, k=k, l=l)
            if not peaks:
                continue
            for peak in peaks:
                x = float(peak.x)
                y = float(peak.y)
                z = float(peak.z)
                z_frame = z + first_frame
                if not keep_off_detector:
                    on_detector = 0 <= x <= detector_size and 0 <= y <= detector_size
                    in_scan = accepted_frame_min <= z_frame <= accepted_frame_max
                    if not (on_detector and in_scan):
                        continue
                predictions.append(
                    {
                        "h": h,
                        "k": k,
                        "l": l,
                        "x": x,
                        "y": y,
                        "z": z,
                        "z_frame": z_frame,
                    }
                )
    return predictions


def plan_scan_orientations(
    settings: dict[str, Any],
    repo_root: str | Path,
    nxprocess_lab,
    target_hkls,
    chi_values,
    omega_values=(None,),
    detector_size: int | None = None,
):
    """Find chi/omega settings where target HKLs hit the detector during the scan."""

    require_values(settings, ["file", "output_directory", "phi_start", "phi_end", "phi_step"])
    if detector_size is None:
        detector_size = int(settings.get("imsize", 3450))

    first_frame = int(settings.get("imstart", 1))
    phi_start = float(settings["phi_start"])
    phi_end = float(settings["phi_end"])
    phi_step = float(settings["phi_step"])
    output_dir = Path(repo_root) / settings["output_directory"]
    hits = []

    with working_directory(output_dir):
        nxroot = nxprocess_lab.nxload(settings["file"] + ".nxs", "r")
        refine = nxprocess_lab.nxrefine_lab.NXRefine(nxroot)
        original_chi = refine.chi
        original_omega = refine.omega

        try:
            for chi in chi_values:
                refine.chi = float(chi)
                for omega in omega_values:
                    if omega is not None:
                        refine.omega = float(omega)
                    for h, k, l in target_hkls:
                        for peak in refine.get_xyz(h, k, l):
                            x = float(peak.x)
                            y = float(peak.y)
                            z = float(peak.z)
                            phi = phi_start + z * phi_step
                            if not (0 <= x <= detector_size and 0 <= y <= detector_size):
                                continue
                            if not (phi_start <= phi <= phi_end):
                                continue
                            hits.append(
                                {
                                    "h": h,
                                    "k": k,
                                    "l": l,
                                    "chi": float(chi),
                                    "omega": float(refine.omega),
                                    "phi": phi,
                                    "z": z,
                                    "z_frame": first_frame + z,
                                    "x": x,
                                    "y": y,
                                }
                            )
        finally:
            refine.chi = original_chi
            refine.omega = original_omega

    return hits


def get_hkl_at_pixel(settings: dict[str, Any], repo_root: str | Path, nxprocess_lab, x, y, z_frame):
    """Return HKL and angles for a detector pixel using LabRefine_gethkl."""

    require_values(settings, ["file", "output_directory"])
    output_dir = Path(repo_root) / settings["output_directory"]
    stream = io.StringIO()
    with working_directory(output_dir):
        with redirect_stdout(stream):
            nxprocess_lab.LabRefine_gethkl(
                file=settings["file"],
                x=x,
                y=y,
                z_frame=z_frame,
            )
    text = stream.getvalue()
    hkl_match = re.search(
        r"H\s*=\s*([-\d.]+),\s*K\s*=\s*([-\d.]+),\s*L\s*=\s*([-\d.]+)",
        text,
    )
    angle_match = re.search(
        r"polar\s*=\s*\[?([-\d.]+)\]?,\s*azimuthal\s*=\s*\[?([-\d.]+)\]?",
        text,
    )
    if not hkl_match:
        raise ValueError(f"Could not parse HKL from LabRefine_gethkl output:\n{text}")

    result = {
        "h": float(hkl_match.group(1)),
        "k": float(hkl_match.group(2)),
        "l": float(hkl_match.group(3)),
        "raw_output": text,
    }
    if angle_match:
        result["polar"] = float(angle_match.group(1))
        result["azimuthal"] = float(angle_match.group(2))
    return result


def ubcopy_kwargs(
    reference_settings: dict[str, Any],
    scan_settings: dict[str, Any],
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build keyword arguments for nxprocess_lab.UBcopy."""

    require_values(reference_settings, ["file", "output_directory"])
    require_values(scan_settings, ["file", "output_directory"])
    if repo_root is None:
        parent_file = reference_settings["file"]
        child_file = scan_settings["file"]
    else:
        root = Path(repo_root)
        parent_file = root / reference_settings["output_directory"] / reference_settings["file"]
        child_file = root / scan_settings["output_directory"] / scan_settings["file"]
    return {
        "parent_file": str(parent_file),
        "file": str(child_file),
    }


def copy_reference_ub_to_strain_scans(
    prepared: dict[str, dict[str, Any]],
    repo_root: str | Path,
    nxprocess_lab,
    reference_scan_id: str | None = None,
) -> dict[str, list[str]]:
    """Copy the finalized reference UB/orientation matrix to every strain scan.

    The notebook builds a ``prepared`` dictionary where each non-reference scan
    already has the parent/child file paths needed by ``nxprocess_lab.UBcopy``.
    This helper applies that copy consistently across the whole scan set and
    reports what happened, so a newly added batch cannot accidentally keep the
    identity orientation matrix.
    """

    repo_root = Path(repo_root)
    copied: list[str] = []
    skipped_reference: list[str] = []
    missing: list[str] = []

    if reference_scan_id is None:
        reference_scan_ids = [
            scan_id
            for scan_id, stage_args in prepared.items()
            if stage_args["settings"].get("role") == "reference"
        ]
        if len(reference_scan_ids) != 1:
            raise ValueError(
                "Pass reference_scan_id when prepared does not contain exactly one reference scan"
            )
        reference_scan_id = reference_scan_ids[0]

    reference_settings = prepared[reference_scan_id]["settings"]
    reference_file = (
        repo_root
        / reference_settings["output_directory"]
        / f"{reference_settings['file']}.nxs"
    )
    if not reference_file.exists():
        raise FileNotFoundError(f"Missing reference .nxs for UB copy: {reference_file}")

    for scan_id, stage_args in prepared.items():
        settings = stage_args["settings"]
        if settings.get("role") == "reference":
            skipped_reference.append(scan_id)
            continue

        ubcopy = stage_args.get("ubcopy")
        if ubcopy is None:
            ubcopy = ubcopy_kwargs(reference_settings, settings, repo_root)

        child_file = Path(ubcopy["file"] + ".nxs")
        if not child_file.exists():
            print(f"Skipping UB copy to {scan_id}; missing child .nxs: {child_file}")
            missing.append(scan_id)
            continue

        print(f"Copying reference UB to {scan_id}")
        nxprocess_lab.UBcopy(**ubcopy)
        copied.append(scan_id)

    return {
        "copied": copied,
        "skipped_reference": skipped_reference,
        "missing": missing,
    }
