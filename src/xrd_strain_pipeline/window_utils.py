"""Window-bound normalization shared by plotting and transform helpers."""

from __future__ import annotations

import json
from pathlib import Path
import warnings


def normalize_bounds(start: int, end: int, axis_name: str) -> tuple[int, int]:
    """Return increasing integer bounds, warning when input order is reversed."""

    start = int(start)
    end = int(end)
    if start == end:
        raise ValueError(f"{axis_name} window has zero width: start and end are both {start}")
    if start > end:
        warnings.warn(
            f"{axis_name} start ({start}) is larger than end ({end}); switching them.",
            UserWarning,
            stacklevel=2,
        )
        start, end = end, start
    return start, end


def normalize_detector_window(
    xstart: int,
    xend: int,
    ystart: int,
    yend: int,
    zstart: int | None = None,
    zend: int | None = None,
):
    """Normalize X/Y and optional Z detector-window bounds."""

    xstart, xend = normalize_bounds(xstart, xend, "X")
    ystart, yend = normalize_bounds(ystart, yend, "Y")
    if (zstart is None) != (zend is None):
        raise ValueError("Provide both zstart and zend, or neither")
    if zstart is None:
        return xstart, xend, ystart, yend
    zstart, zend = normalize_bounds(zstart, zend, "Z")
    return xstart, xend, ystart, yend, zstart, zend


def save_window_definition(
    path: str | Path,
    peak_id: str,
    reference_scan_id: str,
    bounds,
    center,
    target_hkl=None,
    parent_bragg_hkl=None,
    extra_metadata: dict | None = None,
) -> Path:
    """Save one reusable reference detector-window definition as JSON."""

    path = Path(path)
    xstart, xend, ystart, yend, zstart, zend = normalize_detector_window(*bounds)
    document = {
        "schema_version": 1,
        "peak_id": str(peak_id),
        "reference_scan_id": str(reference_scan_id),
        "coordinate_convention": "Albula: top-left origin; x right; y down; z is real frame number",
        "center": {
            "x": float(center[0]),
            "y": float(center[1]),
            "z_frame": float(center[2]),
        },
        "bounds": {
            "xstart": xstart,
            "xend": xend,
            "ystart": ystart,
            "yend": yend,
            "zstart": zstart,
            "zend": zend,
        },
        "target_hkl": list(target_hkl) if target_hkl is not None else None,
        "parent_bragg_hkl": list(parent_bragg_hkl) if parent_bragg_hkl is not None else None,
    }
    if extra_metadata:
        document.update(extra_metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def load_window_definition(path: str | Path) -> dict:
    """Load a saved detector-window definition."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    if document.get("schema_version") != 1:
        raise ValueError(f"Unsupported detector-window schema in {path}")
    return document
