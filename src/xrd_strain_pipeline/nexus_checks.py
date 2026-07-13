"""Validation helpers for the NeXus structure expected by the lab pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


CREATED_REQUIRED_PATHS = (
    "/entry/data/intensity",
    "/entry/data/frame_number",
    "/entry/instrument/calibration/refinement/parameters/Wavelength",
    "/entry/instrument/detector/beam_center_x",
    "/entry/instrument/detector/beam_center_y",
    "/entry/instrument/detector/distance",
    "/entry/instrument/detector/orientation_matrix",
    "/entry/instrument/detector/pixel_size",
    "/entry/instrument/detector/shape",
    "/entry/instrument/goniometer/chi",
    "/entry/instrument/goniometer/omega",
    "/entry/instrument/goniometer/phi",
    "/entry/instrument/goniometer/two_theta",
    "/entry/sample/name",
    "/entry/sample/temperature",
    "/entry/sample/unit_cell_group",
    "/entry/sample/lattice_centring",
    "/entry/sample/unitcell_a",
    "/entry/sample/unitcell_b",
    "/entry/sample/unitcell_c",
    "/entry/sample/unitcell_alpha",
    "/entry/sample/unitcell_beta",
    "/entry/sample/unitcell_gamma",
)

POSTPEAK_REQUIRED_PATHS = (
    "/entry/peaks",
    "/entry/postpeaks",
    "/entry/postpeaks/x",
    "/entry/postpeaks/y",
    "/entry/postpeaks/z",
    "/entry/postpeaks/z_frame",
    "/entry/postpeaks/polar_angle",
    "/entry/postpeaks/azimuthal_angle",
    "/entry/postpeaks/intensity",
)

REFINED_REQUIRED_PATHS = (
    "/entry/instrument/detector/orientation_matrix",
    "/entry/postpeaks/h",
    "/entry/postpeaks/k",
    "/entry/postpeaks/l",
)


@dataclass(frozen=True)
class ValidationResult:
    """Result of checking an opened NeXus root object."""

    stage: str
    missing_paths: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing_paths


def missing_paths(nxroot, required_paths: Iterable[str]) -> tuple[str, ...]:
    """Return required NeXus paths that are absent from an opened nxroot."""

    missing = []
    for path in required_paths:
        try:
            nxroot[path]
        except Exception:
            missing.append(path)
    return tuple(missing)


def validate_created(nxroot) -> ValidationResult:
    """Validate the structure expected after LabCreate."""

    return ValidationResult(
        stage="created",
        missing_paths=missing_paths(nxroot, CREATED_REQUIRED_PATHS),
    )


def validate_postpeaks(nxroot) -> ValidationResult:
    """Validate the structure expected after peak cleanup/selection."""

    return ValidationResult(
        stage="postpeaks",
        missing_paths=missing_paths(nxroot, POSTPEAK_REQUIRED_PATHS),
    )


def validate_refined(nxroot) -> ValidationResult:
    """Validate the structure expected after orientation/HKL assignment."""

    return ValidationResult(
        stage="refined",
        missing_paths=missing_paths(nxroot, REFINED_REQUIRED_PATHS),
    )
