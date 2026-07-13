# Pipeline Design

For notebook helper functions, see [helper_function_reference.md](helper_function_reference.md).

## Experimental Model

Each scan is a full theta scan over the same angular range while strain is varied. In the ideal workflow, theta and strain are the two experimental degrees of freedom after calibration and sample setup are fixed.

- Theta scan: probes Bragg and CDW reflections.
- Strain axis: used to study how CDW intensity, position, and width change under applied strain.

## Existing Lab Capabilities To Reuse

The local lab modules already provide the core diffraction operations:

- `nxprocess_lab.LabCreate`: create `.nxs`, write sample/instrument metadata, and link or build detector data.
- `nxprocess_lab.LabReduce`: calculate max/summed data, find peaks, and write `/entry/peaks`.
- `nxprocess_lab.LabReduce_peakselect`: create cleaned `/entry/postpeaks` from selected peaks.
- `nxprocess_lab.LabReduce_peaklist`: create `/entry/postpeaks` from a manual peak list.
- `nxprocess_lab.LabReduce_check`: export radial/powder-style peak checks.
- `nxprocess_lab.LabReduce_unitcell`: update refined unit-cell metadata.
- `nxprocess_lab.LabRefine_prepare`: generate candidate orientation matrices.
- `nxprocess_lab.LabRefine_check`: report peaks that do not index near integer HKL.
- `nxprocess_lab.LabRefine_grain`: choose and write a candidate orientation matrix.
- `nxprocess_lab.LabRefine`: refine cell/goniometer/orientation parameters.
- `nxprocess_lab.UBcopy`: copy the reference orientation matrix to another scan.
- `nxprocess_lab.LabRefine_gethkl` and `LabRefine_getxyz`: convert detector coordinates and HKL.
- `nxprocess_lab.LabRefine_transform_*`: transform detector data into reciprocal space.
- `nxprocess_lab.HKmap`, `KLmap`, `HLmap`, `Hscan`, `Kscan`, `Lscan`: inspect transformed maps.

## Required NeXus Structure

Before any stage runs, the pipeline should validate that the required groups exist.

### After Creation

```text
/entry/data/intensity
/entry/data/frame_number
/entry/instrument/calibration/refinement/parameters/Wavelength
/entry/instrument/detector/beam_center_x
/entry/instrument/detector/beam_center_y
/entry/instrument/detector/distance
/entry/instrument/detector/orientation_matrix
/entry/instrument/detector/pixel_size
/entry/instrument/detector/shape
/entry/instrument/goniometer/chi
/entry/instrument/goniometer/omega
/entry/instrument/goniometer/phi
/entry/instrument/goniometer/two_theta
/entry/sample/name
/entry/sample/temperature
/entry/sample/unit_cell_group
/entry/sample/lattice_centring
/entry/sample/unitcell_a
/entry/sample/unitcell_b
/entry/sample/unitcell_c
/entry/sample/unitcell_alpha
/entry/sample/unitcell_beta
/entry/sample/unitcell_gamma
```

### After Peak Selection

```text
/entry/peaks
/entry/postpeaks
/entry/postpeaks/x
/entry/postpeaks/y
/entry/postpeaks/z
/entry/postpeaks/z_frame
/entry/postpeaks/polar_angle
/entry/postpeaks/azimuthal_angle
/entry/postpeaks/intensity
```

### After Orientation/Refinement

```text
/entry/instrument/detector/orientation_matrix
/entry/postpeaks/h
/entry/postpeaks/k
/entry/postpeaks/l
```

## Reference Scan Strategy

The reference scan is the only scan that should require careful manual indexing.

1. Create and reduce the reference scan.
2. Remove beryllium-dome and saturated artifacts from the peak list.
3. Use manually chosen reliable Bragg peaks to establish UB if auto-indexing fails.
4. Apply the refined GSAS-II unit cell to the `.nxs` metadata.
5. Refine the reference orientation.
6. Save the selected orientation matrix and final unit cell.
7. Copy this UB matrix to all strain scans.

The working assumption is that strain changes intensities and produces small peak shifts, but does not invalidate the orientation matrix enough to require re-indexing every scan.

## CDW Target Strategy

For each indexed Bragg peak, generate CDW satellite targets using selected q vectors.

Typical target form:

```text
H_cdw = H_bragg +/- q_h
K_cdw = K_bragg +/- q_k
L_cdw = L_bragg +/- q_l
```

The pipeline should store each target with:

- parent Bragg HKL
- q-vector label
- predicted CDW HKL
- predicted detector coordinates, if visible in scan range
- detector transform window
- reciprocal-space integration window
- quality flags such as saturated, boundary-touching, missing, weak, or overlapping

## Window Selection Problem

Manual `z, y, x` detector windows are the current bottleneck. The first implementation should support saved window definitions from a manifest. Later, a small GUI can be added to choose detector windows interactively.

Manual peak-list files are expected to use columns `z_frame, y, x, intensity`, where `z_frame` is the real TIFF image number. Detector coordinates follow image convention: origin at the top-left, `x` increases right, and `y` increases downward. Overlay PNGs in `Max_of_All/` should use the same coordinate convention.

Window selection should become reproducible:

- every chosen window is saved with scan id, target id, and bounds
- the same window rule can be reused across strain scans
- window changes are explicit and auditable

## Reference-to-Series Analysis Stages

The repeatable analysis unit is:

```text
(scan_id, peak_id)
```

Each `peak_id` identifies one CDW target/window. A scan can contain several peak IDs, and every peak ID can be processed across several strain/temperature scans.

### Stage 5: Reference Reciprocal-Space Transformation

- select and save detector/frame windows on the reference scan
- assign every window a unique `peak_id`
- save definitions under `configs/peak_windows/<peak_id>.json`
- transform each reference window to per-pixel H/K/L
- save reference transforms under `Processed_Data/<reference_scan_id>/Transforms/`

### Stage 6: Reference Momentum-Space Analysis

- determine suitable `deltah`, `deltak`, and `deltal` for each peak
- inspect H/K/L maps and momentum scans
- establish reciprocal-space ROI and background rules
- save accepted analysis parameters by `peak_id`

### Stage 7: Apply Across All Scans

- load every accepted peak/window definition
- apply the same transform and momentum-analysis rules to each configured scan
- organize outputs by `(scan_id, peak_id)`
- collect intensity, centroid, width, background, and quality flags into summary tables

The first implementation can reuse fixed detector/frame bounds. This assumes strain, temperature, geometry, and sample mounting do not move the peak outside the reference window. A later recentering mode may shift the detector window to each scan's predicted peak center while preserving its dimensions and the reference momentum-space ROI; every applied shift must be recorded.

Parallel execution should operate across scans using separate worker processes. Peaks within one scan should initially run sequentially because the copied lab functions open the same NeXus/HDF5 file in read-write mode, which is not a safe target for concurrent workers.

## Integration Strategy

The integration method should not depend on subjective visual picking for every scan.

Baseline objective outputs:

- integrated intensity
- background-subtracted integrated intensity
- centroid in H, K, L
- standard deviation or covariance in H, K, L
- peak maximum
- background level
- number of included voxels
- flags for double-peak behavior, saturation, or poor fit

Because the x-ray source can create double peaks, the first robust approach should avoid assuming a single Gaussian peak. Start with a mask/threshold or moment-based integration inside a predefined reciprocal-space ROI, then add optional model fitting later.

## Directory Layout

Proposed project-managed data layout:

```text
project_root/
  configs/
    experiment.yaml
    scans.csv
  data/
    raw/
    processed/
      scan_000_reference/
        scan.nxs
        scan.hdf5
        peaks/
        transforms/
        integration/
      scan_001_strain_.../
  results/
    cdw_summary.csv
    strain_series.csv
    figures/
```

During active development, raw TIFFs should live directly inside this project under:

```text
Raw_Data/
  <DATA_directory>/
```

Each scan in the YAML sets `DATA_directory`, and the pipeline turns it into the `datafile_path` expected by `LabCreate`.

## Immediate Implementation Steps

1. Define a manifest format for scans, q vectors, and analysis targets.
2. Add `.nxs` validation helpers.
3. Add wrappers for `LabCreate`, `LabReduce_unitcell`, `UBcopy`, and HKL prediction.
4. Add a saved detector-window format.
5. Add local transform and objective integration routines.
6. Add summary tables for strain-dependent CDW intensity and centroid shifts.
