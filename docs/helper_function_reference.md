# Helper Function Reference

This document describes the helper functions written for the pipeline. These helpers keep `main_pipeline.ipynb` readable and translate YAML settings into calls to the existing lab code in `nxprocess_lab.py` and `nxrefine_lab.py`.

The helpers are not meant to replace the lab code. They mainly handle paths, config validation, notebook ergonomics, plotting, and NeXus structure checks.

## `config.py`

Configuration loading and normalization.

### `ConfigError`

Custom error raised when the YAML configuration is missing required information or has inconsistent scan IDs.

### `load_experiment_config(path)`

Loads the experiment YAML file and returns it as a Python dictionary.

Use this at the start of the notebook:

```python
config = load_experiment_config(repo_root / "configs" / "experiment_template.yaml")
```

### `scan_by_id(config, scan_id)`

Returns one scan block from `config["scans"]` by matching `scan_id`.

Raises `ConfigError` if the scan ID does not exist.

### `reference_scan(config)`

Returns the scan listed by:

```yaml
reference_scan_id: ...
```

### `scan_ids(config)`

Returns all scan IDs in YAML order. Also checks that every scan has a `scan_id`.

### `merged_scan_settings(config, scan_id)`

Combines global YAML settings with one scan-specific entry.

This is where several derived paths are created:

- `datafile_path`: relative path from `Processed_Data/<scan_id>/` back to `Raw_Data/<DATA_directory>/`
- `output_directory`: usually `Processed_Data/<scan_id>`
- `calibration_path_without_extension`: relative path from the scan output folder to the calibration file base
- `unitcell_*`: initial unit-cell values used during `.nxs` creation
- `refined_unitcell_*`: refined unit-cell values used during the unit-cell update step

### `all_scan_settings(config)`

Returns `merged_scan_settings(...)` for every scan in YAML order.

### `require_values(settings, keys)`

Checks that required keys exist and are not `None`.

Raises `ConfigError` with the missing key names.

## `lab_wrappers.py`

Adapters between pipeline settings and existing `nxprocess_lab` functions.

### `add_lab_module_path(repo_root, lab_module_path)`

Adds the folder containing `nxprocess_lab.py` and `nxrefine_lab.py` to `sys.path`.

In this project, `lab_module_path` is usually:

```yaml
lab_module_path: .
```

### `labcreate_kwargs(settings)`

Builds keyword arguments for:

```python
nxprocess_lab.LabCreate(...)
```

It uses the initial unit cell from YAML, not the refined one.

### `labcreate_output_paths(settings, base_dir=".")`

Returns the `.nxs` and `.hdf5` output paths expected from `LabCreate`.

Used before creation to detect whether files already exist.

### `confirm_existing_outputs(scan_id, settings, base_dir)`

Interactive safety prompt for existing `.nxs` / `.hdf5` files.

Returns one of:

```text
create
delete
skip
stop
```

If the user chooses `delete`, it deletes the existing generated output files for that scan.

### `create_scan_files(prepared, repo_root, nxprocess_lab)`

Loops through prepared scans and calls `LabCreate` inside each scan output directory.

This is why generated `.nxs` / `.hdf5` files go into:

```text
Processed_Data/<scan_id>/
```

### `working_directory(path)`

Context manager that temporarily changes the current working directory.

Used because many lab functions open files by base name, so they need to run from the scan output folder.

Example:

```python
with working_directory(output_dir):
    nxprocess_lab.LabCreate(...)
```

### `unitcell_kwargs(settings)`

Builds keyword arguments for:

```python
nxprocess_lab.LabReduce_unitcell(...)
```

It uses the refined unit cell from YAML:

```yaml
unit_cell_refined:
```

### `labreduce_kwargs(settings, threshold=None)`

Builds starter keyword arguments for:

```python
nxprocess_lab.LabReduce(...)
```

This is the automatic peak-finding path. In the current workflow, manual peak lists are preferred because automatic peak detection can be unreliable.

### `labreduce_peaklist_kwargs(settings, peaklist)`

Builds keyword arguments for:

```python
nxprocess_lab.LabReduce_peaklist(...)
```

This is the manual peak-list path.

### `write_manual_peaklists(prepared, repo_root, nxprocess_lab, np)`

Loads each scan's configured manual peak-list file and writes `/entry/postpeaks` into that scan's `.nxs` file.

Manual peak-list files must have four columns:

```text
z_frame, y, x, intensity
```

`z_frame` is the real TIFF frame number.

### `prepare_orientation(...)`

Runs:

```python
nxprocess_lab.LabRefine_prepare(...)
```

This loads the scan `.nxs` file and prepares the refinement/orientation context.

For automatic orientation, use:

```python
hklselect=True
```

For manual orientation fallback, use:

```python
hklselect=False
```

### `check_orientation(refinevars, nxprocess_lab, ...)`

Runs:

```python
nxprocess_lab.LabRefine_check(...)
```

Prints automatic orientation candidate diagnostics.

### `select_orientation_grain(refinevars, nxprocess_lab, grain=0)`

Runs:

```python
nxprocess_lab.LabRefine_grain(...)
```

Writes the selected automatic grain/orientation to the reference `.nxs` file.

### `list_manual_peak_hkls(refinevars, nxprocess_lab, i, j, ring_tolerance=0)`

Runs:

```python
nxprocess_lab.UBpeaks(...)
```

Prints possible HKL assignments for two selected manual peaks.

`i` and `j` are peak indices from the sorted/loaded postpeak list, not detector coordinates.

### `check_manual_peak_orientation(...)`

Runs:

```python
nxprocess_lab.UBorientcheck(...)
```

Checks whether a chosen pair of peak indices and ring assignments gives a sensible orientation.

### `prepare_manual_ub(...)`

Generates candidate manual UB matrices from two selected peaks.

If `ring_tolerance is None`, it calls:

```python
nxprocess_lab.UBmanual_prepare(...)
```

If `ring_tolerance` is an integer, it calls:

```python
nxprocess_lab.UBmanual_ring_tolerance_prepare(...)
```

Returns `Ulist`, the list of candidate UB matrices.

### `select_manual_ub(refinevars, nxprocess_lab, ulist, iU=0)`

Runs:

```python
nxprocess_lab.UBmanual(...)
```

Writes one selected manual UB matrix from `Ulist` into the reference `.nxs` file.

### `predict_bragg_positions(settings, repo_root, nxprocess_lab, bragg_list, ...)`

Projects configured Bragg HKLs to detector pixel positions using the current orientation matrix in the `.nxs` file.

Internally calls:

```python
nxprocess_lab.LabRefine_getxyz(...)
```

Returns a list of dictionaries like:

```python
{
    "h": 0,
    "k": 0,
    "l": -1,
    "x": ...,
    "y": ...,
    "z": ...,
    "z_frame": ...,
}
```

By default, it keeps predictions only if they are on the detector and within the scan frame range.

### `plan_scan_orientations(settings, repo_root, nxprocess_lab, target_hkls, chi_values, omega_values, ...)`

Uses the stored reference `.nxs` file and current UB/orientation matrix to test which fixed `chi` / `omega` settings allow selected target HKLs to appear during the configured theta/phi scan.

This is for scan planning. It temporarily changes `chi` and `omega` in memory, calls the same Ewald-condition logic used by `LabRefine_getxyz`, and returns hits that are:

- inside the configured scan range
- on the detector

It does not modify the `.nxs` file and does not predict whether the peak intensity will be strong enough to measure.

### `prepare_transform(settings, repo_root, nxprocess_lab, ...)`

Prepares one scan for reciprocal-space transformation.

It:

- ensures `/entry/data:last` contains the usable frame count
- calls `nxprocess_lab.LabRefine_transform_prepare(...)` inside the scan output directory
- writes `/entry/transform_prepare` mask/chunk metadata
- returns the stored frame count and transform-preparation dimensions

This step does not transform detector intensity into H/K/L. Local intensity transformation is performed later with `LabRefine_transform_local` and `LabRefine_combine_transform`.

For a 3450x3450 detector, `chunkx=1` and `chunky=1` create roughly 12 million detector chunks. The pipeline defaults to `chunkx=5`, `chunky=5`, and `chunkz=1` for a more practical preparation pass.

### `transform_local_window(settings, repo_root, nxprocess_lab, ...)`

Runs `LabRefine_transform_local` for one bounded detector/frame window and saves the flattened intensity and per-pixel H/K/L arrays as a compressed NPZ file under the scan output directory.

The required `peak_id` becomes part of the NPZ filename, allowing several peaks to be transformed without overwriting one another:

```text
Processed_Data/<scan_id>/Transforms/<peak_id>_transform_local.npz
```

The X/Y bounds use Albula detector coordinates. Z bounds are real image frame numbers.

### `estimate_momentum_bin_sizes(local_transform_path, ...)`

Reads a saved `<peak_id>_transform_local.npz` file and estimates starting H/K/L bin sizes from adjacent-pixel reciprocal-space spacings.

The returned `deltah`, `deltak`, and `deltal` are intended as first-pass values for `combine_local_transform`. If the HK/KL/HL maps show empty white bins through the peak plane, increase the corresponding delta and re-run only the binning/map cells. The local detector-to-HKL transform does not need to be repeated.

### `prepare_reference_momentum_binning(...)`

High-level helper in `xrd_strain_pipeline.helper` that keeps the notebook compact before binning. It resolves the selected reference scan settings, finds the saved local transform path, optionally runs `estimate_momentum_bin_sizes`, validates fixed `deltah`, `deltak`, and `deltal`, and returns the values needed by `combine_local_transform`.

### `combine_local_transform(settings, repo_root, nxprocess_lab, ...)`

Loads a saved local transform, calls `nxprocess_lab.LabRefine_combine_transform`, and saves the binned regular H/K/L grid as:

```text
Processed_Data/<scan_id>/Transforms/<peak_id>_momentum_grid.npz
```

### `momentum_line_scans(transform_data, Hrange, Krange, Lrange, ...)`

Calculates H, K, and L line scans through the brightest binned momentum voxel using the same summing convention as `nxprocess_lab.Hscan`, `Kscan`, and `Lscan`.

Unlike the original lab functions, this helper returns the scan axes and intensities and can save them as:

```text
Processed_Data/<scan_id>/Transforms/<peak_id>_momentum_line_scans.npz
```

Use these line scans as the first pass for choosing reproducible reciprocal-space integration-window bounds.

### `momentum_line_scan_background_levels(line_scans, background_per_voxel)`

Scales a background level measured per occupied momentum voxel into the correct horizontal background level for each summed H/K/L line scan.

This matters because each line-scan point is not one voxel; it is a sum over a small perpendicular H/K/L window.

### `momentum_integration_bounds_from_background(line_scans, background_level, ...)`

Uses line-scan/background intersections to estimate H/K/L integration bounds. `background_level` may be either a single scalar or the per-axis dictionary returned by `momentum_line_scan_background_levels`.

The returned bounds can populate:

```python
MOMENTUM_INTEGRATION_HMIN
MOMENTUM_INTEGRATION_HMAX
MOMENTUM_INTEGRATION_KMIN
MOMENTUM_INTEGRATION_KMAX
MOMENTUM_INTEGRATION_LMIN
MOMENTUM_INTEGRATION_LMAX
```

### `integrate_momentum_roi(transform_data, Hrange, Krange, Lrange, bounds, ...)`

Integrates a rectangular H/K/L ROI from a binned momentum grid.

It reports:

- raw summed intensity over occupied voxels
- background total and background-subtracted net intensity
- positive background-subtracted intensity
- centroid of positive background-subtracted signal
- brightest voxel HKL
- requested and actual snapped H/K/L bounds

The reference notebook cell saves this summary as:

```text
Processed_Data/<scan_id>/Transforms/<peak_id>_momentum_integration.json
```

### `select_transform_bin_background_patch(...)`

High-level helper in `xrd_strain_pipeline.helper` for choosing a separate detector patch that samples only background.

It uses the same detector-window selection style as the CDW patch:

- preview and optional interactive X/Y detector selection
- save a reusable detector-window JSON
- run `LabRefine_transform_local`
- bin the transformed patch with the same `deltah`, `deltak`, and `deltal`
- report background counts per detector pixel and per occupied momentum voxel

This helper does not subtract background from the CDW signal. It only prepares and saves the background measurement.

### `q_vectors_from_config(config)`

Reads the configured CDW q vectors from:

```yaml
q_vectors:
```

Returns a list of `(h, k, l)` tuples in reciprocal-lattice units.

### `generate_cdw_satellites(bragg_list, q_vectors)`

Builds CDW satellite HKL targets around each parent Bragg peak:

```text
Bragg + q
Bragg - q
```

Duplicate satellite targets are removed after rounding to avoid plotting the same calculated position repeatedly.

### `parent_cdw_target_groups(parent_bragg_list, q_vectors)`

Builds one target group for each candidate parent Bragg peak. Each group contains:

- the parent Bragg HKL
- the six `Bragg +/- q` CDW satellite targets

Use this when scan planning across many possible parent Bragg peaks.

### `unique_hkls(target_groups)`

Returns one deduplicated HKL list from several parent/CDW target groups. This avoids recalculating the same HKL more than once during scan planning.

### `score_parent_orientation_hits(hits, target_groups)`

Scores scan-planning hits by parent Bragg peak and fixed sample orientation.

The returned rows answer:

- which parent Bragg peak was tested
- which `chi` / `omega` orientation was tested
- whether the parent Bragg peak is visible
- how many CDW satellites around that parent are visible
- the phi/theta range where those targets appear

### `get_hkl_at_pixel(settings, repo_root, nxprocess_lab, x, y, z_frame)`

Wraps:

```python
nxprocess_lab.LabRefine_gethkl(...)
```

Returns the calculated HKL and angles for a detector pixel/frame position.

Useful for checking a manually selected peak:

```python
get_hkl_at_pixel(reference_settings, repo_root, nxprocess_lab, x=1631, y=1894, z_frame=239)
```

### `ubcopy_kwargs(reference_settings, scan_settings, repo_root=None)`

Builds keyword arguments for:

```python
nxprocess_lab.UBcopy(...)
```

When `repo_root` is provided, it creates full paths to `.nxs` files inside:

```text
Processed_Data/<scan_id>/
```

The `.nxs` extension is intentionally omitted because `UBcopy` appends `.nxs` internally.

## `plotting.py`

Plotting helpers for manual peak inspection and Bragg projection checks.

### `load_peaklist(path)`

Loads a TXT or CSV peak list.

Expected columns:

```text
z_frame, y, x, intensity
```

Comma-delimited and whitespace-delimited files are both accepted.

### `plot_peaklist(...)`

Plots manual peak-list detector positions, optionally over a PNG image from `Max_of_All/`.

Important coordinate convention:

```text
origin = top-left
x increases right
y increases downward
```

If `save_path` is provided, the figure is saved automatically and parent directories are created.

### `plot_bragg_predictions(...)`

Plots predicted Bragg positions, usually over a max/overview image.

Used to visually check whether the selected UB/orientation matrix is reasonable.

If `save_path` is provided, the figure is saved automatically.

### `plot_cdw_predictions(...)`

Plots both parent Bragg predictions and CDW satellite predictions over a detector overview image.

Current convention:

- Bragg peaks are red hollow circles.
- CDW satellites are cyan hollow triangles.
- Bragg labels are shown by default.
- CDW labels are hidden by default to avoid clutter.
- Detector origin is top-left, so `y` increases downward.

If `save_path` is provided, the figure is saved automatically.

### `plot_detector_window(...)`

Displays the selected local-transform X/Y rectangle on two views of a `Max_of_All` PNG:

- full detector view
- zoomed view around the selected rectangle

The plot uses the top-left detector origin, validates the rectangle against the image dimensions, and verifies that the selected peak center lies inside the window.

Reversed X/Y bounds are automatically reordered with a warning.

### `select_detector_window_interactive(...)`

Opens the full detector image in a separate Matplotlib GUI window and uses a `RectangleSelector` to choose X/Y bounds.

The default popup backend is `TkAgg`, avoiding Conda/PyQt platform-plugin path conflicts on Windows.

- left-drag draws or resizes the rectangle
- mouse wheel zooms around the cursor
- right-drag pans the detector image
- `Z` zooms to the current selection
- `R` resets the full-detector view
- Enter accepts the selection
- Escape cancels

The returned bounds are normalized to increasing integer values. Z-frame bounds remain manual because a max-projection PNG contains no frame axis.

### `resolve_overlay_png(...)`

Resolves an explicitly named overlay or `<scan_id>.png` from `Max_of_All`. If the scan-named file is absent and the folder contains exactly one PNG, that PNG is used with a printed notice.

## `window_utils.py`

Shared detector-window normalization.

### `normalize_detector_window(...)`

Normalizes X/Y and optional Z bounds. When a start value is larger than its end value, the values are switched and a warning is emitted. Equal start/end values remain an error because they define a zero-width window.

### `save_window_definition(...)`

Saves one versioned JSON window manifest containing `peak_id`, reference scan, detector/frame bounds, peak center, coordinate convention, and optional target/parent HKLs. Stage 7 can load these definitions and apply the same analysis target across scans.

### `load_window_definition(...)`

Loads and validates the schema version of a saved peak-window JSON file.

## `nexus_checks.py`

Validation helpers for `.nxs` file structure.

These checks only verify that required paths exist. They do not prove that the orientation or indexing is physically correct.

### `ValidationResult`

Dataclass returned by validation functions.

Fields:

```python
stage
missing_paths
```

Property:

```python
ok
```

is `True` when `missing_paths` is empty.

### `missing_paths(nxroot, required_paths)`

Checks an opened NeXus root object and returns any required paths that are missing.

### `validate_created(nxroot)`

Checks the structure expected after `LabCreate`.

This includes detector image data, calibration, detector geometry, goniometer angles, sample metadata, and unit-cell fields.

### `validate_postpeaks(nxroot)`

Checks the structure expected after peak cleanup/manual peak-list writing.

This includes:

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

### `validate_refined(nxroot)`

Checks the structure expected after UB/orientation assignment.

Currently checks:

```text
/entry/instrument/detector/orientation_matrix
/entry/postpeaks/h
/entry/postpeaks/k
/entry/postpeaks/l
```

This only means those fields exist. Use Bragg projection overlays to judge whether the UB matrix is correct.
