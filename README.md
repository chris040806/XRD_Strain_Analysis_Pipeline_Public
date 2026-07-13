# XRD Strain Analysis Pipeline

Public, code-only demonstration of a notebook-driven X-ray diffraction strain analysis workflow.

This repository is intentionally sanitized for public sharing. It contains the pipeline code, configuration structure, and documentation, but it does not include raw detector images, processed NeXus/HDF5 files, calibration images, peak lists, saved detector windows, or unpublished experimental results.

AI tools were used only for documentation cleanup and debugging support. Scientific choices, validation, and interpretation require human inspection.

## What This Shows

The project demonstrates how to organize an XRD strain/CDW analysis around a reproducible YAML configuration and reusable Python helpers. The workflow covers:

- creating NeXus/HDF5 scan files from detector image stacks
- validating created scan files
- refining or manually selecting a reference UB/orientation matrix
- copying the reference UB matrix to strain scans
- projecting Bragg/CDW satellite positions
- selecting reusable detector windows for signal and background regions
- transforming local detector windows into reciprocal space
- integrating H/K/L regions with background handling
- plotting intensity trends across a strain or voltage series

The notebook is meant as a workflow template. To run it on real data, provide private raw data and edit `configs/experiment_template.yaml` locally.

## Public Data Policy

No experimental data are included in this public repository.

The following paths are placeholders only and are ignored by Git:

```text
Raw_Data/
Processed_Data/
Calibration/
Max_of_All/
Peak_Lists/
configs/peak_windows/
```

These folders are left as local-use placeholders for users who want to run the workflow on their own data.

## Credits and Upstream Tools

This repository is a workflow wrapper around established scientific Python tools and bundled lab-processing scripts. Credit for those upstream tools belongs to their authors and maintainers.

- The bundled `nxprocess_lab.py` and `nxrefine_lab.py` files provide the lab-specific NeXus creation, reduction, orientation, and reciprocal-space transform operations used by the notebook. These lab-processing/refinement scripts are credited to Raymond Osborn at Argonne National Laboratory. They are included here as local research workflow code rather than claimed as newly designed pipeline infrastructure. The public copy preserves them so the wrapper notebook can be read in context, and attribution to the original lab-code author should be retained.
- The NeXus file API is provided by [`nexusformat`](https://pypi.org/project/nexusformat/), authored by Raymond Osborn and maintained through the [NeXpy](https://github.com/nexpy) project.
- Detector calibration and azimuthal-integration functionality comes from [`pyFAI`](https://pypi.org/project/pyFAI/), authored by Jérôme Kieffer and maintained by the silx/ESRF ecosystem.
- Peak finding and crystallographic orientation routines use [`ImageD11`](https://pypi.org/project/ImageD11/).
- TIFF I/O uses [`tifffile`](https://pypi.org/project/tifffile/), authored by Christoph Gohlke.
- Nonlinear fitting support comes from [`lmfit`](https://pypi.org/project/lmfit/), maintained by the LMFit Development Team.
- The workflow also depends on standard scientific Python packages including NumPy, SciPy, Matplotlib, h5py, and PyYAML.

If this code is reused, please keep these upstream credits and follow the licenses of the bundled code and third-party packages. The repository license applies to the public wrapper code, documentation, and configuration scaffold authored for this demo; ownership and licensing of bundled third-party/lab scripts remain with their original authors unless otherwise stated.

### Edits to Bundled Lab Scripts

The original bundled lab scripts are credited to Raymond Osborn. Direct edits to those files are intentionally kept minimal and documented separately in `docs/lab_library_changes.md`.

- `nxprocess_lab.py`: one project edit is recorded in `LabRefine_transform_prepare(...)`. The voxel-direction sampling list was corrected from sampling the detector-frame/Z direction twice to sampling one step along detector X, detector Y, and frame/Z. This affects the estimated H/K/L step sizes used during reciprocal-space transform preparation.
- `nxrefine_lab.py`: no direct project modifications are currently recorded.

## Repository Layout

```text
.
├── main_pipeline.ipynb              # Stepwise analysis workflow template
├── configs/
│   └── experiment_template.yaml     # Sanitized demo configuration
├── src/xrd_strain_pipeline/         # Reusable pipeline helpers
├── docs/                            # Design notes and helper references
├── nxprocess_lab.py                 # Bundled lab processing code
├── nxrefine_lab.py                  # Bundled lab refinement code
├── environment.yml                  # Conda environment
└── pyproject.toml                   # Python package metadata
```

## Environment

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate xrd-strain-analysis
python -m ipykernel install --user --name xrd-strain-analysis --display-name "xrd-strain-analysis"
```

If the environment already exists:

```bash
conda env update -f environment.yml --prune
```

## Using This With Private Data

1. Clone the public code repository.
2. Keep private data outside Git or inside the ignored placeholder folders.
3. Copy `configs/experiment_template.yaml` to a private local config if desired.
4. Fill in scan names, image ranges, calibration metadata, and analysis settings.
5. Run `main_pipeline.ipynb` stage by stage.

The `.gitignore` is deliberately conservative so public commits remain code-only.

## Main Pipeline Tutorial

`main_pipeline.ipynb` is organized as a sequence of gated stages. Most cells define controls first, then a later cell executes only when the corresponding `RUN_*` flag is set to `True`. This makes the notebook safer for experimental workflows: inspect the settings, turn on one stage, run it, check the output, then continue.

### Stage 1: Preparation

Load the YAML configuration, choose which scans to process, and preview the arguments that will be passed into the lab processing functions.

Typical use:

1. Edit `configs/experiment_template.yaml` with private scan folders, image ranges, calibration metadata, and scan labels.
2. Start with a small `SELECTED_SCAN_IDS` list while debugging.
3. Run the argument preview before creating any files.
4. Set `RUN_CREATE = True` only after the scan paths and image ranges look right.
5. Run validation to confirm that the created `.nxs`/`.hdf5` files contain the expected NeXus groups.

Important controls:

- `SELECTED_SCAN_IDS`: limits processing to a subset of YAML scans.
- `RUN_CREATE`: creates scan files from raw detector images.
- `RUN_VALIDATE_CREATED`: checks the created files.
- `RUN_UNITCELL_UPDATE`: writes refined unit-cell values into existing files.

### Stage 2: Peak Finding and Peak Lists

Find or load Bragg peak positions for the reference scan. Automatic peak finding is available, but manually curated peak lists are often more reliable when detector artifacts or sample environment scattering are present.

Typical use:

1. Put curated peak lists in `Peak_Lists/`.
2. Use the peak-list inspection cell with a detector overlay PNG from `Max_of_All/`.
3. Confirm that the peak coordinates use detector image convention: `x` right, `y` down, and `z_frame` as the real TIFF image number.
4. Load the curated peak list into `/entry/postpeaks` before orientation work.

Important controls:

- `RUN_REDUCE`: runs automatic peak finding.
- Manual peak-list path cells: inspect and load curated peak coordinates.

### Stage 3: UB Matrix Determination

Determine the reference orientation matrix. The notebook supports automatic UB search and a manual fallback based on two selected peaks.

Typical use:

1. Try automatic orientation preparation and check candidate grains.
2. If automatic indexing is unreliable, use the manual UB fallback.
3. Choose two trustworthy reference peaks and adjust the ring tolerance.
4. Select the UB candidate that gives physically sensible Bragg projections.

Important controls:

- `RUN_ORIENTATION_PREPARE`
- `RUN_ORIENTATION_CHECK`
- `RUN_ORIENTATION_SELECT`
- `RUN_MANUAL_UB_PREPARE`
- `RUN_MANUAL_UB_SELECT`
- `MANUAL_PEAK_I`, `MANUAL_PEAK_J`, `MANUAL_SELECTED_UB`

### Stage 4: Projection and UB Validation

Project known Bragg peaks and configured CDW satellite targets onto detector overlays. Once the reference UB matrix is accepted, copy it to every strain scan and validate that refined HKL state exists.

Typical use:

1. Run Bragg projection and compare predicted positions with detector overlay features.
2. Run CDW projection to estimate where satellite peaks should appear.
3. After the reference UB is finalized, set `RUN_COPY_REFERENCE_UB = True`.
4. Validate every scan before moving into reciprocal-space transforms.

Important controls:

- `RUN_BRAGG_PROJECTION`
- `RUN_CDW_PROJECTION`
- `RUN_COPY_REFERENCE_UB`
- `RUN_VALIDATE_REFINED`

### Stage 5: Reference Reciprocal-Space Transformation

Prepare transform metadata, select a detector-space signal window, and transform a local detector region around a reference CDW peak into reciprocal space.

Typical use:

1. Choose `WINDOW_CDW_ID` for the peak family being inspected.
2. Start with `RUN_INTERACTIVE_WINDOW_SELECTOR = True` when selecting a new detector ROI.
3. After saving a good ROI, set `USE_SAVED_CDW_WINDOW = True` and `RUN_INTERACTIVE_WINDOW_SELECTOR = False`.
4. Reuse saved windows from `configs/peak_windows/` for repeat analysis.

Important controls:

- `WINDOW_CDW_ID`
- `RUN_TRANSFORM_PREPARE`
- `RUN_INTERACTIVE_WINDOW_SELECTOR`
- `USE_SAVED_CDW_WINDOW`
- `RUN_LOCAL_TRANSFORM`

### Stage 6: Reference Momentum-Space Analysis

Bin the local transform in H/K/L, select a background detector patch, estimate background levels, and choose integration bounds in reciprocal space.

Typical use:

1. Run binning for the reference local transform.
2. Select a background detector patch away from the signal.
3. Generate H/K/L line scans for the reference peak.
4. Use the line scans and background crossing estimates to choose integration bounds.
5. Run reference integration and inspect the summary plot before applying the same method to all scans.

Important controls:

- `RUN_REFERENCE_MOMENTUM_BIN_ESTIMATE`
- `RUN_BACKGROUND_PATCH_TRANSFORM`
- `USE_SAVED_BACKGROUND_WINDOW`
- `RUN_REFERENCE_MOMENTUM_SCANS`
- `RUN_REFERENCE_MOMENTUM_INTEGRATION`
- `MOMENTUM_INTEGRATION_HMIN/HMAX`, `KMIN/KMAX`, `LMIN/LMAX`

### Stage 7: Apply Analysis Across All Scans

Process the selected CDW series across the configured strain scans. This stage reuses saved signal/background detector windows and H/K/L integration bounds.

Typical use:

1. Confirm that the signal and background windows exist for the selected `CDW_SERIES_ID`.
2. Confirm that the integration bounds are set and not `None`.
3. Keep `CDW_SERIES_AUTO_INTEGRATION_BOUNDS = False` when you want one consistent reciprocal-space ROI across scans.
4. Set overwrite controls deliberately when rerunning after a UB/window change.
5. Use diagnostics to compare detector windows and H/K/L line scans across strain.

Important controls:

- `RUN_CDW_STRAIN_SERIES`
- `CDW_SERIES_ID`
- `CDW_SERIES_GROUP_IDS`
- `CDW_SERIES_INTEGRATION_BOUNDS`
- `CDW_SERIES_AUTO_INTEGRATION_BOUNDS`
- `CDW_SERIES_OVERWRITE_EXISTING`
- `RUN_CDW_SERIES_DIAGNOSTICS`

### Stage 8: Final Plotting

Load saved CDW strain-series summaries and create final intensity-versus-control-variable plots. The public template treats the YAML `strain` column as a generic scan-control value; users can later add a real voltage/resistance-to-strain conversion.

Typical use:

1. Run Stage 7 for each CDW family that should appear in the final comparison.
2. Add completed series IDs to `FINAL_CDW_SERIES_IDS`.
3. Choose `FINAL_INTENSITY_KEY`, such as `net_intensity` or `positive_net_intensity`.
4. Turn on final plotting after the summary CSV files exist.

Important controls:

- `RUN_FINAL_VOLTAGE_PLOTS`
- `FINAL_CDW_SERIES_IDS`
- `FINAL_INTENSITY_KEY`
- `FINAL_ERROR_KEY`
- `RUN_FINAL_ACTUAL_STRAIN_PLOTS`

### Session Snapshot

The final snapshot section records notebook state for reproducibility. Use it when an analysis pass is worth preserving, but keep generated snapshots out of the public repository if they include private scan names, file paths, or results.

## Status

This is a research workflow template, not a packaged one-command application. The code reflects an active analysis pipeline and is shared to demonstrate structure, reproducibility practices, and scientific programming style without exposing unpublished data.
