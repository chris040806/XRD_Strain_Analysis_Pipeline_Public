# Lab Library Change Log

The project should add new behavior in `src/xrd_strain_pipeline/` whenever possible. Any direct edit to the copied lab libraries must be recorded here so users can distinguish upstream lab code from pipeline changes.

## `nxprocess_lab.py`

### 2026-06-21: Transform voxel-direction sampling correction

Function:

```python
LabRefine_transform_prepare(...)
```

Original direction list:

```python
ranarr = [[0, 0, 1], [0, 1, 0], [0, 0, 1]]
```

Project copy:

```python
ranarr = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
```

Reason: the original sampled the frame/Z direction twice and never sampled detector X when estimating changes in H/K/L. The corrected list samples one step along X, Y, and Z.

## `nxrefine_lab.py`

No direct project modifications recorded.
