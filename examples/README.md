# Examples

This folder contains runnable scripts for the main analysis modes in the project.

## Overview

| Script | Purpose | Typical output |
|---|---|---|
| `demo_single_case.py` | Smallest end-to-end sanity check for a single plate/load case with 4 fixed supports | mesh plot, 2D result plot, 3D plot, NPZ file |
| `demo_benchmark.py` | Fixed-support benchmark sweep across multiple load cases | per-case plots, `benchmark_summary.csv`, `benchmark_summary.md`, overview plot |
| `demo_benchmark_springs.py` | Same benchmark sweep but with spring supports | per-case plots and benchmark summaries for the spring-supported variant |
| `demo_foundation_patch.py` | Compression-only foundation patch example with concrete and timber zones | mesh plot, result plots, NPZ export, console contact history |
| `demo_foundation_patch_3d.py` | More explicit contact/lift-off demo with 3D visualisation and text summary | mesh plot, 2D/3D result plots, NPZ export, `contact_summary.txt` |
| `demo_benchmark_material.py` | Material sensitivity benchmark for foundation bedding stiffness | `material_benchmark_summary.csv`, Markdown summary, overview plots |
| `demo_benchmark_matrix.py` | Consolidated support-model matrix (`fixed`, `spring_anchors`, `foundation_patch_*`) on shared load cases | `benchmark_matrix_summary.csv`, Markdown summary, overview plot, technical note |
| `verify_benchmark_csv.py` | Post-processes the fixed benchmark summary to add equilibrium error checks | `benchmark_verification.csv` |
| `demo_mesh_convergence.py` | Coarse/medium/fine convergence study for a representative Fz+Mx case, with optional refinement boxes | `mesh_convergence_summary.csv`, Markdown summary, overview plot |

## Recommended order

1. `python examples/demo_single_case.py`
2. `python examples/demo_benchmark.py`
3. `python examples/demo_benchmark_springs.py`
4. `python examples/demo_foundation_patch_3d.py`
5. `python examples/demo_benchmark_material.py`
6. `python examples/demo_benchmark_matrix.py`
7. `python examples/demo_mesh_convergence.py --mode both`

## What each example is good for

### `demo_single_case.py`
Use this first to check that the installation works and that plots/NPZ export are functional.

### `demo_benchmark.py`
Use this when you want a reproducible set of fixed-support load cases and a compact summary table.

### `demo_benchmark_springs.py`
Use this when you want to compare the same benchmark against vertical spring supports.

### `demo_foundation_patch.py`
Use this when you want a simple mixed-bedding contact example without extra reporting overhead.

### `demo_foundation_patch_3d.py`
Use this when you specifically want to inspect lift-off and active contact zones in a more visual way.

### `demo_benchmark_material.py`
Use this when you want to compare how foundation stiffness changes contact area, deflection, and utilisation across materials.
The benchmark summary now records the support model metadata (`model_name`, parameters, notes) for traceability.

### `verify_benchmark_csv.py`
Use this after `demo_benchmark.py` to add a quick equilibrium audit layer to the benchmark summary.

### `demo_benchmark_matrix.py`
Use this when you need an apples-to-apples matrix across support assumptions (fixed/spring vs foundation patch materials) under identical load cases.
The output keeps `model_type` explicit so discrete and hybrid contact models are not conflated.

## Output handling

All examples write generated files into `outputs/...`.
That directory is ignored by Git and should be treated as generated content.


### `demo_mesh_convergence.py`
Use this to run a reproducible coarse/medium/fine mesh convergence study and justify a practical default `target_h_mm` using global metrics (`w_max`, reaction sum) instead of only stress peaks.
