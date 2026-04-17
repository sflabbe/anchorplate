# Anchorplate Morley

A lightweight finite element prototype for anchor plates using a Morley plate element in `scikit-fem`.

The repository focuses on the **out-of-plane plate bending** subproblem of anchor plates with:

- coupled reference-point style loading distributed onto two lines
- fixed or spring-supported anchors
- Winkler foundation patches
- compression-only contact via active-set iteration
- local mesh refinement around anchors, load introduction zones, and contact regions
- benchmark scripts for load-case sweeps and material stiffness comparisons

## Scope

This code is useful when you want to study **plate deformation, support reactions, contact area, lift-off, and plate stresses** for a steel anchor plate under vertical force and bending.

It is **not** a full anchor design package. The following topics are intentionally outside the current scope:

- anchor shear design in `Fx` / `Fy`
- in-plane torsion `Mz` as a full anchor-group design problem
- concrete cone, pry-out, edge breakout, pull-out, and related fastener checks
- tangential friction/contact
- material nonlinearity of the steel plate

## Features

- Kirchhoff-Love plate solver with `ElementTriMorley`
- nodal point supports as either fixed supports or vertical springs `Kz`
- reference-point style load transfer to two parallel lines from `Fz`, `Mx`, and `My`
- bedding models for concrete, timber, steel layers, and calibrated user input
- compression-only foundation patches with active/inactive contact masks
- 2D and 3D plotting utilities
- NPZ export for post-processing and debugging
- reproducible benchmark scripts that write CSV and Markdown summaries

## Installation

```bash
pip install -e .
```

Main runtime dependencies:

- `numpy`
- `scipy`
- `matplotlib`
- `scikit-fem`
- `pandas`

## Quick start

Run the smallest example first:

```bash
python examples/demo_single_case.py
```

Then try the benchmark sweep:

```bash
python examples/demo_benchmark.py
```

Outputs are written to `outputs/...` and are intentionally ignored by Git.

## Examples

The repository ships with example scripts in `examples/`.
A short index is available in [`examples/README.md`](examples/README.md).

### `demo_single_case.py`
Minimal sanity-check model with 4 fixed supports and one coupled vertical load.

Writes mesh and result plots plus an NPZ result bundle.

### `demo_benchmark.py`
Runs a PROFIS-like load-case sweep for a plate with 4 fixed corner supports.

Writes per-case plots and summary files such as:

- `benchmark_summary.csv`
- `benchmark_summary.md`
- `benchmark_overview.png`

### `demo_benchmark_springs.py`
Same sweep as above, but with spring-supported anchors instead of fixed supports.

Useful to study how reaction extraction and rigid-body compliance change the response.

### `demo_foundation_patch.py`
Mixed bedding example with two compression-only foundation zones:

- concrete zone
- timber zone

Shows contact iteration and lift-off behaviour with a combined `Fz + Mx` load.

### `demo_foundation_patch_3d.py`
More documented foundation/contact example with:

- 3D deformed-shape plot
- NPZ export
- contact summary text output

Use this when you want to inspect active vs inactive contact regions in detail.

### `demo_benchmark_material.py`
Material sensitivity benchmark for the compression-only foundation patch.

Compares default bedding materials across multiple load cases and writes summary tables and overview plots.

### `verify_benchmark_csv.py`
Post-processing helper that reads the benchmark CSV and adds equilibrium error columns.

Useful as a quick audit step after a benchmark run.

## Repository layout

```text
src/anchorplate/
  __init__.py
  benchmark.py            PROFIS-like benchmark sweep
  benchmark_material.py   material stiffness benchmark
  loading.py              coupled load transfer and equivalent line loading
  mesh.py                 mesh generation and local refinement helpers
  model.py                dataclasses and model input definitions
  plotting.py             2D/3D visualisation and NPZ export
  postprocess.py          moments, stresses, and result recovery
  solver.py               assembly, constraints, and solve routine
  support.py              bedding and support stiffness helper functions

examples/
  README.md
  demo_single_case.py
  demo_benchmark.py
  demo_benchmark_springs.py
  demo_foundation_patch.py
  demo_foundation_patch_3d.py
  demo_benchmark_material.py
  verify_benchmark_csv.py

docs/
  notes.md
  contact_liftoff_guide.md
  bugfix_spring_reactions.md

tests/
  test_equivalent_line_load.py
  test_support_models.py
  test_spring_reactions.py
  test_foundation_contact.py
  test_benchmark_material.py
```

## Units and conventions

The code uses:

- `mm`
- `N`
- `MPa = N/mm²`

For the foundation-patch examples:

- `w > 0` means motion into the support/foundation, i.e. compression/contact
- `w <= 0` means lift-off, so compression-only bedding is inactive there

## Testing

```bash
pytest -q
```

## Notes for publication

This repository archive has been cleaned for GitHub upload:

- generated `outputs/` were removed
- `__pycache__` and `*.egg-info` folders were removed
- test files were consolidated under `tests/`
- documentation notes were moved into `docs/`

# anchorplate
