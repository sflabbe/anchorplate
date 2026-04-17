# anchorplate

Finite element prototype for **steel anchor plates** using a **Kirchhoff-Love plate model** with the **Morley triangular element** via `scikit-fem`.

The project is aimed at the **out-of-plane plate-bending subproblem** around anchor groups and contact patches. It is useful for studying plate deformation, support reactions, lift-off, and local plate stresses under vertical force and bending moments.

## What this repository does

- Models a steel plate as a thin bending plate with `ElementTriMorley`
- Supports **fixed anchors**, **linear vertical springs**, and **tension-only springs**
- Transfers load from a **reference point** to **two parallel straight lines** to emulate a profile or double-flat-bar load introduction
- Supports **compression-only Winkler foundation patches** with active-set contact iteration
- Includes simple support-material models for:
  - concrete
  - timber
  - stacked steel layers
  - calibrated user-defined bedding stiffness
- Provides:
  - 2D plots
  - 3D deformed-shape plots
  - NPZ export for post-processing
  - benchmark runners
  - mesh-refinement helpers
  - basic tests for support models, contact logic, and import hygiene

## What this repository does **not** do

This is **not** a full anchor-design package.

Out of scope today:

- full anchor-group design for `Fx`, `Fy`, and `Mz`
- concrete cone, pull-out, pry-out, edge breakout, etc.
- frictional tangential contact
- nonlinear steel plasticity
- detailed timber orthotropy/contact as a full advanced material model
- full shell-shell or solid-solid contact mechanics for timber/steel assemblies

`Fx` and `Fy` can be projected into equivalent plate moments using a stand-off `e_out`, but that is still only the **plate-bending subproblem**, not full anchor verification.

## Core modeling assumptions

### Plate model

The steel plate is modeled as a **Kirchhoff-Love plate** in bending. The main unknown is transverse deflection `w`.

Units used throughout the repo:

- length: `mm`
- force: `N`
- stress: `MPa = N/mm²`

### Support models

The code currently supports three main support idealizations:

1. **Fixed discrete supports**
   - nodal `w = 0` at anchor positions

2. **Spring anchors**
   - nodal vertical springs with stiffness `kz [N/mm]`
   - reaction extracted as `R = kz * w`

3. **Foundation patches**
   - distributed bedding stiffness `k_area [N/mm³]`
   - optional **compression-only** behavior using active-set iteration
   - useful for grout, concrete bearing, timber compression-perpendicular simplifications, or calibrated interfaces

### Load introduction

The repo includes a **reference-point style coupled load** that distributes `Fz`, `Mx`, and `My` to **two straight lines**. This is intended as a practical approximation of a profile introducing load into the plate, similar in spirit to an RP plus coupling in a 3D FE model.

### Bedding / support-material models

The support module currently includes simple but traceable helper laws:

- `concrete_simple`: `k = E_cm / h_eff`
- `concrete_advanced`: geometry-corrected concrete helper
- `timber_simple`: `k = spread_factor * E_90 / h_eff`
- `steel_layers_simple`: `1 / k = Σ(t_i / E_i)`
- `calibrated`: direct user-defined `k`

These are engineering approximations, not universal constitutive truth. The repo keeps the chosen model name, parameters, and notes so benchmark outputs remain traceable.

## Repository layout

```text
src/anchorplate/
  __init__.py
  benchmark.py            PROFIS-like plate benchmark
  benchmark_material.py   bedding-material sensitivity benchmark
  benchmark_matrix.py     consolidated support-model matrix benchmark
  loading.py              RP-to-line load transfer helpers
  mesh.py                 mesh generation and refinement helpers
  model.py                dataclasses and analysis options
  plotting.py             2D/3D plots and NPZ export
  postprocess.py          moment/stress recovery helpers
  solver.py               FE assembly, contact iteration, reactions
  support.py              support-material and bedding helpers

examples/
  demo_single_case.py
  demo_benchmark.py
  demo_benchmark_springs.py
  demo_foundation_patch.py
  demo_foundation_patch_3d.py
  demo_benchmark_material.py
  demo_benchmark_matrix.py
  demo_mesh_convergence.py
  verify_benchmark_csv.py

examples/README.md        short example index

docs/
  notes.md
  bugfix_spring_reactions.md
  contact_liftoff_guide.md
  timber_advanced_roadmap.md

tests/
  test_benchmark_material.py
  test_equivalent_line_load.py
  test_foundation_contact.py
  test_import_hygiene.py
  test_spring_reactions.py
  test_support_models.py
```

## Installation

Python requirement from `pyproject.toml`:

- Python `>= 3.11`

Install in editable mode:

```bash
pip install -e .
```

Main dependencies:

- `numpy`
- `scipy`
- `matplotlib`
- `scikit-fem`
- `pandas`

## Import hygiene and optional FE dependency

The package is organized so lightweight submodules can be imported without pulling the full FE stack immediately.

Lightweight imports:

```python
import anchorplate.model
import anchorplate.support
```

FE-heavy functionality remains in explicit submodules or lazy exports:

```python
from anchorplate.solver import solve_anchor_plate
from anchorplate.plotting import plot_result_3d
from anchorplate.benchmark import run_profis_like_benchmark
```

If `scikit-fem` is missing, FE features will fail with a clear error, but support-model utilities should remain usable.

## Quick start

Run the smallest end-to-end example first:

```bash
python examples/demo_single_case.py
```

Then try the main fixed-support benchmark:

```bash
python examples/demo_benchmark.py
```

Then compare against spring anchors:

```bash
python examples/demo_benchmark_springs.py
```

Then inspect lift-off/contact in 3D:

```bash
python examples/demo_foundation_patch_3d.py
```

Then material sensitivity and consolidated benchmark matrix:

```bash
python examples/demo_benchmark_material.py
python examples/demo_benchmark_matrix.py
```

Finally, run a practical mesh-convergence check:

```bash
python examples/demo_mesh_convergence.py --mode both
```

Generated results are written to `outputs/...` and are intended as disposable artifacts.

## Example guide

### `demo_single_case.py`
Small sanity check for:

- plate setup
- mesh generation
- coupled line load transfer
- result plotting
- NPZ export

### `demo_benchmark.py`
Runs a PROFIS-like sweep on a plate with **fixed supports**.

Typical outputs:

- per-case result plots
- `benchmark_summary.csv`
- `benchmark_summary.md`
- `benchmark_overview.png`

### `demo_benchmark_springs.py`
Same idea as above, but using **spring-supported anchors**.
Useful for checking reaction extraction and the effect of finite support stiffness.

### `demo_foundation_patch.py`
Mixed-bedding contact example with distributed support patches.

### `demo_foundation_patch_3d.py`
Best entry point for checking:

- active vs inactive contact zones
- lift-off
- deformed shape relative to `z = 0`
- exported NPZ masks (`active`, `inactive`, `in_patch`)

### `demo_benchmark_material.py`
Material sensitivity benchmark for foundation bedding stiffness.
Currently compares simplified support-material models such as grout/concrete, steel, and timber.

### `demo_benchmark_matrix.py`
Consolidated matrix across different support assumptions, including:

- fixed anchors
- spring anchors
- spring anchors + full foundation patch with different materials

This is the best script for a fast cross-model comparison under the same load cases.

### `demo_mesh_convergence.py`
Coarse/medium/fine convergence study for a representative bending case.
Useful for checking whether global outputs such as `w_max`, reaction sums, and plate stress indicators stabilize with mesh refinement.

### `verify_benchmark_csv.py`
Post-processes benchmark CSV files to add equilibrium-check columns.
Useful as a quick audit step after a benchmark run.

## Typical outputs

Depending on the example, the repo can generate:

- 2D result plots
- 3D deformed-shape plots
- mesh plots
- CSV summaries
- Markdown summaries
- NPZ result bundles for custom post-processing

Foundation/contact examples also export or reconstruct:

- active foundation mask
- inactive foundation mask
- in-patch mask
- contact-iteration history

## Testing

Run the test suite with:

```bash
pytest -q
```

Note that FE-heavy tests require the FE dependency stack installed.

## Known limitations and current maturity

This repository is already useful for:

- plate-level comparison of support assumptions
- studying lift-off vs contact area
- checking reaction patterns for fixed vs spring supports
- comparing simplified concrete/steel/timber bedding assumptions
- building reproducible benchmark plots and CSV summaries

It should still be treated as a **prototype / engineering sandbox**, not as a production-certified verification tool.

In particular:

- local stress peaks near nodal supports or concentrated load-introduction zones are mesh-sensitive
- simplified bedding laws for concrete/timber/steel are engineering surrogates
- timber is not yet modeled as a full orthotropic contact body
- foundation patches represent distributed normal support, not full interface mechanics

## Documentation notes

Useful supporting docs shipped with the repo:

- `docs/bugfix_spring_reactions.md`
- `docs/spring_tension_only.md`
- `docs/contact_liftoff_guide.md`
- `docs/timber_advanced_roadmap.md`
- `docs/notes.md`

## Practical recommendation

Recommended order for someone new to the repo:

1. `demo_single_case.py`
2. `demo_benchmark.py`
3. `demo_benchmark_springs.py`
4. `demo_foundation_patch_3d.py`
5. `demo_benchmark_material.py`
6. `demo_benchmark_matrix.py`
7. `demo_mesh_convergence.py`

That sequence tends to surface installation issues, reaction issues, contact interpretation, material sensitivity, and mesh sensitivity in a sensible order.
