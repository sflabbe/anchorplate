# anchorplate

Finite-element prototype for **steel anchor plates** based on a **Kirchhoff–Love plate bending model** (`ElementTriMorley`, `scikit-fem`).

The repository targets the **out-of-plane bending subproblem** around anchor groups and support/contact zones. It is intended for transparent engineering studies (reaction split, lift-off patterns, sensitivity to support assumptions), not full anchor-code verification.

## Scope of the model (what it solves)

`anchorplate` solves:

- Transverse plate deflection `w(x, y)` and derived plate-bending stress indicators.
- Vertical reaction transfer through:
  - discrete supports (`fixed`, `spring`, `spring_tension_only`), and/or
  - distributed `foundation_patch` support (Winkler-type bedding, optional compression-only active set).
- Coupled vertical loading introduced through a reference-point style line-coupling (`Fz`, `Mx`, `My` mapped to two load lines).
- Comparative benchmarks for support assumptions, materials (as equivalent stiffness), and mesh sensitivity.

## Modeling boundaries / not covered

This section is explicit by design so readers do not infer capabilities that are not modeled.

- **No in-plane plate membrane model** (`u, v` in-plane fields are not solved; this is a bending-only plate submodel).
- **No direct anchor shear verification** (`Fx`, `Fy` anchor shear checks are out of scope).
- **No concrete failure verification** (no cone breakout, pryout, pull-out, side-face blowout, edge breakout checks).
- **No full 3D interface contact mechanics** for steel/timber foundation pieces.
  - Steel/timber/concrete foundation patches are implemented as **equivalent normal stiffness models** (`k_area`), not full 3D deformable bodies with tangential/frictional contact.
- **No nonlinear steel plasticity** (elastic plate model).
- **No frictional tangential contact law** (normal support only in `foundation_patch`).

## Support models available

### 1) `fixed`

- Nodal `w = 0` at anchor coordinates.
- Useful as a rigid-boundary reference case.

### 2) `spring`

- Linear bidirectional vertical spring at each anchor: `R = kz * w`.
- Always active.
- Useful as a linearized discrete-anchor model.

### 3) `spring_tension_only`

- Unilateral spring model:
  - active in tension (`w > +tol` under solver sign convention),
  - inactive in compression (`w < -tol`),
  - hysteresis in `[-tol, +tol]` to reduce chattering.
- Useful when uplift realism is needed and anchors should not provide fictitious compression resistance.

### 4) `foundation_patch` (equivalent distributed support)

- Patch-wise bedding stiffness `k_area [N/mm³]` converted to nodal stiffness using tributary areas.
- Optional compression-only contact active set (`w > tol` active, otherwise lift-off/inactive).
- Can coexist with discrete anchors (hybrid model).

## What “equivalent stiffness benchmark” means here

The material-related benchmarks are **not** constitutive material validation in a full 3D sense.

They compare **equivalent support stiffness inputs** (`k_area`) derived from simple engineering models, e.g. concrete, timber, or steel-layer surrogates, and quantify how those assumptions change:

- contact active area,
- reaction split (anchors vs foundation),
- `w_max` and stress indicators.

Interpretation: this is a **model-assumption sensitivity benchmark**, not a direct proof of real material failure resistance.

## What “anchor-dominant benchmark” means here

`demo_anchor_dominant.py` separates two regimes:

- **anchor-dominant**: load path primarily through discrete anchors (no patch),
- **hybrid with small/soft patch**: patch present but intentionally weak/limited.

Purpose: keep comparisons physically legible by showing when conclusions come from anchor behavior versus distributed support contribution.

## Repository layout

```text
src/anchorplate/
  model.py                      core dataclasses and analysis options
  solver.py                     FE assembly, support/contact active-set logic, reactions
  support.py                    equivalent support-material helpers (k_area)
  loading.py                    reference-point to line-load transfer
  mesh.py                       mesh generation/refinement helpers
  postprocess.py                moment/stress recovery helpers
  plotting.py                   2D/3D plots and NPZ export
  benchmark.py                  fixed/spring benchmark core
  benchmark_material.py         equivalent-stiffness material benchmark
  benchmark_matrix.py           support-model matrix benchmark
  benchmark_anchor_dominant.py  anchor-dominant benchmark
  benchmark_backend.py          tri_morley vs quad_bfs experimental comparison

examples/
  demo_single_case.py
  demo_benchmark.py
  demo_benchmark_springs.py
  demo_foundation_patch.py
  demo_foundation_patch_3d.py
  demo_benchmark_material.py
  demo_benchmark_matrix.py
  demo_anchor_dominant.py
  demo_mesh_convergence.py
  demo_mesh_backend_benchmark.py
  verify_benchmark_csv.py

docs/
  contact_liftoff_guide.md
  spring_tension_only.md
  anchor_dominant_note.md
  hybrid_anchor_support_modes.md
  notes.md
  ...
```

## Benchmark examples available

- `demo_benchmark.py`: baseline sweep (fixed supports).
- `demo_benchmark_springs.py`: same benchmark family with spring anchors.
- `demo_benchmark_material.py`: equivalent-stiffness material sensitivity for foundation patches (`spring` and `spring_tension_only` variants).
- `demo_benchmark_matrix.py`: consolidated matrix across `fixed`, `spring_anchors`, and `foundation_patch_*` models for both discrete support modes.
- `demo_anchor_dominant.py`: focused benchmark for anchor-dominant vs small/soft-patch behavior.
- `demo_mesh_convergence.py`: coarse/medium/fine convergence study (with/without refinement boxes).
- `demo_mesh_backend_benchmark.py`: experimental comparison `tri_morley` vs `quad_bfs` for plate-only load cases (`Fz`, `Fz+Mx`).

## Mini interpretation guide (selected scripts)

### `examples/demo_mesh_convergence.py`

- Use to justify a practical `target_h_mm`.
- Prioritize convergence of global metrics (`w_max`, `reaction_sum_kN`) over local maxima alone.
- Main outputs in `outputs/demo_mesh_convergence/`:
  - `mesh_convergence_summary.csv`
  - `mesh_convergence_summary.md`
  - `mesh_convergence_overview.png`
  - per-level mesh plots under mode subfolders.

### `examples/demo_benchmark_material.py`

- Interprets support materials as **equivalent `k_area` models**.
- Compare `support_type` (`spring` vs `spring_tension_only`) and reaction split columns.
- Respect validity gating columns before comparing metrics:
  `solve_status`, `valid_solution`, `metrics_comparable`, `failure_reason`,
  `equilibrium_error_kN`, `equilibrium_ok`.
- Outputs in:
  - `outputs/material_benchmark/spring/...`
  - `outputs/material_benchmark/spring_tension_only/...`
  with per-support-type summary files (`material_benchmark_summary.csv/.md`) and overview plots.

### `examples/demo_benchmark_matrix.py`

- Use for side-by-side support-model comparison under identical load cases.
- Do not mix discrete-only and hybrid conclusions without checking `model_type` and `support_type` columns.
- Only rows with `valid_solution=true` and `metrics_comparable=true` should be used
  for benchmark comparisons.
- Outputs in:
  - `outputs/benchmark_matrix/spring/...`
  - `outputs/benchmark_matrix/spring_tension_only/...`
  each with `benchmark_matrix_summary.csv/.md`, `benchmark_matrix_overview.png`, `benchmark_matrix_note.md`.

See `docs/hybrid_benchmark_validity_policy.md` for the explicit validity policy and tolerances.

### `examples/demo_anchor_dominant.py`

- Read `foundation_share_pct` to quantify when the patch contribution is secondary vs relevant.
- Useful regression check for discrete-anchor behavior under `Fz+Mx` and `Fz+Mx+My`.
- Outputs in `outputs/anchor_dominant/`:
  - `anchor_dominant_summary.csv`
  - `anchor_dominant_summary.md`
  - `anchor_dominant_overview.png`
  - `anchor_dominant_note.md`
  - per-case folders with 2D/3D plots and NPZ.

### `examples/demo_foundation_patch_3d.py`

- Reference contact/lift-off inspection case.
- Validate mask logic from NPZ:
  - `active_foundation_mask`
  - `inactive_foundation_mask`
  - `in_patch_mask`
- Outputs in `outputs/demo_foundation_patch_3d/`:
  - `mesh.png`
  - `demo_foundation_patch_3d.png`
  - `demo_foundation_patch_3d_3d.png`
  - `demo_foundation_patch_3d_result.npz`
  - `contact_summary.txt`

See `docs/contact_liftoff_guide.md` for detailed sign convention and mask interpretation.

## Experimental mesh backend selector

`analysis_options.mesh_backend` accepts:

- `"tri_morley"` (default, production reference)
- `"quad_bfs"` (experimental 2D Kirchhoff plate backend)

This is a **2D plate backend switch** only. It is intentionally not a 3D-solid (`hex`) formulation swap.

## Main outputs

Depending on script/configuration, generated artifacts include:

- summary CSV/Markdown tables,
- overview plots,
- per-case 2D/3D result plots,
- NPZ bundles for post-processing,
- contact/lift-off metrics and active-set iteration history.

Outputs are written under `outputs/...` and treated as generated artifacts.


## Declarative TOML input (single case + studies)

You can now run analyses without editing Python scripts by using TOML input files and the new runner:

```bash
python -m anchorplate.run_case examples/toml/simple_case.toml
python -m anchorplate.run_case examples/toml/parametric_study.toml
```

Supported top-level sections include:

- `[plate]`
- `[analysis_options]`
- `[[supports]]`
- `[[point_loads]]`
- `[[coupled_line_loads]]`
- `[[foundation_patches]]`
- `[[refinement_boxes]]`
- optional `[support_material_model]`
- optional `[[sweeps]]` for parametric studies

See `docs/input_toml.md` for schema details and `examples/toml/*.toml` for minimal templates.

## Install and run

Requirements (from `pyproject.toml`): Python `>=3.11`.

```bash
pip install -e .
```

Suggested first run order:

```bash
python examples/demo_single_case.py
python examples/demo_benchmark.py
python examples/demo_benchmark_springs.py
python examples/demo_foundation_patch_3d.py
python examples/demo_benchmark_material.py
python examples/demo_benchmark_matrix.py
python examples/demo_anchor_dominant.py
python examples/demo_mesh_convergence.py --mode both
```

## Tests

```bash
pytest -q
```

## Related docs

- `examples/README.md` (compact runnable index)
- `docs/contact_liftoff_guide.md`
- `docs/spring_tension_only.md`
- `docs/hybrid_anchor_support_modes.md`
- `docs/anchor_dominant_note.md`
- `docs/notes.md`
- `docs/quad_backend_experimental_note.md`
