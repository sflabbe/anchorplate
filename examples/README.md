# Examples

Runnable scripts for the main analysis/benchmark modes.

> Scope note: detailed modeling boundaries and interpretation guidance now live in the repository root `README.md`. This file stays intentionally compact.

## Script index

| Script | Main purpose | Main output root |
|---|---|---|
| `demo_single_case.py` | Minimal end-to-end sanity run | `outputs/demo_single_case/` |
| `demo_benchmark.py` | Fixed-support benchmark sweep | `outputs/demo_benchmark/` |
| `demo_benchmark_springs.py` | Spring-support benchmark sweep | `outputs/demo_benchmark_springs/` |
| `demo_foundation_patch.py` | Basic compression-only foundation patch case | `outputs/demo_foundation_patch/` |
| `demo_foundation_patch_3d.py` | Contact/lift-off visualization + NPZ masks | `outputs/demo_foundation_patch_3d/` |
| `demo_benchmark_material.py` | Equivalent-stiffness material benchmark (`spring` and `spring_tension_only`) | `outputs/material_benchmark/` |
| `demo_benchmark_matrix.py` | Consolidated support-model matrix benchmark | `outputs/benchmark_matrix/` |
| `demo_anchor_dominant.py` | Anchor-dominant vs small/soft patch comparison | `outputs/anchor_dominant/` |
| `demo_mesh_convergence.py` | Coarse/medium/fine mesh convergence study | `outputs/demo_mesh_convergence/` |
| `verify_benchmark_csv.py` | Equilibrium audit post-process | writes verification CSV in selected benchmark folder |

## Recommended run order

1. `python examples/demo_single_case.py`
2. `python examples/demo_benchmark.py`
3. `python examples/demo_benchmark_springs.py`
4. `python examples/demo_foundation_patch_3d.py`
5. `python examples/demo_benchmark_material.py`
6. `python examples/demo_benchmark_matrix.py`
7. `python examples/demo_anchor_dominant.py`
8. `python examples/demo_mesh_convergence.py --mode both`

## Pointers

- Contact/lift-off sign convention and mask interpretation: `docs/contact_liftoff_guide.md`.
- Tension-only anchor behavior: `docs/spring_tension_only.md`.
- Benchmark positioning (hybrid modes, anchor-dominant framing):
  - `docs/hybrid_anchor_support_modes.md`
  - `docs/anchor_dominant_note.md`

All files under `outputs/...` are generated artifacts.
