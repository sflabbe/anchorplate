"""
demo_benchmark_matrix.py
========================
Benchmark matrix consolidado para comparar modelos de soporte bajo los mismos
load cases (compactos): Fz céntrico, Fz+Mx, Fz+My y Mx puro.

Modelos incluidos:
- fixed
- spring_anchors
- foundation_patch_concrete
- foundation_patch_steel
- foundation_patch_timber

Run
---
    python examples/demo_benchmark_matrix.py

Outputs (outputs/benchmark_matrix/)
-----------------------------------
- benchmark_matrix_summary.csv
- benchmark_matrix_summary.md
- benchmark_matrix_overview.png
- benchmark_matrix_note.md
"""

from __future__ import annotations

from pathlib import Path

from anchorplate.benchmark_matrix import run_support_model_matrix_benchmark
from anchorplate.model import AnalysisOptions, SteelPlate


def main() -> None:
    outdir = Path("outputs/benchmark_matrix")
    plate = SteelPlate(length_mm=300.0, width_mm=300.0, thickness_mm=15.0)
    options = AnalysisOptions(
        target_h_mm=8.0,
        output_dir=str(outdir),
        save_plots=False,
        save_result_npz=False,
        save_3d_plots=False,
    )

    rows = []
    for support_kind in ("spring", "spring_tension_only"):
        rows.extend(
            run_support_model_matrix_benchmark(
                plate=plate,
                options=options,
                outdir=outdir / support_kind,
                hybrid_support_kind=support_kind,
            )
        )

    print(f"Casos resueltos: {len(rows)}")
    print(
        f"\n{'Modelo':<45} {'LC':<20} {'Anchors':>10} {'ΣR_anchor [kN]':>14} {'ΣR_found [kN]':>13}"
    )
    print("-" * 115)
    for r in rows:
        if "foundation_patch" not in r.model_name:
            continue
        print(
            f"{r.model_name:<45} {r.load_case:<20} "
            f"{f'{r.anchor_active_count}/{r.anchor_inactive_count}':>10} "
            f"{r.sum_spring_reactions_kN:>14.2f} {r.sum_foundation_reactions_kN:>13.2f}"
        )
    print(f"Resultados en: {outdir.resolve()}")


if __name__ == "__main__":
    main()
