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

    rows = run_support_model_matrix_benchmark(
        plate=plate,
        options=options,
        outdir=outdir,
    )

    print(f"Casos resueltos: {len(rows)}")
    print(f"Resultados en: {outdir.resolve()}")


if __name__ == "__main__":
    main()
