"""
demo_anchor_dominant.py
=======================
Benchmark pedagógico para separar claramente:
- modelo dominado por anclajes discretos (`anchor_dominant_no_patch`)
- modelo híbrido con patch pequeño/blando (`anchor_dominant_small_or_soft_patch`)

Run
---
    python examples/demo_anchor_dominant.py

Outputs (outputs/anchor_dominant/)
-----------------------------------
- anchor_dominant_summary.csv
- anchor_dominant_summary.md
- anchor_dominant_overview.png
- anchor_dominant_note.md
- Un subdirectorio por caso, cada uno con plot 2D, plot 3D y NPZ.
"""

from __future__ import annotations

from pathlib import Path

from anchorplate.benchmark_anchor_dominant import run_anchor_dominant_benchmark
from anchorplate.model import AnalysisOptions, SteelPlate


def main() -> None:
    outdir = Path("outputs/anchor_dominant")
    plate = SteelPlate(length_mm=300.0, width_mm=300.0, thickness_mm=15.0)
    options = AnalysisOptions(
        target_h_mm=8.0,
        output_dir=str(outdir),
        save_plots=True,
        save_result_npz=True,
        save_3d_plots=True,
        z_plot_scale=60.0,
    )

    rows = run_anchor_dominant_benchmark(
        plate=plate,
        options=options,
        outdir=outdir,
    )

    print(f"Casos resueltos: {len(rows)}")
    print(f"Resultados en: {outdir.resolve()}")


if __name__ == "__main__":
    main()
