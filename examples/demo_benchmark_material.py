"""
demo_benchmark_material.py
==========================
Material benchmark for the compression-only foundation patch.

Compares three bedding materials (grout / steel shim / timber) across four
PROFIS-like load cases and writes a summary CSV, Markdown table, and two
overview plots.

Run
---
    python examples/demo_benchmark_material.py

Output structure
----------------
    outputs/material_benchmark/
        material_benchmark_summary.csv
        material_benchmark_summary.md
        overview_utilisation_contact.png
        overview_contact_vs_stiffness.png
        grout/LC01_Fz_centric/   (+ subfolders per case if save_plots=True)
        steel/...
        timber/...
"""
from __future__ import annotations

from pathlib import Path

from anchorplate.benchmark_material import (
    default_load_cases,
    default_materials,
    run_material_benchmark,
)
from anchorplate.model import AnalysisOptions, PointSupport, SteelPlate


def main() -> None:
    plate = SteelPlate(length_mm=300.0, width_mm=300.0, thickness_mm=15.0)

    supports = [
        PointSupport(30.0,  30.0,  kind="spring", kz_n_per_mm=150_000.0, label="A1"),
        PointSupport(270.0, 30.0,  kind="spring", kz_n_per_mm=150_000.0, label="A2"),
        PointSupport(30.0,  270.0, kind="spring", kz_n_per_mm=150_000.0, label="A3"),
        PointSupport(270.0, 270.0, kind="spring", kz_n_per_mm=150_000.0, label="A4"),
    ]

    outdir = Path("outputs/material_benchmark")
    options = AnalysisOptions(
        target_h_mm=8.0,
        output_dir=str(outdir),
        save_plots=True,
        save_result_npz=True,
        save_3d_plots=True,
        z_plot_scale=50.0,
    )

    materials  = default_materials()
    load_cases = default_load_cases()

    print("Running material benchmark …")
    print(f"  {len(materials)} materials × {len(load_cases)} load cases = {len(materials)*len(load_cases)} analyses\n")

    rows = run_material_benchmark(
        plate=plate,
        supports=supports,
        materials=materials,
        load_cases=load_cases,
        options=options,
        outdir=outdir,
    )

    # Console summary table
    print(f"\n{'Material':<30} {'LC':<20} {'k [N/mm³]':>10}  {'Contact%':>8}  {'w_max [mm]':>10}  {'η':>6}  {'ΣR [kN]':>8}")
    print("-" * 100)
    for r in rows:
        print(
            f"{r.material:<30} {r.load_case:<20} {r.k_area_n_mm3:>10.1f}  "
            f"{r.pct_active:>7.1f}%  {r.w_max_mm:>10.4f}  {r.eta_plate:>6.3f}  {r.sum_total_reactions_kN:>8.2f}"
        )

    print(f"\nAll {len(rows)} cases converged: {all(r.converged for r in rows)}")
    print(f"Results written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
