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
    build_corner_supports,
    default_load_cases,
    material_spec_from_model_result,
    run_material_benchmark,
)
from anchorplate.model import AnalysisOptions, SteelPlate
from anchorplate.support import (
    support_material_concrete_simple,
    support_material_steel_layers_simple,
    support_material_timber_simple,
)
from anchorplate.model import SteelLayer


def main() -> None:
    plate = SteelPlate(length_mm=300.0, width_mm=300.0, thickness_mm=15.0)

    outdir = Path("outputs/material_benchmark")
    options = AnalysisOptions(
        target_h_mm=8.0,
        output_dir=str(outdir),
        save_plots=True,
        save_result_npz=True,
        save_3d_plots=True,
        z_plot_scale=50.0,
    )

    materials = [
        material_spec_from_model_result(
            name="Grout C25/30 h=50 mm",
            label="grout",
            model=support_material_concrete_simple(e_cm_mpa=32_000.0, h_eff_mm=50.0),
            description="E_cm=32000 MPa, h_eff=50 mm",
        ),
        material_spec_from_model_result(
            name="Steel S235 t=10 mm",
            label="steel",
            model=support_material_steel_layers_simple(
                [SteelLayer(thickness_mm=10.0, youngs_modulus_mpa=210_000.0)]
            ),
            description="E=210000 MPa, t=10 mm",
        ),
        material_spec_from_model_result(
            name="Timber GL24h h=50 mm",
            label="timber",
            model=support_material_timber_simple(e90_mpa=390.0, h_eff_mm=50.0),
            description="E_90=390 MPa, h_eff=50 mm",
        ),
    ]
    load_cases = default_load_cases()

    rows = []
    for support_kind in ("spring", "spring_tension_only"):
        supports = build_corner_supports(kind=support_kind)
        kind_outdir = outdir / support_kind
        print(f"Running material benchmark ({support_kind}) …")
        print(
            f"  {len(materials)} materials × {len(load_cases)} load cases "
            f"= {len(materials)*len(load_cases)} analyses\n"
        )
        rows.extend(
            run_material_benchmark(
                plate=plate,
                supports=supports,
                materials=materials,
                load_cases=load_cases,
                options=options,
                outdir=kind_outdir,
                hybrid_support_kind=support_kind,
            )
        )

    # Console summary table
    print(
        f"\n{'Support':<20} {'Material':<30} {'LC':<20} {'k [N/mm³]':>10}  "
        f"{'Anchors':>10}  {'ΣR_anchor [kN]':>14}  {'ΣR_found [kN]':>13}"
    )
    print("-" * 100)
    for r in rows:
        print(
            f"{r.support_type:<20} {r.material:<30} {r.load_case:<20} {r.k_area_n_mm3:>10.1f}  "
            f"{f'{r.anchor_active_count}/{r.anchor_inactive_count}':>10}  "
            f"{r.sum_spring_reactions_kN:>14.2f}  {r.sum_foundation_reaction_kN:>13.2f}"
        )

    print("\nDirect comparison for uplift-sensitive case (LC04_Mx_pure):")
    print(f"{'Support':<20} {'Material':<20} {'Anchors':>10} {'ΣR_anchor [kN]':>14} {'ΣR_found [kN]':>13}")
    for r in rows:
        if r.load_case != "LC04_Mx_pure":
            continue
        print(
            f"{r.support_type:<20} {r.material:<20} "
            f"{f'{r.anchor_active_count}/{r.anchor_inactive_count}':>10} "
            f"{r.sum_spring_reactions_kN:>14.2f} {r.sum_foundation_reaction_kN:>13.2f}"
        )

    print(f"\nAll {len(rows)} cases converged: {all(r.converged for r in rows)}")
    print(f"Results written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
