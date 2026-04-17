"""
demo_foundation_patch_3d.py
===========================
Demonstration of the foundation-patch (compression-only Winkler bedding) submodel
with two material zones — concrete and timber — and a coupled line load with moment.

This is the reference case for the material-benchmark series (concrete / steel / timber).

Run
---
    python examples/demo_foundation_patch_3d.py

Outputs (in outputs/demo_foundation_patch_3d/)
-----------------------------------------------
    mesh.png                              — mesh + geometry overview
    demo_foundation_patch_3d.png          — 2×2 results panel
    demo_foundation_patch_3d_3d.png       — 3D deformed shape with contact overlay
    demo_foundation_patch_3d_result.npz   — full result for post-processing
    contact_summary.txt                   — plain-text contact statistics

Reading the .npz
----------------
    import numpy as np
    d = np.load("demo_foundation_patch_3d_result.npz")
    active   = d["active_foundation_mask"].astype(bool)   # in contact
    inactive = d["inactive_foundation_mask"].astype(bool) # lift-off
    w        = d["w_mm"]
    # sanity: active nodes have w > 0, inactive nodes have w <= 0
    assert w[active].min() > 0
    assert w[inactive].max() <= 0

See docs/contact_liftoff_guide.md for interpretation guidance.
"""
from __future__ import annotations

from pathlib import Path

from anchorplate.model import (
    AnalysisOptions,
    CoupledLineLoad,
    FoundationPatch,
    MeshRefinementBox,
    PointSupport,
    SteelPlate,
)
from anchorplate.plotting import export_result_npz, plot_mesh, plot_result, plot_result_3d
from anchorplate.solver import solve_anchor_plate
from anchorplate.support import bedding_concrete_simple, bedding_timber_simple


def _print_contact_summary(summary: dict, outpath: Path) -> None:
    lines = [
        "Foundation contact summary",
        "=" * 40,
        f"  Patch nodes total  : {summary['n_patch_total']}",
        f"  In contact (active): {summary['n_active']}  ({summary['pct_active']:.1f}%)",
        f"  Lift-off (inactive): {summary['n_inactive']}  ({summary['pct_inactive']:.1f}%)",
        "",
        f"  w  active  range : [{summary['w_active_min']:.4f}, {summary['w_active_max']:.4f}] mm",
        f"  w  lift-off range: [{summary['w_inactive_min']:.4f}, {summary['w_inactive_max']:.4f}] mm",
        "",
        f"  Iterations       : {summary['n_iterations']}",
        f"  Converged        : {summary['converged']}",
        f"  History (delta/iter): {summary['history_changes']}",
        "=" * 40,
    ]
    text = "\n".join(lines)
    print(text)
    outpath.write_text(text, encoding="utf-8")


def main() -> None:
    # Model definition
    plate = SteelPlate(length_mm=300.0, width_mm=300.0, thickness_mm=15.0)

    # Four spring anchors at corners
    supports = [
        PointSupport(30.0,  30.0,  kind="spring", kz_n_per_mm=150_000.0, label="A1"),
        PointSupport(270.0, 30.0,  kind="spring", kz_n_per_mm=150_000.0, label="A2"),
        PointSupport(30.0,  270.0, kind="spring", kz_n_per_mm=150_000.0, label="A3"),
        PointSupport(270.0, 270.0, kind="spring", kz_n_per_mm=150_000.0, label="A4"),
    ]

    # Eccentric load: Fz=40 kN + Mx=3 kN·m (rocking load, induces lift-off)
    loads = [
        CoupledLineLoad(
            ref_x_mm=150.0, ref_y_mm=150.0,
            force_n=40_000.0, mx_nmm=3.0e6,
            line_spacing_mm=150.0, line_length_mm=100.0,
            label="RP",
        )
    ]

    # Winkler bedding: concrete (bottom) and timber (top) zones.
    # k_concrete = 32000/200 = 160 N/mm3  (stiff grout)
    # k_timber   = 370/100   =   3.7 N/mm3 (soft packing)
    # The 40x stiffness contrast drives the contact asymmetry.
    k_concrete = bedding_concrete_simple(e_cm_mpa=32_000.0, h_eff_mm=200.0)
    k_timber   = bedding_timber_simple(e90_mpa=370.0, h_eff_mm=100.0, spread_factor=1.0)

    foundation = [
        FoundationPatch(0.0, 300.0, 0.0,   150.0, k_area_n_per_mm3=k_concrete,
                        compression_only=True, label="concrete-zone"),
        FoundationPatch(0.0, 300.0, 150.0, 300.0, k_area_n_per_mm3=k_timber,
                        compression_only=True, label="timber-zone"),
    ]
    refinement = [MeshRefinementBox(40.0, 260.0, 80.0, 220.0, h_mm=4.0, label="load-zone")]

    options = AnalysisOptions(
        target_h_mm=8.0,
        output_dir="outputs/demo_foundation_patch_3d",
        z_plot_scale=80.0,
        save_3d_plots=True,
    )

    # Solve
    result = solve_anchor_plate(
        plate=plate, supports=supports, coupled_loads=loads,
        options=options, foundation_patches=foundation,
        refinement_boxes=refinement, name="demo_foundation_patch_3d",
    )

    # Outputs
    outdir = Path(options.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    plot_mesh(result.mesh, plate, supports, loads, [],
              refinement_boxes=refinement, foundation_patches=foundation,
              outpath=outdir / "mesh.png")
    plot_result(plate, supports, [], loads, result, options)
    plot_result_3d(plate, supports, result, options)

    npz_path, contact_summary = export_result_npz(
        result, outdir / "demo_foundation_patch_3d_result.npz"
    )
    _print_contact_summary(contact_summary, outdir / "contact_summary.txt")

    print(f"\nOutputs written to: {outdir.resolve()}")
    print(f"  .npz : {npz_path.name}")


if __name__ == "__main__":
    main()
