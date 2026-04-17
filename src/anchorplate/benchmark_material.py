"""
benchmark_material.py
=====================
Benchmark runner comparing three bedding materials (Grout / Steel / Timber) across
four PROFIS-like load cases. Uses the compression-only foundation_patch submodel.

Physical scenario
-----------------
300 × 300 × 15 mm steel plate (S355), four M16 anchor bolts at corners (springs,
kz = 150 kN/mm each). A single full-plate foundation patch, material varies.
Load applied as a CoupledLineLoad (two bolt lines, vertical orientation).

Load cases
----------
  LC01  Fz = 50 kN  (centric)
  LC02  Fz = 40 kN + Mx = 4 kN·m  (eccentric → one-sided lift-off)
  LC03  Fz = 40 kN + My = 4 kN·m
  LC04  Mx = 6 kN·m  (pure moment → symmetric lift-off)

Materials
---------
  concrete  — Zementmörtel C25/30, h_eff = 50 mm    → k = 640  N/mm³
  steel     — Stahlpake S235, t = 10 mm              → k = 21 000 N/mm³
  timber    — GL24h (E90 = 390 MPa), h_eff = 50 mm  → k = 7.8  N/mm³
"""

from __future__ import annotations

import csv
import json
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .model import (
    AnalysisOptions,
    CoupledLineLoad,
    FoundationPatch,
    PointSupport,
    SteelLayer,
    SteelPlate,
)
from .benchmark_validity import classify_case_validity
from .support import (
    SupportMaterialModelResult,
    support_material_concrete_simple,
    support_material_steel_layers_simple,
    support_material_timber_simple,
)

if TYPE_CHECKING:
    from .solver import Result


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MaterialSpec:
    """One bedding material definition."""
    name: str                   # display name
    label: str                  # short ASCII key for filenames
    k_area_n_per_mm3: float     # Winkler stiffness [N/mm³]
    description: str = ""
    model_name: str = "legacy"
    parameters: dict[str, object] | None = None
    notes: str = ""


@dataclass(frozen=True)
class MaterialLoadCase:
    """One applied load combination."""
    name: str
    description: str
    force_n: float = 0.0
    mx_nmm: float  = 0.0
    my_nmm: float  = 0.0
    ref_x_mm: float = 150.0
    ref_y_mm: float = 150.0
    line_spacing_mm: float = 150.0
    line_length_mm: float  = 100.0
    orientation: str = "vertical"


@dataclass
class MaterialBenchmarkRow:
    material:                    str
    model_name:                  str
    k_area_n_mm3:                float
    model_parameters_json:       str
    model_notes:                 str
    support_type:                str
    solve_status:                str
    valid_solution:              bool
    metrics_comparable:          bool
    solve_error:                 str
    failure_reason:              str
    load_case:                   str
    load_description:            str
    force_kN:                    float
    mx_kNm:                      float
    my_kNm:                      float
    # Contact
    n_patch_total:               int
    n_active:                    int
    n_inactive:                  int
    pct_active:                  float
    pct_inactive:                float
    n_iterations:                int
    converged:                   bool
    # Structural
    w_max_mm:                    float
    sigma_vm_max_mpa:            float
    eta_plate:                   float
    # Reactions — anchors (discrete springs) only
    sum_spring_reactions_kN:     float
    max_spring_reaction_kN:      float
    min_spring_reaction_kN:      float
    anchor_active_count:         int
    anchor_inactive_count:       int
    anchor_reactions_kN_json:    str
    # Reactions — continuous foundation (Winkler bedding) total
    sum_foundation_reaction_kN:  float
    # Global equilibrium check: springs + foundation ≈ applied Fz
    sum_total_reactions_kN:      float
    equilibrium_error_kN:        float
    equilibrium_ok:              bool
    equilibrium_tol_kN:          float


# ---------------------------------------------------------------------------
# Default materials and load cases
# ---------------------------------------------------------------------------

def default_materials() -> list[MaterialSpec]:
    concrete = support_material_concrete_simple(e_cm_mpa=32_000.0, h_eff_mm=50.0)
    steel = support_material_steel_layers_simple(
        [SteelLayer(thickness_mm=10.0, youngs_modulus_mpa=210_000.0)]
    )
    timber = support_material_timber_simple(e90_mpa=390.0, h_eff_mm=50.0)
    return [
        _material_spec_from_model_result(
            name="Grout C25/30 h=50 mm",
            label="grout",
            description="E_cm=32000 MPa, h_eff=50 mm",
            model=concrete,
        ),
        _material_spec_from_model_result(
            name="Steel S235 t=10 mm",
            label="steel",
            description="E=210000 MPa, t=10 mm",
            model=steel,
        ),
        _material_spec_from_model_result(
            name="Timber GL24h h=50 mm",
            label="timber",
            description="E_90=390 MPa, h_eff=50 mm",
            model=timber,
        ),
    ]


def _material_spec_from_model_result(
    name: str,
    label: str,
    description: str,
    model: SupportMaterialModelResult,
) -> MaterialSpec:
    return MaterialSpec(
        name=name,
        label=label,
        k_area_n_per_mm3=model.k_area_n_per_mm3,
        description=description,
        model_name=model.model_name,
        parameters=model.parameters,
        notes=model.notes,
    )


def material_spec_from_model_result(
    name: str,
    label: str,
    model: SupportMaterialModelResult,
    description: str = "",
) -> MaterialSpec:
    """
    Public helper: create benchmark MaterialSpec from SupportMaterialModelResult.
    """
    return _material_spec_from_model_result(
        name=name,
        label=label,
        description=description,
        model=model,
    )


def default_load_cases() -> list[MaterialLoadCase]:
    return [
        MaterialLoadCase("LC01_Fz_centric",
                         "Centric Fz = 50 kN",
                         force_n=50_000.0),
        MaterialLoadCase("LC02_Fz_Mx",
                         "Fz = 40 kN + Mx = 4 kN·m",
                         force_n=40_000.0, mx_nmm=4.0e6),
        MaterialLoadCase("LC03_Fz_My",
                         "Fz = 40 kN + My = 4 kN·m",
                         force_n=40_000.0, my_nmm=4.0e6,
                         orientation="horizontal"),
        MaterialLoadCase("LC04_Mx_pure",
                         "Pure Mx = 6 kN·m",
                         mx_nmm=6.0e6),
    ]


def _foundation_total_reaction(result: "Result", foundation_patches: Sequence[FoundationPatch]) -> float:
    """
    Compute the total upward reaction force from all active foundation nodes.

    R_foundation = Σ_i  k_area_i · A_tributary_i · w_i   (sum over active nodes)

    This is required for global equilibrium:
        sum(spring_reactions) + R_foundation ≈ applied_Fz
    """
    from .mesh import nodal_tributary_areas

    tributary = nodal_tributary_areas(result.mesh)
    w = result.w_vertex_mm
    total = 0.0
    for patch, active_ids in zip(foundation_patches, result.foundation_state.active_vertices):
        if not active_ids:
            continue
        ids = np.array(sorted(active_ids), dtype=int)
        k_nodal = patch.k_area_n_per_mm3 * tributary[ids]
        total += float(np.sum(k_nodal * w[ids]))
    return total


def default_spring_supports() -> list[PointSupport]:
    """Default 4-corner M16 spring supports used in the benchmark scenario."""
    return [
        PointSupport(30.0,  30.0,  kind="spring", kz_n_per_mm=150_000.0, label="A1"),
        PointSupport(270.0, 30.0,  kind="spring", kz_n_per_mm=150_000.0, label="A2"),
        PointSupport(30.0,  270.0, kind="spring", kz_n_per_mm=150_000.0, label="A3"),
        PointSupport(270.0, 270.0, kind="spring", kz_n_per_mm=150_000.0, label="A4"),
    ]


def build_corner_supports(kind: str = "spring") -> list[PointSupport]:
    """4-corner supports with configurable discrete support kind."""
    if kind not in {"spring", "spring_tension_only"}:
        raise ValueError(f"Unsupported support kind for benchmark: {kind}")
    return [
        PointSupport(30.0, 30.0, kind=kind, kz_n_per_mm=150_000.0, label="A1"),
        PointSupport(270.0, 30.0, kind=kind, kz_n_per_mm=150_000.0, label="A2"),
        PointSupport(30.0, 270.0, kind=kind, kz_n_per_mm=150_000.0, label="A3"),
        PointSupport(270.0, 270.0, kind=kind, kz_n_per_mm=150_000.0, label="A4"),
    ]


def _supports_with_kind(
    supports: Sequence[PointSupport],
    kind: str,
) -> list[PointSupport]:
    """Clone supports while forcing spring-like supports to the requested kind."""
    if kind not in {"spring", "spring_tension_only"}:
        raise ValueError(f"Unsupported support kind for benchmark: {kind}")
    mapped: list[PointSupport] = []
    for s in supports:
        mapped_kind = kind if s.kind in {"spring", "spring_tension_only"} else s.kind
        mapped.append(
            PointSupport(
                x_mm=s.x_mm,
                y_mm=s.y_mm,
                kind=mapped_kind,
                kz_n_per_mm=s.kz_n_per_mm,
                label=s.label,
            )
        )
    return mapped


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_material_benchmark(
    plate: SteelPlate,
    supports: Sequence[PointSupport] | None = None,
    materials: Sequence[MaterialSpec] | None = None,
    load_cases: Sequence[MaterialLoadCase] | None = None,
    options: AnalysisOptions | None = None,
    outdir: Path | None = None,
    compression_only: bool = True,
    hybrid_support_kind: str = "spring",
) -> list[MaterialBenchmarkRow]:
    """
    Run the material benchmark and return one row per (material, load_case) combination.

    Parameters
    ----------
    plate           : SteelPlate geometry and material.
    supports        : List of PointSupport (spring or fixed). If None, default
                      4-corner spring supports are used.
    materials       : Bedding materials to compare; defaults to grout / steel / timber.
    load_cases      : Load combinations; defaults to LC01–LC04.
    options         : AnalysisOptions template (output_dir is overridden per case).
    outdir          : Root output directory.
    compression_only: Whether to use the compression-only (lift-off) contact model.
    hybrid_support_kind:
                    Discrete support law used by hybrid benchmark:
                    "spring" (bidirectional linear) or "spring_tension_only".

    Returns
    -------
    List of MaterialBenchmarkRow — one per (material × load case).
    Sorted: material-first, then load case.
    """
    from .plotting import (
        _contact_summary,
        _foundation_masks,
        export_result_npz,
        plot_result,
        plot_result_3d,
    )
    from .solver import solve_anchor_plate

    materials  = list(materials  or default_materials())
    load_cases = list(load_cases or default_load_cases())
    supports   = list(supports or default_spring_supports())
    supports   = _supports_with_kind(supports, hybrid_support_kind)
    options    = options or AnalysisOptions()
    outdir     = outdir or Path(options.output_dir) / "material_benchmark"
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list[MaterialBenchmarkRow] = []

    for mat in materials:
        for lc in load_cases:
            case_name = f"{mat.label}__{lc.name}"
            case_dir  = outdir / mat.label / lc.name
            case_dir.mkdir(parents=True, exist_ok=True)

            # Override output_dir; inherit other options
            case_opts = AnalysisOptions(
                target_h_mm=options.target_h_mm,
                output_dir=str(case_dir),
                save_plots=options.save_plots,
                show_plots=False,
                save_result_npz=options.save_result_npz,
                save_3d_plots=options.save_3d_plots,
                z_plot_scale=options.z_plot_scale,
                foundation_iterations_max=options.foundation_iterations_max,
                foundation_contact_tol_mm=options.foundation_contact_tol_mm,
                equilibrium_tol_abs_kN=options.equilibrium_tol_abs_kN,
                equilibrium_tol_rel=options.equilibrium_tol_rel,
            )

            # Build load — skip cases with no representable load
            if abs(lc.force_n) < 1.0 and abs(lc.mx_nmm) < 1.0 and abs(lc.my_nmm) < 1.0:
                continue
            coupled_load = CoupledLineLoad(
                ref_x_mm=lc.ref_x_mm,
                ref_y_mm=lc.ref_y_mm,
                force_n=lc.force_n,
                mx_nmm=lc.mx_nmm,
                my_nmm=lc.my_nmm,
                line_spacing_mm=lc.line_spacing_mm,
                line_length_mm=lc.line_length_mm,
                orientation=lc.orientation,
                label=lc.name,
            )

            foundation = [
                FoundationPatch(
                    x_min_mm=0.0, x_max_mm=plate.length_mm,
                    y_min_mm=0.0, y_max_mm=plate.width_mm,
                    k_area_n_per_mm3=mat.k_area_n_per_mm3,
                    compression_only=compression_only,
                    label=mat.label,
                )
            ]

            solve_error = ""
            try:
                result = solve_anchor_plate(
                    plate=plate,
                    supports=supports,
                    coupled_loads=[coupled_load],
                    options=case_opts,
                    foundation_patches=foundation,
                    name=case_name,
                )
                initial_status = "ok"
            except RuntimeError as exc:
                result = None
                initial_status = "failed"
                solve_error = str(exc)

            if result is not None and case_opts.save_plots:
                plot_result(plate, supports, [], [coupled_load], result, case_opts)
            if result is not None and case_opts.save_3d_plots:
                plot_result_3d(plate, supports, result, case_opts)

            # NPZ + contact summary
            if result is not None and case_opts.save_result_npz:
                export_result_npz(result, case_dir / f"{case_name}_result.npz")
                metadata = {
                    "material_name": mat.name,
                    "material_label": mat.label,
                    "model_name": mat.model_name,
                    "parameters": mat.parameters or {},
                    "notes": mat.notes,
                    "support_type": hybrid_support_kind,
                }
                (case_dir / f"{case_name}_material_model.json").write_text(
                    json.dumps(metadata, indent=2, sort_keys=True),
                    encoding="utf-8",
                )

            # Collect contact stats
            if result is not None:
                active_mask, inactive_mask = _foundation_masks(result)
                summary = _contact_summary(result, active_mask, inactive_mask)
                R_kN = result.support_reactions_n / 1000.0
                found_R_kN = _foundation_total_reaction(result, foundation) / 1000.0
                anchor_reaction_map = {
                    (s.label or f"A{i+1}"): float(R_kN[i]) for i, s in enumerate(supports)
                }
                anchor_active_count = int(np.sum(result.support_active))
                anchor_inactive_count = int(len(result.support_active) - anchor_active_count)
                sum_spring_kN = float(np.sum(R_kN))
                sum_total_kN = sum_spring_kN + found_R_kN
                validity = classify_case_validity(
                    initial_solve_status=initial_status,
                    solve_error=solve_error,
                    expected_vertical_load_kN=lc.force_n / 1000.0,
                    total_reactions_kN=sum_total_kN,
                    contact_converged=bool(summary["converged"]),
                    requires_contact_convergence=True,
                    has_foundation_patch=True,
                    support_type=hybrid_support_kind,
                    force_n=lc.force_n,
                    mx_nmm=lc.mx_nmm,
                    my_nmm=lc.my_nmm,
                    equilibrium_tol_abs_kN=case_opts.equilibrium_tol_abs_kN,
                    equilibrium_tol_rel=case_opts.equilibrium_tol_rel,
                )
            else:
                summary = {
                    "n_patch_total": 0,
                    "n_active": 0,
                    "n_inactive": 0,
                    "pct_active": 0.0,
                    "pct_inactive": 0.0,
                    "n_iterations": case_opts.foundation_iterations_max,
                    "converged": False,
                }
                found_R_kN = float("nan")
                anchor_reaction_map = {}
                anchor_active_count = 0
                anchor_inactive_count = len(supports)
                sum_spring_kN = float("nan")
                sum_total_kN = float("nan")
                validity = classify_case_validity(
                    initial_solve_status=initial_status,
                    solve_error=solve_error,
                    expected_vertical_load_kN=lc.force_n / 1000.0,
                    total_reactions_kN=0.0,
                    contact_converged=False,
                    requires_contact_convergence=True,
                    has_foundation_patch=True,
                    support_type=hybrid_support_kind,
                    force_n=lc.force_n,
                    mx_nmm=lc.mx_nmm,
                    my_nmm=lc.my_nmm,
                    equilibrium_tol_abs_kN=case_opts.equilibrium_tol_abs_kN,
                    equilibrium_tol_rel=case_opts.equilibrium_tol_rel,
                )

            rows.append(MaterialBenchmarkRow(
                material=mat.name,
                model_name=mat.model_name,
                k_area_n_mm3=mat.k_area_n_per_mm3,
                model_parameters_json=json.dumps(mat.parameters or {}, sort_keys=True),
                model_notes=mat.notes,
                support_type=hybrid_support_kind,
                solve_status=validity.solve_status,
                valid_solution=validity.valid_solution,
                metrics_comparable=validity.metrics_comparable,
                solve_error=solve_error,
                failure_reason=validity.failure_reason,
                load_case=lc.name,
                load_description=lc.description,
                force_kN=lc.force_n / 1000.0,
                mx_kNm=lc.mx_nmm / 1.0e6,
                my_kNm=lc.my_nmm / 1.0e6,
                n_patch_total=summary["n_patch_total"],
                n_active=summary["n_active"],
                n_inactive=summary["n_inactive"],
                pct_active=summary["pct_active"],
                pct_inactive=summary["pct_inactive"],
                n_iterations=summary["n_iterations"],
                converged=summary["converged"],
                w_max_mm=result.max_deflection_mm if result is not None else float("nan"),
                sigma_vm_max_mpa=result.max_von_mises_mpa if result is not None else float("nan"),
                eta_plate=(result.max_von_mises_mpa / plate.fy_d_mpa) if result is not None else float("nan"),
                sum_spring_reactions_kN=sum_spring_kN,
                max_spring_reaction_kN=float(np.max(R_kN)) if result is not None else float("nan"),
                min_spring_reaction_kN=float(np.min(R_kN)) if result is not None else float("nan"),
                anchor_active_count=anchor_active_count,
                anchor_inactive_count=anchor_inactive_count,
                anchor_reactions_kN_json=json.dumps(anchor_reaction_map, sort_keys=True),
                sum_foundation_reaction_kN=found_R_kN,
                sum_total_reactions_kN=sum_total_kN,
                equilibrium_error_kN=validity.equilibrium_error_kN,
                equilibrium_ok=validity.equilibrium_ok,
                equilibrium_tol_kN=validity.equilibrium_tol_kN,
            ))

    _save_csv(rows, outdir / "material_benchmark_summary.csv")
    _save_markdown(rows, outdir / "material_benchmark_summary.md")
    _save_overview_plots(rows, materials, load_cases, outdir)
    return rows


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _save_csv(rows: Sequence[MaterialBenchmarkRow], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(_serialize_row(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(_serialize_row(row))


def _serialize_row(row: MaterialBenchmarkRow) -> dict[str, object]:
    """Stable dataclass serialization helper for benchmark summary outputs."""
    return asdict(row)


def _save_markdown(rows: Sequence[MaterialBenchmarkRow], path: Path) -> None:
    if not rows:
        return
    header = textwrap.dedent("""\
        # Material benchmark — foundation patch (compression-only)

        | Material | Model | Support type | Solve | Valid | Comparable metrics | Failure reason | k [N/mm³] | Load case | Fz [kN] | Mx [kNm] | My [kNm] | Contact [%] | Lift-off [%] | Iter | Anchors active/inactive | w_max [mm] | σ_v,max [MPa] | η | ΣR_anker [kN] | ΣR_found [kN] | ΣR_total [kN] | Eq. err [kN] | Eq. ok |
        |---|---|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
    """)
    lines = [header.rstrip()]
    for r in rows:
        lines.append(
            f"| {r.material} | {r.model_name} | {r.support_type} | {r.solve_status} | {r.valid_solution} | {r.metrics_comparable} | {r.failure_reason or '—'} | {r.k_area_n_mm3:.1f} | {r.load_case} "
            f"| {r.force_kN:.0f} | {r.mx_kNm:.1f} | {r.my_kNm:.1f} "
            f"| {r.pct_active:.1f} | {r.pct_inactive:.1f} | {r.n_iterations} "
            f"| {r.anchor_active_count}/{r.anchor_inactive_count} | {r.w_max_mm:.4f} | {r.sigma_vm_max_mpa:.1f} | {r.eta_plate:.3f} "
            f"| {r.sum_spring_reactions_kN:.2f} | {r.sum_foundation_reaction_kN:.2f} "
            f"| {r.sum_total_reactions_kN:.2f} | {r.equilibrium_error_kN:.3f} | {r.equilibrium_ok} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_overview_plots(
    rows: Sequence[MaterialBenchmarkRow],
    materials: Sequence[MaterialSpec],
    load_cases: Sequence[MaterialLoadCase],
    outdir: Path,
) -> None:
    """
    Two overview figures:
      1. Grid: one subplot per load case, bars per material for η and contact%.
      2. Contact% vs k_area scatter (log-x scale, one series per load case).
    """
    if not rows:
        return

    mat_labels  = [m.label for m in materials]
    mat_names   = [m.name  for m in materials]
    lc_names    = [lc.name for lc in load_cases]
    n_mat       = len(mat_labels)
    n_lc        = len(lc_names)
    colors      = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # ── Figure 1: η and contact% per load case ───────────────────────────────
    fig, axes = plt.subplots(n_lc, 2, figsize=(12, 3 * n_lc), constrained_layout=True)
    if n_lc == 1:
        axes = [axes]

    for i, lc_name in enumerate(lc_names):
        lc_rows = [r for r in rows if r.load_case == lc_name]
        lc_desc = lc_rows[0].load_description if lc_rows else lc_name

        eta_vals = [next((r.eta_plate for r in lc_rows if r.material == m.name and r.metrics_comparable), np.nan) for m in materials]
        contact_vals = [next((r.pct_active for r in lc_rows if r.material == m.name and r.metrics_comparable), np.nan) for m in materials]
        liftoff_vals = [next((r.pct_inactive for r in lc_rows if r.material == m.name and r.metrics_comparable), np.nan) for m in materials]

        ax_eta = axes[i][0]
        ax_c   = axes[i][1]

        bars = ax_eta.bar(mat_names, eta_vals, color=colors[:n_mat])
        ax_eta.axhline(1.0, color="k", lw=0.8, ls="--")
        ax_eta.set_ylabel("η  [-]")
        ax_eta.set_title(f"{lc_name}: Plate utilisation η")
        ax_eta.set_ylim(bottom=0)
        for bar, val in zip(bars, eta_vals):
            ax_eta.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        x = np.arange(n_mat)
        width = 0.4
        b1 = ax_c.bar(x - width/2, contact_vals, width, label="In contact", color="#2ca02c", alpha=0.8)
        b2 = ax_c.bar(x + width/2, liftoff_vals, width, label="Lift-off",   color="#d62728", alpha=0.8)
        ax_c.set_xticks(x)
        ax_c.set_xticklabels(mat_names, fontsize=8)
        ax_c.set_ylabel("Nodes [%]")
        ax_c.set_title(f"{lc_name}: Contact state")
        ax_c.set_ylim(0, 105)
        ax_c.legend(fontsize=8)
        for bar, val in zip(list(b1) + list(b2), contact_vals + liftoff_vals):
            ax_c.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                      f"{val:.0f}%", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Material benchmark — foundation patch (compression-only)", fontsize=11)
    fig.savefig(outdir / "overview_utilisation_contact.png", dpi=180)
    plt.close(fig)

    # ── Figure 2: contact% vs k_area (log scale) ─────────────────────────────
    fig2, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)

    for j, lc_name in enumerate(lc_names):
        lc_rows   = [r for r in rows if r.load_case == lc_name]
        valid_rows = [r for r in lc_rows if r.metrics_comparable]
        k_vals = [r.k_area_n_mm3 for r in valid_rows]
        pct_vals = [r.pct_active for r in valid_rows]
        w_vals = [r.w_max_mm for r in valid_rows]
        ax_top.plot(k_vals, pct_vals, "o-", label=lc_name, color=colors[j % len(colors)])
        ax_bot.plot(k_vals, w_vals,   "s--", label=lc_name, color=colors[j % len(colors)])

    for ax in (ax_top, ax_bot):
        ax.set_xscale("log")
        ax.set_xlabel("k_area [N/mm³]  (log scale)")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

    ax_top.set_ylabel("Contact area [%]")
    ax_top.set_title("Contact percentage vs bedding stiffness")
    ax_bot.set_ylabel("w_max [mm]")
    ax_bot.set_title("Max deflection vs bedding stiffness")
    fig2.suptitle("Material benchmark — contact vs stiffness", fontsize=11)
    fig2.savefig(outdir / "overview_contact_vs_stiffness.png", dpi=180)
    plt.close(fig2)
