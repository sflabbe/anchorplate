from __future__ import annotations

"""
demo_mesh_convergence.py
========================
Estudio de convergencia de malla (coarse/medium/fine) para dos familias:
- base (solo anclajes)
- foundation (anclajes + patch de fundación)

Outputs raíz: `outputs/demo_mesh_convergence/<mode>/` con CSVs de métricas,
plots comparativos y subcarpetas por nivel de malla.
"""

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from anchorplate.mesh import element_areas
from anchorplate.model import AnalysisOptions, CoupledLineLoad, MeshRefinementBox, PointSupport, SteelPlate
from anchorplate.plotting import plot_mesh
from anchorplate.solver import Result, solve_anchor_plate


@dataclass(frozen=True)
class MeshLevel:
    name: str
    target_h_mm: float


@dataclass
class ConvergenceRow:
    mode: str
    level: str
    target_h_mm: float
    n_nodes: int
    n_elements: int
    h_char_mm: float
    h_min_mm: float
    h_max_mm: float
    w_max_mm: float
    sigma_vm_max_mpa: float
    reaction_min_kN: float
    reaction_max_kN: float
    reaction_sum_kN: float
    delta_w_vs_fine_pct: float
    delta_sigma_vs_fine_pct: float
    delta_rsum_vs_fine_pct: float


REFINEMENT_BOX_GEOMETRY = [
    ("load-zone", 80.0, 220.0, 80.0, 220.0),
    ("A1", 0.0, 70.0, 0.0, 70.0),
    ("A2", 230.0, 300.0, 0.0, 70.0),
    ("A3", 0.0, 70.0, 230.0, 300.0),
    ("A4", 230.0, 300.0, 230.0, 300.0),
]


def _refinement_boxes_for_level(level_name: str) -> list[MeshRefinementBox]:
    # Estrategia monotónica: cada nivel reduce h objetivo local en todas las cajas.
    h_by_level = {"coarse": 8.0, "medium": 6.0, "fine": 4.0}
    h_local = h_by_level[level_name]
    return [
        MeshRefinementBox(x0, x1, y0, y1, h_mm=h_local, label=label)
        for (label, x0, x1, y0, y1) in REFINEMENT_BOX_GEOMETRY
    ]


def _mesh_size_stats_mm(result: Result) -> tuple[float, float, float]:
    areas = element_areas(result.mesh)
    h_eq = np.sqrt(2.0 * areas)
    return float(np.mean(h_eq)), float(np.min(h_eq)), float(np.max(h_eq))


def _relative_delta_pct(value: float, reference: float) -> float:
    denom = max(abs(reference), 1e-12)
    return 100.0 * (value - reference) / denom


def _run_one_level(
    plate: SteelPlate,
    supports: Sequence[PointSupport],
    load: CoupledLineLoad,
    level: MeshLevel,
    outdir: Path,
    refinement_boxes: Sequence[MeshRefinementBox] | None,
) -> tuple[ConvergenceRow, Result]:
    level_outdir = outdir / level.name
    level_outdir.mkdir(parents=True, exist_ok=True)
    options = AnalysisOptions(
        target_h_mm=level.target_h_mm,
        output_dir=str(level_outdir),
        save_plots=False,
        show_plots=False,
        save_result_npz=False,
        save_3d_plots=False,
    )
    result = solve_anchor_plate(
        plate=plate,
        supports=supports,
        coupled_loads=[load],
        options=options,
        refinement_boxes=refinement_boxes,
        name=f"mesh_{level.name}",
    )

    plot_mesh(
        result.mesh,
        plate,
        supports,
        [load],
        [],
        refinement_boxes=refinement_boxes,
        outpath=level_outdir / "mesh.png",
    )

    rz_kN = result.support_reactions_n / 1000.0
    h_char_mm, h_min_mm, h_max_mm = _mesh_size_stats_mm(result)

    row = ConvergenceRow(
        mode="",
        level=level.name,
        target_h_mm=level.target_h_mm,
        n_nodes=int(result.mesh.p.shape[1]),
        n_elements=int(result.mesh.t.shape[1]),
        h_char_mm=h_char_mm,
        h_min_mm=h_min_mm,
        h_max_mm=h_max_mm,
        w_max_mm=float(result.max_deflection_mm),
        sigma_vm_max_mpa=float(result.max_von_mises_mpa),
        reaction_min_kN=float(np.min(rz_kN)),
        reaction_max_kN=float(np.max(rz_kN)),
        reaction_sum_kN=float(np.sum(rz_kN)),
        delta_w_vs_fine_pct=np.nan,
        delta_sigma_vs_fine_pct=np.nan,
        delta_rsum_vs_fine_pct=np.nan,
    )
    return row, result


def _annotate_vs_fine(rows: list[ConvergenceRow]) -> None:
    fine = next(r for r in rows if r.level == "fine")
    for r in rows:
        r.delta_w_vs_fine_pct = _relative_delta_pct(r.w_max_mm, fine.w_max_mm)
        r.delta_sigma_vs_fine_pct = _relative_delta_pct(r.sigma_vm_max_mpa, fine.sigma_vm_max_mpa)
        r.delta_rsum_vs_fine_pct = _relative_delta_pct(r.reaction_sum_kN, fine.reaction_sum_kN)


def _save_csv(rows: Sequence[ConvergenceRow], outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _save_markdown(rows: Sequence[ConvergenceRow], outpath: Path, recommendation: str) -> None:
    lines = [
        "# Mesh convergence summary",
        "",
        "Caso: placa 300×300×15 mm, 4 supports fijos y carga acoplada Fz + Mx.",
        "",
        "| Mode | Level | target_h [mm] | n nodes | n elems | h_char [mm] | h_min [mm] | h_max [mm] | w_max [mm] | sigma_vm,max [MPa] | Rmin [kN] | Rmax [kN] | ΣR [kN] | Δw vs fine [%] | Δσvm vs fine [%] | ΔΣR vs fine [%] |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.mode} | {r.level} | {r.target_h_mm:.2f} | {r.n_nodes} | {r.n_elements} | {r.h_char_mm:.2f} | {r.h_min_mm:.2f} | {r.h_max_mm:.2f} | {r.w_max_mm:.4f} | {r.sigma_vm_max_mpa:.2f} | {r.reaction_min_kN:.2f} | {r.reaction_max_kN:.2f} | {r.reaction_sum_kN:.2f} | {r.delta_w_vs_fine_pct:+.2f} | {r.delta_sigma_vs_fine_pct:+.2f} | {r.delta_rsum_vs_fine_pct:+.2f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretación rápida",
            "- Convergencia principal: usa `w_max` y `ΣR` como métricas globales.",
            "- `h_min`/`h_max` ayudan a verificar jerarquía coarse→medium→fine de forma objetiva.",
            "- `sigma_vm_max` puede estar contaminado por singularidades locales y no debe usarse como único criterio.",
            f"- Recomendación de `target_h_mm` por defecto para este caso: **{recommendation}**.",
        ]
    )
    outpath.write_text("\n".join(lines), encoding="utf-8")


def _save_overview_plot(rows: Sequence[ConvergenceRow], outpath: Path) -> None:
    grouped: dict[str, list[ConvergenceRow]] = {}
    for row in rows:
        grouped.setdefault(row.mode, []).append(row)
    for mode in grouped:
        grouped[mode].sort(key=lambda r: r.target_h_mm, reverse=True)

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), constrained_layout=True)

    for mode, mode_rows in grouped.items():
        h = np.array([r.h_char_mm for r in mode_rows], dtype=float)
        w = np.array([r.w_max_mm for r in mode_rows], dtype=float)
        sigma = np.array([r.sigma_vm_max_mpa for r in mode_rows], dtype=float)
        rsum = np.array([r.reaction_sum_kN for r in mode_rows], dtype=float)

        axes[0].plot(h, w, marker="o", label=mode)
        axes[1].plot(h, sigma, marker="o", label=mode)
        axes[2].plot(h, rsum, marker="o", label=mode)

    axes[0].set_ylabel("w_max [mm]")
    axes[1].set_ylabel("sigma_vm,max [MPa]")
    axes[2].set_ylabel("ΣR [kN]")
    axes[2].set_xlabel("h_char [mm] (de coarse a fine)")
    axes[0].set_title("Mesh convergence overview")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.invert_xaxis()
        ax.legend()

    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def _recommend_target_h(rows: Sequence[ConvergenceRow]) -> str:
    preferred_mode = [r for r in rows if r.mode == "with_boxes"]
    if len(preferred_mode) < 3:
        preferred_mode = list(rows)
    candidates = {r.level: r for r in preferred_mode}
    medium = candidates.get("medium")
    fine = candidates.get("fine")
    if medium and fine:
        delta_w = abs(_relative_delta_pct(medium.w_max_mm, fine.w_max_mm))
        delta_r = abs(_relative_delta_pct(medium.reaction_sum_kN, fine.reaction_sum_kN))
        if delta_w <= 3.0 and delta_r <= 2.0:
            return f"{medium.target_h_mm:.1f} mm (nivel medium)"
    if fine:
        return f"{fine.target_h_mm:.1f} mm (nivel fine)"
    return "revisar estudio"


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo de convergencia de malla coarse/medium/fine")
    parser.add_argument(
        "--mode",
        choices=["both", "with-boxes", "without-boxes"],
        default="both",
        help="Ejecuta estudio con y/o sin refinement boxes",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/demo_mesh_convergence",
        help="Directorio base para resultados",
    )
    args = parser.parse_args()

    plate = SteelPlate(length_mm=300.0, width_mm=300.0, thickness_mm=15.0)
    supports = [
        PointSupport(30.0, 30.0, kind="fixed", label="A1"),
        PointSupport(270.0, 30.0, kind="fixed", label="A2"),
        PointSupport(30.0, 270.0, kind="fixed", label="A3"),
        PointSupport(270.0, 270.0, kind="fixed", label="A4"),
    ]
    load = CoupledLineLoad(
        ref_x_mm=150.0,
        ref_y_mm=150.0,
        force_n=50_000.0,
        mx_nmm=4.0e6,
        label="LC_Fz_plus_Mx",
    )
    levels = [
        MeshLevel("coarse", target_h_mm=12.0),
        MeshLevel("medium", target_h_mm=8.0),
        MeshLevel("fine", target_h_mm=6.0),
    ]

    root_out = Path(args.output_dir)
    rows: list[ConvergenceRow] = []

    modes: list[tuple[str, Sequence[MeshRefinementBox] | None]] = []
    if args.mode in {"both", "with-boxes"}:
        # Se define por nivel para forzar jerarquía monotónica también dentro de cajas.
        modes.append(("with_boxes", None))
    if args.mode in {"both", "without-boxes"}:
        modes.append(("without_boxes", None))

    for mode_name, base_refinement_boxes in modes:
        mode_rows: list[ConvergenceRow] = []
        mode_dir = root_out / mode_name
        mode_dir.mkdir(parents=True, exist_ok=True)

        for level in levels:
            row, _ = _run_one_level(
                plate=plate,
                supports=supports,
                load=load,
                level=level,
                outdir=mode_dir,
                refinement_boxes=(_refinement_boxes_for_level(level.name) if mode_name == "with_boxes" else base_refinement_boxes),
            )
            row.mode = mode_name
            mode_rows.append(row)

        _annotate_vs_fine(mode_rows)
        rows.extend(mode_rows)

    rows.sort(key=lambda r: (r.mode, ["coarse", "medium", "fine"].index(r.level)))

    recommendation = _recommend_target_h(rows)
    _save_csv(rows, root_out / "mesh_convergence_summary.csv")
    _save_markdown(rows, root_out / "mesh_convergence_summary.md", recommendation)
    _save_overview_plot(rows, root_out / "mesh_convergence_overview.png")

    print(f"Rows: {len(rows)}")
    print(f"Summary CSV: {root_out / 'mesh_convergence_summary.csv'}")
    print(f"Summary MD: {root_out / 'mesh_convergence_summary.md'}")
    print(f"Overview plot: {root_out / 'mesh_convergence_overview.png'}")
    print(f"Recommended target_h_mm: {recommendation}")


if __name__ == "__main__":
    main()
