from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .model import AnalysisOptions, CoupledLineLoad, PointSupport, SteelPlate
from .plotting import export_result_npz, plot_result, plot_result_3d
from .solver import solve_anchor_plate


@dataclass(frozen=True)
class ProfisLikeCase:
    name: str
    description: str
    fz_n: float = 0.0
    fx_n: float = 0.0
    fy_n: float = 0.0
    mx_nmm: float = 0.0
    my_nmm: float = 0.0
    mz_nmm: float = 0.0
    e_out_mm: float = 180.0
    ref_x_mm: float = 150.0
    ref_y_mm: float = 150.0
    line_spacing_mm: float = 150.0
    line_length_mm: float = 100.0
    orientation: str = "vertical"


@dataclass
class BenchmarkRow:
    name: str
    description: str
    fx_kN: float
    fy_kN: float
    fz_kN: float
    mx_kNm: float
    my_kNm: float
    mz_kNm: float
    e_out_mm: float
    mx_eq_kNm: float
    my_eq_kNm: float
    unsupported_terms: str
    max_deflection_mm: float
    max_von_mises_mpa: float
    eta_plate: float
    reaction_min_kN: float
    reaction_max_kN: float
    reaction_absmax_kN: float
    sum_reactions_kN: float
    plot_path: str


def default_cases() -> list[ProfisLikeCase]:
    return [
        ProfisLikeCase("LC01_Fz_centric", "Pure centric transverse load", fz_n=50_000.0),
        ProfisLikeCase("LC02_Fz_plus_Mx", "Fz plus bending about x", fz_n=50_000.0, mx_nmm=4.0e6),
        ProfisLikeCase("LC03_Fz_plus_My", "Fz plus bending about y", fz_n=50_000.0, my_nmm=4.0e6),
        ProfisLikeCase("LC04_Fz_plus_Mxy", "Fz plus biaxial bending", fz_n=50_000.0, mx_nmm=3.0e6, my_nmm=5.0e6),
        ProfisLikeCase("LC05_pure_Mx", "Pure moment about x", mx_nmm=8.0e6),
        ProfisLikeCase("LC06_pure_My", "Pure moment about y", my_nmm=8.0e6),
        ProfisLikeCase("LC07_Fx_with_standoff", "Horizontal Fx at stand-off", fx_n=20_000.0, e_out_mm=180.0),
        ProfisLikeCase("LC08_Fy_with_standoff", "Horizontal Fy at stand-off", fy_n=20_000.0, e_out_mm=180.0),
        ProfisLikeCase("LC09_combo", "Combined Fx, Fy, Fz with direct moments", fx_n=12_000.0, fy_n=8_000.0, fz_n=40_000.0, mx_nmm=1.5e6, my_nmm=2.0e6),
        ProfisLikeCase("LC10_service", "Milder service-like combination", fx_n=5_000.0, fy_n=4_000.0, fz_n=20_000.0, mx_nmm=0.5e6, my_nmm=0.7e6, e_out_mm=150.0),
        ProfisLikeCase("LC11_Mz_only_note", "Pure Mz intentionally unsupported", mz_nmm=4.0e6),
    ]


def project_case_to_plate(case: ProfisLikeCase):
    unsupported: list[str] = []
    mx_eq = case.mx_nmm - case.fy_n * case.e_out_mm
    my_eq = case.my_nmm + case.fx_n * case.e_out_mm
    if abs(case.mz_nmm) > 1e-9:
        unsupported.append("Mz")
    representable = abs(case.fz_n) > 1e-9 or abs(mx_eq) > 1e-9 or abs(my_eq) > 1e-9
    if not representable:
        return None, unsupported, mx_eq, my_eq
    load = CoupledLineLoad(
        ref_x_mm=case.ref_x_mm,
        ref_y_mm=case.ref_y_mm,
        force_n=case.fz_n,
        mx_nmm=mx_eq,
        my_nmm=my_eq,
        line_spacing_mm=case.line_spacing_mm,
        line_length_mm=case.line_length_mm,
        orientation=case.orientation,
        label=case.name,
    )
    return load, unsupported, mx_eq, my_eq


def _save_summary_csv(rows: Sequence[BenchmarkRow], outpath: Path) -> None:
    with outpath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _save_summary_markdown(rows: Sequence[BenchmarkRow], outpath: Path) -> None:
    lines = [
        "# Anchor plate benchmark summary",
        "| Case | Description | mx,eq [kNm] | my,eq [kNm] | w_max [mm] | sigma_v,max [MPa] | eta | Rmin [kN] | Rmax [kN] | Unsupported |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.name} | {r.description} | {r.mx_eq_kNm:.2f} | {r.my_eq_kNm:.2f} | {r.max_deflection_mm:.3f} | {r.max_von_mises_mpa:.1f} | {r.eta_plate:.2f} | {r.reaction_min_kN:.2f} | {r.reaction_max_kN:.2f} | {r.unsupported_terms or ''} |"
        )
    outpath.write_text("\n".join(lines), encoding="utf-8")


def _save_overview_plot(rows: Sequence[BenchmarkRow], outpath: Path) -> None:
    names = [r.name.replace("LC", "") for r in rows]
    eta = np.array([r.eta_plate for r in rows], dtype=float)
    wmax = np.array([r.max_deflection_mm for r in rows], dtype=float)
    rabs = np.array([r.reaction_absmax_kN for r in rows], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    axes[0].bar(names, eta)
    axes[0].axhline(1.0, lw=1.0, color="0.2")
    axes[0].set_ylabel("η_placa [-]")
    axes[0].set_title("Benchmark overview")

    axes[1].bar(names, wmax)
    axes[1].set_ylabel("w_max [mm]")

    axes[2].bar(names, rabs)
    axes[2].set_ylabel("|R| max [kN]")
    axes[2].set_xlabel("Case")

    for ax in axes:
        ax.tick_params(axis="x", rotation=35)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def run_profis_like_benchmark(plate: SteelPlate, supports: Sequence[PointSupport], options: AnalysisOptions, outdir: Path, cases: Sequence[ProfisLikeCase] | None = None):
    outdir.mkdir(parents=True, exist_ok=True)
    cases = list(cases or default_cases())
    rows: list[BenchmarkRow] = []
    point_loads = []

    for case in cases:
        load, unsupported, mx_eq, my_eq = project_case_to_plate(case)
        if load is None:
            row = BenchmarkRow(
                name=case.name,
                description=case.description,
                fx_kN=case.fx_n / 1000.0,
                fy_kN=case.fy_n / 1000.0,
                fz_kN=case.fz_n / 1000.0,
                mx_kNm=case.mx_nmm / 1e6,
                my_kNm=case.my_nmm / 1e6,
                mz_kNm=case.mz_nmm / 1e6,
                e_out_mm=case.e_out_mm,
                mx_eq_kNm=mx_eq / 1e6,
                my_eq_kNm=my_eq / 1e6,
                unsupported_terms=", ".join(unsupported),
                max_deflection_mm=0.0,
                max_von_mises_mpa=0.0,
                eta_plate=0.0,
                reaction_min_kN=0.0,
                reaction_max_kN=0.0,
                reaction_absmax_kN=0.0,
                sum_reactions_kN=0.0,
                plot_path="",
            )
            rows.append(row)
            continue

        case_options = AnalysisOptions(**{**vars(options), 'output_dir': str(outdir / case.name)})
        result = solve_anchor_plate(
            plate=plate,
            supports=supports,
            point_loads=point_loads,
            coupled_loads=[load],
            options=case_options,
            name=case.name,
        )
        plot_path = plot_result(plate, supports, point_loads, [load], result, case_options)
        if case_options.save_3d_plots:
            plot_result_3d(plate, supports, result, case_options)
        if case_options.save_result_npz:
            export_result_npz(result, Path(case_options.output_dir) / f"{case.name.lower()}_result.npz")
        rz_kN = result.support_reactions_n / 1000.0
        rows.append(
            BenchmarkRow(
                name=case.name,
                description=case.description,
                fx_kN=case.fx_n / 1000.0,
                fy_kN=case.fy_n / 1000.0,
                fz_kN=case.fz_n / 1000.0,
                mx_kNm=case.mx_nmm / 1e6,
                my_kNm=case.my_nmm / 1e6,
                mz_kNm=case.mz_nmm / 1e6,
                e_out_mm=case.e_out_mm,
                mx_eq_kNm=mx_eq / 1e6,
                my_eq_kNm=my_eq / 1e6,
                unsupported_terms=", ".join(unsupported),
                max_deflection_mm=result.max_deflection_mm,
                max_von_mises_mpa=result.max_von_mises_mpa,
                eta_plate=result.max_von_mises_mpa / plate.fy_d_mpa,
                reaction_min_kN=float(np.min(rz_kN)),
                reaction_max_kN=float(np.max(rz_kN)),
                reaction_absmax_kN=float(np.max(np.abs(rz_kN))),
                sum_reactions_kN=float(np.sum(rz_kN)),
                plot_path=str(plot_path),
            )
        )

    _save_summary_csv(rows, outdir / "benchmark_summary.csv")
    _save_summary_markdown(rows, outdir / "benchmark_summary.md")
    _save_overview_plot(rows, outdir / "benchmark_overview.png")
    return rows
