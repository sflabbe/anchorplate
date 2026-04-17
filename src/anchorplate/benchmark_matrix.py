from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .benchmark import ProfisLikeCase, project_case_to_plate
from .benchmark_material import _foundation_total_reaction
from .model import AnalysisOptions, FoundationPatch, PointSupport, SteelLayer, SteelPlate
from .plotting import _contact_summary, _foundation_masks
from .solver import solve_anchor_plate
from .support import (
    support_material_concrete_simple,
    support_material_steel_layers_simple,
    support_material_timber_simple,
)


@dataclass(frozen=True)
class SupportModelSpec:
    key: str
    display_name: str
    model_type: str
    supports: tuple[PointSupport, ...]
    foundation_patch: FoundationPatch | None = None
    model_notes: str = ""


@dataclass
class BenchmarkMatrixRow:
    model_key: str
    model_name: str
    model_type: str
    model_notes: str
    load_case: str
    load_description: str
    fz_kN: float
    mx_kNm: float
    my_kNm: float
    w_max_mm: float
    sigma_vm_max_mpa: float
    eta_plate: float
    sum_reactions_kN: float
    sum_spring_reactions_kN: float
    sum_foundation_reactions_kN: float
    rz_min_kN: float
    rz_max_kN: float
    contact_active_pct: float | None
    contact_iterations: int | None
    contact_converged: bool | None
    k_area_n_per_mm3: float | None


def default_matrix_load_cases() -> list[ProfisLikeCase]:
    return [
        ProfisLikeCase("LC01_Fz_centric", "Fz céntrico", fz_n=50_000.0),
        ProfisLikeCase("LC02_Fz_plus_Mx", "Fz + Mx", fz_n=40_000.0, mx_nmm=4.0e6),
        ProfisLikeCase("LC03_Fz_plus_My", "Fz + My", fz_n=40_000.0, my_nmm=4.0e6),
        ProfisLikeCase("LC04_pure_Mx", "Mx puro", mx_nmm=6.0e6),
    ]


def default_matrix_models() -> list[SupportModelSpec]:
    spring_anchors = (
        PointSupport(30.0, 30.0, kind="spring", kz_n_per_mm=150_000.0, label="A1"),
        PointSupport(270.0, 30.0, kind="spring", kz_n_per_mm=150_000.0, label="A2"),
        PointSupport(30.0, 270.0, kind="spring", kz_n_per_mm=150_000.0, label="A3"),
        PointSupport(270.0, 270.0, kind="spring", kz_n_per_mm=150_000.0, label="A4"),
    )
    fixed_anchors = tuple(
        PointSupport(s.x_mm, s.y_mm, kind="fixed", kz_n_per_mm=0.0, label=s.label) for s in spring_anchors
    )

    k_concrete = support_material_concrete_simple(e_cm_mpa=32_000.0, h_eff_mm=50.0).k_area_n_per_mm3
    k_steel = support_material_steel_layers_simple(
        [SteelLayer(thickness_mm=10.0, youngs_modulus_mpa=210_000.0)]
    ).k_area_n_per_mm3
    k_timber = support_material_timber_simple(e90_mpa=390.0, h_eff_mm=50.0).k_area_n_per_mm3

    patch = dict(x_min_mm=0.0, x_max_mm=300.0, y_min_mm=0.0, y_max_mm=300.0, compression_only=True)

    return [
        SupportModelSpec(
            key="fixed",
            display_name="fixed",
            model_type="discrete_only",
            supports=fixed_anchors,
            model_notes="4 anclajes fijos (sin rigidez distribuida).",
        ),
        SupportModelSpec(
            key="spring_anchors",
            display_name="spring_anchors",
            model_type="discrete_only",
            supports=spring_anchors,
            model_notes="4 anclajes elásticos (kz=150 kN/mm por anclaje).",
        ),
        SupportModelSpec(
            key="foundation_patch_concrete",
            display_name="foundation_patch_concrete",
            model_type="hybrid_springs_plus_foundation_patch",
            supports=spring_anchors,
            foundation_patch=FoundationPatch(k_area_n_per_mm3=k_concrete, label="concrete", **patch),
            model_notes="Anclajes elásticos + parche de contacto hormigón (compresión-only).",
        ),
        SupportModelSpec(
            key="foundation_patch_steel",
            display_name="foundation_patch_steel",
            model_type="hybrid_springs_plus_foundation_patch",
            supports=spring_anchors,
            foundation_patch=FoundationPatch(k_area_n_per_mm3=k_steel, label="steel", **patch),
            model_notes="Anclajes elásticos + parche de contacto acero (compresión-only).",
        ),
        SupportModelSpec(
            key="foundation_patch_timber",
            display_name="foundation_patch_timber",
            model_type="hybrid_springs_plus_foundation_patch",
            supports=spring_anchors,
            foundation_patch=FoundationPatch(k_area_n_per_mm3=k_timber, label="timber", **patch),
            model_notes="Anclajes elásticos + parche de contacto madera (compresión-only).",
        ),
    ]


def run_support_model_matrix_benchmark(
    plate: SteelPlate,
    options: AnalysisOptions,
    outdir: Path,
    models: Sequence[SupportModelSpec] | None = None,
    load_cases: Sequence[ProfisLikeCase] | None = None,
) -> list[BenchmarkMatrixRow]:
    outdir.mkdir(parents=True, exist_ok=True)
    models = list(models or default_matrix_models())
    load_cases = list(load_cases or default_matrix_load_cases())

    rows: list[BenchmarkMatrixRow] = []

    for model in models:
        for lc in load_cases:
            load, unsupported, _, _ = project_case_to_plate(lc)
            if load is None:
                continue
            if unsupported:
                raise ValueError(f"Load case {lc.name} has unsupported terms for matrix benchmark: {unsupported}")

            case_dir = outdir / model.key / lc.name
            case_options = AnalysisOptions(**{**vars(options), "output_dir": str(case_dir)})
            foundation = [model.foundation_patch] if model.foundation_patch else []

            result = solve_anchor_plate(
                plate=plate,
                supports=model.supports,
                coupled_loads=[load],
                options=case_options,
                foundation_patches=foundation,
                name=f"{model.key}__{lc.name}",
            )

            r_spring_kN = result.support_reactions_n / 1000.0
            foundation_kN = _foundation_total_reaction(result, foundation) / 1000.0 if foundation else 0.0
            total_kN = float(np.sum(r_spring_kN)) + float(foundation_kN)

            if foundation:
                active_mask, inactive_mask = _foundation_masks(result)
                contact = _contact_summary(result, active_mask, inactive_mask)
                pct_active: float | None = float(contact["pct_active"])
                n_iter: int | None = int(contact["n_iterations"])
                converged: bool | None = bool(contact["converged"])
                k_area = float(model.foundation_patch.k_area_n_per_mm3)
            else:
                pct_active = None
                n_iter = None
                converged = None
                k_area = None

            rows.append(
                BenchmarkMatrixRow(
                    model_key=model.key,
                    model_name=model.display_name,
                    model_type=model.model_type,
                    model_notes=model.model_notes,
                    load_case=lc.name,
                    load_description=lc.description,
                    fz_kN=lc.fz_n / 1000.0,
                    mx_kNm=lc.mx_nmm / 1e6,
                    my_kNm=lc.my_nmm / 1e6,
                    w_max_mm=float(result.max_deflection_mm),
                    sigma_vm_max_mpa=float(result.max_von_mises_mpa),
                    eta_plate=float(result.max_von_mises_mpa / plate.fy_d_mpa),
                    sum_reactions_kN=total_kN,
                    sum_spring_reactions_kN=float(np.sum(r_spring_kN)),
                    sum_foundation_reactions_kN=float(foundation_kN),
                    rz_min_kN=float(np.min(r_spring_kN)),
                    rz_max_kN=float(np.max(r_spring_kN)),
                    contact_active_pct=pct_active,
                    contact_iterations=n_iter,
                    contact_converged=converged,
                    k_area_n_per_mm3=k_area,
                )
            )

    _save_matrix_csv(rows, outdir / "benchmark_matrix_summary.csv")
    _save_matrix_markdown(rows, outdir / "benchmark_matrix_summary.md")
    _save_overview_plot(rows, outdir / "benchmark_matrix_overview.png")
    _save_technical_note(rows, outdir / "benchmark_matrix_note.md")
    return rows


def _save_matrix_csv(rows: Sequence[BenchmarkMatrixRow], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _fmt_optional(value: float | int | bool | None, fmt: str = "{:.2f}") -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "sí" if value else "no"
    if isinstance(value, int):
        return f"{value:d}"
    return fmt.format(value)


def _save_matrix_markdown(rows: Sequence[BenchmarkMatrixRow], path: Path) -> None:
    if not rows:
        return
    lines = [
        "# Benchmark matrix de modelos de soporte",
        "",
        "Comparación consolidada de `fixed`, `spring_anchors` y variantes `foundation_patch_*` bajo los mismos load cases.",
        "",
        "| Modelo | Tipo de modelo | Caso | w_max [mm] | σ_vm,max [MPa] | η_plate | ΣR [kN] | Rz_min [kN] | Rz_max [kN] | Contacto activo [%] | Iter. contacto | k_area [N/mm³] |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.model_name} | {r.model_type} | {r.load_case} | {r.w_max_mm:.4f} | {r.sigma_vm_max_mpa:.1f} | {r.eta_plate:.3f} "
            f"| {r.sum_reactions_kN:.2f} | {r.rz_min_kN:.2f} | {r.rz_max_kN:.2f} | {_fmt_optional(r.contact_active_pct, '{:.1f}')} "
            f"| {_fmt_optional(r.contact_iterations, '{}')} | {_fmt_optional(r.k_area_n_per_mm3, '{:.1f}')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_overview_plot(rows: Sequence[BenchmarkMatrixRow], path: Path) -> None:
    if not rows:
        return

    models = sorted({r.model_name for r in rows})
    cases = sorted({r.load_case for r in rows})
    idx = np.arange(len(cases), dtype=float)
    width = 0.14

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for m_idx, model in enumerate(models):
        sub = [r for r in rows if r.model_name == model]
        eta = [next((r.eta_plate for r in sub if r.load_case == c), np.nan) for c in cases]
        w = [next((r.w_max_mm for r in sub if r.load_case == c), np.nan) for c in cases]
        s = [next((r.sigma_vm_max_mpa for r in sub if r.load_case == c), np.nan) for c in cases]
        x = idx + (m_idx - (len(models) - 1) / 2.0) * width
        axes[0].bar(x, eta, width=width, label=model, color=colors[m_idx % len(colors)])
        axes[1].bar(x, w, width=width, color=colors[m_idx % len(colors)])
        axes[2].bar(x, s, width=width, color=colors[m_idx % len(colors)])

    axes[0].axhline(1.0, color="0.2", lw=1.0, ls="--")
    axes[0].set_ylabel("η_plate [-]")
    axes[0].set_title("Benchmark matrix overview por modelo de soporte")
    axes[1].set_ylabel("w_max [mm]")
    axes[2].set_ylabel("σ_vm,max [MPa]")
    axes[2].set_xlabel("Load case")

    for ax in axes:
        ax.set_xticks(idx)
        ax.set_xticklabels(cases, rotation=20)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(ncols=2, fontsize=8)

    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_technical_note(rows: Sequence[BenchmarkMatrixRow], path: Path) -> None:
    if not rows:
        return

    def _avg(model_name: str, attr: str) -> float:
        vals = [getattr(r, attr) for r in rows if r.model_name == model_name]
        return float(np.mean(vals)) if vals else float("nan")

    refs = {
        name: {
            "eta": _avg(name, "eta_plate"),
            "w": _avg(name, "w_max_mm"),
            "sigma": _avg(name, "sigma_vm_max_mpa"),
        }
        for name in sorted({r.model_name for r in rows})
    }

    lines = [
        "# Nota técnica — interpretación rápida",
        "",
        "Esta matrix NO mezcla físicamente modelos discretos (`fixed`, `spring_anchors`) con modelos híbridos",
        "(`foundation_patch_*`: anclajes + apoyo distribuido de compresión). Se comparan como alternativas de modelado,",
        "manteniendo iguales geometría y cargas.",
        "",
        "## Tendencias observadas (promedio sobre los 4 casos)",
        "",
    ]
    for model, stats in refs.items():
        lines.append(
            f"- **{model}** → η̄={stats['eta']:.3f}, w̄_max={stats['w']:.4f} mm, σ̄_vm,max={stats['sigma']:.1f} MPa."
        )

    lines.extend([
        "",
        "## Lectura de ingeniería",
        "",
        "- Los modelos con `foundation_patch` tienden a redistribuir mejor la compresión cuando hay contacto activo,",
        "  reduciendo picos locales frente a modelos solo discretos.",
        "- `foundation_patch_steel` (k_area más alto) suele limitar más la flecha, mientras que `timber` permite",
        "  mayor deformación y potencialmente más no linealidad de contacto (lift-off).",
        "- `fixed` es útil como cota rígida idealizada; `spring_anchors` introduce flexibilidad discreta más realista",
        "  para anclajes, pero sin capturar reacción de apoyo continuo.",
    ])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
