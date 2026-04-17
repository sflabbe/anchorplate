from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .benchmark_material import _foundation_total_reaction
from .model import AnalysisOptions, CoupledLineLoad, FoundationPatch, PointSupport, SteelPlate
from .plotting import _contact_summary, _foundation_masks, export_result_npz, plot_result, plot_result_3d
from .solver import solve_anchor_plate


@dataclass(frozen=True)
class AnchorDominantVariant:
    key: str
    description: str
    supports: tuple[PointSupport, ...]
    foundation_patches: tuple[FoundationPatch, ...] = ()


@dataclass(frozen=True)
class AnchorDominantLoadCase:
    name: str
    description: str
    fz_n: float
    mx_nmm: float = 0.0
    my_nmm: float = 0.0


@dataclass
class AnchorDominantRow:
    variant: str
    variant_description: str
    load_case: str
    load_description: str
    fz_kN: float
    mx_kNm: float
    my_kNm: float
    w_max_mm: float
    sigma_vm_max_mpa: float
    rz_a1_kN: float
    rz_a2_kN: float
    rz_a3_kN: float
    rz_a4_kN: float
    sum_anchor_reactions_kN: float
    sum_foundation_reactions_kN: float
    sum_total_reactions_kN: float
    foundation_share_pct: float
    contact_active_pct: float | None
    contact_iterations: int | None


def default_anchor_dominant_variants() -> list[AnchorDominantVariant]:
    supports = (
        PointSupport(30.0, 30.0, kind="spring", kz_n_per_mm=150_000.0, label="A1"),
        PointSupport(270.0, 30.0, kind="spring", kz_n_per_mm=150_000.0, label="A2"),
        PointSupport(30.0, 270.0, kind="spring", kz_n_per_mm=150_000.0, label="A3"),
        PointSupport(270.0, 270.0, kind="spring", kz_n_per_mm=150_000.0, label="A4"),
    )

    return [
        AnchorDominantVariant(
            key="anchor_dominant_no_patch",
            description="Solo anclajes elásticos discretos; sin foundation patch.",
            supports=supports,
        ),
        AnchorDominantVariant(
            key="anchor_dominant_small_or_soft_patch",
            description=(
                "Anclajes elásticos + patch pequeño y blando (núcleo central 120×120 mm, "
                "k=1.5 N/mm³, compresión-only)."
            ),
            supports=supports,
            foundation_patches=(
                FoundationPatch(
                    x_min_mm=90.0,
                    x_max_mm=210.0,
                    y_min_mm=90.0,
                    y_max_mm=210.0,
                    k_area_n_per_mm3=1.5,
                    compression_only=True,
                    label="soft-core",
                ),
            ),
        ),
    ]


def default_anchor_dominant_loads() -> list[AnchorDominantLoadCase]:
    return [
        AnchorDominantLoadCase(
            name="LC01_Fz_plus_Mx",
            description="Fz + Mx",
            fz_n=35_000.0,
            mx_nmm=5.5e6,
        ),
        AnchorDominantLoadCase(
            name="LC02_Fz_plus_Mx_plus_My",
            description="Fz + Mx + My",
            fz_n=35_000.0,
            mx_nmm=4.0e6,
            my_nmm=3.0e6,
        ),
    ]


def run_anchor_dominant_benchmark(
    plate: SteelPlate,
    options: AnalysisOptions,
    outdir: Path,
    variants: Sequence[AnchorDominantVariant] | None = None,
    load_cases: Sequence[AnchorDominantLoadCase] | None = None,
) -> list[AnchorDominantRow]:
    outdir.mkdir(parents=True, exist_ok=True)
    variants = list(variants or default_anchor_dominant_variants())
    load_cases = list(load_cases or default_anchor_dominant_loads())

    rows: list[AnchorDominantRow] = []

    for variant in variants:
        for load_case in load_cases:
            case_name = f"{variant.key}__{load_case.name}"
            case_outdir = outdir / case_name
            case_options = AnalysisOptions(
                **{**vars(options), "output_dir": str(case_outdir), "save_plots": True, "save_3d_plots": True}
            )
            coupled_load = CoupledLineLoad(
                ref_x_mm=150.0,
                ref_y_mm=150.0,
                force_n=load_case.fz_n,
                mx_nmm=load_case.mx_nmm,
                my_nmm=load_case.my_nmm,
                line_spacing_mm=150.0,
                line_length_mm=100.0,
                orientation="vertical",
                label=load_case.name,
            )

            result = solve_anchor_plate(
                plate=plate,
                supports=variant.supports,
                coupled_loads=[coupled_load],
                options=case_options,
                foundation_patches=variant.foundation_patches,
                name=case_name,
            )

            plot_result(plate, variant.supports, [], [coupled_load], result, case_options)
            plot_result_3d(plate, variant.supports, result, case_options)
            export_result_npz(result, case_outdir / f"{case_name}_result.npz")

            anchor_kN = np.asarray(result.support_reactions_n, dtype=float) / 1000.0
            if anchor_kN.size != 4:
                raise ValueError("Anchor-dominant benchmark expects exactly four anchors.")
            foundation_kN = 0.0
            contact_active_pct: float | None = None
            contact_iterations: int | None = None
            if variant.foundation_patches:
                foundation_kN = _foundation_total_reaction(result, variant.foundation_patches) / 1000.0
                active_mask, inactive_mask = _foundation_masks(result)
                contact = _contact_summary(result, active_mask, inactive_mask)
                contact_active_pct = float(contact["pct_active"])
                contact_iterations = int(contact["n_iterations"])

            sum_anchor = float(np.sum(anchor_kN))
            sum_total = sum_anchor + float(foundation_kN)
            foundation_share = 0.0 if abs(sum_total) < 1e-9 else 100.0 * foundation_kN / sum_total

            rows.append(
                AnchorDominantRow(
                    variant=variant.key,
                    variant_description=variant.description,
                    load_case=load_case.name,
                    load_description=load_case.description,
                    fz_kN=load_case.fz_n / 1000.0,
                    mx_kNm=load_case.mx_nmm / 1e6,
                    my_kNm=load_case.my_nmm / 1e6,
                    w_max_mm=float(result.max_deflection_mm),
                    sigma_vm_max_mpa=float(result.max_von_mises_mpa),
                    rz_a1_kN=float(anchor_kN[0]),
                    rz_a2_kN=float(anchor_kN[1]),
                    rz_a3_kN=float(anchor_kN[2]),
                    rz_a4_kN=float(anchor_kN[3]),
                    sum_anchor_reactions_kN=sum_anchor,
                    sum_foundation_reactions_kN=float(foundation_kN),
                    sum_total_reactions_kN=float(sum_total),
                    foundation_share_pct=float(foundation_share),
                    contact_active_pct=contact_active_pct,
                    contact_iterations=contact_iterations,
                )
            )

    _save_anchor_dominant_csv(rows, outdir / "anchor_dominant_summary.csv")
    _save_anchor_dominant_markdown(rows, outdir / "anchor_dominant_summary.md")
    _save_anchor_dominant_overview(rows, outdir / "anchor_dominant_overview.png")
    _save_anchor_dominant_note(outdir / "anchor_dominant_note.md")
    return rows


def _save_anchor_dominant_csv(rows: Sequence[AnchorDominantRow], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _fmt_optional(value: float | int | None, fmt: str = "{:.1f}") -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return f"{value:d}"
    return fmt.format(value)


def _save_anchor_dominant_markdown(rows: Sequence[AnchorDominantRow], path: Path) -> None:
    if not rows:
        return
    lines = [
        "# Anchor-dominant benchmark",
        "",
        "Comparación pedagógica entre modelos **dominados por anclajes discretos** y un modelo híbrido con patch pequeño/blando.",
        "",
        "| Variant | Load case | w_max [mm] | σ_vm,max [MPa] | R_A1 [kN] | R_A2 [kN] | R_A3 [kN] | R_A4 [kN] | ΣR_anchor [kN] | ΣR_found [kN] | Foundation share [%] | Contact active [%] |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.variant} | {row.load_case} | {row.w_max_mm:.4f} | {row.sigma_vm_max_mpa:.1f} | {row.rz_a1_kN:.2f} | "
            f"{row.rz_a2_kN:.2f} | {row.rz_a3_kN:.2f} | {row.rz_a4_kN:.2f} | {row.sum_anchor_reactions_kN:.2f} | "
            f"{row.sum_foundation_reactions_kN:.2f} | {row.foundation_share_pct:.2f} | {_fmt_optional(row.contact_active_pct)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_anchor_dominant_overview(rows: Sequence[AnchorDominantRow], path: Path) -> None:
    if not rows:
        return

    labels = [f"{row.variant}\n{row.load_case}" for row in rows]
    anchor = np.array([row.sum_anchor_reactions_kN for row in rows], dtype=float)
    foundation = np.array([row.sum_foundation_reactions_kN for row in rows], dtype=float)
    w_max = np.array([row.w_max_mm for row in rows], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    x = np.arange(len(rows), dtype=float)
    axes[0].bar(x, anchor, label="Anchors")
    axes[0].bar(x, foundation, bottom=anchor, label="Foundation patch")
    axes[0].set_ylabel("ΣR [kN]")
    axes[0].set_title("Reaction split — anchors vs foundation")
    axes[0].legend()

    axes[1].bar(x, w_max)
    axes[1].set_ylabel("w_max [mm]")
    axes[1].set_title("Maximum deflection")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.2)

    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_anchor_dominant_note(path: Path) -> None:
    lines = [
        "# Nota técnica — benchmark anchor-dominant",
        "",
        "## ¿Cuándo domina el foundation patch?",
        "",
        "- Cuando el área de patch activa es grande y `k_area` es alto, la cama distribuida capta gran parte de `Fz`.",
        "- En ese régimen, las reacciones por anclaje bajan y el modelo se comporta como apoyo continuo con anclajes secundarios.",
        "",
        "## ¿Cuándo gobiernan los anclajes?",
        "",
        "- Cuando no hay patch, o el patch es pequeño/blando, la ruta principal de carga es discreta.",
        "- El reparto entre A1…A4 refleja con claridad la combinación `Fz + Mx (+ My)` y el uplift asociado.",
        "",
        "## ¿Por qué importa para interpretar benchmarks?",
        "",
        "- Evita mezclar conclusiones: un benchmark patch-dominant no valida automáticamente un modelo anchor-dominant, ni al revés.",
        "- Facilita usar este set como caso pedagógico y como regression test físico: el rol de cada mecanismo de soporte queda explícito.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
