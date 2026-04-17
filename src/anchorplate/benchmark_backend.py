from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np

from .model import AnalysisOptions, CoupledLineLoad, PointSupport, SteelPlate
from .solver import solve_anchor_plate


@dataclass(frozen=True)
class BackendCase:
    name: str
    description: str
    load: CoupledLineLoad


@dataclass(frozen=True)
class BackendBenchmarkRow:
    case_name: str
    backend: Literal["tri_morley", "quad_bfs"]
    n_nodes: int
    n_elements: int
    w_max_mm: float
    sigma_vm_max_mpa: float
    sum_reactions_kN: float
    solve_time_s: float


def default_backend_cases(ref_x_mm: float = 150.0, ref_y_mm: float = 150.0) -> list[BackendCase]:
    return [
        BackendCase(
            name="Fz_only",
            description="Carga transversal pura Fz",
            load=CoupledLineLoad(
                ref_x_mm=ref_x_mm,
                ref_y_mm=ref_y_mm,
                force_n=50_000.0,
                mx_nmm=0.0,
                my_nmm=0.0,
                line_spacing_mm=150.0,
                line_length_mm=100.0,
                orientation="vertical",
                label="Fz_only",
            ),
        ),
        BackendCase(
            name="Fz_plus_Mx",
            description="Carga combinada Fz + Mx",
            load=CoupledLineLoad(
                ref_x_mm=ref_x_mm,
                ref_y_mm=ref_y_mm,
                force_n=50_000.0,
                mx_nmm=4_000_000.0,
                my_nmm=0.0,
                line_spacing_mm=150.0,
                line_length_mm=100.0,
                orientation="vertical",
                label="Fz_plus_Mx",
            ),
        ),
    ]


def run_backend_benchmark(
    plate: SteelPlate,
    supports: list[PointSupport],
    base_options: AnalysisOptions,
    cases: list[BackendCase] | None = None,
    backends: tuple[Literal["tri_morley", "quad_bfs"], ...] = ("tri_morley", "quad_bfs"),
) -> list[BackendBenchmarkRow]:
    rows: list[BackendBenchmarkRow] = []
    for case in (cases or default_backend_cases()):
        for backend in backends:
            options = AnalysisOptions(**{**vars(base_options), "mesh_backend": backend})
            t0 = perf_counter()
            result = solve_anchor_plate(
                plate=plate,
                supports=supports,
                point_loads=[],
                coupled_loads=[case.load],
                options=options,
                name=f"{case.name}_{backend}",
            )
            elapsed = perf_counter() - t0
            rows.append(
                BackendBenchmarkRow(
                    case_name=case.name,
                    backend=backend,
                    n_nodes=int(result.mesh.p.shape[1]),
                    n_elements=int(result.mesh.t.shape[1]),
                    w_max_mm=float(result.max_deflection_mm),
                    sigma_vm_max_mpa=float(result.max_von_mises_mpa),
                    sum_reactions_kN=float(np.sum(result.support_reactions_n) / 1000.0),
                    solve_time_s=float(elapsed),
                )
            )
    return rows


def backend_benchmark_markdown(rows: list[BackendBenchmarkRow]) -> str:
    lines = [
        "| Caso | Backend | Nodos | Elementos | w_max [mm] | sigma_vm_max [MPa] | ΣRz [kN] | t_solve [s] |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.case_name} | {row.backend} | {row.n_nodes} | {row.n_elements} | "
            f"{row.w_max_mm:.4f} | {row.sigma_vm_max_mpa:.2f} | {row.sum_reactions_kN:.3f} | {row.solve_time_s:.3f} |"
        )
    return "\n".join(lines)
