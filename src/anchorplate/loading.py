from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .mesh import line_vertex_ids, nearest_vertex_ids, vertex_dofs_for_ids
from .model import AnalysisOptions, CoupledLineLoad, PointLoad


@dataclass
class LoadTransferRecord:
    vertex_ids: np.ndarray
    dofs: np.ndarray
    nodal_forces_n: np.ndarray
    coords_mm: np.ndarray
    description: str


def trapezoidal_tributary_lengths(coords_1d: np.ndarray) -> np.ndarray:
    if coords_1d.size == 1:
        return np.ones(1, dtype=float)
    ds = np.diff(coords_1d)
    w = np.empty_like(coords_1d, dtype=float)
    w[0] = 0.5 * ds[0]
    w[-1] = 0.5 * ds[-1]
    if coords_1d.size > 2:
        w[1:-1] = 0.5 * (ds[:-1] + ds[1:])
    return w


def add_point_loads(mesh, basis, rhs: np.ndarray, loads: Sequence[PointLoad]) -> list[LoadTransferRecord]:
    if not loads:
        return []
    xy = np.array([[p.x_mm, p.y_mm] for p in loads], dtype=float)
    vids = nearest_vertex_ids(mesh, xy)
    dofs = vertex_dofs_for_ids(basis, vids)

    records: list[LoadTransferRecord] = []
    for load, vid, dof in zip(loads, vids, dofs, strict=True):
        rhs[dof] += load.force_n
        records.append(
            LoadTransferRecord(
                vertex_ids=np.array([vid], dtype=int),
                dofs=np.array([dof], dtype=int),
                nodal_forces_n=np.array([load.force_n], dtype=float),
                coords_mm=mesh.p[:, [vid]].T.copy(),
                description=f"point load {load.label or ''}".strip(),
            )
        )
    return records


def _minimum_norm_force_distribution(
    coords_mm: np.ndarray,
    weights: np.ndarray,
    ref_x_mm: float,
    ref_y_mm: float,
    force_n: float,
    mx_nmm: float,
    my_nmm: float,
) -> np.ndarray:
    dx = coords_mm[:, 0] - ref_x_mm
    dy = coords_mm[:, 1] - ref_y_mm

    a = np.vstack([
        np.ones(coords_mm.shape[0]),
        dy,
        -dx,
    ])
    b = np.array([force_n, mx_nmm, my_nmm], dtype=float)

    awat = a @ (weights[:, None] * a.T)
    lam = np.linalg.solve(awat, b)
    return weights * (a.T @ lam)


def add_coupled_line_loads(
    mesh,
    basis,
    rhs: np.ndarray,
    coupled_loads: Sequence[CoupledLineLoad],
    options: AnalysisOptions,
) -> list[LoadTransferRecord]:
    records: list[LoadTransferRecord] = []

    for load in coupled_loads:
        if load.orientation == "vertical":
            x1 = load.ref_x_mm - 0.5 * load.line_spacing_mm
            x2 = load.ref_x_mm + 0.5 * load.line_spacing_mm
            y0 = load.ref_y_mm - 0.5 * load.line_length_mm
            y1 = load.ref_y_mm + 0.5 * load.line_length_mm
            ids1 = line_vertex_ids(mesh, x_const=x1, y_const=None, span_min=y0, span_max=y1, tol=options.line_pick_tol_mm)
            ids2 = line_vertex_ids(mesh, x_const=x2, y_const=None, span_min=y0, span_max=y1, tol=options.line_pick_tol_mm)
            weights = np.zeros(np.unique(np.concatenate((ids1, ids2))).size, dtype=float)
        else:
            y1 = load.ref_y_mm - 0.5 * load.line_spacing_mm
            y2 = load.ref_y_mm + 0.5 * load.line_spacing_mm
            x0 = load.ref_x_mm - 0.5 * load.line_length_mm
            x1 = load.ref_x_mm + 0.5 * load.line_length_mm
            ids1 = line_vertex_ids(mesh, x_const=None, y_const=y1, span_min=x0, span_max=x1, tol=options.line_pick_tol_mm)
            ids2 = line_vertex_ids(mesh, x_const=None, y_const=y2, span_min=x0, span_max=x1, tol=options.line_pick_tol_mm)
            weights = np.zeros(np.unique(np.concatenate((ids1, ids2))).size, dtype=float)

        if ids1.size == 0 or ids2.size == 0:
            raise RuntimeError(
                "No mesh vertices found on one of the coupling lines. Reduce target_h_mm or refine locally."
            )

        vertex_ids = np.unique(np.concatenate((ids1, ids2))).astype(int)
        coords = mesh.p[:, vertex_ids].T.copy()

        if load.orientation == "vertical":
            coords_y = mesh.p[1]
            w1 = trapezoidal_tributary_lengths(coords_y[ids1])
            w2 = trapezoidal_tributary_lengths(coords_y[ids2])
            for local_ids, local_w in ((ids1, w1), (ids2, w2)):
                pos = np.searchsorted(vertex_ids, local_ids)
                weights[pos] = local_w
        else:
            coords_x = mesh.p[0]
            w1 = trapezoidal_tributary_lengths(coords_x[ids1])
            w2 = trapezoidal_tributary_lengths(coords_x[ids2])
            for local_ids, local_w in ((ids1, w1), (ids2, w2)):
                pos = np.searchsorted(vertex_ids, local_ids)
                weights[pos] = local_w

        nodal_forces = _minimum_norm_force_distribution(
            coords_mm=coords,
            weights=weights,
            ref_x_mm=load.ref_x_mm,
            ref_y_mm=load.ref_y_mm,
            force_n=load.force_n,
            mx_nmm=load.mx_nmm,
            my_nmm=load.my_nmm,
        )
        dofs = vertex_dofs_for_ids(basis, vertex_ids)
        rhs[dofs] += nodal_forces

        records.append(
            LoadTransferRecord(
                vertex_ids=vertex_ids,
                dofs=dofs,
                nodal_forces_n=nodal_forces,
                coords_mm=coords,
                description=(
                    f"coupled dual-line load {load.label or ''} "
                    f"(Fz={load.force_n/1000:.1f} kN, Mx={load.mx_nmm/1e6:.2f} kNm, My={load.my_nmm/1e6:.2f} kNm)"
                ).strip(),
            )
        )
    return records
