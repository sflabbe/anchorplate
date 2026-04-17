from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from skfem import MeshTri

from .model import AnalysisOptions, CoupledLineLoad, MeshRefinementBox, PointLoad, PointSupport, SteelPlate


def _unique_sorted(values: Iterable[float], ndigits: int = 10) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    arr = np.unique(np.round(arr, ndigits))
    arr.sort()
    return arr


def make_axis_grid(length_mm: float, target_h_mm: float, seeds_mm: Sequence[float]) -> np.ndarray:
    n = max(int(np.ceil(length_mm / target_h_mm)), 2)
    base = np.linspace(0.0, length_mm, n + 1)
    seeded = np.concatenate((base, np.asarray(seeds_mm, dtype=float), np.array([0.0, length_mm])))
    seeded = seeded[(seeded >= -1e-9) & (seeded <= length_mm + 1e-9)]
    return _unique_sorted(seeded)


def seeds_from_boxes(boxes: Sequence[MeshRefinementBox]) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for box in boxes:
        n_x = max(int(np.ceil((box.x_max_mm - box.x_min_mm) / box.h_mm)), box.n_div_min)
        n_y = max(int(np.ceil((box.y_max_mm - box.y_min_mm) / box.h_mm)), box.n_div_min)
        xs.extend(np.linspace(box.x_min_mm, box.x_max_mm, n_x + 1).tolist())
        ys.extend(np.linspace(box.y_min_mm, box.y_max_mm, n_y + 1).tolist())
    return xs, ys


def build_mesh(
    plate: SteelPlate,
    supports: Sequence[PointSupport],
    point_loads: Sequence[PointLoad],
    coupled_loads: Sequence[CoupledLineLoad],
    options: AnalysisOptions,
    refinement_boxes: Sequence[MeshRefinementBox] | None = None,
) -> MeshTri:
    x_seeds: list[float] = []
    y_seeds: list[float] = []

    for s in supports:
        x_seeds.append(s.x_mm)
        y_seeds.append(s.y_mm)

    for p in point_loads:
        x_seeds.append(p.x_mm)
        y_seeds.append(p.y_mm)

    for cl in coupled_loads:
        if cl.orientation == "vertical":
            x_seeds.extend([
                cl.ref_x_mm - 0.5 * cl.line_spacing_mm,
                cl.ref_x_mm + 0.5 * cl.line_spacing_mm,
                cl.ref_x_mm,
            ])
            y_seeds.extend([
                cl.ref_y_mm - 0.5 * cl.line_length_mm,
                cl.ref_y_mm + 0.5 * cl.line_length_mm,
                cl.ref_y_mm,
            ])
        else:
            x_seeds.extend([
                cl.ref_x_mm - 0.5 * cl.line_length_mm,
                cl.ref_x_mm + 0.5 * cl.line_length_mm,
                cl.ref_x_mm,
            ])
            y_seeds.extend([
                cl.ref_y_mm - 0.5 * cl.line_spacing_mm,
                cl.ref_y_mm + 0.5 * cl.line_spacing_mm,
                cl.ref_y_mm,
            ])

    if refinement_boxes:
        bx, by = seeds_from_boxes(refinement_boxes)
        x_seeds.extend(bx)
        y_seeds.extend(by)

    x = make_axis_grid(plate.length_mm, options.target_h_mm, x_seeds)
    y = make_axis_grid(plate.width_mm, options.target_h_mm, y_seeds)
    return MeshTri.init_tensor(x, y)


def nearest_vertex_ids(mesh: MeshTri, xy_mm: np.ndarray) -> np.ndarray:
    vertices = mesh.p.T
    out = []
    for xy in xy_mm:
        d2 = np.sum((vertices - xy[None, :]) ** 2, axis=1)
        out.append(int(np.argmin(d2)))
    return np.asarray(out, dtype=int)


def vertex_dofs_for_ids(basis, vertex_ids: np.ndarray) -> np.ndarray:
    return basis.nodal_dofs[0, vertex_ids].astype(int)


def line_vertex_ids(
    mesh: MeshTri,
    x_const: float | None,
    y_const: float | None,
    span_min: float,
    span_max: float,
    tol: float = 1e-9,
) -> np.ndarray:
    x = mesh.p[0]
    y = mesh.p[1]
    if x_const is not None:
        mask = np.isclose(x, x_const, atol=tol) & (y >= span_min - tol) & (y <= span_max + tol)
        ids = np.flatnonzero(mask)
        order = np.argsort(y[ids])
        return ids[order]
    mask = np.isclose(y, y_const, atol=tol) & (x >= span_min - tol) & (x <= span_max + tol)
    ids = np.flatnonzero(mask)
    order = np.argsort(x[ids])
    return ids[order]


def triangle_connectivity(mesh: MeshTri) -> np.ndarray:
    return mesh.t[:3, :].T.copy()


def triangle_areas(mesh: MeshTri) -> np.ndarray:
    tri = triangle_connectivity(mesh)
    p = mesh.p.T
    a = p[tri[:, 0]]
    b = p[tri[:, 1]]
    c = p[tri[:, 2]]
    return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


def nodal_tributary_areas(mesh: MeshTri) -> np.ndarray:
    tri = triangle_connectivity(mesh)
    areas = triangle_areas(mesh)
    n_vertices = mesh.p.shape[1]
    nodal = np.zeros(n_vertices, dtype=float)
    for local in range(3):
        np.add.at(nodal, tri[:, local], areas / 3.0)
    return nodal
