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


def _clip_to_domain(value: float, length_mm: float) -> float:
    return float(np.clip(value, 0.0, length_mm))


def _sanitize_box_axis(
    min_mm: float,
    max_mm: float,
    h_mm: float,
    n_div_min: int,
    length_mm: float,
) -> tuple[float, float, float] | None:
    lo = _clip_to_domain(min(min_mm, max_mm), length_mm)
    hi = _clip_to_domain(max(min_mm, max_mm), length_mm)
    span = hi - lo
    if span <= 1e-12:
        return None
    n_div = max(int(n_div_min), 1)
    h_by_div = span / n_div
    h_eff = max(min(float(h_mm), h_by_div), 1e-9)
    return lo, hi, h_eff


def _axis_refinement_specs(
    boxes: Sequence[MeshRefinementBox],
    axis: str,
    length_mm: float,
) -> list[tuple[float, float, float]]:
    specs: list[tuple[float, float, float]] = []
    for box in boxes:
        if axis == "x":
            spec = _sanitize_box_axis(box.x_min_mm, box.x_max_mm, box.h_mm, box.n_div_min, length_mm)
        else:
            spec = _sanitize_box_axis(box.y_min_mm, box.y_max_mm, box.h_mm, box.n_div_min, length_mm)
        if spec is not None:
            specs.append(spec)
    specs.sort(key=lambda it: (it[0], it[1], it[2]))
    return specs


def _segment_target_h(a: float, b: float, global_h: float, specs: Sequence[tuple[float, float, float]]) -> float:
    h_target = float(global_h)
    for lo, hi, h_local in specs:
        overlaps = (a < hi - 1e-12) and (b > lo + 1e-12)
        if overlaps:
            h_target = min(h_target, h_local)
    return max(h_target, 1e-9)


def _points_for_segment(a: float, b: float, target_h: float) -> np.ndarray:
    span = max(b - a, 0.0)
    n_div = max(int(np.ceil(span / target_h)), 1)
    return np.linspace(a, b, n_div + 1)


def make_axis_grid(
    length_mm: float,
    target_h_mm: float,
    seeds_mm: Sequence[float],
    refinement_specs: Sequence[tuple[float, float, float]] | None = None,
) -> np.ndarray:
    specs = list(refinement_specs or [])
    breakpoints = [0.0, length_mm]
    for v in seeds_mm:
        if -1e-9 <= v <= length_mm + 1e-9:
            breakpoints.append(_clip_to_domain(float(v), length_mm))
    for lo, hi, _ in specs:
        breakpoints.extend([lo, hi])

    bp = _unique_sorted(breakpoints)
    out: list[float] = [float(bp[0])]
    for i in range(len(bp) - 1):
        a = float(bp[i])
        b = float(bp[i + 1])
        if b - a <= 1e-12:
            continue
        seg_h = _segment_target_h(a, b, target_h_mm, specs)
        seg = _points_for_segment(a, b, seg_h)
        out.extend(seg[1:].tolist())
    return _unique_sorted(out)


def seeds_from_boxes(boxes: Sequence[MeshRefinementBox]) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for box in boxes:
        xs.extend([box.x_min_mm, box.x_max_mm])
        ys.extend([box.y_min_mm, box.y_max_mm])
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

    x_specs: list[tuple[float, float, float]] = []
    y_specs: list[tuple[float, float, float]] = []
    if refinement_boxes:
        bx, by = seeds_from_boxes(refinement_boxes)
        x_seeds.extend(bx)
        y_seeds.extend(by)
        x_specs = _axis_refinement_specs(refinement_boxes, axis="x", length_mm=plate.length_mm)
        y_specs = _axis_refinement_specs(refinement_boxes, axis="y", length_mm=plate.width_mm)

    x = make_axis_grid(plate.length_mm, options.target_h_mm, x_seeds, refinement_specs=x_specs)
    y = make_axis_grid(plate.width_mm, options.target_h_mm, y_seeds, refinement_specs=y_specs)
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
