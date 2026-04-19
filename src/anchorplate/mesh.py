from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from skfem import MeshQuad, MeshTri

from .model import AnalysisOptions, CoupledLineLoad, LoadTransferDefinition, MeshRefinementBox, PointLoad, PointSupport, SteelPlate


MeshType = MeshTri | MeshQuad


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


def _add_segment_seeds(
    x_seeds: list[float],
    y_seeds: list[float],
    p1_mm: tuple[float, float],
    p2_mm: tuple[float, float],
    target_h_mm: float,
) -> None:
    x1, y1 = p1_mm
    x2, y2 = p2_mm
    x_seeds.extend([float(x1), float(x2)])
    y_seeds.extend([float(y1), float(y2)])

    # For oblique flanges, add paired parametric points so the tensor mesh has
    # vertices lying exactly on the arbitrary transfer segment. Axis-aligned
    # flanges are already handled by the axis grids, preserving legacy meshing.
    if abs(float(x2) - float(x1)) <= 1e-12 or abs(float(y2) - float(y1)) <= 1e-12:
        return
    length = float(np.hypot(float(x2) - float(x1), float(y2) - float(y1)))
    if length <= 1e-12:
        return
    n_div = max(int(np.ceil(length / max(float(target_h_mm), 1e-9))), 1)
    ts = np.linspace(0.0, 1.0, n_div + 1)
    x_seeds.extend((float(x1) + (float(x2) - float(x1)) * ts).tolist())
    y_seeds.extend((float(y1) + (float(y2) - float(y1)) * ts).tolist())


def build_mesh(
    plate: SteelPlate,
    supports: Sequence[PointSupport],
    point_loads: Sequence[PointLoad],
    coupled_loads: Sequence[CoupledLineLoad],
    options: AnalysisOptions,
    load_transfers: Sequence[LoadTransferDefinition] | None = None,
    refinement_boxes: Sequence[MeshRefinementBox] | None = None,
) -> MeshType:
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

    for transfer in load_transfers or []:
        x_seeds.append(transfer.ref_x_mm)
        y_seeds.append(transfer.ref_y_mm)
        for flange in transfer.flanges:
            _add_segment_seeds(x_seeds, y_seeds, flange.p1_mm, flange.p2_mm, options.target_h_mm)

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
    if options.mesh_backend == "quad_bfs":
        return MeshQuad.init_tensor(x, y)
    return MeshTri.init_tensor(x, y)


def nearest_vertex_ids(mesh: MeshType, xy_mm: np.ndarray) -> np.ndarray:
    vertices = mesh.p.T
    out = []
    for xy in xy_mm:
        d2 = np.sum((vertices - xy[None, :]) ** 2, axis=1)
        out.append(int(np.argmin(d2)))
    return np.asarray(out, dtype=int)


def vertex_dofs_for_ids(basis, vertex_ids: np.ndarray) -> np.ndarray:
    return basis.nodal_dofs[0, vertex_ids].astype(int)


def line_vertex_ids(
    mesh: MeshType,
    x_const: float | None = None,
    y_const: float | None = None,
    span_min: float | None = None,
    span_max: float | None = None,
    p1_mm: tuple[float, float] | None = None,
    p2_mm: tuple[float, float] | None = None,
    tol: float = 1e-9,
) -> np.ndarray:
    """
    Return mesh vertices on a line segment, ordered from p1 to p2.

    `tol` is applied to the normal distance from each vertex to the segment.
    The scalar projection parameter is also bounded by
    `[-tol / segment_length, 1 + tol / segment_length]`. The legacy
    x-constant/y-constant calling form is retained for existing callers and is
    internally converted to an equivalent segment.
    """

    if p1_mm is None or p2_mm is None:
        if span_min is None or span_max is None:
            raise ValueError("line_vertex_ids requires p1_mm/p2_mm or span_min/span_max")
        lo = float(min(span_min, span_max))
        hi = float(max(span_min, span_max))
        if x_const is not None:
            p1_mm = (float(x_const), lo)
            p2_mm = (float(x_const), hi)
        elif y_const is not None:
            p1_mm = (lo, float(y_const))
            p2_mm = (hi, float(y_const))
        else:
            raise ValueError("line_vertex_ids legacy form requires x_const or y_const")

    p1 = np.asarray(p1_mm, dtype=float)
    p2 = np.asarray(p2_mm, dtype=float)
    v = p2 - p1
    length = float(np.linalg.norm(v))
    if length <= 0.0:
        raise ValueError("line segment length must be positive")

    vertices = mesh.p.T
    rel = vertices - p1[None, :]
    t = (rel @ v) / (length * length)
    closest = p1[None, :] + t[:, None] * v[None, :]
    normal_dist = np.linalg.norm(vertices - closest, axis=1)
    tol_s = float(tol) / length
    mask = (normal_dist <= float(tol)) & (t >= -tol_s) & (t <= 1.0 + tol_s)
    ids = np.flatnonzero(mask)
    order = np.argsort(t[ids], kind="mergesort")
    return ids[order]


def element_connectivity(mesh: MeshType) -> np.ndarray:
    return mesh.t.T.copy()


def triangle_connectivity(mesh: MeshType) -> np.ndarray:
    conn = element_connectivity(mesh)
    if conn.shape[1] == 3:
        return conn
    if conn.shape[1] == 4:
        tri1 = conn[:, [0, 1, 2]]
        tri2 = conn[:, [0, 2, 3]]
        return np.vstack([tri1, tri2])
    raise ValueError(f"Unsupported element vertex count for triangulation: {conn.shape[1]}")


def element_areas(mesh: MeshType) -> np.ndarray:
    conn = element_connectivity(mesh)
    p = mesh.p.T
    poly = p[conn]
    x = poly[:, :, 0]
    y = poly[:, :, 1]
    x_next = np.roll(x, -1, axis=1)
    y_next = np.roll(y, -1, axis=1)
    return 0.5 * np.abs(np.sum(x * y_next - y * x_next, axis=1))


def nodal_tributary_areas(mesh: MeshType) -> np.ndarray:
    conn = element_connectivity(mesh)
    areas = element_areas(mesh)
    n_vertices = mesh.p.shape[1]
    nodal = np.zeros(n_vertices, dtype=float)
    n_local = conn.shape[1]
    for local in range(n_local):
        np.add.at(nodal, conn[:, local], areas / n_local)
    return nodal


def triangle_areas(mesh: MeshType) -> np.ndarray:
    """Backward-compatible alias for triangular element area calculation.

    Historically examples imported ``triangle_areas`` from this module. The
    mesh internals were generalized to mixed element topologies and renamed to
    :func:`element_areas`; keep this alias to avoid breaking existing scripts.
    """
    return element_areas(mesh)
