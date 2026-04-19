from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .mesh import line_vertex_ids, nearest_vertex_ids, vertex_dofs_for_ids
from .model import AnalysisOptions, CoupledLineLoad, FlangeTransferLine, LoadTransferDefinition, PointLoad


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
    try:
        lam = np.linalg.solve(awat, b)
    except np.linalg.LinAlgError:
        lam, *_ = np.linalg.lstsq(awat, b, rcond=None)
    forces = weights * (a.T @ lam)
    residual = a @ forces - b
    scale = max(float(np.linalg.norm(b, ord=np.inf)), 1.0)
    if float(np.linalg.norm(residual, ord=np.inf)) > 1e-8 * scale:
        raise RuntimeError(
            "Load transfer resultant is not compatible with the selected flange nodes. "
            "Add flanges, move the reference point, or reduce the requested moments."
        )
    return forces


def coupled_line_load_to_transfer(load: CoupledLineLoad) -> LoadTransferDefinition:
    if load.orientation == "vertical":
        x_left = load.ref_x_mm - 0.5 * load.line_spacing_mm
        x_right = load.ref_x_mm + 0.5 * load.line_spacing_mm
        y0 = load.ref_y_mm - 0.5 * load.line_length_mm
        y1 = load.ref_y_mm + 0.5 * load.line_length_mm
        flanges = (
            FlangeTransferLine(p1_mm=(x_left, y0), p2_mm=(x_left, y1), weight_scale=1.0, label="legacy_1"),
            FlangeTransferLine(p1_mm=(x_right, y0), p2_mm=(x_right, y1), weight_scale=1.0, label="legacy_2"),
        )
    elif load.orientation == "horizontal":
        y_bottom = load.ref_y_mm - 0.5 * load.line_spacing_mm
        y_top = load.ref_y_mm + 0.5 * load.line_spacing_mm
        x0 = load.ref_x_mm - 0.5 * load.line_length_mm
        x1 = load.ref_x_mm + 0.5 * load.line_length_mm
        flanges = (
            FlangeTransferLine(p1_mm=(x0, y_bottom), p2_mm=(x1, y_bottom), weight_scale=1.0, label="legacy_1"),
            FlangeTransferLine(p1_mm=(x0, y_top), p2_mm=(x1, y_top), weight_scale=1.0, label="legacy_2"),
        )
    else:
        raise ValueError(f"Unsupported coupled line orientation: {load.orientation}")

    return LoadTransferDefinition(
        ref_x_mm=load.ref_x_mm,
        ref_y_mm=load.ref_y_mm,
        force_n=load.force_n,
        mx_nmm=load.mx_nmm,
        my_nmm=load.my_nmm,
        flanges=flanges,
        label=load.label,
    )


def _segment_length(p1_mm: tuple[float, float], p2_mm: tuple[float, float]) -> float:
    return float(np.hypot(float(p2_mm[0]) - float(p1_mm[0]), float(p2_mm[1]) - float(p1_mm[1])))


def _flange_tributary_lengths(mesh, vertex_ids: np.ndarray, flange: FlangeTransferLine) -> np.ndarray:
    p1 = np.asarray(flange.p1_mm, dtype=float)
    p2 = np.asarray(flange.p2_mm, dtype=float)
    v = p2 - p1
    length = float(np.linalg.norm(v))
    coords = mesh.p[:, vertex_ids].T
    t = ((coords - p1[None, :]) @ v) / (length * length)
    return trapezoidal_tributary_lengths(t * length)


def add_flange_group_load(
    mesh,
    basis,
    rhs: np.ndarray,
    load_transfer: LoadTransferDefinition,
    options: AnalysisOptions,
    description_prefix: str = "load transfer",
) -> LoadTransferRecord:
    """
    Apply one rigid load transfer through one or more arbitrary flange segments.

    Nodes are picked on each segment with `options.line_pick_tol_mm`; nodal
    weights are tributary lengths multiplied by each flange's `weight_scale`.
    A single minimum-norm solve over the union of all flange nodes enforces the
    transfer resultant at the reference point.
    """

    if not load_transfer.flanges:
        raise ValueError("A load transfer requires at least one flange")

    selected_ids: list[np.ndarray] = []
    for idx, flange in enumerate(load_transfer.flanges, start=1):
        if flange.weight_scale <= 0.0:
            raise ValueError(f"load transfer flange #{idx}: weight_scale must be > 0")
        if _segment_length(flange.p1_mm, flange.p2_mm) <= max(options.line_pick_tol_mm, 1e-12):
            raise ValueError(f"load transfer flange #{idx}: segment length must be greater than line_pick_tol_mm")
        ids = line_vertex_ids(
            mesh,
            p1_mm=flange.p1_mm,
            p2_mm=flange.p2_mm,
            tol=options.line_pick_tol_mm,
        )
        if ids.size == 0:
            raise RuntimeError(
                "No mesh vertices found on a load-transfer flange. Reduce target_h_mm or refine locally."
            )
        selected_ids.append(ids.astype(int))

    for i in range(len(selected_ids)):
        ids_i = set(int(v) for v in selected_ids[i])
        for j in range(i + 1, len(selected_ids)):
            if ids_i.intersection(int(v) for v in selected_ids[j]):
                raise ValueError("Flanges in the same load transfer overlap or have common mesh nodes")

    vertex_ids = np.unique(np.concatenate(selected_ids)).astype(int)
    coords = mesh.p[:, vertex_ids].T.copy()
    weights = np.zeros(vertex_ids.size, dtype=float)
    for flange, ids in zip(load_transfer.flanges, selected_ids, strict=True):
        local_w = _flange_tributary_lengths(mesh, ids, flange) * float(flange.weight_scale)
        pos = np.searchsorted(vertex_ids, ids)
        weights[pos] += local_w

    nodal_forces = _minimum_norm_force_distribution(
        coords_mm=coords,
        weights=weights,
        ref_x_mm=load_transfer.ref_x_mm,
        ref_y_mm=load_transfer.ref_y_mm,
        force_n=load_transfer.force_n,
        mx_nmm=load_transfer.mx_nmm,
        my_nmm=load_transfer.my_nmm,
    )
    dofs = vertex_dofs_for_ids(basis, vertex_ids)
    rhs[dofs] += nodal_forces

    return LoadTransferRecord(
        vertex_ids=vertex_ids,
        dofs=dofs,
        nodal_forces_n=nodal_forces,
        coords_mm=coords,
        description=(
            f"{description_prefix} {load_transfer.label or ''} "
            f"(Fz={load_transfer.force_n/1000:.1f} kN, "
            f"Mx={load_transfer.mx_nmm/1e6:.2f} kNm, "
            f"My={load_transfer.my_nmm/1e6:.2f} kNm)"
        ).strip(),
    )


def add_coupled_line_loads(
    mesh,
    basis,
    rhs: np.ndarray,
    coupled_loads: Sequence[CoupledLineLoad],
    options: AnalysisOptions,
) -> list[LoadTransferRecord]:
    records: list[LoadTransferRecord] = []

    for load in coupled_loads:
        transfer = coupled_line_load_to_transfer(load)
        records.append(
            add_flange_group_load(
                mesh,
                basis,
                rhs,
                transfer,
                options,
                description_prefix="coupled dual-line load",
            )
        )
    return records
