from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.sparse import csr_matrix, diags
from skfem import Basis, BilinearForm, ElementTriMorley, condense, solve
from skfem.helpers import dd, ddot, eye, trace

from .loading import LoadTransferRecord, add_coupled_line_loads, add_point_loads
from .mesh import build_mesh, nearest_vertex_ids, nodal_tributary_areas, triangle_connectivity, vertex_dofs_for_ids
from .model import AnalysisOptions, AnchorSpringState, CoupledLineLoad, FoundationPatch, FoundationState, MeshRefinementBox, PointLoad, PointSupport, SteelPlate
from .postprocess import recover_moments_and_stress, vertex_deflections


@dataclass
class Result:
    name: str
    mesh: object
    basis: object
    solution: np.ndarray
    total_stiffness: csr_matrix
    rhs: np.ndarray
    support_vertex_ids: np.ndarray
    support_dofs: np.ndarray
    support_kinds: np.ndarray
    support_labels: np.ndarray
    support_reactions_n: np.ndarray
    support_active: np.ndarray
    load_records: list[LoadTransferRecord]
    foundation_state: FoundationState
    anchor_spring_state: AnchorSpringState
    w_vertex_mm: np.ndarray
    mxx_nmm_per_mm: np.ndarray
    myy_nmm_per_mm: np.ndarray
    mxy_nmm_per_mm: np.ndarray
    von_mises_mpa: np.ndarray
    max_deflection_mm: float
    max_von_mises_mpa: float


def support_reaction_rows(result: Result) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i, (vid, dof, reaction, active, kind, label) in enumerate(
        zip(
            result.support_vertex_ids,
            result.support_dofs,
            result.support_reactions_n,
            result.support_active,
            result.support_kinds,
            result.support_labels,
            strict=True,
        ),
        start=1,
    ):
        rows.append(
            {
                "index": i,
                "label": str(label) if label else f"S{i}",
                "kind": str(kind),
                "vertex_id": int(vid),
                "dof": int(dof),
                "reaction_n": float(reaction),
                "active": bool(active),
            }
        )
    return rows


def export_support_reactions_json(result: Result, outpath: Path | str) -> Path:
    out = Path(outpath)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_name": result.name,
        "sign_convention": {
            "w_positive": "downward",
            "spring_force_relation": "R = kz * w",
            "spring_tension_only_active_if": "w > +tol",
        },
        "supports": support_reaction_rows(result),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def export_support_reactions_csv(result: Result, outpath: Path | str) -> Path:
    out = Path(outpath)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = support_reaction_rows(result)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "label", "kind", "vertex_id", "dof", "reaction_n", "active"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return out


def _plate_constitutive_tensor(tensor, e: float, nu: float):
    return e / (1.0 + nu) * (tensor + nu / (1.0 - nu) * eye(trace(tensor), 2))


@BilinearForm
def kirchhoff_bending(u, v, w):
    return w.t**3 / 12.0 * ddot(_plate_constitutive_tensor(dd(u), w.e, w.nu), dd(v))


def assemble_plate_stiffness(basis: Basis, plate: SteelPlate) -> csr_matrix:
    return kirchhoff_bending.assemble(
        basis,
        e=plate.youngs_modulus_mpa,
        nu=plate.poisson,
        t=plate.thickness_mm,
    ).tocsr()


def build_support_data(
    mesh,
    basis,
    supports: Sequence[PointSupport],
    system_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not supports:
        raise RuntimeError("At least one support is required")
    xy = np.array([[s.x_mm, s.y_mm] for s in supports], dtype=float)
    vids = nearest_vertex_ids(mesh, xy)
    dofs = vertex_dofs_for_ids(basis, vids)

    support_kz = np.zeros(len(supports), dtype=float)
    support_is_linear_spring = np.zeros(len(supports), dtype=bool)
    support_is_tension_only = np.zeros(len(supports), dtype=bool)
    fixed_dofs: list[int] = []
    for idx, (support, dof) in enumerate(zip(supports, dofs, strict=True)):
        if support.kind == "fixed":
            fixed_dofs.append(int(dof))
        elif support.kind == "spring":
            support_kz[idx] = float(support.kz_n_per_mm)
            support_is_linear_spring[idx] = True
        elif support.kind == "spring_tension_only":
            support_kz[idx] = float(support.kz_n_per_mm)
            support_is_tension_only[idx] = True
        else:
            raise ValueError(f"Unsupported support kind: {support.kind}")
    spring_diag_linear = np.zeros(system_size, dtype=float)
    if np.any(support_is_linear_spring):
        np.add.at(spring_diag_linear, dofs[support_is_linear_spring], support_kz[support_is_linear_spring])

    return (
        vids,
        dofs,
        support_kz,
        support_is_linear_spring,
        support_is_tension_only,
        diags(spring_diag_linear, format="csr"),
        np.asarray(fixed_dofs, dtype=int),
    )


def assemble_tension_only_spring_matrix(
    system_size: int,
    support_dofs: np.ndarray,
    support_kz: np.ndarray,
    support_is_tension_only: np.ndarray,
    active_tension_only: np.ndarray,
) -> csr_matrix:
    diag = np.zeros(system_size, dtype=float)
    mask = support_is_tension_only & active_tension_only
    if np.any(mask):
        np.add.at(diag, support_dofs[mask], support_kz[mask])
    return diags(diag, format="csr")


def _patch_vertex_ids(mesh, patch: FoundationPatch) -> set[int]:
    x = mesh.p[0]
    y = mesh.p[1]
    mask = (
        (x >= patch.x_min_mm - 1e-9)
        & (x <= patch.x_max_mm + 1e-9)
        & (y >= patch.y_min_mm - 1e-9)
        & (y <= patch.y_max_mm + 1e-9)
    )
    return set(np.flatnonzero(mask).astype(int).tolist())


def build_foundation_data(mesh, basis, foundation_patches: Sequence[FoundationPatch], system_size: int) -> tuple[list[set[int]], np.ndarray, np.ndarray]:
    n_vertices = mesh.p.shape[1]
    nodal_k = np.zeros(n_vertices, dtype=float)
    vertex_sets: list[set[int]] = []
    tributary = nodal_tributary_areas(mesh)
    nodal_w_dofs = basis.nodal_dofs[0, :n_vertices].astype(int)
    foundation_dofs = nodal_w_dofs.copy()

    for patch in foundation_patches:
        ids = _patch_vertex_ids(mesh, patch)
        vertex_sets.append(ids)
        if not ids:
            continue
        ids_arr = np.array(sorted(ids), dtype=int)
        nodal_k[ids_arr] += patch.k_area_n_per_mm3 * tributary[ids_arr]
    return vertex_sets, foundation_dofs, nodal_k


def assemble_foundation_matrix(system_size: int, foundation_dofs: np.ndarray, nodal_k: np.ndarray, active_vertices: list[set[int]]) -> csr_matrix:
    diag = np.zeros(system_size, dtype=float)
    active_union = set().union(*active_vertices) if active_vertices else set()
    if active_union:
        ids = np.array(sorted(active_union), dtype=int)
        diag[foundation_dofs[ids]] = nodal_k[ids]
    return diags(diag, format="csr")


def iterate_foundation_contact(
    mesh,
    basis,
    foundation_patches: Sequence[FoundationPatch],
    nodal_k: np.ndarray,
    foundation_dofs: np.ndarray,
    k_base: csr_matrix,
    rhs: np.ndarray,
    fixed_dofs: np.ndarray,
    options: AnalysisOptions,
) -> tuple[np.ndarray, FoundationState, csr_matrix]:
    if not foundation_patches:
        k_total = k_base.tocsr()
        solution = solve(*condense(k_total, rhs, D=fixed_dofs)) if fixed_dofs.size > 0 else solve(k_total, rhs)
        return solution, FoundationState(active_vertices=[], history_changes=[]), k_total

    # Pre-compute the full vertex set for every patch once — needed for inactive mask.
    all_patch_vertices: list[set[int]] = [_patch_vertex_ids(mesh, p) for p in foundation_patches]

    active_vertices = [set(ids) for ids in all_patch_vertices]

    history: list[int] = []
    k_total = k_base.tocsr()
    solution = np.zeros(k_base.shape[0], dtype=float)
    n_vertices = mesh.p.shape[1]
    nodal_w_dofs = basis.nodal_dofs[0, :n_vertices].astype(int)

    for _ in range(options.foundation_iterations_max):
        k_found = assemble_foundation_matrix(k_base.shape[0], foundation_dofs, nodal_k, active_vertices)
        k_total = (k_base + k_found).tocsr()
        solution = solve(*condense(k_total, rhs, D=fixed_dofs)) if fixed_dofs.size > 0 else solve(k_total, rhs)
        wv = solution[nodal_w_dofs]

        changes = 0
        next_active: list[set[int]] = []
        for patch, ids in zip(foundation_patches, active_vertices, strict=True):
            if not patch.compression_only:
                next_active.append(set(ids))
                continue
            patch_ids = _patch_vertex_ids(mesh, patch)
            new_ids = {vid for vid in patch_ids if wv[vid] > options.foundation_contact_tol_mm}
            changes += len(new_ids.symmetric_difference(ids))
            if not new_ids and patch_ids:
                candidate = max(patch_ids, key=lambda vid: wv[vid])
                new_ids = {candidate}
            next_active.append(new_ids)
        history.append(changes)
        active_vertices = next_active
        if changes == 0:
            break
    else:
        raise RuntimeError("Foundation contact iteration did not converge")

    return solution, FoundationState(active_vertices=active_vertices, history_changes=history, all_patch_vertices=all_patch_vertices), k_total


def iterate_contact_with_tension_only_springs(
    mesh,
    basis,
    foundation_patches: Sequence[FoundationPatch],
    nodal_k: np.ndarray,
    foundation_dofs: np.ndarray,
    k_base: csr_matrix,
    rhs: np.ndarray,
    fixed_dofs: np.ndarray,
    options: AnalysisOptions,
    support_dofs: np.ndarray,
    support_kz: np.ndarray,
    support_is_tension_only: np.ndarray,
) -> tuple[np.ndarray, FoundationState, AnchorSpringState, csr_matrix, csr_matrix]:
    n_supports = support_dofs.size
    n_vertices = mesh.p.shape[1]
    nodal_w_dofs = basis.nodal_dofs[0, :n_vertices].astype(int)

    has_foundation = bool(foundation_patches)
    has_tension_only = bool(np.any(support_is_tension_only))

    if not has_foundation and not has_tension_only:
        k_total = k_base.tocsr()
        solution = solve(*condense(k_total, rhs, D=fixed_dofs)) if fixed_dofs.size > 0 else solve(k_total, rhs)
        return (
            solution,
            FoundationState(active_vertices=[], history_changes=[]),
            AnchorSpringState(active=np.ones(n_supports, dtype=bool), history_changes=[], tension_only_indices=np.zeros(0, dtype=int)),
            k_total,
            diags(np.zeros(k_base.shape[0], dtype=float), format="csr"),
        )

    all_patch_vertices: list[set[int]] = [_patch_vertex_ids(mesh, p) for p in foundation_patches]
    active_foundation = [set(ids) for ids in all_patch_vertices]
    active_tension_only = np.ones(n_supports, dtype=bool)
    active_tension_only[~support_is_tension_only] = False

    foundation_history: list[int] = []
    spring_history: list[int] = []

    k_total = k_base.tocsr()
    k_tension = diags(np.zeros(k_base.shape[0], dtype=float), format="csr")
    solution = np.zeros(k_base.shape[0], dtype=float)
    seen_states: set[tuple[tuple[tuple[int, ...], ...], tuple[bool, ...]]] = set()

    for _ in range(options.foundation_iterations_max):
        foundation_key = tuple(tuple(sorted(s)) for s in active_foundation)
        spring_key = tuple(bool(v) for v in active_tension_only.tolist())
        seen_states.add((foundation_key, spring_key))

        k_found = assemble_foundation_matrix(k_base.shape[0], foundation_dofs, nodal_k, active_foundation) if has_foundation else diags(np.zeros(k_base.shape[0], dtype=float), format="csr")
        k_tension = assemble_tension_only_spring_matrix(
            system_size=k_base.shape[0],
            support_dofs=support_dofs,
            support_kz=support_kz,
            support_is_tension_only=support_is_tension_only,
            active_tension_only=active_tension_only,
        )
        k_total = (k_base + k_found + k_tension).tocsr()
        solution = solve(*condense(k_total, rhs, D=fixed_dofs)) if fixed_dofs.size > 0 else solve(k_total, rhs)
        wv = solution[nodal_w_dofs]

        foundation_changes = 0
        next_foundation: list[set[int]] = []
        if has_foundation:
            for patch, ids in zip(foundation_patches, active_foundation, strict=True):
                if not patch.compression_only:
                    next_foundation.append(set(ids))
                    continue
                patch_ids = _patch_vertex_ids(mesh, patch)
                new_ids = {vid for vid in patch_ids if wv[vid] > options.foundation_contact_tol_mm}
                foundation_changes += len(new_ids.symmetric_difference(ids))
                if not new_ids and patch_ids:
                    candidate = max(patch_ids, key=lambda vid: wv[vid])
                    new_ids = {candidate}
                next_foundation.append(new_ids)
        else:
            next_foundation = []

        spring_changes = 0
        next_tension_only = active_tension_only.copy()
        if has_tension_only:
            tol = max(options.foundation_contact_tol_mm, 1e-8)
            for i in np.flatnonzero(support_is_tension_only):
                w_i = solution[support_dofs[i]]
                was_active = bool(active_tension_only[i])
                if was_active and w_i < -tol:
                    is_active = False
                elif (not was_active) and w_i > tol:
                    is_active = True
                else:
                    is_active = was_active
                if is_active != bool(active_tension_only[i]):
                    spring_changes += 1
                next_tension_only[i] = is_active

        next_foundation_key = tuple(tuple(sorted(s)) for s in next_foundation)
        next_spring_key = tuple(bool(v) for v in next_tension_only.tolist())
        if (next_foundation_key, next_spring_key) in seen_states:
            foundation_history.append(foundation_changes)
            spring_history.append(spring_changes)
            break

        foundation_history.append(foundation_changes)
        spring_history.append(spring_changes)
        active_foundation = next_foundation
        active_tension_only = next_tension_only
        if foundation_changes == 0 and spring_changes == 0:
            break
    else:
        raise RuntimeError("Foundation/spring contact iteration did not converge")

    foundation_state = FoundationState(
        active_vertices=active_foundation,
        history_changes=foundation_history,
        all_patch_vertices=all_patch_vertices,
    )
    anchor_state = AnchorSpringState(
        active=active_tension_only,
        history_changes=spring_history,
        tension_only_indices=np.flatnonzero(support_is_tension_only),
    )
    return solution, foundation_state, anchor_state, k_total, k_tension


def solve_anchor_plate(
    plate: SteelPlate,
    supports: Sequence[PointSupport],
    point_loads: Sequence[PointLoad] | None = None,
    coupled_loads: Sequence[CoupledLineLoad] | None = None,
    options: AnalysisOptions | None = None,
    foundation_patches: Sequence[FoundationPatch] | None = None,
    refinement_boxes: Sequence[MeshRefinementBox] | None = None,
    name: str = "anchor_plate_case",
) -> Result:
    point_loads = list(point_loads or [])
    coupled_loads = list(coupled_loads or [])
    foundation_patches = list(foundation_patches or [])
    options = options or AnalysisOptions()

    mesh = build_mesh(plate, supports, point_loads, coupled_loads, options, refinement_boxes=refinement_boxes)
    basis = Basis(mesh, ElementTriMorley())
    k_plate = assemble_plate_stiffness(basis, plate)

    rhs = np.zeros(k_plate.shape[0], dtype=float)
    load_records: list[LoadTransferRecord] = []
    load_records.extend(add_point_loads(mesh, basis, rhs, point_loads))
    load_records.extend(add_coupled_line_loads(mesh, basis, rhs, coupled_loads, options))

    (
        support_vertex_ids,
        support_dofs,
        support_kz,
        support_is_linear_spring,
        support_is_tension_only,
        k_springs_linear,
        fixed_dofs,
    ) = build_support_data(mesh, basis, supports, k_plate.shape[0])
    k_base = (k_plate + k_springs_linear).tocsr()

    foundation_dofs = np.zeros(0, dtype=int)
    nodal_k = np.zeros(mesh.p.shape[1], dtype=float)
    if foundation_patches:
        _, foundation_dofs, nodal_k = build_foundation_data(mesh, basis, foundation_patches, k_plate.shape[0])

    solution, foundation_state, anchor_spring_state, k_total, k_springs_tension = iterate_contact_with_tension_only_springs(
        mesh=mesh,
        basis=basis,
        foundation_patches=foundation_patches,
        nodal_k=nodal_k,
        foundation_dofs=foundation_dofs,
        k_base=k_base,
        rhs=rhs,
        fixed_dofs=fixed_dofs,
        options=options,
        support_dofs=support_dofs,
        support_kz=support_kz,
        support_is_tension_only=support_is_tension_only,
    )

    # Reaction extraction:
    # - Fixed DOFs: condensed out of the solve → K·u ≠ f at those DOFs → residual gives the reaction.
    # - Spring DOFs: free DOFs, K·u = f exactly → residual = 0 there → MUST use R = k·u directly.
    # The two contributions are non-overlapping (spring_diag is 0 at fixed DOFs).
    residual = k_total @ solution - rhs
    spring_reactions = ((k_springs_linear + k_springs_tension) @ solution)[support_dofs]   # R = kz · w  (N), zero for fixed dofs
    fixed_reactions  = -residual[support_dofs]                # correct for fixed, zero for springs
    support_reactions = spring_reactions + fixed_reactions
    support_active = np.ones(len(supports), dtype=bool)
    spring_active_tol_n = 1e-6
    support_active[support_is_tension_only] = np.abs(spring_reactions[support_is_tension_only]) > spring_active_tol_n
    anchor_spring_state.active = support_active.copy()

    wv = vertex_deflections(mesh, basis, solution)
    mxx, myy, mxy, sigma_vm = recover_moments_and_stress(mesh, wv, plate)

    return Result(
        name=name,
        mesh=mesh,
        basis=basis,
        solution=solution,
        total_stiffness=k_total,
        rhs=rhs,
        support_vertex_ids=support_vertex_ids,
        support_dofs=support_dofs,
        support_kinds=np.array([s.kind for s in supports], dtype=object),
        support_labels=np.array([s.label for s in supports], dtype=object),
        support_reactions_n=support_reactions,
        support_active=support_active,
        load_records=load_records,
        foundation_state=foundation_state,
        anchor_spring_state=anchor_spring_state,
        w_vertex_mm=wv,
        mxx_nmm_per_mm=mxx,
        myy_nmm_per_mm=myy,
        mxy_nmm_per_mm=mxy,
        von_mises_mpa=sigma_vm,
        max_deflection_mm=float(np.max(wv)),
        max_von_mises_mpa=float(np.max(sigma_vm)),
    )
