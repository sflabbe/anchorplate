from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.sparse import csr_matrix, diags
from skfem import Basis, BilinearForm, ElementTriMorley, condense, solve
from skfem.helpers import dd, ddot, eye, trace

from .loading import LoadTransferRecord, add_coupled_line_loads, add_point_loads
from .mesh import build_mesh, nearest_vertex_ids, nodal_tributary_areas, triangle_connectivity, vertex_dofs_for_ids
from .model import AnalysisOptions, CoupledLineLoad, FoundationPatch, FoundationState, MeshRefinementBox, PointLoad, PointSupport, SteelPlate
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
    support_reactions_n: np.ndarray
    load_records: list[LoadTransferRecord]
    foundation_state: FoundationState
    w_vertex_mm: np.ndarray
    mxx_nmm_per_mm: np.ndarray
    myy_nmm_per_mm: np.ndarray
    mxy_nmm_per_mm: np.ndarray
    von_mises_mpa: np.ndarray
    max_deflection_mm: float
    max_von_mises_mpa: float


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


def build_support_data(mesh, basis, supports: Sequence[PointSupport], system_size: int) -> tuple[np.ndarray, np.ndarray, csr_matrix, np.ndarray]:
    if not supports:
        raise RuntimeError("At least one support is required")
    xy = np.array([[s.x_mm, s.y_mm] for s in supports], dtype=float)
    vids = nearest_vertex_ids(mesh, xy)
    dofs = vertex_dofs_for_ids(basis, vids)

    spring_diag = np.zeros(system_size, dtype=float)
    fixed_dofs: list[int] = []
    for support, dof in zip(supports, dofs, strict=True):
        if support.kind == "fixed":
            fixed_dofs.append(int(dof))
        elif support.kind == "spring":
            spring_diag[int(dof)] += float(support.kz_n_per_mm)
        else:
            raise ValueError(f"Unsupported support kind: {support.kind}")
    return vids, dofs, diags(spring_diag, format="csr"), np.asarray(fixed_dofs, dtype=int)


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

    support_vertex_ids, support_dofs, k_springs, fixed_dofs = build_support_data(mesh, basis, supports, k_plate.shape[0])
    k_base = (k_plate + k_springs).tocsr()

    if foundation_patches:
        _, foundation_dofs, nodal_k = build_foundation_data(mesh, basis, foundation_patches, k_plate.shape[0])
        solution, foundation_state, k_total = iterate_foundation_contact(
            mesh=mesh,
            basis=basis,
            foundation_patches=foundation_patches,
            nodal_k=nodal_k,
            foundation_dofs=foundation_dofs,
            k_base=k_base,
            rhs=rhs,
            fixed_dofs=fixed_dofs,
            options=options,
        )
    else:
        k_total = k_base
        solution = solve(*condense(k_total, rhs, D=fixed_dofs)) if fixed_dofs.size > 0 else solve(k_total, rhs)
        foundation_state = FoundationState(active_vertices=[], history_changes=[])

    # Reaction extraction:
    # - Fixed DOFs: condensed out of the solve → K·u ≠ f at those DOFs → residual gives the reaction.
    # - Spring DOFs: free DOFs, K·u = f exactly → residual = 0 there → MUST use R = k·u directly.
    # The two contributions are non-overlapping (spring_diag is 0 at fixed DOFs).
    residual = k_total @ solution - rhs
    spring_reactions = (k_springs @ solution)[support_dofs]   # R = kz · w  (N), zero for fixed dofs
    fixed_reactions  = -residual[support_dofs]                # correct for fixed, zero for springs
    support_reactions = spring_reactions + fixed_reactions

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
        support_reactions_n=support_reactions,
        load_records=load_records,
        foundation_state=foundation_state,
        w_vertex_mm=wv,
        mxx_nmm_per_mm=mxx,
        myy_nmm_per_mm=myy,
        mxy_nmm_per_mm=mxy,
        von_mises_mpa=sigma_vm,
        max_deflection_mm=float(np.max(wv)),
        max_von_mises_mpa=float(np.max(sigma_vm)),
    )
