from __future__ import annotations

import numpy as np

from .mesh import triangle_connectivity
from .model import SteelPlate


def vertex_deflections(mesh, basis, solution: np.ndarray) -> np.ndarray:
    n_vertices = mesh.p.shape[1]
    return solution[basis.nodal_dofs[0, :n_vertices]]


def build_vertex_adjacency(mesh) -> list[set[int]]:
    tri = triangle_connectivity(mesh)
    n_vertices = mesh.p.shape[1]
    adj = [set([i]) for i in range(n_vertices)]
    for a, b, c in tri:
        adj[a].update([b, c])
        adj[b].update([a, c])
        adj[c].update([a, b])
    enriched: list[set[int]] = []
    for i in range(n_vertices):
        patch = set(adj[i])
        for j in list(adj[i]):
            patch.update(adj[j])
        enriched.append(patch)
    return enriched


def recover_curvatures_by_quadratic_patch(mesh, w_vertex_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = mesh.p.T
    n_vertices = p.shape[0]
    adjacency = build_vertex_adjacency(mesh)

    kxx = np.zeros(n_vertices, dtype=float)
    kyy = np.zeros(n_vertices, dtype=float)
    kxy = np.zeros(n_vertices, dtype=float)

    for i in range(n_vertices):
        ids = np.array(sorted(adjacency[i]), dtype=int)
        if ids.size < 6:
            d2 = np.sum((p - p[i][None, :]) ** 2, axis=1)
            ids = np.argsort(d2)[:12]

        xy = p[ids] - p[i][None, :]
        z = w_vertex_mm[ids]
        a = np.column_stack([
            np.ones(ids.size),
            xy[:, 0],
            xy[:, 1],
            xy[:, 0] ** 2,
            xy[:, 0] * xy[:, 1],
            xy[:, 1] ** 2,
        ])
        coeff, *_ = np.linalg.lstsq(a, z, rcond=None)
        kxx[i] = 2.0 * coeff[3]
        kxy[i] = coeff[4]
        kyy[i] = 2.0 * coeff[5]

    return kxx, kyy, kxy


def recover_moments_and_stress(mesh, w_vertex_mm: np.ndarray, plate: SteelPlate) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tri = triangle_connectivity(mesh)
    kv_xx, kv_yy, kv_xy = recover_curvatures_by_quadratic_patch(mesh, w_vertex_mm)

    kxx_e = kv_xx[tri].mean(axis=1)
    kyy_e = kv_yy[tri].mean(axis=1)
    kxy_e = kv_xy[tri].mean(axis=1)

    d = plate.rigidity_d_nmm
    nu = plate.poisson

    mxx = -d * (kxx_e + nu * kyy_e)
    myy = -d * (kyy_e + nu * kxx_e)
    mxy = -d * (1.0 - nu) * kxy_e

    t = plate.thickness_mm
    sigma_x = 6.0 * mxx / (t**2)
    sigma_y = 6.0 * myy / (t**2)
    tau_xy = 6.0 * mxy / (t**2)
    sigma_vm = np.sqrt(np.maximum(sigma_x**2 + sigma_y**2 - sigma_x * sigma_y + 3.0 * tau_xy**2, 0.0))
    return mxx, myy, mxy, sigma_vm


def nodal_average_from_element_field(mesh, elem_values: np.ndarray) -> np.ndarray:
    tri = triangle_connectivity(mesh)
    n_vertices = mesh.p.shape[1]
    out = np.zeros(n_vertices, dtype=float)
    cnt = np.zeros(n_vertices, dtype=float)
    for local in range(3):
        ids = tri[:, local]
        np.add.at(out, ids, elem_values)
        np.add.at(cnt, ids, 1.0)
    cnt[cnt == 0.0] = 1.0
    return out / cnt
