from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .mesh import triangle_connectivity
from .model import AnalysisOptions, CoupledLineLoad, FoundationPatch, MeshRefinementBox, PointLoad, PointSupport, SteelPlate
from .postprocess import nodal_average_from_element_field


def _foundation_masks(result) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Return (active_mask, inactive_mask) as boolean arrays over mesh vertices.

    active   — vertex is inside a foundation patch AND in contact (w > 0 at convergence).
    inactive — vertex is inside a foundation patch AND NOT in contact (w ≤ 0, lift-off).

    Returns (active, None) when no foundation patches are present.
    The two masks are guaranteed to be complementary within the patch domain:
        active | inactive == in_any_patch
        active & inactive == False (everywhere)
    """
    n_vertices = result.mesh.p.shape[1]
    fs = result.foundation_state

    if not fs.active_vertices:
        return np.zeros(n_vertices, dtype=bool), None

    # Build active mask from the converged active vertex sets.
    active = np.zeros(n_vertices, dtype=bool)
    for ids in fs.active_vertices:
        if ids:
            active[np.array(sorted(ids), dtype=int)] = True

    # Build the full "in-patch" mask from all_patch_vertices when available.
    # Fall back to active only when all_patch_vertices is missing (old serialised data).
    if fs.all_patch_vertices:
        all_patch_sets = fs.all_patch_vertices
    else:
        # Legacy fallback: can't reconstruct inactive properly.
        all_patch_sets = fs.active_vertices

    all_patch_ids = sorted(set().union(*all_patch_sets))
    in_patch = np.zeros(n_vertices, dtype=bool)
    if all_patch_ids:
        in_patch[np.array(all_patch_ids, dtype=int)] = True

    # inactive = in patch AND not active
    inactive = in_patch & ~active
    return active, inactive


def _contact_summary(result, active_mask: np.ndarray, inactive_mask: np.ndarray | None) -> dict:
    """
    Build a plain dict with scalar contact statistics suitable for printing and npz storage.

    Keys
    ----
    n_patch_total       : total nodes across all patches (with overlap counted once)
    n_active            : nodes in contact (w > tol)
    n_inactive          : nodes with lift-off (w ≤ tol inside patch)
    pct_active          : n_active / n_patch_total * 100
    pct_inactive        : n_inactive / n_patch_total * 100
    w_active_min/max    : deflection range of active nodes [mm]
    w_inactive_min/max  : deflection range of inactive (lift-off) nodes [mm]
    n_iterations        : number of active-set iterations performed
    converged           : True when last history_changes entry is 0
    history_changes     : list of per-iteration active-set change counts
    """
    fs = result.foundation_state
    w = result.w_vertex_mm

    n_active = int(active_mask.sum())
    n_inactive = int(inactive_mask.sum()) if inactive_mask is not None else 0
    n_patch_total = n_active + n_inactive

    summary: dict = {
        "n_patch_total": n_patch_total,
        "n_active": n_active,
        "n_inactive": n_inactive,
        "pct_active": round(100.0 * n_active / max(n_patch_total, 1), 2),
        "pct_inactive": round(100.0 * n_inactive / max(n_patch_total, 1), 2),
        "w_active_min": float(w[active_mask].min()) if n_active else float("nan"),
        "w_active_max": float(w[active_mask].max()) if n_active else float("nan"),
        "w_inactive_min": float(w[inactive_mask].min()) if n_inactive else float("nan"),
        "w_inactive_max": float(w[inactive_mask].max()) if n_inactive else float("nan"),
        "n_iterations": len(fs.history_changes),
        "converged": bool(fs.history_changes and fs.history_changes[-1] == 0),
        "history_changes": list(fs.history_changes),
    }
    return summary


def export_result_npz(result, outpath: Path) -> tuple[Path, dict]:
    """
    Export the full result to a compressed .npz file and return (path, contact_summary).

    NPZ arrays
    ----------
    x_mm, y_mm              : mesh vertex coordinates
    triangles               : element connectivity (n_elem × 3)
    w_mm                    : vertex deflections
    sigma_vm_mpa            : von Mises stress (nodal average from elements)
    support_vertex_ids      : mesh vertex indices of supports
    support_reactions_n     : support reactions [N] (R = kz·w for springs, residual for fixed)
    support_active_mask     : uint8 per support — 1 if spring is active (always 1 for fixed/linear spring)
    support_kinds           : support kind string per support
    support_labels          : support label per support
    active_foundation_mask  : uint8 per vertex — 1 if in contact with foundation
    inactive_foundation_mask: uint8 per vertex — 1 if inside patch but NOT in contact (lift-off)
    in_patch_mask           : uint8 per vertex — 1 if inside any foundation patch (active | inactive)
    foundation_history      : active-set change count per iteration
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    tri = triangle_connectivity(result.mesh)
    vm_nodal = nodal_average_from_element_field(result.mesh, result.von_mises_mpa)
    active_mask, inactive_mask = _foundation_masks(result)

    _inactive = inactive_mask if inactive_mask is not None else np.zeros_like(active_mask)
    in_patch = active_mask | _inactive

    summary = _contact_summary(result, active_mask, inactive_mask)

    np.savez_compressed(
        outpath,
        x_mm=result.mesh.p[0],
        y_mm=result.mesh.p[1],
        triangles=tri,
        w_mm=result.w_vertex_mm,
        sigma_vm_mpa=vm_nodal,
        support_vertex_ids=result.support_vertex_ids,
        support_reactions_n=result.support_reactions_n,
        support_active_mask=result.support_active.astype(np.uint8),
        support_kinds=result.support_kinds,
        support_labels=result.support_labels,
        active_foundation_mask=active_mask.astype(np.uint8),
        inactive_foundation_mask=_inactive.astype(np.uint8),
        in_patch_mask=in_patch.astype(np.uint8),
        foundation_history=np.array(result.foundation_state.history_changes, dtype=int),
    )
    return outpath, summary


def plot_mesh(
    mesh,
    plate: SteelPlate,
    supports: Sequence[PointSupport],
    coupled_loads: Sequence[CoupledLineLoad],
    point_loads: Sequence[PointLoad],
    refinement_boxes: Sequence[MeshRefinementBox] | None = None,
    foundation_patches: Sequence[FoundationPatch] | None = None,
    outpath: Path | None = None,
):
    tri = triangle_connectivity(mesh)
    triang = mtri.Triangulation(mesh.p[0], mesh.p[1], tri)
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ax.triplot(triang, color="0.8", linewidth=0.4)
    ax.plot([0, plate.length_mm, plate.length_mm, 0, 0], [0, 0, plate.width_mm, plate.width_mm, 0], "k-", lw=1.0)

    for i, s in enumerate(supports, start=1):
        marker = "s" if s.kind == "fixed" else "^"
        ax.plot(s.x_mm, s.y_mm, marker=marker, ms=7, linestyle="None")
        ax.annotate(s.label or f"S{i}", (s.x_mm, s.y_mm), xytext=(4, 4), textcoords="offset points")

    for p in point_loads:
        ax.plot(p.x_mm, p.y_mm, marker="o", ms=6, linestyle="None")
        ax.annotate(p.label or "P", (p.x_mm, p.y_mm), xytext=(4, 4), textcoords="offset points")

    for cl in coupled_loads:
        ax.plot(cl.ref_x_mm, cl.ref_y_mm, marker="x", ms=8, linestyle="None")
        ax.annotate(cl.label, (cl.ref_x_mm, cl.ref_y_mm), xytext=(5, 5), textcoords="offset points")
        if cl.orientation == "vertical":
            x1 = cl.ref_x_mm - 0.5 * cl.line_spacing_mm
            x2 = cl.ref_x_mm + 0.5 * cl.line_spacing_mm
            y0 = cl.ref_y_mm - 0.5 * cl.line_length_mm
            y1 = cl.ref_y_mm + 0.5 * cl.line_length_mm
            ax.plot([x1, x1], [y0, y1], lw=3, alpha=0.8)
            ax.plot([x2, x2], [y0, y1], lw=3, alpha=0.8)
        else:
            y1 = cl.ref_y_mm - 0.5 * cl.line_spacing_mm
            y2 = cl.ref_y_mm + 0.5 * cl.line_spacing_mm
            x0 = cl.ref_x_mm - 0.5 * cl.line_length_mm
            x1 = cl.ref_x_mm + 0.5 * cl.line_length_mm
            ax.plot([x0, x1], [y1, y1], lw=3, alpha=0.8)
            ax.plot([x0, x1], [y2, y2], lw=3, alpha=0.8)

    for box in refinement_boxes or []:
        xs = [box.x_min_mm, box.x_max_mm, box.x_max_mm, box.x_min_mm, box.x_min_mm]
        ys = [box.y_min_mm, box.y_min_mm, box.y_max_mm, box.y_max_mm, box.y_min_mm]
        ax.plot(xs, ys, linestyle="--", lw=1.3)

    for patch in foundation_patches or []:
        xs = [patch.x_min_mm, patch.x_max_mm, patch.x_max_mm, patch.x_min_mm, patch.x_min_mm]
        ys = [patch.y_min_mm, patch.y_min_mm, patch.y_max_mm, patch.y_max_mm, patch.y_min_mm]
        ax.fill(xs, ys, alpha=0.12)

    ax.set_title("Mesh and model geometry")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    if outpath is not None:
        fig.savefig(outpath, dpi=180)
        plt.close(fig)
        return outpath
    return fig, ax


def plot_result_3d(
    plate: SteelPlate,
    supports: Sequence[PointSupport],
    result,
    options: AnalysisOptions,
    outpath: Path | None = None,
):
    """
    3D plot of the deformed plate with foundation contact state.

    Visual layers (bottom to top in z)
    ------------------------------------
    z = 0 plane    — translucent grey: the rigid foundation / support plane.
    Deformed plate — coloured by w [mm] with RdYlBu_r.
    Green dots     — nodes in contact (active, w > 0) shown at z = 0.
    Red dots       — nodes with lift-off (inactive, w ≤ 0) shown at z = 0.
    Orange lines   — spring support markers.
    """
    outdir = Path(options.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outpath or (outdir / (result.name.lower().replace(" ", "_") + "_3d.png"))

    tri = triangle_connectivity(result.mesh)
    x = result.mesh.p[0]
    y = result.mesh.p[1]
    scale = options.z_plot_scale
    z = scale * result.w_vertex_mm

    active_mask, inactive_mask = _foundation_masks(result)
    has_foundation = active_mask.any() or (inactive_mask is not None and inactive_mask.any())

    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    # ── z = 0 support plane ──────────────────────────────────────────────────
    if has_foundation:
        ax.plot_surface(
            np.array([[0.0, plate.length_mm], [0.0, plate.length_mm]]),
            np.array([[0.0, 0.0], [plate.width_mm, plate.width_mm]]),
            np.zeros((2, 2)),
            alpha=0.08, color="grey", linewidth=0, zorder=0,
        )
    bx = np.array([0.0, plate.length_mm, plate.length_mm, 0.0, 0.0])
    by = np.array([0.0, 0.0, plate.width_mm, plate.width_mm, 0.0])
    ax.plot(bx, by, np.zeros_like(bx), "k--", lw=1.0,
            label="z = 0  (support plane)", zorder=2)

    # ── Deformed plate ────────────────────────────────────────────────────────
    surf = ax.plot_trisurf(
        x, y, z, triangles=tri,
        cmap="RdYlBu_r", linewidth=0.0, antialiased=True, alpha=0.88, zorder=3,
    )
    cbar = fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.08)
    cbar.set_label(f"w × {scale:.0f}  [mm·{scale:.0f}]" if scale != 1.0 else "w [mm]")

    # ── Contact state markers at z = 0 ───────────────────────────────────────
    if has_foundation:
        n_active   = int(active_mask.sum())
        n_inactive = int(inactive_mask.sum()) if inactive_mask is not None else 0
        n_patch    = n_active + n_inactive
        if n_active > 0:
            ax.scatter(
                x[active_mask], y[active_mask], np.zeros(n_active),
                s=6, color="#2ca02c", alpha=0.65, depthshade=False, zorder=4,
                label=f"In contact: {n_active} nodes ({100*n_active/max(n_patch,1):.0f}%)",
            )
        if inactive_mask is not None and n_inactive > 0:
            ax.scatter(
                x[inactive_mask], y[inactive_mask], np.zeros(n_inactive),
                s=6, color="#d62728", alpha=0.55, depthshade=False, zorder=4,
                label=f"Lift-off:   {n_inactive} nodes ({100*n_inactive/max(n_patch,1):.0f}%)",
            )

    # ── Support markers ────────────────────────────────────────────────────────
    for i, s in enumerate(supports, start=1):
        vid = int(result.support_vertex_ids[i - 1])
        zi = z[vid]
        if s.kind == "spring":
            color = "#ff7f0e"
        elif s.kind == "spring_tension_only":
            color = "#9467bd"
        else:
            color = "#d62728"
        ax.plot([s.x_mm, s.x_mm], [s.y_mm, s.y_mm], [0.0, zi],
                color=color, lw=2.0, zorder=5)
        ax.scatter([s.x_mm], [s.y_mm], [zi], color=color, s=40,
                   depthshade=False, zorder=6)
        R_kN = result.support_reactions_n[i - 1] / 1000.0
        ax.text(s.x_mm + 4, s.y_mm + 4, zi,
                f"{s.label or f'S{i}'}\n{R_kN:+.1f} kN", fontsize=7, zorder=7)

    ax.set_title(f"{result.name}: deformed plate vs z = 0 support plane", pad=8)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel(f"w × {scale:.0f} [mm]" if scale != 1.0 else "w [mm]")
    ax.view_init(elev=28, azim=-52)
    try:
        ax.set_box_aspect((plate.length_mm, plate.width_mm,
                           max(plate.length_mm, plate.width_mm) * 0.28))
    except Exception:
        pass
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(outpath, dpi=180)
    if options.show_plots:
        plt.show()
    else:
        plt.close(fig)
    return outpath


def plot_result(plate: SteelPlate, supports: Sequence[PointSupport], point_loads: Sequence[PointLoad], coupled_loads: Sequence[CoupledLineLoad], result, options: AnalysisOptions):
    outdir = Path(options.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    tri = triangle_connectivity(result.mesh)
    triang = mtri.Triangulation(result.mesh.p[0], result.mesh.p[1], tri)
    vm_nodal = nodal_average_from_element_field(result.mesh, result.von_mises_mpa)
    active_mask, inactive_mask = _foundation_masks(result)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)

    ax = axes[0, 0]
    ax.set_title("Geometry, supports, RP and coupled lines")
    ax.triplot(triang, color="0.85", linewidth=0.35)
    ax.plot([0, plate.length_mm, plate.length_mm, 0, 0], [0, 0, plate.width_mm, plate.width_mm, 0], "k-", lw=1.0)
    for i, s in enumerate(supports, start=1):
        marker = "s" if s.kind == "fixed" else "^"
        ax.plot(s.x_mm, s.y_mm, marker=marker, ms=7, linestyle="None")
        ax.annotate(s.label or f"S{i}", (s.x_mm, s.y_mm), xytext=(5, 5), textcoords="offset points")
    for p in point_loads:
        ax.plot(p.x_mm, p.y_mm, marker="o", ms=6, linestyle="None")
        ax.annotate(p.label or "P", (p.x_mm, p.y_mm), xytext=(5, -10), textcoords="offset points")
    for cl in coupled_loads:
        ax.plot(cl.ref_x_mm, cl.ref_y_mm, marker="x", ms=8, linestyle="None")
        ax.annotate(cl.label, (cl.ref_x_mm, cl.ref_y_mm), xytext=(6, 6), textcoords="offset points")
        if cl.orientation == "vertical":
            x1 = cl.ref_x_mm - 0.5 * cl.line_spacing_mm
            x2 = cl.ref_x_mm + 0.5 * cl.line_spacing_mm
            y0 = cl.ref_y_mm - 0.5 * cl.line_length_mm
            y1 = cl.ref_y_mm + 0.5 * cl.line_length_mm
            ax.plot([x1, x1], [y0, y1], lw=3, alpha=0.8)
            ax.plot([x2, x2], [y0, y1], lw=3, alpha=0.8)
        else:
            y1 = cl.ref_y_mm - 0.5 * cl.line_spacing_mm
            y2 = cl.ref_y_mm + 0.5 * cl.line_spacing_mm
            x0 = cl.ref_x_mm - 0.5 * cl.line_length_mm
            x1 = cl.ref_x_mm + 0.5 * cl.line_length_mm
            ax.plot([x0, x1], [y1, y1], lw=3, alpha=0.8)
            ax.plot([x0, x1], [y2, y2], lw=3, alpha=0.8)
    if np.any(active_mask):
        ax.scatter(result.mesh.p[0][active_mask], result.mesh.p[1][active_mask], s=6, alpha=0.4, label="foundation active")
    if inactive_mask is not None and np.any(inactive_mask):
        ax.scatter(result.mesh.p[0][inactive_mask], result.mesh.p[1][inactive_mask], s=4, alpha=0.2, label="foundation inactive")
    if np.any(active_mask) or (inactive_mask is not None and np.any(inactive_mask)):
        ax.legend(loc="upper right")
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")

    ax = axes[0, 1]
    ax.set_title("Deflection w [mm]")
    tcf = ax.tricontourf(triang, result.w_vertex_mm, levels=20)
    fig.colorbar(tcf, ax=ax, label="w [mm]")
    if np.min(result.w_vertex_mm) < 0.0 < np.max(result.w_vertex_mm):
        ax.tricontour(triang, result.w_vertex_mm, levels=[0.0], linewidths=1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")

    ax = axes[1, 0]
    ax.set_title("Approx. plate von Mises [MPa]")
    tcf = ax.tricontourf(triang, vm_nodal, levels=20)
    fig.colorbar(tcf, ax=ax, label="σ_v [MPa]")
    ax.text(
        0.02,
        0.98,
        f"f_y,d = {plate.fy_d_mpa:.0f} MPa\nη = {result.max_von_mises_mpa / plate.fy_d_mpa:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")

    ax = axes[1, 1]
    ax.set_title("Support reactions")
    labels = [s.label or f"S{i}" for i, s in enumerate(supports, start=1)]
    ax.bar(labels, result.support_reactions_n)
    ax.axhline(0.0, color="0.2", lw=1.0)
    ax.set_ylabel("Rz [N]")
    for i, val in enumerate(result.support_reactions_n):
        ax.text(i, val, f"{val/1000:.1f} kN", ha="center", va="bottom" if val >= 0 else "top")

    fig.suptitle(result.name)
    outpath = outdir / (result.name.lower().replace(" ", "_") + ".png")
    fig.savefig(outpath, dpi=180)
    if options.show_plots:
        plt.show()
    else:
        plt.close(fig)
    return outpath
