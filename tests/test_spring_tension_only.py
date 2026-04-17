from __future__ import annotations

import csv
import json

import numpy as np

from anchorplate.mesh import nodal_tributary_areas
from anchorplate.model import AnalysisOptions, CoupledLineLoad, FoundationPatch, PointSupport, SteelPlate
from anchorplate.solver import export_support_reactions_csv, export_support_reactions_json, solve_anchor_plate


PLATE = SteelPlate(length_mm=300.0, width_mm=300.0, thickness_mm=15.0)
KZ = 150_000.0
FZ = 50_000.0
MX = 8.0e6
OPTIONS = AnalysisOptions(
    target_h_mm=8.0,
    output_dir="outputs/tests",
    save_plots=False,
    save_result_npz=False,
    save_3d_plots=False,
)


def _supports(kind: str) -> list[PointSupport]:
    return [
        PointSupport(30.0, 30.0, kind=kind, kz_n_per_mm=KZ, label="A1"),
        PointSupport(270.0, 30.0, kind=kind, kz_n_per_mm=KZ, label="A2"),
        PointSupport(30.0, 270.0, kind=kind, kz_n_per_mm=KZ, label="A3"),
        PointSupport(270.0, 270.0, kind=kind, kz_n_per_mm=KZ, label="A4"),
    ]


def _foundation_total_reaction(result, patches: list[FoundationPatch]) -> float:
    tributary = nodal_tributary_areas(result.mesh)
    total = 0.0
    for patch, active_ids in zip(patches, result.foundation_state.active_vertices, strict=True):
        if not active_ids:
            continue
        ids = np.array(sorted(active_ids), dtype=int)
        total += float(np.sum(patch.k_area_n_per_mm3 * tributary[ids] * result.w_vertex_mm[ids]))
    return total


def test_tension_only_centric_fz_activates_under_solver_sign_convention():
    result = solve_anchor_plate(
        plate=PLATE,
        supports=_supports("spring_tension_only"),
        coupled_loads=[CoupledLineLoad(ref_x_mm=150.0, ref_y_mm=150.0, force_n=FZ, label="Fz_centric")],
        options=OPTIONS,
        name="to_centric_fz",
    )

    assert np.all(result.support_active)
    assert np.sum(result.support_reactions_n) > 0.0


def test_tension_only_uplift_by_moment_deactivates_some_anchors():
    result = solve_anchor_plate(
        plate=PLATE,
        supports=_supports("spring_tension_only"),
        coupled_loads=[CoupledLineLoad(ref_x_mm=150.0, ref_y_mm=150.0, force_n=0.0, mx_nmm=MX, label="Mx_pure")],
        options=OPTIONS,
        name="to_uplift_moment",
    )

    n_active = int(np.sum(result.support_active))
    assert 0 < n_active < len(result.support_active)
    reactions_active = result.support_reactions_n[result.support_active]
    assert np.max(np.abs(reactions_active)) > 1e-6


def test_spring_vs_tension_only_behaviour_differs_for_pure_moment():
    linear = solve_anchor_plate(
        plate=PLATE,
        supports=_supports("spring"),
        coupled_loads=[CoupledLineLoad(ref_x_mm=150.0, ref_y_mm=150.0, force_n=0.0, mx_nmm=MX, label="Mx_pure")],
        options=OPTIONS,
        name="linear_mx",
    )
    tension_only = solve_anchor_plate(
        plate=PLATE,
        supports=_supports("spring_tension_only"),
        coupled_loads=[CoupledLineLoad(ref_x_mm=150.0, ref_y_mm=150.0, force_n=0.0, mx_nmm=MX, label="Mx_pure")],
        options=OPTIONS,
        name="to_mx",
    )

    assert np.any(linear.support_reactions_n > 0.0)
    assert np.any(linear.support_reactions_n < 0.0)
    assert np.max(np.abs(tension_only.support_reactions_n[tension_only.support_active])) > 1e-6
    assert int(np.sum(tension_only.support_active)) < len(tension_only.support_active)


def test_equilibrium_with_foundation_and_active_tension_only_springs():
    patches = [
        FoundationPatch(
            x_min_mm=0.0,
            x_max_mm=300.0,
            y_min_mm=0.0,
            y_max_mm=300.0,
            k_area_n_per_mm3=20.0,
            compression_only=True,
            label="full_patch",
        )
    ]
    load = CoupledLineLoad(ref_x_mm=150.0, ref_y_mm=150.0, force_n=40_000.0, mx_nmm=4.0e6, label="Fz_Mx")
    result = solve_anchor_plate(
        plate=PLATE,
        supports=_supports("spring_tension_only"),
        coupled_loads=[load],
        foundation_patches=patches,
        options=OPTIONS,
        name="to_foundation_hybrid",
    )
    foundation_r = _foundation_total_reaction(result, patches)
    total_reaction = float(np.sum(result.support_reactions_n)) + foundation_r
    assert abs(total_reaction - load.force_n) < load.force_n * 1e-3
    assert np.any(result.support_active)


def test_export_support_reactions_json_and_csv(tmp_path):
    result = solve_anchor_plate(
        plate=PLATE,
        supports=_supports("spring_tension_only"),
        coupled_loads=[CoupledLineLoad(ref_x_mm=150.0, ref_y_mm=150.0, force_n=0.0, mx_nmm=MX, label="Mx_pure")],
        options=OPTIONS,
        name="to_export",
    )
    json_path = export_support_reactions_json(result, tmp_path / "support_reactions.json")
    csv_path = export_support_reactions_csv(result, tmp_path / "support_reactions.csv")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["case_name"] == "to_export"
    assert len(payload["supports"]) == 4
    assert "active" in payload["supports"][0]

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    assert set(rows[0].keys()) == {"index", "label", "kind", "vertex_id", "dof", "reaction_n", "active"}
