import numpy as np

from anchorplate.model import AnalysisOptions, CoupledLineLoad, PointSupport, SteelPlate
from anchorplate.solver import solve_anchor_plate


def _base_case_options(mesh_backend: str) -> AnalysisOptions:
    return AnalysisOptions(
        target_h_mm=20.0,
        save_plots=False,
        show_plots=False,
        save_result_npz=False,
        save_3d_plots=False,
        mesh_backend=mesh_backend,
    )


def _supports() -> list[PointSupport]:
    return [
        PointSupport(x_mm=50.0, y_mm=50.0, kind="fixed"),
        PointSupport(x_mm=250.0, y_mm=50.0, kind="fixed"),
        PointSupport(x_mm=50.0, y_mm=250.0, kind="fixed"),
        PointSupport(x_mm=250.0, y_mm=250.0, kind="fixed"),
    ]


def _load_fz() -> CoupledLineLoad:
    return CoupledLineLoad(
        ref_x_mm=150.0,
        ref_y_mm=150.0,
        force_n=40_000.0,
        line_spacing_mm=150.0,
        line_length_mm=100.0,
        orientation="vertical",
    )


def test_quad_bfs_backend_runs_and_balances_vertical_load():
    plate = SteelPlate(length_mm=300.0, width_mm=300.0, thickness_mm=12.0)

    res_quad = solve_anchor_plate(
        plate=plate,
        supports=_supports(),
        coupled_loads=[_load_fz()],
        options=_base_case_options("quad_bfs"),
        name="quad_case",
    )

    assert res_quad.mesh_backend == "quad_bfs"
    assert res_quad.mesh.t.shape[0] == 4
    assert np.isclose(np.sum(res_quad.support_reactions_n), 40_000.0, rtol=1e-6)
    assert res_quad.max_deflection_mm > 0.0


def test_tri_and_quad_backends_are_close_for_fz_only():
    plate = SteelPlate(length_mm=300.0, width_mm=300.0, thickness_mm=12.0)

    res_tri = solve_anchor_plate(
        plate=plate,
        supports=_supports(),
        coupled_loads=[_load_fz()],
        options=_base_case_options("tri_morley"),
        name="tri_case",
    )
    res_quad = solve_anchor_plate(
        plate=plate,
        supports=_supports(),
        coupled_loads=[_load_fz()],
        options=_base_case_options("quad_bfs"),
        name="quad_case",
    )

    assert np.isclose(res_tri.max_deflection_mm, res_quad.max_deflection_mm, rtol=0.08)
