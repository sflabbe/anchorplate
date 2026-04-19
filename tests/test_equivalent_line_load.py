import numpy as np
import pytest

from anchorplate.model import (
    AnalysisOptions,
    CoupledLineLoad,
    FlangeTransferLine,
    LoadTransferDefinition,
    PointSupport,
    SteelPlate,
)
from anchorplate.solver import solve_anchor_plate


def minimum_norm_force_distribution(coords_mm, weights, ref_x_mm, ref_y_mm, force_n, mx_nmm, my_nmm):
    dx = coords_mm[:, 0] - ref_x_mm
    dy = coords_mm[:, 1] - ref_y_mm
    a = np.vstack([np.ones(coords_mm.shape[0]), dy, -dx])
    b = np.array([force_n, mx_nmm, my_nmm], dtype=float)
    awat = a @ (weights[:, None] * a.T)
    lam = np.linalg.solve(awat, b)
    return weights * (a.T @ lam)


def test_force_and_moment_recovery():
    coords = np.array([
        [75.0, 100.0],
        [75.0, 150.0],
        [75.0, 200.0],
        [225.0, 100.0],
        [225.0, 150.0],
        [225.0, 200.0],
    ])
    weights = np.array([25.0, 50.0, 25.0, 25.0, 50.0, 25.0])
    f = minimum_norm_force_distribution(coords, weights, 150.0, 150.0, 50000.0, 3.0e6, -2.0e6)
    dx = coords[:, 0] - 150.0
    dy = coords[:, 1] - 150.0
    assert abs(f.sum() - 50000.0) < 1e-8
    assert abs((dy * f).sum() - 3.0e6) < 1e-6
    assert abs((-dx * f).sum() + 2.0e6) < 1e-6


def _plate():
    return SteelPlate(length_mm=300.0, width_mm=300.0, thickness_mm=15.0)


def _supports():
    return [
        PointSupport(30.0, 30.0, kind="fixed", label="A1"),
        PointSupport(270.0, 30.0, kind="fixed", label="A2"),
        PointSupport(30.0, 270.0, kind="fixed", label="A3"),
        PointSupport(270.0, 270.0, kind="fixed", label="A4"),
    ]


def _options():
    return AnalysisOptions(
        target_h_mm=50.0,
        save_plots=False,
        save_3d_plots=False,
        save_result_npz=False,
    )


def _new_transfer_equivalent_to_legacy(load: CoupledLineLoad) -> LoadTransferDefinition:
    if load.orientation == "vertical":
        ref_x = load.ref_x_mm
        ref_y = load.ref_y_mm
        spacing = load.line_spacing_mm
        length = load.line_length_mm
        flanges = (
            FlangeTransferLine(
                p1_mm=(ref_x - spacing / 2.0, ref_y - length / 2.0),
                p2_mm=(ref_x - spacing / 2.0, ref_y + length / 2.0),
                weight_scale=1.0,
            ),
            FlangeTransferLine(
                p1_mm=(ref_x + spacing / 2.0, ref_y - length / 2.0),
                p2_mm=(ref_x + spacing / 2.0, ref_y + length / 2.0),
                weight_scale=1.0,
            ),
        )
    else:
        ref_x = load.ref_x_mm
        ref_y = load.ref_y_mm
        spacing = load.line_spacing_mm
        length = load.line_length_mm
        flanges = (
            FlangeTransferLine(
                p1_mm=(ref_x - length / 2.0, ref_y - spacing / 2.0),
                p2_mm=(ref_x + length / 2.0, ref_y - spacing / 2.0),
                weight_scale=1.0,
            ),
            FlangeTransferLine(
                p1_mm=(ref_x - length / 2.0, ref_y + spacing / 2.0),
                p2_mm=(ref_x + length / 2.0, ref_y + spacing / 2.0),
                weight_scale=1.0,
            ),
        )
    return LoadTransferDefinition(
        ref_x_mm=load.ref_x_mm,
        ref_y_mm=load.ref_y_mm,
        force_n=load.force_n,
        mx_nmm=load.mx_nmm,
        my_nmm=load.my_nmm,
        flanges=flanges,
        label=load.label,
    )


def _assert_record_resultant(record, transfer):
    coords = record.coords_mm
    f = record.nodal_forces_n
    dx = coords[:, 0] - transfer.ref_x_mm
    dy = coords[:, 1] - transfer.ref_y_mm
    np.testing.assert_allclose(f.sum(), transfer.force_n, rtol=0.0, atol=1e-8)
    np.testing.assert_allclose((dy * f).sum(), transfer.mx_nmm, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose((-dx * f).sum(), transfer.my_nmm, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
def test_legacy_coupled_line_load_matches_new_two_flange_transfer(orientation):
    legacy_load = CoupledLineLoad(
        ref_x_mm=150.0,
        ref_y_mm=150.0,
        force_n=50000.0,
        mx_nmm=3.0e6,
        my_nmm=-2.0e6,
        line_spacing_mm=150.0,
        line_length_mm=100.0,
        orientation=orientation,
        label=f"legacy_{orientation}",
    )
    new_transfer = _new_transfer_equivalent_to_legacy(legacy_load)

    legacy_result = solve_anchor_plate(
        plate=_plate(),
        supports=_supports(),
        coupled_loads=[legacy_load],
        options=_options(),
        name=f"legacy_{orientation}",
    )
    new_result = solve_anchor_plate(
        plate=_plate(),
        supports=_supports(),
        load_transfers=[new_transfer],
        options=_options(),
        name=f"new_{orientation}",
    )

    legacy_record = legacy_result.load_records[0]
    new_record = new_result.load_records[0]
    np.testing.assert_allclose(legacy_result.mesh.p, new_result.mesh.p, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(legacy_record.vertex_ids, new_record.vertex_ids)
    np.testing.assert_allclose(legacy_record.coords_mm, new_record.coords_mm, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(legacy_record.nodal_forces_n, new_record.nodal_forces_n)
    np.testing.assert_array_equal(legacy_result.rhs, new_result.rhs)
    _assert_record_resultant(legacy_record, new_transfer)
    _assert_record_resultant(new_record, new_transfer)


def test_solver_accepts_single_flange_transfer():
    transfer = LoadTransferDefinition(
        ref_x_mm=150.0,
        ref_y_mm=150.0,
        force_n=10000.0,
        flanges=(
            FlangeTransferLine(p1_mm=(100.0, 150.0), p2_mm=(200.0, 150.0)),
        ),
        label="single",
    )
    result = solve_anchor_plate(
        plate=_plate(),
        supports=_supports(),
        load_transfers=[transfer],
        options=_options(),
        name="single_flange",
    )
    assert len(result.load_records) == 1
    _assert_record_resultant(result.load_records[0], transfer)


def test_oblique_flange_picking_and_resultant():
    transfer = LoadTransferDefinition(
        ref_x_mm=150.0,
        ref_y_mm=150.0,
        force_n=10000.0,
        flanges=(
            FlangeTransferLine(p1_mm=(100.0, 100.0), p2_mm=(200.0, 200.0)),
        ),
        label="oblique",
    )
    result = solve_anchor_plate(
        plate=_plate(),
        supports=_supports(),
        load_transfers=[transfer],
        options=_options(),
        name="oblique_flange",
    )
    record = result.load_records[0]
    assert record.vertex_ids.size >= 3
    np.testing.assert_allclose(record.coords_mm[:, 0], record.coords_mm[:, 1], rtol=0.0, atol=1e-9)
    _assert_record_resultant(record, transfer)
