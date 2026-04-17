from anchorplate.model import ConcreteAdvancedInput, SteelLayer
from anchorplate.support import (
    bedding_calibrated,
    bedding_concrete_advanced,
    bedding_concrete_simple,
    bedding_nodal_from_area,
    bedding_steel_layers,
    bedding_timber_simple,
)


def test_concrete_simple():
    assert abs(bedding_concrete_simple(32000.0, 200.0) - 160.0) < 1e-12


def test_timber_simple():
    assert abs(bedding_timber_simple(370.0, 100.0) - 3.7) < 1e-12
    assert abs(bedding_timber_simple(370.0, 100.0, spread_factor=1.5) - 5.55) < 1e-12


def test_steel_layers_series():
    k = bedding_steel_layers([SteelLayer(10.0, 210000.0), SteelLayer(15.0, 210000.0)])
    assert abs(k - 8400.0) < 1e-9


def test_concrete_advanced_positive():
    k = bedding_concrete_advanced(
        ConcreteAdvancedInput(
            e_cm_mpa=32000.0,
            nu=0.2,
            a_eff_mm2=300 * 300,
            a_ref_mm2=300 * 300,
            h_block_mm=200.0,
            d_plate_mm=300.0,
        )
    )
    assert k > 0.0


def test_nodal_conversion():
    assert abs(bedding_nodal_from_area(3.7, 100.0) - 370.0) < 1e-12
    assert abs(bedding_calibrated(12.5) - 12.5) < 1e-12
