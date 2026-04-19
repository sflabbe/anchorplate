from anchorplate.model import ConcreteAdvancedInput, FlangeTransferLine, LoadTransferDefinition, SteelLayer
from anchorplate.support import (
    SupportMaterialModelResult,
    bedding_calibrated,
    bedding_concrete_advanced,
    bedding_concrete_simple,
    bedding_nodal_from_area,
    bedding_steel_layers,
    bedding_timber_simple,
    support_material_calibrated,
    support_material_concrete_advanced,
    support_material_concrete_simple,
    support_material_steel_layers_simple,
    support_material_timber_simple,
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


def test_wrapper_concrete_simple_metadata():
    result = support_material_concrete_simple(e_cm_mpa=32000.0, h_eff_mm=200.0)
    assert isinstance(result, SupportMaterialModelResult)
    assert result.k_area_n_per_mm3 == bedding_concrete_simple(32000.0, 200.0)
    assert result.model_name == "concrete_simple"
    assert result.parameters["e_cm_mpa"] == 32000.0
    assert result.parameters["h_eff_mm"] == 200.0
    assert result.notes


def test_wrapper_timber_simple_metadata():
    result = support_material_timber_simple(e90_mpa=390.0, h_eff_mm=50.0, spread_factor=1.2)
    assert result.k_area_n_per_mm3 == bedding_timber_simple(390.0, 50.0, spread_factor=1.2)
    assert result.model_name == "timber_simple"
    assert result.parameters["spread_factor"] == 1.2


def test_wrapper_steel_layers_simple_metadata():
    layers = [SteelLayer(thickness_mm=10.0, youngs_modulus_mpa=210000.0)]
    result = support_material_steel_layers_simple(layers)
    assert result.k_area_n_per_mm3 == bedding_steel_layers(layers)
    assert result.model_name == "steel_layers_simple"
    assert result.parameters["layers"][0]["thickness_mm"] == 10.0


def test_wrapper_concrete_advanced_metadata():
    inp = ConcreteAdvancedInput(
        e_cm_mpa=32000.0,
        nu=0.2,
        a_eff_mm2=300 * 300,
        a_ref_mm2=300 * 300,
        h_block_mm=200.0,
        d_plate_mm=300.0,
    )
    result = support_material_concrete_advanced(inp)
    assert result.k_area_n_per_mm3 == bedding_concrete_advanced(inp)
    assert result.model_name == "concrete_advanced"
    assert "nu" in result.parameters


def test_wrapper_calibrated_metadata():
    result = support_material_calibrated(12.5)
    assert result.k_area_n_per_mm3 == bedding_calibrated(12.5)
    assert result.model_name == "calibrated"
    assert result.parameters["k_area_n_per_mm3"] == 12.5


def test_load_transfer_model_uses_nested_flange_segments():
    transfer = LoadTransferDefinition(
        ref_x_mm=150.0,
        ref_y_mm=150.0,
        force_n=10000.0,
        flanges=(
            FlangeTransferLine(
                p1_mm=(100.0, 150.0),
                p2_mm=(200.0, 150.0),
                weight_scale=1.5,
                label="flange",
            ),
        ),
        label="body",
    )
    assert transfer.label == "body"
    assert transfer.flanges[0].p1_mm == (100.0, 150.0)
    assert transfer.flanges[0].weight_scale == 1.5
