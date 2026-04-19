from pathlib import Path

import pytest

from anchorplate.inputio import InputValidationError, expand_cases, load_input_config


def _base_case_toml(body: str) -> str:
    return f'''
mode = "single_case"

[plate]
length_mm = 300.0
width_mm = 300.0
thickness_mm = 15.0

[analysis_options]
target_h_mm = 25.0
line_pick_tol_mm = 1e-9

[[anchors]]
x_mm = 30.0
y_mm = 30.0
kind = "fixed"

{body}
'''


def _load_config_from_text(tmp_path: Path, text: str):
    p = tmp_path / "case.toml"
    p.write_text(text, encoding="utf-8")
    return load_input_config(p)


def test_parse_dual_flange_load_transfer(tmp_path: Path):
    cfg = _load_config_from_text(
        tmp_path,
        _base_case_toml(
            '''
[[load_transfers]]
ref_x_mm = 150.0
ref_y_mm = 150.0
force_n = 50000.0
mx_nmm = 3000000.0
label = "column"

[[load_transfers.flanges]]
p1_mm = [75.0, 100.0]
p2_mm = [75.0, 200.0]
label = "left"

[[load_transfers.flanges]]
p1_mm = [225.0, 100.0]
p2_mm = [225.0, 200.0]
weight_scale = 1.25
label = "right"
'''
        ),
    )
    assert len(cfg.case.load_transfers) == 1
    transfer = cfg.case.load_transfers[0]
    assert transfer.label == "column"
    assert len(transfer.flanges) == 2
    assert transfer.flanges[1].weight_scale == 1.25
    assert cfg.case.supports[0].x_mm == 30.0


def test_parse_single_flange_load_transfer(tmp_path: Path):
    cfg = _load_config_from_text(
        tmp_path,
        _base_case_toml(
            '''
[[load_transfers]]
ref_x_mm = 150.0
ref_y_mm = 150.0
force_n = 12000.0
label = "single"

[[load_transfers.flanges]]
p1_mm = [100.0, 150.0]
p2_mm = [200.0, 150.0]
'''
        ),
    )
    assert len(cfg.case.load_transfers) == 1
    assert len(cfg.case.load_transfers[0].flanges) == 1


def test_supports_legacy_warns_but_still_maps_to_supports(tmp_path: Path):
    toml_text = '''
mode = "single_case"

[plate]
length_mm = 300.0
width_mm = 300.0
thickness_mm = 15.0

[[supports]]
x_mm = 30.0
y_mm = 30.0
kind = "fixed"
'''
    p = tmp_path / "legacy.toml"
    p.write_text(toml_text, encoding="utf-8")

    with pytest.warns(DeprecationWarning):
        cfg = load_input_config(p)
    assert len(cfg.case.supports) == 1
    assert cfg.case.supports[0].x_mm == 30.0


def test_anchors_and_supports_conflict(tmp_path: Path):
    toml_text = '''
mode = "single_case"

[plate]
length_mm = 300.0
width_mm = 300.0
thickness_mm = 15.0

[[anchors]]
x_mm = 30.0
y_mm = 30.0
kind = "fixed"

[[supports]]
x_mm = 40.0
y_mm = 40.0
kind = "fixed"
'''
    p = tmp_path / "conflict.toml"
    p.write_text(toml_text, encoding="utf-8")

    with pytest.raises(InputValidationError, match="anchors.*supports"):
        load_input_config(p)


def test_load_transfers_and_coupled_line_loads_conflict(tmp_path: Path):
    toml_text = _base_case_toml(
        '''
[[load_transfers]]
ref_x_mm = 150.0
ref_y_mm = 150.0
force_n = 12000.0

[[load_transfers.flanges]]
p1_mm = [100.0, 150.0]
p2_mm = [200.0, 150.0]

[[coupled_line_loads]]
ref_x_mm = 150.0
ref_y_mm = 150.0
force_n = 12000.0
'''
    )
    p = tmp_path / "conflict.toml"
    p.write_text(toml_text, encoding="utf-8")

    with pytest.raises(InputValidationError, match="load_transfers.*coupled_line_loads"):
        load_input_config(p)


def test_load_transfer_rejects_nonpositive_weight_scale(tmp_path: Path):
    toml_text = _base_case_toml(
        '''
[[load_transfers]]
ref_x_mm = 150.0
ref_y_mm = 150.0
force_n = 12000.0

[[load_transfers.flanges]]
p1_mm = [100.0, 150.0]
p2_mm = [200.0, 150.0]
weight_scale = 0.0
'''
    )
    with pytest.raises(InputValidationError, match="weight_scale"):
        _load_config_from_text(tmp_path, toml_text)


def test_load_transfer_rejects_degenerate_flange(tmp_path: Path):
    toml_text = _base_case_toml(
        '''
[[load_transfers]]
ref_x_mm = 150.0
ref_y_mm = 150.0
force_n = 12000.0

[[load_transfers.flanges]]
p1_mm = [100.0, 150.0]
p2_mm = [100.0, 150.0]
'''
    )
    with pytest.raises(InputValidationError, match="non-degenerate"):
        _load_config_from_text(tmp_path, toml_text)


def test_load_transfer_requires_at_least_one_flange(tmp_path: Path):
    toml_text = _base_case_toml(
        '''
[[load_transfers]]
ref_x_mm = 150.0
ref_y_mm = 150.0
force_n = 12000.0
'''
    )
    with pytest.raises(InputValidationError, match="at least one"):
        _load_config_from_text(tmp_path, toml_text)


def test_load_transfer_rejects_overlapping_flanges_on_mesh_nodes(tmp_path: Path):
    toml_text = _base_case_toml(
        '''
[[load_transfers]]
ref_x_mm = 150.0
ref_y_mm = 150.0
force_n = 12000.0

[[load_transfers.flanges]]
p1_mm = [100.0, 150.0]
p2_mm = [200.0, 150.0]

[[load_transfers.flanges]]
p1_mm = [100.0, 150.0]
p2_mm = [200.0, 150.0]
'''
    )
    with pytest.raises(InputValidationError, match="overlap|common"):
        _load_config_from_text(tmp_path, toml_text)


def test_load_single_case_uses_support_material_for_patch_k(tmp_path: Path):
    toml_text = '''
mode = "single_case"

[plate]
length_mm = 300.0
width_mm = 300.0
thickness_mm = 15.0

[support_material_model]
model = "calibrated"
k_area_n_per_mm3 = 123.0

[[supports]]
x_mm = 30.0
y_mm = 30.0
kind = "fixed"

[[foundation_patches]]
x_min_mm = 100.0
x_max_mm = 200.0
y_min_mm = 100.0
y_max_mm = 200.0
compression_only = true
'''
    p = tmp_path / "case.toml"
    p.write_text(toml_text, encoding="utf-8")

    cfg = load_input_config(p)
    assert cfg.mode == "single_case"
    assert cfg.case.foundation_patches[0].k_area_n_per_mm3 == 123.0
    assert cfg.case.support_material_model is not None
    assert cfg.case.support_material_model.model_name == "calibrated"


def test_expand_study_product_generates_expected_case_count(tmp_path: Path):
    toml_text = '''
mode = "study"

[case]
name = "study"

[plate]
length_mm = 300.0
width_mm = 300.0
thickness_mm = 12.0

[[supports]]
x_mm = 30.0
y_mm = 30.0
kind = "spring"
kz_n_per_mm = 10000.0

[[coupled_line_loads]]
ref_x_mm = 150.0
ref_y_mm = 150.0
force_n = 30000.0

[[foundation_patches]]
x_min_mm = 120.0
x_max_mm = 180.0
y_min_mm = 120.0
y_max_mm = 180.0
k_area_n_per_mm3 = 90.0

[[sweeps]]
name = "basic"
strategy = "product"

[sweeps.plate]
thickness_mm = [12.0, 16.0]

[sweeps.supports]
kz_n_per_mm = [10000.0, 20000.0]
'''
    p = tmp_path / "study.toml"
    p.write_text(toml_text, encoding="utf-8")

    cfg = load_input_config(p)
    expanded = expand_cases(cfg)
    assert len(expanded) == 4
    assert len({e.name for e in expanded}) == 4


def test_invalid_input_error_message_is_clear(tmp_path: Path):
    p = tmp_path / "bad.toml"
    p.write_text("mode='single_case'\n", encoding="utf-8")

    with pytest.raises(InputValidationError, match="Missing required table '\\[plate\\]'"):
        load_input_config(p)
