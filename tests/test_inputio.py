from pathlib import Path

import pytest

from anchorplate.inputio import InputValidationError, expand_cases, load_input_config


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
