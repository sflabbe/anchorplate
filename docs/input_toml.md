# TOML input format

`anchorplate` supports external declarative input files using TOML (`tomllib` from Python stdlib).

## Run

```bash
python -m anchorplate.run_case examples/toml/simple_case.toml
python -m anchorplate.run_case examples/toml/parametric_study.toml
python -m anchorplate.run_case examples/toml/parametric_study.toml --dry-run
```

## Modes

- `mode = "single_case"`: runs one case.
- `mode = "study"`: expands `[[sweeps]]` into multiple cases.

If omitted, mode defaults to:
- `single_case` when there are no sweeps,
- `study` when `[[sweeps]]` exists.

## Core sections

```toml
[case]
name = "my_case"

[plate]
length_mm = 300.0
width_mm = 300.0
thickness_mm = 15.0

[analysis_options]
output_dir = "outputs/my_case"

[[supports]]
x_mm = 30.0
y_mm = 30.0
kind = "fixed" # or spring / spring_tension_only
kz_n_per_mm = 12000.0

[[point_loads]]
...

[[coupled_line_loads]]
...

[[foundation_patches]]
...

[[refinement_boxes]]
...
```

## Optional support material model

When using foundation patches, you can define `k_area` directly per patch, or provide a global model in `[support_material_model]`.

Supported models:

- `calibrated`
- `concrete_simple`
- `timber_simple`
- `steel_layers_simple`
- `concrete_advanced`

Example:

```toml
[support_material_model]
model = "concrete_simple"
e_cm_mpa = 32000.0
h_eff_mm = 250.0
```

If a `[[foundation_patches]]` item omits `k_area_n_per_mm3`, the value is filled from the support material model.

## Parametric studies (`[[sweeps]]`)

Each sweep can use:

- `strategy = "product"` (cartesian product, default)
- `strategy = "zip"` (all axes same length)

Supported sweep axes:

- `[sweeps.plate] thickness_mm = [...]`
- `[sweeps.supports] kz_n_per_mm = [...]` (applies to spring/tension-only supports)
- `[sweeps.foundation_patches] k_area_n_per_mm3 = [...]`
- `[sweeps.foundation_patches] size_mm = [...]` (resizes each patch around its center)
- `[sweeps.coupled_line_loads] line_spacing_mm = [...]`
- `[sweeps.coupled_line_loads] line_length_mm = [...]`

Each expanded case gets:

- a unique name (`<base>__<sweep>_<idx>`),
- case metadata (`case_metadata.json`),
- its own output subfolder.

Metadata includes support type, material model (if any), and foundation `k_area` values.
