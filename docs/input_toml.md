# TOML input format

`anchorplate` supports external declarative input files using TOML (`tomllib` from Python stdlib).

## Run

```bash
python -m anchorplate.run_case examples/toml/simple_case.toml
python -m anchorplate.run_case examples/toml/dual_flange_case.toml
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

[[anchors]]
x_mm = 30.0
y_mm = 30.0
kind = "fixed" # or spring / spring_tension_only
kz_n_per_mm = 12000.0

[[point_loads]]
x_mm = 150.0
y_mm = 150.0
force_n = 1000.0

[[load_transfers]]
ref_x_mm = 150.0
ref_y_mm = 150.0
force_n = 50000.0
mx_nmm = 3000000.0
my_nmm = -2000000.0
label = "column"

[[load_transfers.flanges]]
p1_mm = [75.0, 100.0]
p2_mm = [75.0, 200.0]

[[load_transfers.flanges]]
p1_mm = [225.0, 100.0]
p2_mm = [225.0, 200.0]

[[foundation_patches]]
...

[[refinement_boxes]]
...
```

## Anchors

Use `[[anchors]]` for point anchors in TOML. Each entry maps to the existing
Python `PointSupport` model and to `case.supports` internally; the Python API is
not renamed in this refactor.

`[[supports]]` is still accepted as a legacy alias and emits a
`DeprecationWarning`. Do not provide both `[[anchors]]` and `[[supports]]` in the
same file.

## Load Transfers

Use `[[load_transfers]]` as an array. A transfer represents one physical rigid
body, for example one column, bracket, or connection shoe. It applies one global
resultant:

- `force_n`
- `mx_nmm`
- `my_nmm`

at the reference point `ref_x_mm`, `ref_y_mm`, and transmits that resultant to
the plate through its nested `[[load_transfers.flanges]]` entries.

The solver performs one minimum-norm distribution over the union of all flange
nodes belonging to that same transfer. Two separate columns therefore require
two separate `[[load_transfers]]` entries. Do not mix flanges from different
rigid bodies inside one transfer.

Each flange is an arbitrary segment:

```toml
[[load_transfers.flanges]]
p1_mm = [100.0, 120.0]
p2_mm = [220.0, 180.0]
weight_scale = 1.0
label = "front"
```

`weight_scale` must be greater than zero. It multiplies the tributary nodal
weights of that flange before the transfer-level minimum-norm solve. That means
it changes the distribution between flanges of the same transfer; it is not a
local force multiplier applied after the solve.

With a single flange, moments are distributed along that line. They are not
split between two lines, because there is only one transfer segment.

Flange picking uses `analysis_options.line_pick_tol_mm` as the normal-distance
tolerance to the segment. The segment projection parameter must also fall within
`[-tol_s, 1 + tol_s]`, where `tol_s = line_pick_tol_mm / segment_length`.

## Examples

Dual flange transfer:

```toml
[[load_transfers]]
ref_x_mm = 150.0
ref_y_mm = 150.0
force_n = 50000.0
mx_nmm = 3000000.0
my_nmm = -2000000.0
label = "dual_flange_column"

[[load_transfers.flanges]]
p1_mm = [75.0, 100.0]
p2_mm = [75.0, 200.0]

[[load_transfers.flanges]]
p1_mm = [225.0, 100.0]
p2_mm = [225.0, 200.0]
```

Single flange transfer:

```toml
[[load_transfers]]
ref_x_mm = 150.0
ref_y_mm = 120.0
force_n = 12000.0
mx_nmm = 0.0
my_nmm = 0.0
label = "balustrade_post"

[[load_transfers.flanges]]
p1_mm = [90.0, 120.0]
p2_mm = [210.0, 120.0]
weight_scale = 1.0
```

## Legacy Coupled Line Loads

`[[coupled_line_loads]]` remains supported for backward compatibility. It is a
legacy shorthand for exactly two flanges, both with `weight_scale = 1.0`, and is
translated internally to the new load-transfer path.

Do not provide `[[load_transfers]]` and `[[coupled_line_loads]]` in the same
file.

## Optional Support Material Model

When using foundation patches, you can define `k_area` directly per patch, or
provide a global model in `[support_material_model]`.

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

If a `[[foundation_patches]]` item omits `k_area_n_per_mm3`, the value is filled
from the support material model.

## Parametric Studies (`[[sweeps]]`)

Each sweep can use:

- `strategy = "product"` (cartesian product, default)
- `strategy = "zip"` (all axes same length)

Supported sweep axes:

- `[sweeps.plate] thickness_mm = [...]`
- `[sweeps.supports] kz_n_per_mm = [...]` (applies to spring/tension-only anchors)
- `[sweeps.foundation_patches] k_area_n_per_mm3 = [...]`
- `[sweeps.foundation_patches] size_mm = [...]` (resizes each patch around its center)
- `[sweeps.coupled_line_loads] line_spacing_mm = [...]`
- `[sweeps.coupled_line_loads] line_length_mm = [...]`

Sweeps over `coupled_line_loads.line_spacing_mm` and
`coupled_line_loads.line_length_mm` apply only to the legacy coupled-line route.
Extending sweeps to `load_transfers` or flange `p1_mm`/`p2_mm` is outside the
scope of this refactor.

Each expanded case gets:

- a unique name (`<base>__<sweep>_<idx>`),
- case metadata (`case_metadata.json`),
- its own output subfolder.

Metadata includes support type, material model (if any), and foundation `k_area` values.
