from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
import copy
import warnings
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only for Python < 3.11
    import tomli as tomllib
from typing import Any, Literal

from .model import (
    AnalysisOptions,
    ConcreteAdvancedInput,
    CoupledLineLoad,
    FlangeTransferLine,
    FoundationPatch,
    LoadTransferDefinition,
    MeshRefinementBox,
    PointLoad,
    PointSupport,
    SteelLayer,
    SteelPlate,
)
from .support import (
    SupportMaterialModelResult,
    support_material_calibrated,
    support_material_concrete_advanced,
    support_material_concrete_simple,
    support_material_steel_layers_simple,
    support_material_timber_simple,
)


class InputValidationError(ValueError):
    """Raised when an external input file is invalid."""


@dataclass(frozen=True)
class CaseDefinition:
    """
    Parsed case definition.

    TOML input now uses `[[anchors]]` for point anchors, but the internal Python
    representation remains `supports` with `PointSupport` objects for backward
    compatibility. `load_transfers` represent rigid bodies that distribute one
    resultant over their own flange set; do not mix flanges from separate rigid
    bodies in a single transfer.
    """

    name: str
    plate: SteelPlate
    analysis_options: AnalysisOptions
    supports: list[PointSupport]
    point_loads: list[PointLoad]
    coupled_line_loads: list[CoupledLineLoad]
    load_transfers: tuple[LoadTransferDefinition, ...]
    foundation_patches: list[FoundationPatch]
    refinement_boxes: list[MeshRefinementBox]
    support_material_model: SupportMaterialModelResult | None


@dataclass(frozen=True)
class SweepDefinition:
    name: str
    strategy: Literal["product", "zip"]
    parameter_axes: dict[str, list[float]]


@dataclass(frozen=True)
class InputConfig:
    mode: Literal["single_case", "study"]
    case: CaseDefinition
    sweeps: list[SweepDefinition]


@dataclass(frozen=True)
class ExpandedCase:
    name: str
    case: CaseDefinition
    metadata: dict[str, Any]


def load_input_config(path: str | Path) -> InputConfig:
    input_path = Path(path)
    try:
        data = tomllib.loads(input_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise InputValidationError(f"Input file does not exist: {input_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise InputValidationError(f"Invalid TOML syntax in '{input_path}': {exc}") from exc

    plate_tbl = _require_table(data, "plate")
    plate = SteelPlate(**_pick_fields("plate", plate_tbl, ["length_mm", "width_mm", "thickness_mm", "youngs_modulus_mpa", "poisson", "fy_d_mpa"]))

    analysis_options_tbl = _optional_table(data, "analysis_options")
    analysis_options = AnalysisOptions(**_pick_fields("analysis_options", analysis_options_tbl, list(AnalysisOptions.__dataclass_fields__.keys())))

    if "anchors" in data and "supports" in data:
        raise InputValidationError("Use either [[anchors]] or legacy [[supports]], not both")
    if "supports" in data:
        warnings.warn(
            "[[supports]] is deprecated in TOML input; use [[anchors]] instead. "
            "The Python API still uses PointSupport and case.supports.",
            DeprecationWarning,
            stacklevel=2,
        )
        support_key = "supports"
    else:
        support_key = "anchors"
    supports = [
        _parse_support(x, i, table_name=support_key)
        for i, x in enumerate(_optional_array_of_tables(data, support_key), start=1)
    ]
    if not supports:
        raise InputValidationError("At least one [[anchors]] entry is required")

    point_loads = [_parse_point_load(x, i) for i, x in enumerate(_optional_array_of_tables(data, "point_loads"), start=1)]
    if "load_transfer" in data:
        raise InputValidationError("Use [[load_transfers]] array syntax; [load_transfer] is not supported")
    if "load_transfers" in data and "coupled_line_loads" in data:
        raise InputValidationError("Use either [[load_transfers]] or legacy [[coupled_line_loads]], not both")
    load_transfers = tuple(
        _parse_load_transfer(x, i)
        for i, x in enumerate(_optional_array_of_tables(data, "load_transfers"), start=1)
    )
    coupled_line_loads = [_parse_coupled_line_load(x, i) for i, x in enumerate(_optional_array_of_tables(data, "coupled_line_loads"), start=1)]
    foundation_patches_raw = _optional_array_of_tables(data, "foundation_patches")
    refinement_boxes = [_parse_refinement_box(x, i) for i, x in enumerate(_optional_array_of_tables(data, "refinement_boxes"), start=1)]

    support_material_model = _parse_support_material_model(data.get("support_material_model"))
    foundation_patches = [
        _parse_foundation_patch(raw, idx, support_material_model)
        for idx, raw in enumerate(foundation_patches_raw, start=1)
    ]

    case_tbl = _optional_table(data, "case")
    default_name = input_path.stem
    case_name = str(case_tbl.get("name", default_name))
    case = CaseDefinition(
        name=case_name,
        plate=plate,
        analysis_options=analysis_options,
        supports=supports,
        point_loads=point_loads,
        coupled_line_loads=coupled_line_loads,
        load_transfers=load_transfers,
        foundation_patches=foundation_patches,
        refinement_boxes=refinement_boxes,
        support_material_model=support_material_model,
    )
    if load_transfers:
        _validate_load_transfer_mesh_overlaps(case)

    sweeps = [_parse_sweep(x, i) for i, x in enumerate(_optional_array_of_tables(data, "sweeps"), start=1)]
    mode = str(data.get("mode", "study" if sweeps else "single_case"))
    if mode not in {"single_case", "study"}:
        raise InputValidationError(f"Invalid mode '{mode}'. Use 'single_case' or 'study'.")
    if mode == "study" and not sweeps:
        raise InputValidationError("mode='study' requires at least one [[sweeps]] section")

    return InputConfig(mode=mode, case=case, sweeps=sweeps)


def expand_cases(config: InputConfig) -> list[ExpandedCase]:
    base_metadata = _base_metadata(config.case)
    if config.mode == "single_case":
        return [ExpandedCase(name=config.case.name, case=config.case, metadata=base_metadata)]

    expanded: list[ExpandedCase] = []
    for sweep_idx, sweep in enumerate(config.sweeps, start=1):
        assignments = _expand_sweep_assignments(sweep)
        for case_idx, assignment in enumerate(assignments, start=1):
            c = _apply_assignment(config.case, assignment)
            case_name = f"{config.case.name}__{sweep.name}_{case_idx:03d}"
            meta = {
                **base_metadata,
                "sweep": sweep.name,
                "sweep_index": sweep_idx,
                "case_index": case_idx,
                "parameters": assignment,
            }
            expanded.append(ExpandedCase(name=case_name, case=replace(c, name=case_name), metadata=meta))
    return expanded


def _base_metadata(case: CaseDefinition) -> dict[str, Any]:
    support_kinds = sorted({s.kind for s in case.supports})
    material = case.support_material_model
    payload: dict[str, Any] = {
        "support_type": ",".join(support_kinds),
        "foundation_patch_count": len(case.foundation_patches),
    }
    if case.foundation_patches:
        payload["foundation_k_area_n_per_mm3"] = [p.k_area_n_per_mm3 for p in case.foundation_patches]
    if material is not None:
        payload["material_model"] = material.model_name
        payload["material_model_k_area_n_per_mm3"] = material.k_area_n_per_mm3
        payload["material_model_parameters"] = material.parameters
    return payload


def _expand_sweep_assignments(sweep: SweepDefinition) -> list[dict[str, float]]:
    keys = list(sweep.parameter_axes.keys())
    axes = [sweep.parameter_axes[k] for k in keys]
    if sweep.strategy == "zip":
        lengths = {len(v) for v in axes}
        if len(lengths) != 1:
            raise InputValidationError(
                f"Sweep '{sweep.name}' with strategy='zip' requires all axes with same length"
            )
        return [dict(zip(keys, values, strict=True)) for values in zip(*axes, strict=True)]
    return [dict(zip(keys, values, strict=True)) for values in product(*axes)]


def _apply_assignment(case: CaseDefinition, assignment: dict[str, float]) -> CaseDefinition:
    updated = replace(case)
    plate = updated.plate
    supports = copy.deepcopy(updated.supports)
    loads = copy.deepcopy(updated.coupled_line_loads)
    patches = copy.deepcopy(updated.foundation_patches)

    for key, value in assignment.items():
        if key == "plate.thickness_mm":
            plate = replace(plate, thickness_mm=float(value))
        elif key == "supports.kz_n_per_mm":
            supports = [
                replace(s, kz_n_per_mm=float(value)) if s.kind in {"spring", "spring_tension_only"} else s
                for s in supports
            ]
        elif key == "foundation_patches.k_area_n_per_mm3":
            patches = [replace(p, k_area_n_per_mm3=float(value)) for p in patches]
        elif key == "foundation_patches.size_mm":
            patches = [_resize_patch(p, float(value)) for p in patches]
        elif key == "coupled_line_loads.line_spacing_mm":
            loads = [replace(cl, line_spacing_mm=float(value)) for cl in loads]
        elif key == "coupled_line_loads.line_length_mm":
            loads = [replace(cl, line_length_mm=float(value)) for cl in loads]
        else:
            raise InputValidationError(f"Unsupported sweep parameter: {key}")

    return replace(updated, plate=plate, supports=supports, coupled_line_loads=loads, foundation_patches=patches)


def _resize_patch(patch: FoundationPatch, size_mm: float) -> FoundationPatch:
    if size_mm <= 0.0:
        raise InputValidationError("foundation_patches.size_mm values must be > 0")
    cx = 0.5 * (patch.x_min_mm + patch.x_max_mm)
    cy = 0.5 * (patch.y_min_mm + patch.y_max_mm)
    h = 0.5 * size_mm
    return replace(patch, x_min_mm=cx - h, x_max_mm=cx + h, y_min_mm=cy - h, y_max_mm=cy + h)


def _parse_support(tbl: dict[str, Any], idx: int, table_name: str = "anchors") -> PointSupport:
    try:
        item = PointSupport(**_pick_fields(f"{table_name}[{idx}]", tbl, ["x_mm", "y_mm", "kind", "kz_n_per_mm", "label"]))
    except TypeError as exc:
        raise InputValidationError(f"Invalid [[{table_name}]] #{idx}: {exc}") from exc
    if item.kind in {"spring", "spring_tension_only"} and item.kz_n_per_mm <= 0.0:
        raise InputValidationError(f"[[{table_name}]] #{idx}: kz_n_per_mm must be > 0 for kind='{item.kind}'")
    return item


def _parse_point_load(tbl: dict[str, Any], idx: int) -> PointLoad:
    try:
        return PointLoad(**_pick_fields(f"point_loads[{idx}]", tbl, ["x_mm", "y_mm", "force_n", "label"]))
    except TypeError as exc:
        raise InputValidationError(f"Invalid [[point_loads]] #{idx}: {exc}") from exc


def _parse_coupled_line_load(tbl: dict[str, Any], idx: int) -> CoupledLineLoad:
    try:
        return CoupledLineLoad(
            **_pick_fields(
                f"coupled_line_loads[{idx}]",
                tbl,
                [
                    "ref_x_mm",
                    "ref_y_mm",
                    "force_n",
                    "mx_nmm",
                    "my_nmm",
                    "line_spacing_mm",
                    "line_length_mm",
                    "orientation",
                    "label",
                ],
            )
        )
    except TypeError as exc:
        raise InputValidationError(f"Invalid [[coupled_line_loads]] #{idx}: {exc}") from exc


def _as_point_pair(ctx: str, value: Any) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise InputValidationError(f"{ctx} must be a two-number array")
    if not all(isinstance(item, (int, float)) for item in value):
        raise InputValidationError(f"{ctx} must contain only numbers")
    return (float(value[0]), float(value[1]))


def _parse_flange(tbl: dict[str, Any], transfer_idx: int, idx: int) -> FlangeTransferLine:
    ctx = f"load_transfers[{transfer_idx}].flanges[{idx}]"
    fields = _pick_fields(ctx, tbl, ["p1_mm", "p2_mm", "weight_scale", "label"])
    if "p1_mm" not in fields or "p2_mm" not in fields:
        raise InputValidationError(f"[[load_transfers.flanges]] #{idx}: p1_mm and p2_mm are required")
    p1_mm = _as_point_pair(f"{ctx}.p1_mm", fields["p1_mm"])
    p2_mm = _as_point_pair(f"{ctx}.p2_mm", fields["p2_mm"])
    weight_scale = float(fields.get("weight_scale", 1.0))
    if weight_scale <= 0.0:
        raise InputValidationError(f"{ctx}: weight_scale must be > 0")
    length = ((p2_mm[0] - p1_mm[0]) ** 2 + (p2_mm[1] - p1_mm[1]) ** 2) ** 0.5
    if length <= 1e-12:
        raise InputValidationError(f"{ctx}: p1_mm and p2_mm must define a non-degenerate segment")
    return FlangeTransferLine(
        p1_mm=p1_mm,
        p2_mm=p2_mm,
        weight_scale=weight_scale,
        label=str(fields.get("label", "")),
    )


def _parse_load_transfer(tbl: dict[str, Any], idx: int) -> LoadTransferDefinition:
    ctx = f"load_transfers[{idx}]"
    fields = _pick_fields(
        ctx,
        tbl,
        ["ref_x_mm", "ref_y_mm", "force_n", "mx_nmm", "my_nmm", "flanges", "label"],
    )
    raw_flanges = fields.pop("flanges", [])
    if not isinstance(raw_flanges, list):
        raise InputValidationError(f"{ctx}.flanges must be an array of tables")
    if not raw_flanges:
        raise InputValidationError(f"[[load_transfers]] #{idx}: at least one [[load_transfers.flanges]] entry is required")
    for flange_idx, item in enumerate(raw_flanges, start=1):
        if not isinstance(item, dict):
            raise InputValidationError(f"{ctx}.flanges[{flange_idx}] must be a table")
    flanges = tuple(_parse_flange(item, idx, i) for i, item in enumerate(raw_flanges, start=1))
    try:
        return LoadTransferDefinition(
            ref_x_mm=float(fields["ref_x_mm"]),
            ref_y_mm=float(fields["ref_y_mm"]),
            force_n=float(fields["force_n"]),
            mx_nmm=float(fields.get("mx_nmm", 0.0)),
            my_nmm=float(fields.get("my_nmm", 0.0)),
            flanges=flanges,
            label=str(fields.get("label", "")),
        )
    except KeyError as exc:
        raise InputValidationError(f"[[load_transfers]] #{idx}: missing required key {exc.args[0]!r}") from exc
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"Invalid [[load_transfers]] #{idx}: {exc}") from exc


def _validate_load_transfer_mesh_overlaps(case: CaseDefinition) -> None:
    from .mesh import build_mesh, line_vertex_ids

    mesh = build_mesh(
        case.plate,
        case.supports,
        case.point_loads,
        case.coupled_line_loads,
        case.analysis_options,
        load_transfers=case.load_transfers,
        refinement_boxes=case.refinement_boxes,
    )
    tol = case.analysis_options.line_pick_tol_mm
    for transfer_idx, transfer in enumerate(case.load_transfers, start=1):
        flange_node_sets: list[set[int]] = []
        for flange_idx, flange in enumerate(transfer.flanges, start=1):
            length = ((flange.p2_mm[0] - flange.p1_mm[0]) ** 2 + (flange.p2_mm[1] - flange.p1_mm[1]) ** 2) ** 0.5
            if length <= max(tol, 1e-12):
                raise InputValidationError(
                    f"load_transfers[{transfer_idx}].flanges[{flange_idx}]: "
                    "p1_mm and p2_mm must define a non-degenerate segment"
                )
            ids = line_vertex_ids(mesh, p1_mm=flange.p1_mm, p2_mm=flange.p2_mm, tol=tol)
            flange_node_sets.append(set(int(v) for v in ids))
        for i in range(len(flange_node_sets)):
            for j in range(i + 1, len(flange_node_sets)):
                if flange_node_sets[i].intersection(flange_node_sets[j]):
                    raise InputValidationError(
                        f"[[load_transfers]] #{transfer_idx}: flanges #{i + 1} and #{j + 1} "
                        "overlap or have common mesh nodes within analysis_options.line_pick_tol_mm"
                    )


def _parse_refinement_box(tbl: dict[str, Any], idx: int) -> MeshRefinementBox:
    try:
        box = MeshRefinementBox(**_pick_fields(f"refinement_boxes[{idx}]", tbl, ["x_min_mm", "x_max_mm", "y_min_mm", "y_max_mm", "h_mm", "n_div_min", "label"]))
    except TypeError as exc:
        raise InputValidationError(f"Invalid [[refinement_boxes]] #{idx}: {exc}") from exc
    if box.x_min_mm >= box.x_max_mm or box.y_min_mm >= box.y_max_mm:
        raise InputValidationError(f"[[refinement_boxes]] #{idx}: min coordinates must be smaller than max coordinates")
    return box


def _parse_foundation_patch(tbl: dict[str, Any], idx: int, material: SupportMaterialModelResult | None) -> FoundationPatch:
    kwargs = _pick_fields(
        f"foundation_patches[{idx}]",
        tbl,
        ["x_min_mm", "x_max_mm", "y_min_mm", "y_max_mm", "k_area_n_per_mm3", "compression_only", "label"],
    )
    if "k_area_n_per_mm3" not in kwargs:
        if material is None:
            raise InputValidationError(
                f"[[foundation_patches]] #{idx}: missing k_area_n_per_mm3 and no [support_material_model] provided"
            )
        kwargs["k_area_n_per_mm3"] = material.k_area_n_per_mm3
    try:
        patch = FoundationPatch(**kwargs)
    except TypeError as exc:
        raise InputValidationError(f"Invalid [[foundation_patches]] #{idx}: {exc}") from exc
    if patch.x_min_mm >= patch.x_max_mm or patch.y_min_mm >= patch.y_max_mm:
        raise InputValidationError(f"[[foundation_patches]] #{idx}: min coordinates must be smaller than max coordinates")
    if patch.k_area_n_per_mm3 <= 0.0:
        raise InputValidationError(f"[[foundation_patches]] #{idx}: k_area_n_per_mm3 must be > 0")
    return patch


def _parse_support_material_model(raw: Any) -> SupportMaterialModelResult | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise InputValidationError("[support_material_model] must be a table")

    model = str(raw.get("model", "")).strip()
    if not model:
        raise InputValidationError("[support_material_model] requires 'model'")

    if model == "calibrated":
        return support_material_calibrated(float(raw["k_area_n_per_mm3"]))
    if model == "concrete_simple":
        return support_material_concrete_simple(float(raw["e_cm_mpa"]), float(raw["h_eff_mm"]))
    if model == "timber_simple":
        return support_material_timber_simple(
            float(raw["e90_mpa"]),
            float(raw["h_eff_mm"]),
            float(raw.get("spread_factor", 1.0)),
        )
    if model == "steel_layers_simple":
        layers_raw = raw.get("layers")
        if not isinstance(layers_raw, list) or not layers_raw:
            raise InputValidationError("[support_material_model] steel_layers_simple requires non-empty 'layers' array")
        layers = [
            SteelLayer(
                thickness_mm=float(layer["thickness_mm"]),
                youngs_modulus_mpa=float(layer.get("youngs_modulus_mpa", 210000.0)),
            )
            for layer in layers_raw
        ]
        return support_material_steel_layers_simple(layers)
    if model == "concrete_advanced":
        inp = ConcreteAdvancedInput(
            e_cm_mpa=float(raw["e_cm_mpa"]),
            nu=float(raw["nu"]),
            a_eff_mm2=float(raw["a_eff_mm2"]),
            a_ref_mm2=float(raw["a_ref_mm2"]),
            h_block_mm=float(raw["h_block_mm"]),
            d_plate_mm=float(raw["d_plate_mm"]),
        )
        return support_material_concrete_advanced(inp)

    raise InputValidationError(f"Unsupported support_material_model.model='{model}'")


def _parse_sweep(tbl: dict[str, Any], idx: int) -> SweepDefinition:
    if not isinstance(tbl, dict):
        raise InputValidationError(f"[[sweeps]] #{idx} must be a table")
    name = str(tbl.get("name", f"sweep{idx}"))
    strategy = str(tbl.get("strategy", "product"))
    if strategy not in {"product", "zip"}:
        raise InputValidationError(f"[[sweeps]] '{name}': strategy must be 'product' or 'zip'")

    parameter_axes: dict[str, list[float]] = {}
    plate = tbl.get("plate", {})
    if "thickness_mm" in plate:
        parameter_axes["plate.thickness_mm"] = _as_number_list(f"sweeps[{idx}].plate.thickness_mm", plate["thickness_mm"])

    supports = tbl.get("supports", {})
    if "kz_n_per_mm" in supports:
        parameter_axes["supports.kz_n_per_mm"] = _as_number_list(f"sweeps[{idx}].supports.kz_n_per_mm", supports["kz_n_per_mm"])

    lines = tbl.get("coupled_line_loads", {})
    if "line_spacing_mm" in lines:
        parameter_axes["coupled_line_loads.line_spacing_mm"] = _as_number_list(
            f"sweeps[{idx}].coupled_line_loads.line_spacing_mm", lines["line_spacing_mm"]
        )
    if "line_length_mm" in lines:
        parameter_axes["coupled_line_loads.line_length_mm"] = _as_number_list(
            f"sweeps[{idx}].coupled_line_loads.line_length_mm", lines["line_length_mm"]
        )

    patches = tbl.get("foundation_patches", {})
    if "k_area_n_per_mm3" in patches:
        parameter_axes["foundation_patches.k_area_n_per_mm3"] = _as_number_list(
            f"sweeps[{idx}].foundation_patches.k_area_n_per_mm3", patches["k_area_n_per_mm3"]
        )
    if "size_mm" in patches:
        parameter_axes["foundation_patches.size_mm"] = _as_number_list(
            f"sweeps[{idx}].foundation_patches.size_mm", patches["size_mm"]
        )

    if not parameter_axes:
        raise InputValidationError(f"[[sweeps]] '{name}' has no supported sweep axes")

    return SweepDefinition(name=name, strategy=strategy, parameter_axes=parameter_axes)


def _as_number_list(ctx: str, value: Any) -> list[float]:
    if not isinstance(value, list) or not value:
        raise InputValidationError(f"{ctx} must be a non-empty array")
    out: list[float] = []
    for i, item in enumerate(value, start=1):
        if not isinstance(item, (int, float)):
            raise InputValidationError(f"{ctx}[{i}] must be numeric")
        out.append(float(item))
    return out


def _require_table(root: dict[str, Any], key: str) -> dict[str, Any]:
    if key not in root:
        raise InputValidationError(f"Missing required table '[{key}]'")
    value = root[key]
    if not isinstance(value, dict):
        raise InputValidationError(f"[{key}] must be a table")
    return value


def _optional_table(root: dict[str, Any], key: str) -> dict[str, Any]:
    value = root.get(key, {})
    if not isinstance(value, dict):
        raise InputValidationError(f"[{key}] must be a table")
    return value


def _optional_array_of_tables(root: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = root.get(key, [])
    if not isinstance(value, list):
        raise InputValidationError(f"[[{key}]] must be an array of tables")
    for idx, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise InputValidationError(f"[[{key}]] item #{idx} must be a table")
    return value


def _pick_fields(ctx: str, source: dict[str, Any], names: list[str]) -> dict[str, Any]:
    unknown = sorted(set(source.keys()) - set(names))
    if unknown:
        raise InputValidationError(f"{ctx}: unsupported keys: {', '.join(unknown)}")
    return {name: source[name] for name in names if name in source}
