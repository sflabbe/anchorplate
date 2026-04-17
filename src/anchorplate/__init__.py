from __future__ import annotations

from importlib import import_module

from .model import (
    AnalysisOptions,
    AnchorSpringState,
    ConcreteAdvancedInput,
    CoupledLineLoad,
    FoundationPatch,
    FoundationState,
    MeshRefinementBox,
    PointLoad,
    PointSupport,
    SteelLayer,
    SteelPlate,
)
from .support import (
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

# Names that require optional FE/plotting dependencies.
_LAZY_EXPORTS = {
    "Result": ("anchorplate.solver", "Result"),
    "solve_anchor_plate": ("anchorplate.solver", "solve_anchor_plate"),
    "export_support_reactions_csv": ("anchorplate.solver", "export_support_reactions_csv"),
    "export_support_reactions_json": ("anchorplate.solver", "export_support_reactions_json"),
    "support_reaction_rows": ("anchorplate.solver", "support_reaction_rows"),
    "export_result_npz": ("anchorplate.plotting", "export_result_npz"),
    "plot_result_3d": ("anchorplate.plotting", "plot_result_3d"),
    "run_profis_like_benchmark": ("anchorplate.benchmark", "run_profis_like_benchmark"),
    "run_support_model_matrix_benchmark": ("anchorplate.benchmark_matrix", "run_support_model_matrix_benchmark"),
}

__all__ = [
    "AnalysisOptions",
    "AnchorSpringState",
    "ConcreteAdvancedInput",
    "CoupledLineLoad",
    "FoundationPatch",
    "FoundationState",
    "MeshRefinementBox",
    "PointLoad",
    "PointSupport",
    "SteelLayer",
    "SteelPlate",
    "SupportMaterialModelResult",
    "bedding_calibrated",
    "bedding_concrete_advanced",
    "bedding_concrete_simple",
    "bedding_nodal_from_area",
    "bedding_steel_layers",
    "bedding_timber_simple",
    "support_material_calibrated",
    "support_material_concrete_advanced",
    "support_material_concrete_simple",
    "support_material_steel_layers_simple",
    "support_material_timber_simple",
    *_LAZY_EXPORTS.keys(),
]


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    try:
        module = import_module(module_name)
        value = getattr(module, attr_name)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("skfem"):
            raise ModuleNotFoundError(
                f"'{name}' requiere dependencias FE opcionales. "
                "Instala 'scikit-fem' o importa solo submódulos livianos "
                "como anchorplate.model / anchorplate.support."
            ) from exc
        raise

    globals()[name] = value
    return value
