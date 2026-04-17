from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Sequence

from .model import ConcreteAdvancedInput, SteelLayer


@dataclass(frozen=True)
class SupportMaterialModelResult:
    """Normalized/traceable result of a bedding/support material model."""
    k_area_n_per_mm3: float
    model_name: str
    parameters: dict[str, Any]
    notes: str = ""


def bedding_concrete_simple(e_cm_mpa: float, h_eff_mm: float) -> float:
    if h_eff_mm <= 0.0:
        raise ValueError("h_eff_mm must be > 0")
    return float(e_cm_mpa) / float(h_eff_mm)


def bedding_timber_simple(e90_mpa: float, h_eff_mm: float, spread_factor: float = 1.0) -> float:
    if h_eff_mm <= 0.0:
        raise ValueError("h_eff_mm must be > 0")
    if spread_factor <= 0.0:
        raise ValueError("spread_factor must be > 0")
    return float(spread_factor) * float(e90_mpa) / float(h_eff_mm)


def bedding_steel_layers(layers: Sequence[SteelLayer]) -> float:
    if not layers:
        raise ValueError("At least one steel layer is required")
    compliance = 0.0
    for layer in layers:
        if layer.thickness_mm <= 0.0 or layer.youngs_modulus_mpa <= 0.0:
            raise ValueError("Invalid steel layer properties")
        compliance += layer.thickness_mm / layer.youngs_modulus_mpa
    if compliance <= 0.0:
        raise ValueError("Compliance must be positive")
    return 1.0 / compliance


def bedding_concrete_advanced(inp: ConcreteAdvancedInput) -> float:
    if inp.a_eff_mm2 <= 0.0 or inp.a_ref_mm2 <= 0.0 or inp.h_block_mm <= 0.0 or inp.d_plate_mm <= 0.0:
        raise ValueError("Areas, block height and plate width must be positive")
    a1 = 1.65
    a2 = 0.5
    a3 = 0.3
    a4 = 1.0
    area_factor = sqrt(inp.a_eff_mm2 / inp.a_ref_mm2)
    geom = 1.0 / (inp.h_block_mm / (a2 * inp.d_plate_mm) + a3) + a4
    return inp.e_cm_mpa / ((a1 + inp.nu) * area_factor) * geom


def bedding_calibrated(k_area_n_per_mm3: float) -> float:
    if k_area_n_per_mm3 <= 0.0:
        raise ValueError("k_area_n_per_mm3 must be > 0")
    return float(k_area_n_per_mm3)


def bedding_nodal_from_area(k_area_n_per_mm3: float, tributary_area_mm2: float) -> float:
    if k_area_n_per_mm3 < 0.0 or tributary_area_mm2 < 0.0:
        raise ValueError("Inputs must be non-negative")
    return float(k_area_n_per_mm3) * float(tributary_area_mm2)


def support_material_concrete_simple(e_cm_mpa: float, h_eff_mm: float) -> SupportMaterialModelResult:
    """Wrapper API for bedding_concrete_simple with explicit metadata."""
    return SupportMaterialModelResult(
        k_area_n_per_mm3=bedding_concrete_simple(e_cm_mpa=e_cm_mpa, h_eff_mm=h_eff_mm),
        model_name="concrete_simple",
        parameters={
            "e_cm_mpa": float(e_cm_mpa),
            "h_eff_mm": float(h_eff_mm),
        },
        notes="Simple linear estimate k = E_cm / h_eff.",
    )


def support_material_concrete_advanced(inp: ConcreteAdvancedInput) -> SupportMaterialModelResult:
    """Wrapper API for bedding_concrete_advanced with explicit metadata."""
    return SupportMaterialModelResult(
        k_area_n_per_mm3=bedding_concrete_advanced(inp),
        model_name="concrete_advanced",
        parameters={
            "e_cm_mpa": float(inp.e_cm_mpa),
            "nu": float(inp.nu),
            "a_eff_mm2": float(inp.a_eff_mm2),
            "a_ref_mm2": float(inp.a_ref_mm2),
            "h_block_mm": float(inp.h_block_mm),
            "d_plate_mm": float(inp.d_plate_mm),
        },
        notes="Advanced geometry/area-corrected concrete model used in legacy helper.",
    )


def support_material_timber_simple(
    e90_mpa: float,
    h_eff_mm: float,
    spread_factor: float = 1.0,
) -> SupportMaterialModelResult:
    """Wrapper API for bedding_timber_simple with explicit metadata."""
    return SupportMaterialModelResult(
        k_area_n_per_mm3=bedding_timber_simple(
            e90_mpa=e90_mpa,
            h_eff_mm=h_eff_mm,
            spread_factor=spread_factor,
        ),
        model_name="timber_simple",
        parameters={
            "e90_mpa": float(e90_mpa),
            "h_eff_mm": float(h_eff_mm),
            "spread_factor": float(spread_factor),
        },
        notes="Simple linear estimate k = spread_factor * E90 / h_eff.",
    )


def support_material_steel_layers_simple(layers: Sequence[SteelLayer]) -> SupportMaterialModelResult:
    """Wrapper API for bedding_steel_layers with explicit metadata."""
    return SupportMaterialModelResult(
        k_area_n_per_mm3=bedding_steel_layers(layers),
        model_name="steel_layers_simple",
        parameters={
            "layers": [
                {
                    "thickness_mm": float(layer.thickness_mm),
                    "youngs_modulus_mpa": float(layer.youngs_modulus_mpa),
                }
                for layer in layers
            ],
        },
        notes="Series-compliance model (1/k = Σ(t_i/E_i)) for stacked layers.",
    )


def support_material_calibrated(k_area_n_per_mm3: float) -> SupportMaterialModelResult:
    """Wrapper API for bedding_calibrated with explicit metadata."""
    return SupportMaterialModelResult(
        k_area_n_per_mm3=bedding_calibrated(k_area_n_per_mm3),
        model_name="calibrated",
        parameters={
            "k_area_n_per_mm3": float(k_area_n_per_mm3),
        },
        notes="Direct user-calibrated stiffness, no constitutive back-calculation.",
    )
