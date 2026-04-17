from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Sequence

from .model import ConcreteAdvancedInput, SteelLayer


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
