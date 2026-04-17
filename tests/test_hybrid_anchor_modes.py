from __future__ import annotations

import numpy as np

from anchorplate.benchmark_material import _foundation_total_reaction
from anchorplate.model import AnalysisOptions, CoupledLineLoad, FoundationPatch, PointSupport, SteelPlate
from anchorplate.solver import solve_anchor_plate


PLATE = SteelPlate(length_mm=300.0, width_mm=300.0, thickness_mm=15.0)
OPTIONS = AnalysisOptions(
    target_h_mm=12.0,
    output_dir="outputs/tests",
    save_plots=False,
    save_result_npz=False,
    save_3d_plots=False,
)


def _corner_anchors(kind: str) -> list[PointSupport]:
    return [
        PointSupport(30.0, 30.0, kind=kind, kz_n_per_mm=150_000.0, label="A1"),
        PointSupport(270.0, 30.0, kind=kind, kz_n_per_mm=150_000.0, label="A2"),
        PointSupport(30.0, 270.0, kind=kind, kz_n_per_mm=150_000.0, label="A3"),
        PointSupport(270.0, 270.0, kind=kind, kz_n_per_mm=150_000.0, label="A4"),
    ]


def _hybrid_foundation_patch() -> list[FoundationPatch]:
    return [
        FoundationPatch(
            x_min_mm=0.0,
            x_max_mm=300.0,
            y_min_mm=0.0,
            y_max_mm=300.0,
            k_area_n_per_mm3=20.0,
            compression_only=True,
            label="full_patch",
        )
    ]


def _sum_total_reaction_n(result, patches: list[FoundationPatch]) -> float:
    spring_total = float(np.sum(result.support_reactions_n))
    foundation_total = _foundation_total_reaction(result, patches)
    return spring_total + foundation_total


def test_hybrid_tension_only_avoids_fictitious_anchor_compression() -> None:
    """
    Physical regression case:
    Hybrid support (anchors + compression-only foundation) under Fz+Mx creates
    mixed response with partial uplift. Linear springs can show negative anchor
    reactions (fictitious anchor compression), while tension-only anchors must
    unload those supports and keep only tensile anchor reactions.
    """
    patches = _hybrid_foundation_patch()
    load = CoupledLineLoad(
        ref_x_mm=150.0,
        ref_y_mm=150.0,
        force_n=40_000.0,
        mx_nmm=4.0e6,
        label="Fz_plus_Mx",
    )

    spring_result = solve_anchor_plate(
        plate=PLATE,
        supports=_corner_anchors("spring"),
        coupled_loads=[load],
        foundation_patches=patches,
        options=OPTIONS,
        name="hybrid_spring",
    )
    tension_only_result = solve_anchor_plate(
        plate=PLATE,
        supports=_corner_anchors("spring_tension_only"),
        coupled_loads=[load],
        foundation_patches=patches,
        options=OPTIONS,
        name="hybrid_spring_tension_only",
    )

    # Linear spring model can develop compressive (negative) anchor reactions.
    assert np.any(spring_result.support_reactions_n < -100.0)

    # Tension-only model must not keep anchors active in compression.
    active_reactions = tension_only_result.support_reactions_n[tension_only_result.support_active]
    assert active_reactions.size > 0
    assert np.min(active_reactions) >= -1e-6

    # Mixed-load uplift should deactivate at least one anchor in tension-only mode.
    assert int(np.sum(tension_only_result.support_active)) < len(tension_only_result.support_active)

    # Global equilibrium must still close for both models (springs + foundation).
    tol_n = max(100.0, 1e-3 * abs(load.force_n))
    assert abs(_sum_total_reaction_n(spring_result, patches) - load.force_n) <= tol_n
    assert abs(_sum_total_reaction_n(tension_only_result, patches) - load.force_n) <= tol_n
