from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np


@dataclass(frozen=True)
class SteelPlate:
    length_mm: float
    width_mm: float
    thickness_mm: float
    youngs_modulus_mpa: float = 210000.0
    poisson: float = 0.30
    fy_d_mpa: float = 355.0

    @property
    def rigidity_d_nmm(self) -> float:
        return (
            self.youngs_modulus_mpa
            * self.thickness_mm**3
            / (12.0 * (1.0 - self.poisson**2))
        )


@dataclass(frozen=True)
class PointSupport:
    x_mm: float
    y_mm: float
    kind: Literal["fixed", "spring", "spring_tension_only"] = "fixed"
    kz_n_per_mm: float = 0.0
    label: str = ""


@dataclass(frozen=True)
class PointLoad:
    x_mm: float
    y_mm: float
    force_n: float
    label: str = ""


@dataclass(frozen=True)
class CoupledLineLoad:
    ref_x_mm: float
    ref_y_mm: float
    force_n: float
    mx_nmm: float = 0.0
    my_nmm: float = 0.0
    line_spacing_mm: float = 150.0
    line_length_mm: float = 100.0
    orientation: Literal["vertical", "horizontal"] = "vertical"
    label: str = "RP"


@dataclass(frozen=True)
class MeshRefinementBox:
    x_min_mm: float
    x_max_mm: float
    y_min_mm: float
    y_max_mm: float
    h_mm: float
    n_div_min: int = 2
    label: str = ""


@dataclass(frozen=True)
class FoundationPatch:
    x_min_mm: float
    x_max_mm: float
    y_min_mm: float
    y_max_mm: float
    k_area_n_per_mm3: float
    compression_only: bool = True
    label: str = ""


@dataclass(frozen=True)
class SteelLayer:
    thickness_mm: float
    youngs_modulus_mpa: float = 210000.0


@dataclass(frozen=True)
class ConcreteAdvancedInput:
    e_cm_mpa: float
    nu: float
    a_eff_mm2: float
    a_ref_mm2: float
    h_block_mm: float
    d_plate_mm: float


@dataclass(frozen=True)
class AnalysisOptions:
    target_h_mm: float = 8.0
    output_dir: str = "outputs"
    save_plots: bool = True
    show_plots: bool = False
    save_result_npz: bool = True
    save_3d_plots: bool = True
    z_plot_scale: float = 1.0
    foundation_iterations_max: int = 30
    foundation_contact_tol_mm: float = 1e-10
    line_pick_tol_mm: float = 1e-9


@dataclass
class FoundationState:
    active_vertices: list[set[int]] = field(default_factory=list)
    history_changes: list[int] = field(default_factory=list)
    # Full vertex set for every patch (active ∪ inactive). Populated by the solver.
    # Needed to reconstruct which nodes are in the patch domain but NOT in contact.
    all_patch_vertices: list[set[int]] = field(default_factory=list)


@dataclass
class AnchorSpringState:
    """
    State of discrete anchor springs for nonlinear support models.

    Sign convention
    ---------------
    The solver uses w [mm] positive downward.
    For `spring_tension_only`, spring extension (tension) is defined as:
        delta_tension = w
    Therefore:
      - active (tension):   w > +tol_mm
      - inactive (compression / no tension): w <= -tol_mm
      - in (-tol_mm, +tol_mm), previous active-set state is kept (hysteresis).
    """

    active: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    history_changes: list[int] = field(default_factory=list)
    tension_only_indices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
