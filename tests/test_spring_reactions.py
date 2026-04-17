"""
test_spring_reactions.py
========================
Verification tests for spring-type PointSupport reaction extraction.

Bug fixed (2026-04-17): reactions were computed as:
    residual = k_total @ solution - rhs
    reactions = -residual[support_dofs]
For spring DOFs (free DOFs), K·u = f exactly after the solve, so residual = 0 and
reactions came out as zero regardless of the load. The fix is:
    spring_reactions = (k_springs @ solution)[support_dofs]   # R = kz · w
    fixed_reactions  = -residual[support_dofs]                # valid only for condensed DOFs
    support_reactions = spring_reactions + fixed_reactions
"""

from __future__ import annotations

import numpy as np
import pytest

from anchorplate.model import AnalysisOptions, CoupledLineLoad, PointSupport, SteelPlate
from anchorplate.solver import solve_anchor_plate


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

PLATE = SteelPlate(length_mm=300.0, width_mm=300.0, thickness_mm=15.0)
KZ_STIFF = 150_000.0  # N/mm  — "production" spring stiffness

OPTS = AnalysisOptions(
    target_h_mm=8.0,
    output_dir="outputs/tests",
    save_plots=False,
    save_result_npz=False,
    save_3d_plots=False,
)

FZ = 50_000.0  # N — centric transverse load
MX = 8.0e6    # N·mm — pure bending moment about x


def _springs(kz: float = KZ_STIFF) -> list[PointSupport]:
    return [
        PointSupport(30.0,  30.0,  kind="spring", kz_n_per_mm=kz, label="A1"),
        PointSupport(270.0, 30.0,  kind="spring", kz_n_per_mm=kz, label="A2"),
        PointSupport(30.0,  270.0, kind="spring", kz_n_per_mm=kz, label="A3"),
        PointSupport(270.0, 270.0, kind="spring", kz_n_per_mm=kz, label="A4"),
    ]


def _fixed() -> list[PointSupport]:
    return [
        PointSupport(30.0,  30.0,  kind="fixed", label="A1"),
        PointSupport(270.0, 30.0,  kind="fixed", label="A2"),
        PointSupport(30.0,  270.0, kind="fixed", label="A3"),
        PointSupport(270.0, 270.0, kind="fixed", label="A4"),
    ]


def _load_fz() -> CoupledLineLoad:
    return CoupledLineLoad(
        ref_x_mm=150.0, ref_y_mm=150.0,
        force_n=FZ, mx_nmm=0.0,
        line_spacing_mm=150.0, line_length_mm=100.0,
        label="Fz_centric",
    )


def _load_mx() -> CoupledLineLoad:
    return CoupledLineLoad(
        ref_x_mm=150.0, ref_y_mm=150.0,
        force_n=0.0, mx_nmm=MX,
        line_spacing_mm=150.0, line_length_mm=100.0,
        label="Mx_pure",
    )


# ---------------------------------------------------------------------------
# Case 1: Centric Fz — 4 identical springs
# ---------------------------------------------------------------------------

class TestCase1CentricFz:
    """Centric Fz with 4 identical symmetric springs."""

    def setup_method(self):
        self.result = solve_anchor_plate(
            plate=PLATE,
            supports=_springs(KZ_STIFF),
            coupled_loads=[_load_fz()],
            options=OPTS,
            name="c1_centric_fz",
        )

    def test_global_equilibrium(self):
        """Sum of spring reactions must equal applied Fz."""
        sum_R = np.sum(self.result.support_reactions_n)
        assert abs(sum_R - FZ) < FZ * 1e-4, (
            f"Equilibrium error: sum_R={sum_R:.1f} N, expected {FZ:.1f} N"
        )

    def test_reactions_nonzero(self):
        """Each reaction must be significantly non-zero (bug was: all ≈ 0)."""
        expected_each = FZ / 4
        for i, R in enumerate(self.result.support_reactions_n):
            assert R > expected_each * 0.5, (
                f"Reaction A{i+1}={R:.2f} N is unexpectedly small (bug?)"
            )

    def test_symmetric_reactions(self):
        """For a symmetric geometry+load, all 4 reactions must be equal."""
        R = self.result.support_reactions_n
        assert np.max(R) - np.min(R) < FZ * 1e-3, (
            f"Asymmetric reactions: {R} N — expected all equal"
        )

    def test_reactions_equal_kz_times_w(self):
        """R_i = kz * w_i must hold for every spring support."""
        R = self.result.support_reactions_n
        w_supports = self.result.w_vertex_mm[self.result.support_vertex_ids]
        for i, (Ri, wi) in enumerate(zip(R, w_supports)):
            expected = KZ_STIFF * wi
            assert abs(Ri - expected) < abs(expected) * 1e-6, (
                f"A{i+1}: R={Ri:.4f} N, kz*w={expected:.4f} N — mismatch"
            )

    def test_deflection_positive(self):
        """Plate must deflect downward (w > 0) under downward load."""
        assert self.result.max_deflection_mm > 0.0


# ---------------------------------------------------------------------------
# Case 2: Pure Mx — spring reactions must form a moment couple
# ---------------------------------------------------------------------------

class TestCase2PureMx:
    """Pure bending Mx with 4 identical springs."""

    def setup_method(self):
        self.result = solve_anchor_plate(
            plate=PLATE,
            supports=_springs(KZ_STIFF),
            coupled_loads=[_load_mx()],
            options=OPTS,
            name="c2_pure_mx",
        )

    def test_global_equilibrium_fz_zero(self):
        """Sum of reactions must be ~0 for pure moment (no net vertical force)."""
        sum_R = np.sum(self.result.support_reactions_n)
        assert abs(sum_R) < FZ * 1e-4, (
            f"Sum of reactions={sum_R:.4f} N — should be ~0 for pure Mx"
        )

    def test_moment_couple_sign(self):
        """
        A1, A2 (y=30)  and A3, A4 (y=270) should have opposite signs.
        Mx > 0 rotates about x → compression on y-high side, tension on y-low side.
        """
        R = self.result.support_reactions_n
        # A1, A2 are at y=30 (low); A3, A4 at y=270 (high)
        R_low  = R[0] + R[1]   # A1 + A2
        R_high = R[2] + R[3]   # A3 + A4
        assert R_low * R_high < 0, (
            f"Expected opposite signs for low/high groups: R_low={R_low:.1f}, R_high={R_high:.1f}"
        )

    def test_reactions_nonzero(self):
        """Reactions must not be zero — this was the bug."""
        R = self.result.support_reactions_n
        assert np.max(np.abs(R)) > FZ * 0.1, (
            f"Reactions are suspiciously small: {R}"
        )


# ---------------------------------------------------------------------------
# Case 3: kz sweep — varying kz must change w_max monotonically
# ---------------------------------------------------------------------------

class TestCase3KzSweep:
    """Sweep kz over 3 orders of magnitude; w_max must decrease monotonically."""

    KZ_VALUES = [100.0, 5_000.0, 150_000.0, 5_000_000.0]

    def setup_method(self):
        self.results = []
        for kz in self.KZ_VALUES:
            r = solve_anchor_plate(
                plate=PLATE,
                supports=_springs(kz),
                coupled_loads=[_load_fz()],
                options=OPTS,
                name=f"c3_kz_{kz:.0f}",
            )
            self.results.append(r)

    def test_equilibrium_all_kz(self):
        """Sum of reactions == Fz for every kz value."""
        for kz, r in zip(self.KZ_VALUES, self.results):
            sum_R = np.sum(r.support_reactions_n)
            assert abs(sum_R - FZ) < FZ * 1e-4, (
                f"kz={kz}: sum_R={sum_R:.1f} N != {FZ:.1f} N"
            )

    def test_wmax_decreases_with_kz(self):
        """Higher kz → stiffer → smaller w_max."""
        wmax = [r.max_deflection_mm for r in self.results]
        for i in range(len(wmax) - 1):
            assert wmax[i] > wmax[i + 1], (
                f"w_max not decreasing: kz={self.KZ_VALUES[i]} → {wmax[i]:.4f} mm, "
                f"kz={self.KZ_VALUES[i+1]} → {wmax[i+1]:.4f} mm"
            )

    def test_reactions_scale_with_w(self):
        """For every kz: R_i = kz * w_i must hold (physics check)."""
        for kz, r in zip(self.KZ_VALUES, self.results):
            w_sup = r.w_vertex_mm[r.support_vertex_ids]
            for i, (Ri, wi) in enumerate(zip(r.support_reactions_n, w_sup)):
                expected = kz * wi
                assert abs(Ri - expected) < abs(expected) * 1e-6, (
                    f"kz={kz}, A{i+1}: R={Ri:.4f} N vs kz*w={expected:.4f} N"
                )


# ---------------------------------------------------------------------------
# Case 4: Very stiff spring → must converge to fixed-support result
# ---------------------------------------------------------------------------

class TestCase4SpringToFixed:
    """Spring with very large kz must approximate the fixed-support solution."""

    KZ_VERY_STIFF = 50_000_000.0  # N/mm — ~333x stiffer than test default

    def setup_method(self):
        self.res_spring = solve_anchor_plate(
            plate=PLATE,
            supports=_springs(self.KZ_VERY_STIFF),
            coupled_loads=[_load_fz()],
            options=OPTS,
            name="c4_stiff_spring",
        )
        self.res_fixed = solve_anchor_plate(
            plate=PLATE,
            supports=_fixed(),
            coupled_loads=[_load_fz()],
            options=OPTS,
            name="c4_fixed",
        )

    def test_equilibrium_spring(self):
        sum_R = np.sum(self.res_spring.support_reactions_n)
        assert abs(sum_R - FZ) < FZ * 1e-4

    def test_equilibrium_fixed(self):
        sum_R = np.sum(self.res_fixed.support_reactions_n)
        assert abs(sum_R - FZ) < FZ * 1e-4

    def test_wmax_converges(self):
        """w_max of very-stiff spring must be within 2% of fixed case."""
        wf = self.res_fixed.max_deflection_mm
        ws = self.res_spring.max_deflection_mm
        # spring w_max > fixed w_max (spring adds rigid-body translation),
        # but the difference must be small relative to wf
        assert abs(ws - wf) / wf < 0.02, (
            f"w_max: spring={ws:.4f} mm, fixed={wf:.4f} mm — gap > 2%"
        )

    def test_sigma_vm_converges(self):
        """Max von Mises must match to within 2%."""
        sf = self.res_fixed.max_von_mises_mpa
        ss = self.res_spring.max_von_mises_mpa
        assert abs(ss - sf) / sf < 0.02, (
            f"sigma_vm: spring={ss:.2f} MPa, fixed={sf:.2f} MPa — gap > 2%"
        )

    def test_reactions_converge(self):
        """Reaction distribution must match to within 1% of Fz."""
        Rf = np.sort(self.res_fixed.support_reactions_n)
        Rs = np.sort(self.res_spring.support_reactions_n)
        max_diff = np.max(np.abs(Rf - Rs))
        assert max_diff < FZ * 0.01, (
            f"Max reaction difference = {max_diff:.2f} N (> 1% of Fz)"
        )

    def test_fixed_not_broken(self):
        """Sanity: fixed reactions must still sum to Fz after the bugfix."""
        sum_R = np.sum(self.res_fixed.support_reactions_n)
        assert abs(sum_R - FZ) < FZ * 1e-4
