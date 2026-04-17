"""
test_foundation_contact.py
==========================
Verification tests for the compression-only foundation patch submodel.

Covers
------
- active/inactive mask correctness and complementarity
- physical consistency (w_active > 0, w_inactive <= 0)
- active | inactive == in_patch (no gaps)
- npz export round-trip
- contact summary statistics
- convergence sanity
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pytest

from anchorplate.model import (
    AnalysisOptions,
    CoupledLineLoad,
    FoundationPatch,
    PointSupport,
    SteelPlate,
)
from anchorplate.plotting import _foundation_masks, _contact_summary, export_result_npz
from anchorplate.solver import solve_anchor_plate
from anchorplate.support import bedding_concrete_simple, bedding_timber_simple


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

PLATE = SteelPlate(length_mm=300.0, width_mm=300.0, thickness_mm=15.0)

OPTS = AnalysisOptions(
    target_h_mm=10.0,          # coarser for speed in tests
    output_dir="outputs/tests",
    save_plots=False,
    save_result_npz=False,
    save_3d_plots=False,
)

SPRINGS = [
    PointSupport(30.0,  30.0,  kind="spring", kz_n_per_mm=150_000.0, label="A1"),
    PointSupport(270.0, 30.0,  kind="spring", kz_n_per_mm=150_000.0, label="A2"),
    PointSupport(30.0,  270.0, kind="spring", kz_n_per_mm=150_000.0, label="A3"),
    PointSupport(270.0, 270.0, kind="spring", kz_n_per_mm=150_000.0, label="A4"),
]

K_CONCRETE = bedding_concrete_simple(e_cm_mpa=32_000.0, h_eff_mm=200.0)
K_TIMBER   = bedding_timber_simple(e90_mpa=370.0, h_eff_mm=100.0, spread_factor=1.0)

# Eccentric load → large moment → guaranteed lift-off in at least one zone
LOAD_LIFTOFF = CoupledLineLoad(
    ref_x_mm=150.0, ref_y_mm=150.0,
    force_n=40_000.0, mx_nmm=3.0e6,
    line_spacing_mm=150.0, line_length_mm=100.0, label="RP",
)

# Centric load → symmetric → mostly in contact
LOAD_CENTRIC = CoupledLineLoad(
    ref_x_mm=150.0, ref_y_mm=150.0,
    force_n=40_000.0, mx_nmm=0.0,
    line_spacing_mm=150.0, line_length_mm=100.0, label="RP",
)

FOUNDATION_TWO_ZONE = [
    FoundationPatch(0.0, 300.0, 0.0,   150.0, k_area_n_per_mm3=K_CONCRETE,
                    compression_only=True, label="concrete"),
    FoundationPatch(0.0, 300.0, 150.0, 300.0, k_area_n_per_mm3=K_TIMBER,
                    compression_only=True, label="timber"),
]

FOUNDATION_FULL = [
    FoundationPatch(0.0, 300.0, 0.0, 300.0, k_area_n_per_mm3=K_CONCRETE,
                    compression_only=True, label="full"),
]


def _solve_liftoff():
    return solve_anchor_plate(
        plate=PLATE, supports=SPRINGS, coupled_loads=[LOAD_LIFTOFF],
        options=OPTS, foundation_patches=FOUNDATION_TWO_ZONE,
        name="fp_liftoff",
    )


def _solve_centric():
    return solve_anchor_plate(
        plate=PLATE, supports=SPRINGS, coupled_loads=[LOAD_CENTRIC],
        options=OPTS, foundation_patches=FOUNDATION_FULL,
        name="fp_centric",
    )


# ---------------------------------------------------------------------------
# Tests: FoundationState content
# ---------------------------------------------------------------------------

class TestFoundationState:
    def setup_method(self):
        self.result = _solve_liftoff()

    def test_all_patch_vertices_populated(self):
        """all_patch_vertices must be present and have one set per patch."""
        fs = self.result.foundation_state
        assert len(fs.all_patch_vertices) == len(FOUNDATION_TWO_ZONE), (
            "all_patch_vertices must have one entry per foundation patch"
        )

    def test_all_patch_vertices_superset_of_active(self):
        """Total patch sets must contain all active vertices."""
        fs = self.result.foundation_state
        for i, (total, active) in enumerate(zip(fs.all_patch_vertices, fs.active_vertices)):
            assert active.issubset(total), (
                f"Patch {i}: active set has vertices not in all_patch_vertices"
            )

    def test_converged(self):
        """Active-set iteration must have converged (last delta == 0)."""
        fs = self.result.foundation_state
        assert fs.history_changes, "history_changes is empty"
        assert fs.history_changes[-1] == 0, (
            f"Did not converge: last delta = {fs.history_changes[-1]}"
        )


# ---------------------------------------------------------------------------
# Tests: _foundation_masks — correctness and complementarity
# ---------------------------------------------------------------------------

class TestFoundationMasks:
    def setup_method(self):
        self.result = _solve_liftoff()
        self.active, self.inactive = _foundation_masks(self.result)
        self.w = self.result.w_vertex_mm
        tol = OPTS.foundation_contact_tol_mm

    def test_inactive_mask_not_all_zero(self):
        """Bug was: inactive_mask always zero. Must be non-trivial."""
        assert self.inactive is not None, "inactive mask is None"
        assert self.inactive.any(), (
            "inactive_foundation_mask is all-zero — lift-off not detected"
        )

    def test_active_mask_not_all_zero(self):
        assert self.active.any(), "active_foundation_mask is all-zero"

    def test_masks_disjoint(self):
        """active AND inactive must not overlap."""
        assert self.inactive is not None
        assert not np.any(self.active & self.inactive), (
            "active and inactive masks overlap — a node cannot be both"
        )

    def test_masks_complementary_within_patch(self):
        """active | inactive must equal in_patch (no gaps, no extras)."""
        assert self.inactive is not None
        fs = self.result.foundation_state
        all_ids = sorted(set().union(*fs.all_patch_vertices))
        in_patch = np.zeros(self.result.mesh.p.shape[1], dtype=bool)
        in_patch[np.array(all_ids, dtype=int)] = True

        union = self.active | self.inactive
        assert np.array_equal(union, in_patch), (
            "active | inactive != in_patch — masks have gaps or extras"
        )

    def test_active_nodes_have_positive_w(self):
        """All active nodes must have w > 0 (in contact with foundation)."""
        if self.active.any():
            w_active = self.w[self.active]
            assert w_active.min() > 0.0, (
                f"Active node with w <= 0: min(w_active) = {w_active.min():.6f} mm"
            )

    def test_inactive_nodes_have_nonpositive_w(self):
        """All inactive nodes must have w <= 0 (lifted away from foundation)."""
        assert self.inactive is not None
        if self.inactive.any():
            w_inactive = self.w[self.inactive]
            # The contact tolerance used by the solver is 1e-10; allow a small margin
            assert w_inactive.max() <= OPTS.foundation_contact_tol_mm * 10, (
                f"Inactive node with w > 0: max(w_inactive) = {w_inactive.max():.6e} mm"
            )

    def test_no_foundation_case_returns_none_inactive(self):
        """Without foundation patches, inactive mask must be None."""
        result_no_fp = solve_anchor_plate(
            plate=PLATE, supports=SPRINGS, coupled_loads=[LOAD_CENTRIC],
            options=OPTS, name="no_fp",
        )
        active, inactive = _foundation_masks(result_no_fp)
        assert not active.any(), "active mask should be empty without foundation"
        assert inactive is None, "inactive mask should be None without foundation"


# ---------------------------------------------------------------------------
# Tests: contact summary statistics
# ---------------------------------------------------------------------------

class TestContactSummary:
    def setup_method(self):
        self.result = _solve_liftoff()
        active, inactive = _foundation_masks(self.result)
        self.summary = _contact_summary(self.result, active, inactive)

    def test_counts_consistent(self):
        s = self.summary
        assert s["n_active"] + s["n_inactive"] == s["n_patch_total"]

    def test_percentages_sum_to_100(self):
        s = self.summary
        assert abs(s["pct_active"] + s["pct_inactive"] - 100.0) < 0.1, (
            f"pct_active + pct_inactive = {s['pct_active'] + s['pct_inactive']:.2f}"
        )

    def test_liftoff_present(self):
        """For the eccentric load, lift-off must be non-zero."""
        assert self.summary["n_inactive"] > 0, "Expected lift-off nodes"

    def test_w_ranges_physical(self):
        s = self.summary
        assert s["w_active_min"] > 0.0, "Active w_min should be > 0"
        assert s["w_inactive_max"] <= 0.0, "Inactive w_max should be <= 0"

    def test_converged_flag(self):
        assert self.summary["converged"] is True


# ---------------------------------------------------------------------------
# Tests: npz export round-trip
# ---------------------------------------------------------------------------

class TestNpzExport:
    def setup_method(self):
        self.result = _solve_liftoff()

    def test_export_loads_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, summary = export_result_npz(self.result, Path(tmp) / "out.npz")
            assert path.exists()
            d = np.load(path)
            required = [
                "x_mm", "y_mm", "triangles", "w_mm", "sigma_vm_mpa",
                "support_vertex_ids", "support_reactions_n",
                "active_foundation_mask", "inactive_foundation_mask",
                "in_patch_mask", "foundation_history",
            ]
            for key in required:
                assert key in d, f"Missing key in npz: {key}"

    def test_masks_complementary_from_npz(self):
        """Round-tripped active | inactive must equal in_patch."""
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = export_result_npz(self.result, Path(tmp) / "out.npz")
            d = np.load(path)
            active   = d["active_foundation_mask"].astype(bool)
            inactive = d["inactive_foundation_mask"].astype(bool)
            in_patch = d["in_patch_mask"].astype(bool)
            assert np.array_equal(active | inactive, in_patch)
            assert not np.any(active & inactive)

    def test_w_sign_consistent_in_npz(self):
        """Active nodes must have w > 0, inactive nodes w <= 0 after round-trip."""
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = export_result_npz(self.result, Path(tmp) / "out.npz")
            d = np.load(path)
            active   = d["active_foundation_mask"].astype(bool)
            inactive = d["inactive_foundation_mask"].astype(bool)
            w = d["w_mm"]
            if active.any():
                assert w[active].min() > 0.0
            if inactive.any():
                assert w[inactive].max() <= OPTS.foundation_contact_tol_mm * 10

    def test_inactive_nonzero_in_npz(self):
        """Regression: inactive mask must not be all-zero in the exported file."""
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = export_result_npz(self.result, Path(tmp) / "out.npz")
            d = np.load(path)
            inactive = d["inactive_foundation_mask"].astype(bool)
            assert inactive.any(), (
                "inactive_foundation_mask is all-zero in .npz — regression of the original bug"
            )

    def test_summary_returned(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, summary = export_result_npz(self.result, Path(tmp) / "out.npz")
            assert summary["n_patch_total"] > 0
            assert summary["converged"] is True
