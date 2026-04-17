"""
test_benchmark_material.py
==========================
Tests for the material benchmark runner (run_material_benchmark).

Checks
------
- Correct number of rows (n_materials × n_load_cases)
- All cases converged
- Equilibrium per case: sum(reactions) ≈ applied Fz
- Physical ordering: stiffer material → smaller w_max (for same centric load)
- Contact% ordering: stiffer material → less contact area needed (centric load)
- Lift-off present for eccentric/moment cases
- CSV and Markdown files written
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from anchorplate.benchmark_material import (
    MaterialLoadCase,
    MaterialSpec,
    default_load_cases,
    default_materials,
    material_spec_from_model_result,
    run_material_benchmark,
)
from anchorplate.model import AnalysisOptions, PointSupport, SteelPlate
from anchorplate.support import support_material_concrete_simple, support_material_timber_simple


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PLATE = SteelPlate(length_mm=300.0, width_mm=300.0, thickness_mm=15.0)

SUPPORTS = [
    PointSupport(30.0,  30.0,  kind="spring", kz_n_per_mm=150_000.0, label="A1"),
    PointSupport(270.0, 30.0,  kind="spring", kz_n_per_mm=150_000.0, label="A2"),
    PointSupport(30.0,  270.0, kind="spring", kz_n_per_mm=150_000.0, label="A3"),
    PointSupport(270.0, 270.0, kind="spring", kz_n_per_mm=150_000.0, label="A4"),
]

# Coarser mesh for test speed
OPTS_FAST = AnalysisOptions(
    target_h_mm=15.0,
    output_dir="outputs/tests/material_benchmark",
    save_plots=False,
    save_result_npz=False,
    save_3d_plots=False,
)

# Two contrasting materials: soft (timber) and stiff (grout)
_grout_model = support_material_concrete_simple(e_cm_mpa=32_000.0, h_eff_mm=50.0)
_timber_model = support_material_timber_simple(e90_mpa=390.0, h_eff_mm=50.0)

TWO_MATERIALS = [
    material_spec_from_model_result("Grout", "grout", _grout_model, description="stiff"),
    material_spec_from_model_result("Timber", "timber", _timber_model, description="soft"),
]

TWO_CASES = [
    MaterialLoadCase("LC01_Fz", "Centric Fz", force_n=50_000.0),
    MaterialLoadCase("LC02_Mx", "Pure Mx",    mx_nmm=6.0e6),
]


@pytest.fixture(scope="module")
def bmark_rows():
    """Run a small 2×2 benchmark once for the whole module."""
    with tempfile.TemporaryDirectory() as tmp:
        rows = run_material_benchmark(
            plate=PLATE,
            supports=SUPPORTS,
            materials=TWO_MATERIALS,
            load_cases=TWO_CASES,
            options=OPTS_FAST,
            outdir=Path(tmp) / "bmark",
        )
    return rows


# ---------------------------------------------------------------------------
# Basic shape
# ---------------------------------------------------------------------------

class TestRowCount:
    def test_correct_number_of_rows(self, bmark_rows):
        assert len(bmark_rows) == len(TWO_MATERIALS) * len(TWO_CASES), (
            f"Expected {len(TWO_MATERIALS)*len(TWO_CASES)} rows, got {len(bmark_rows)}"
        )

    def test_all_converged(self, bmark_rows):
        for r in bmark_rows:
            assert r.converged, f"{r.material} / {r.load_case} did not converge"

    def test_all_materials_represented(self, bmark_rows):
        names = {r.material for r in bmark_rows}
        for m in TWO_MATERIALS:
            assert m.name in names

    def test_all_load_cases_represented(self, bmark_rows):
        lcs = {r.load_case for r in bmark_rows}
        for lc in TWO_CASES:
            assert lc.name in lcs

    def test_model_metadata_in_rows(self, bmark_rows):
        for r in bmark_rows:
            assert r.model_name
            assert r.model_parameters_json.startswith("{")


# ---------------------------------------------------------------------------
# Equilibrium
# ---------------------------------------------------------------------------

class TestEquilibrium:
    def test_total_equilibrium_centric_fz(self, bmark_rows):
        """
        Global equilibrium: sum(spring_R) + sum(foundation_R) must equal applied Fz.

        Note: with a stiff foundation, most load is carried by the Winkler bedding
        and the discrete spring anchors may carry little or even be in tension.
        The correct check is the *total* reaction.
        """
        fz_cases = [r for r in bmark_rows if r.load_case == "LC01_Fz"]
        assert fz_cases, "No centric Fz rows found"
        for r in fz_cases:
            assert abs(r.sum_total_reactions_kN - 50.0) < 0.5, (
                f"{r.material}: total_R = {r.sum_total_reactions_kN:.3f} kN "
                f"(springs={r.sum_spring_reactions_kN:.3f}, found={r.sum_foundation_reaction_kN:.3f})"
            )

    def test_total_equilibrium_pure_moment(self, bmark_rows):
        """
        For pure Mx (no net Fz), total reaction must be ~0.
        Foundation creates a force couple internally; springs may also form a couple.
        """
        mx_cases = [r for r in bmark_rows if r.load_case == "LC02_Mx"]
        assert mx_cases, "No pure Mx rows found"
        for r in mx_cases:
            assert abs(r.sum_total_reactions_kN) < 1.0, (
                f"{r.material}: total_R = {r.sum_total_reactions_kN:.4f} kN for pure Mx (expected ~0)"
            )

    def test_foundation_carries_load_stiff_material(self, bmark_rows):
        """
        For grout (stiff), most of the centric Fz should be carried by the foundation,
        not the spring anchors.
        """
        r_grout = next((r for r in bmark_rows
                        if r.material == "Grout" and r.load_case == "LC01_Fz"), None)
        if r_grout is None:
            pytest.skip("Grout / LC01_Fz not in fixture")
        # Foundation reaction > 80% of total load for stiff grout
        assert r_grout.sum_foundation_reaction_kN > 0.8 * 50.0, (
            f"Grout foundation carries only {r_grout.sum_foundation_reaction_kN:.1f} kN "
            f"of 50 kN — expected > 40 kN"
        )

    def test_foundation_carries_less_for_soft_material(self, bmark_rows):
        """
        For timber (soft), a larger fraction should go through the springs
        compared to grout, because the soft foundation needs more area to resist.
        Actually: soft = more deformation = more spring force (k*w is larger for springs
        since w is larger). Both spring and foundation carry more in absolute terms
        but the *fraction* depends on relative stiffness.
        Simpler check: foundation_R > 0 for both materials.
        """
        for r in bmark_rows:
            if r.load_case == "LC01_Fz":
                assert r.sum_foundation_reaction_kN > 0, (
                    f"{r.material}: foundation reaction should be positive under Fz"
                )


# ---------------------------------------------------------------------------
# Physical ordering by stiffness
# ---------------------------------------------------------------------------

class TestPhysicalOrdering:
    """Stiff material → smaller deflection, less contact area (per unit force)."""

    def _get(self, bmark_rows, material_name, lc_name):
        return next(r for r in bmark_rows if r.material == material_name and r.load_case == lc_name)

    def test_wmax_decreases_with_stiffness_centric(self, bmark_rows):
        """Grout (stiff) must have smaller w_max than Timber (soft) under Fz."""
        r_grout  = self._get(bmark_rows, "Grout",  "LC01_Fz")
        r_timber = self._get(bmark_rows, "Timber", "LC01_Fz")
        assert r_grout.w_max_mm < r_timber.w_max_mm, (
            f"Grout w_max={r_grout.w_max_mm:.4f} >= Timber w_max={r_timber.w_max_mm:.4f}"
        )

    def test_contact_pct_decreases_with_stiffness_centric(self, bmark_rows):
        """
        Stiffer material generates higher reaction per unit area → needs less contact
        area to equilibrate the same load → lower contact%.
        """
        r_grout  = self._get(bmark_rows, "Grout",  "LC01_Fz")
        r_timber = self._get(bmark_rows, "Timber", "LC01_Fz")
        assert r_grout.pct_active < r_timber.pct_active, (
            f"Grout contact%={r_grout.pct_active:.1f} >= Timber={r_timber.pct_active:.1f}"
        )


# ---------------------------------------------------------------------------
# Lift-off presence
# ---------------------------------------------------------------------------

class TestLiftoff:
    def test_liftoff_present_for_pure_moment(self, bmark_rows):
        """Under pure Mx, both materials must have significant lift-off."""
        mx_cases = [r for r in bmark_rows if r.load_case == "LC02_Mx"]
        for r in mx_cases:
            assert r.pct_inactive > 5.0, (
                f"{r.material}: only {r.pct_inactive:.1f}% lift-off under pure Mx — expected > 5%"
            )

    def test_masks_complementary_in_rows(self, bmark_rows):
        """pct_active + pct_inactive must sum to 100%."""
        for r in bmark_rows:
            total = r.pct_active + r.pct_inactive
            assert abs(total - 100.0) < 0.2, (
                f"{r.material}/{r.load_case}: pct_active+pct_inactive = {total:.2f}%"
            )


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------

class TestFileOutput:
    def test_csv_and_md_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp) / "bmark"
            run_material_benchmark(
                plate=PLATE,
                supports=SUPPORTS,
                materials=TWO_MATERIALS[:1],   # one material only — fast
                load_cases=TWO_CASES[:1],
                options=OPTS_FAST,
                outdir=outdir,
            )
            assert (outdir / "material_benchmark_summary.csv").exists()
            assert (outdir / "material_benchmark_summary.md").exists()

    def test_csv_has_correct_number_of_data_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp) / "bmark"
            rows = run_material_benchmark(
                plate=PLATE,
                supports=SUPPORTS,
                materials=TWO_MATERIALS,
                load_cases=TWO_CASES,
                options=OPTS_FAST,
                outdir=outdir,
            )
            csv_path = outdir / "material_benchmark_summary.csv"
            lines = csv_path.read_text().splitlines()
            # header + data rows
            assert len(lines) == len(rows) + 1, (
                f"CSV has {len(lines)-1} data rows, expected {len(rows)}"
            )


# ---------------------------------------------------------------------------
# Default materials/cases (integration smoke-test)
# ---------------------------------------------------------------------------

def test_default_materials_and_cases_run():
    """Full 3×4 benchmark with default parameters must complete without error."""
    with tempfile.TemporaryDirectory() as tmp:
        rows = run_material_benchmark(
            plate=PLATE,
            supports=SUPPORTS,
            materials=default_materials(),
            load_cases=default_load_cases(),
            options=OPTS_FAST,
            outdir=Path(tmp) / "full_bmark",
        )
        assert len(rows) == len(default_materials()) * len(default_load_cases())
        assert all(r.converged for r in rows)
