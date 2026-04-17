from __future__ import annotations

from pathlib import Path

import numpy as np

from anchorplate.benchmark import ProfisLikeCase
from anchorplate.benchmark_matrix import SupportModelSpec, run_support_model_matrix_benchmark
from anchorplate.model import AnalysisOptions, FoundationPatch, PointSupport, SteelPlate


class _FakeFoundationState:
    def __init__(self) -> None:
        self.active_vertices = [set([0])]
        self.history_changes = [3, 1, 1]


class _FakeAnchorState:
    def __init__(self) -> None:
        self.history_changes = [0]


class _FakeResult:
    def __init__(self) -> None:
        self.support_reactions_n = np.array([1_000.0, 1_000.0, 1_000.0, 1_000.0], dtype=float)
        self.support_active = np.array([True, True, True, True], dtype=bool)
        self.anchor_spring_state = _FakeAnchorState()
        self.foundation_state = _FakeFoundationState()
        self.max_deflection_mm = 0.25
        self.max_von_mises_mpa = 120.0


def _supports() -> tuple[PointSupport, ...]:
    return (
        PointSupport(30.0, 30.0, kind="spring_tension_only", kz_n_per_mm=150_000.0, label="A1"),
        PointSupport(270.0, 30.0, kind="spring_tension_only", kz_n_per_mm=150_000.0, label="A2"),
        PointSupport(30.0, 270.0, kind="spring_tension_only", kz_n_per_mm=150_000.0, label="A3"),
        PointSupport(270.0, 270.0, kind="spring_tension_only", kz_n_per_mm=150_000.0, label="A4"),
    )


def test_non_converged_hybrid_case_is_not_ok(monkeypatch, tmp_path: Path) -> None:
    def _fake_solve_anchor_plate(**kwargs):
        return _FakeResult()

    def _fake_contact_summary(*args, **kwargs):
        return {
            "pct_active": 42.0,
            "n_iterations": 30,
            "converged": False,
        }

    monkeypatch.setattr("anchorplate.benchmark_matrix.solve_anchor_plate", _fake_solve_anchor_plate)
    monkeypatch.setattr("anchorplate.benchmark_matrix._foundation_total_reaction", lambda *a, **k: 0.0)
    monkeypatch.setattr("anchorplate.benchmark_matrix._foundation_masks", lambda result: (None, None))
    monkeypatch.setattr("anchorplate.benchmark_matrix._contact_summary", _fake_contact_summary)

    plate = SteelPlate(length_mm=300.0, width_mm=300.0, thickness_mm=15.0)
    options = AnalysisOptions(target_h_mm=20.0, output_dir=str(tmp_path), save_plots=False, save_result_npz=False, save_3d_plots=False)

    models = [
        SupportModelSpec(
            key="hybrid_case",
            display_name="hybrid_case",
            model_type="hybrid_springs_plus_foundation_patch",
            supports=_supports(),
            support_type="spring_tension_only",
            foundation_patch=FoundationPatch(
                x_min_mm=0.0,
                x_max_mm=300.0,
                y_min_mm=0.0,
                y_max_mm=300.0,
                k_area_n_per_mm3=640.0,
                compression_only=True,
                label="concrete",
            ),
        )
    ]
    cases = [ProfisLikeCase("LC04_pure_Mx", "Mx puro", fz_n=0.0, mx_nmm=6.0e6)]

    rows = run_support_model_matrix_benchmark(
        plate=plate,
        options=options,
        outdir=tmp_path,
        models=models,
        load_cases=cases,
        hybrid_support_kind="spring_tension_only",
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.valid_solution is False
    assert row.solve_status != "ok"
    assert row.metrics_comparable is False
    assert row.failure_reason
