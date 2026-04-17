from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np

from anchorplate.benchmark_anchor_dominant import run_anchor_dominant_benchmark
from anchorplate.model import AnalysisOptions, SteelPlate


def test_anchor_dominant_split_and_required_metrics() -> None:
    plate = SteelPlate(length_mm=300.0, width_mm=300.0, thickness_mm=15.0)

    with tempfile.TemporaryDirectory() as tmp:
        outdir = Path(tmp) / "anchor_dominant"
        options = AnalysisOptions(
            target_h_mm=12.0,
            output_dir=str(outdir),
            save_plots=False,
            save_result_npz=False,
            save_3d_plots=False,
        )
        rows = run_anchor_dominant_benchmark(
            plate=plate,
            options=options,
            outdir=outdir,
        )

        assert len(rows) >= 4
        by_variant: dict[str, list] = {}
        for row in rows:
            by_variant.setdefault(row.variant, []).append(row)
            # Required metric coverage
            assert row.w_max_mm > 0.0
            assert row.sigma_vm_max_mpa > 0.0
            assert np.isfinite(row.sum_anchor_reactions_kN)
            assert np.isfinite(row.sum_foundation_reactions_kN)

        no_patch_rows = by_variant["anchor_dominant_no_patch"]
        for row in no_patch_rows:
            assert abs(row.sum_foundation_reactions_kN) < 1e-9
            assert abs(row.foundation_share_pct) < 1e-9
            assert row.contact_active_pct is None

        soft_patch_rows = by_variant["anchor_dominant_small_or_soft_patch"]
        for row in soft_patch_rows:
            assert row.sum_anchor_reactions_kN > row.sum_foundation_reactions_kN
            assert row.foundation_share_pct < 35.0
            assert row.contact_active_pct is not None
            assert 0.0 <= row.contact_active_pct <= 100.0

        # Ensure outputs requested by the benchmark are generated.
        assert (outdir / "anchor_dominant_summary.csv").exists()
        assert (outdir / "anchor_dominant_summary.md").exists()
        assert (outdir / "anchor_dominant_overview.png").exists()
        assert (outdir / "anchor_dominant_note.md").exists()
