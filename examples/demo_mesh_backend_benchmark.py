from __future__ import annotations

from pathlib import Path

from anchorplate.benchmark_backend import backend_benchmark_markdown, run_backend_benchmark
from anchorplate.model import AnalysisOptions, PointSupport, SteelPlate


def main() -> None:
    plate = SteelPlate(length_mm=300.0, width_mm=300.0, thickness_mm=12.0)
    supports = [
        PointSupport(x_mm=50.0, y_mm=50.0, kind="fixed", label="A1"),
        PointSupport(x_mm=250.0, y_mm=50.0, kind="fixed", label="A2"),
        PointSupport(x_mm=50.0, y_mm=250.0, kind="fixed", label="A3"),
        PointSupport(x_mm=250.0, y_mm=250.0, kind="fixed", label="A4"),
    ]
    options = AnalysisOptions(
        target_h_mm=12.5,
        save_plots=False,
        show_plots=False,
        save_result_npz=False,
        save_3d_plots=False,
    )

    rows = run_backend_benchmark(plate=plate, supports=supports, base_options=options)
    out = Path("outputs") / "benchmark_backend_tri_vs_quad.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# Benchmark tri_morley vs quad_bfs\n\n" + backend_benchmark_markdown(rows) + "\n", encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
