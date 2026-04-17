"""
demo_benchmark.py
=================
PROFIS-like benchmark base con 4 apoyos discretos fijos.

Qué corre:
- familia completa de casos `default_cases()` de `anchorplate.benchmark`

Qué genera y dónde:
- `outputs/demo_benchmark/benchmark_summary.csv`
- `outputs/demo_benchmark/benchmark_summary.md`
- `outputs/demo_benchmark/benchmark_overview.png`
- `outputs/demo_benchmark/<load_case>/...` (plots y opcionalmente 3D/NPZ según options)
"""

from pathlib import Path

from anchorplate.benchmark import run_profis_like_benchmark
from anchorplate.model import AnalysisOptions, PointSupport, SteelPlate


def main() -> None:
    plate = SteelPlate(length_mm=300.0, width_mm=300.0, thickness_mm=15.0)
    supports = [
        PointSupport(30.0, 30.0, kind="fixed", label="A1"),
        PointSupport(270.0, 30.0, kind="fixed", label="A2"),
        PointSupport(30.0, 270.0, kind="fixed", label="A3"),
        PointSupport(270.0, 270.0, kind="fixed", label="A4"),
    ]
    options = AnalysisOptions(target_h_mm=6.0, output_dir="outputs/demo_benchmark", z_plot_scale=20.0)
    rows = run_profis_like_benchmark(plate, supports, options, outdir=Path(options.output_dir))
    print(f"Ran {len(rows)} cases")


if __name__ == "__main__":
    main()
