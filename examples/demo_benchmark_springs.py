from pathlib import Path

from anchorplate.benchmark import run_profis_like_benchmark
from anchorplate.model import AnalysisOptions, PointSupport, SteelPlate


def main() -> None:
    plate = SteelPlate(length_mm=300.0, width_mm=300.0, thickness_mm=15.0)
    supports = [
        PointSupport(30.0, 30.0, kind="spring", kz_n_per_mm=150_000.0, label="A1"),
        PointSupport(270.0, 30.0, kind="spring", kz_n_per_mm=150_000.0, label="A2"),
        PointSupport(30.0, 270.0, kind="spring", kz_n_per_mm=150_000.0, label="A3"),
        PointSupport(270.0, 270.0, kind="spring", kz_n_per_mm=150_000.0, label="A4"),
    ]
    options = AnalysisOptions(target_h_mm=6.0, output_dir="outputs/demo_benchmark_springs", z_plot_scale=20.0)
    rows = run_profis_like_benchmark(plate, supports, options, outdir=Path(options.output_dir))
    print(f"Ran {len(rows)} spring-supported cases")


if __name__ == "__main__":
    main()
