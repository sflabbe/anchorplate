"""
verify_benchmark_csv.py
=======================
Post-proceso de equilibrio vertical para un CSV de benchmark existente.

Por defecto lee:
- `outputs/demo_benchmark/benchmark_summary.csv`

Y genera:
- `benchmark_verification.csv` en la misma carpeta.
"""

from pathlib import Path
import pandas as pd


def classify(row):
    if abs(row.fz_kN) > 1e-9:
        exp = row.fz_kN
    else:
        exp = 0.0
    err = row.sum_reactions_kN - exp
    return err


def main() -> None:
    path = Path("outputs/demo_benchmark/benchmark_summary.csv")
    if not path.exists():
        raise SystemExit(f"Missing {path}")
    df = pd.read_csv(path)
    df["equilibrium_error_kN"] = df.apply(classify, axis=1)
    df["equilibrium_error_pct_of_fz"] = df.apply(lambda r: 0.0 if abs(r.fz_kN) < 1e-9 else 100.0 * r.equilibrium_error_kN / r.fz_kN, axis=1)
    out = path.with_name("benchmark_verification.csv")
    df.to_csv(out, index=False)
    print(df[["name", "fz_kN", "sum_reactions_kN", "equilibrium_error_kN", "equilibrium_error_pct_of_fz", "eta_plate"]])
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
