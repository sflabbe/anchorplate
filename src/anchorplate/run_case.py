from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any

from .inputio import InputValidationError, expand_cases, load_input_config


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run anchorplate cases from TOML input")
    p.add_argument("input_toml", help="Path to TOML file")
    p.add_argument("--dry-run", action="store_true", help="Parse and expand cases without solving")
    return p


def run_from_toml(input_toml: str | Path, dry_run: bool = False) -> int:
    config = load_input_config(input_toml)
    expanded = expand_cases(config)

    if dry_run:
        print(f"Parsed {len(expanded)} case(s) in mode='{config.mode}'")
        for case in expanded:
            print(f" - {case.name}")
        return 0

    from .plotting import export_result_npz, plot_result, plot_result_3d
    from .solver import export_support_reactions_csv, export_support_reactions_json, solve_anchor_plate

    for item in expanded:
        case = item.case
        case_outdir = Path(case.analysis_options.output_dir) / item.name
        options = replace(case.analysis_options, output_dir=str(case_outdir))

        result = solve_anchor_plate(
            plate=case.plate,
            supports=case.supports,
            point_loads=case.point_loads,
            coupled_loads=case.coupled_line_loads,
            foundation_patches=case.foundation_patches,
            refinement_boxes=case.refinement_boxes,
            options=options,
            name=item.name,
        )

        payload: dict[str, Any] = {
            "name": item.name,
            "metadata": item.metadata,
            "max_deflection_mm": result.max_deflection_mm,
            "max_von_mises_mpa": result.max_von_mises_mpa,
        }

        if options.save_plots:
            plot_path = plot_result(case.plate, case.supports, case.point_loads, case.coupled_line_loads, result, options)
            payload["plot_path"] = str(plot_path)
        if options.save_3d_plots:
            plot3d = plot_result_3d(case.plate, case.supports, result, options)
            payload["plot_3d_path"] = str(plot3d)
        if options.save_result_npz:
            npz_path, contact_summary = export_result_npz(result, case_outdir / f"{item.name}_result.npz")
            payload["result_npz"] = str(npz_path)
            payload["foundation_contact_summary"] = contact_summary

        payload["support_reactions_json"] = str(export_support_reactions_json(result, case_outdir / "support_reactions.json"))
        payload["support_reactions_csv"] = str(export_support_reactions_csv(result, case_outdir / "support_reactions.csv"))

        case_outdir.mkdir(parents=True, exist_ok=True)
        (case_outdir / "case_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[ok] {item.name} -> {case_outdir}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return run_from_toml(args.input_toml, dry_run=args.dry_run)
    except InputValidationError as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
