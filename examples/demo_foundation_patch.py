from pathlib import Path

from anchorplate.model import AnalysisOptions, CoupledLineLoad, FoundationPatch, MeshRefinementBox, PointSupport, SteelPlate
from anchorplate.plotting import export_result_npz, plot_mesh, plot_result, plot_result_3d
from anchorplate.solver import solve_anchor_plate
from anchorplate.support import bedding_concrete_simple, bedding_timber_simple


def main() -> None:
    plate = SteelPlate(length_mm=300.0, width_mm=300.0, thickness_mm=15.0)
    supports = [
        PointSupport(30.0, 30.0, kind="spring", kz_n_per_mm=150_000.0, label="A1"),
        PointSupport(270.0, 30.0, kind="spring", kz_n_per_mm=150_000.0, label="A2"),
        PointSupport(30.0, 270.0, kind="spring", kz_n_per_mm=150_000.0, label="A3"),
        PointSupport(270.0, 270.0, kind="spring", kz_n_per_mm=150_000.0, label="A4"),
    ]
    loads = [
        CoupledLineLoad(
            ref_x_mm=150.0,
            ref_y_mm=150.0,
            force_n=40_000.0,
            mx_nmm=3.0e6,
            line_spacing_mm=150.0,
            line_length_mm=100.0,
            label="RP",
        )
    ]
    k_concrete = bedding_concrete_simple(e_cm_mpa=32_000.0, h_eff_mm=200.0)
    k_timber = bedding_timber_simple(e90_mpa=370.0, h_eff_mm=100.0, spread_factor=1.0)
    foundation = [
        FoundationPatch(0.0, 300.0, 0.0, 150.0, k_area_n_per_mm3=k_concrete, compression_only=True, label="concrete-zone"),
        FoundationPatch(0.0, 300.0, 150.0, 300.0, k_area_n_per_mm3=k_timber, compression_only=True, label="timber-zone"),
    ]
    refinement = [MeshRefinementBox(40.0, 260.0, 80.0, 220.0, h_mm=4.0, label="load-zone")]
    options = AnalysisOptions(target_h_mm=8.0, output_dir="outputs/demo_foundation_patch")

    result = solve_anchor_plate(
        plate=plate,
        supports=supports,
        coupled_loads=loads,
        options=options,
        foundation_patches=foundation,
        refinement_boxes=refinement,
        name="demo_foundation_patch",
    )
    outdir = Path(options.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    plot_mesh(result.mesh, plate, supports, loads, [], refinement_boxes=refinement, foundation_patches=foundation, outpath=outdir / "mesh.png")
    plot_result(plate, supports, [], loads, result, options)
    plot_result_3d(plate, supports, result, options)
    export_result_npz(result, Path(options.output_dir) / f"{result.name}_result.npz")
    print("foundation contact history:", result.foundation_state.history_changes)


if __name__ == "__main__":
    main()
