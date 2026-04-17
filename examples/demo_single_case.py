from pathlib import Path

from anchorplate.model import AnalysisOptions, CoupledLineLoad, MeshRefinementBox, PointSupport, SteelPlate
from anchorplate.plotting import export_result_npz, plot_mesh, plot_result, plot_result_3d
from anchorplate.solver import solve_anchor_plate


def main() -> None:
    plate = SteelPlate(length_mm=300.0, width_mm=300.0, thickness_mm=15.0)
    supports = [
        PointSupport(30.0, 30.0, kind="fixed", label="A1"),
        PointSupport(270.0, 30.0, kind="fixed", label="A2"),
        PointSupport(30.0, 270.0, kind="fixed", label="A3"),
        PointSupport(270.0, 270.0, kind="fixed", label="A4"),
    ]
    loads = [
        CoupledLineLoad(
            ref_x_mm=150.0,
            ref_y_mm=150.0,
            force_n=50_000.0,
            line_spacing_mm=150.0,
            line_length_mm=100.0,
            label="RP",
        )
    ]
    refinement = [
        MeshRefinementBox(50.0, 250.0, 80.0, 220.0, h_mm=4.0, label="profile-zone"),
        MeshRefinementBox(0.0, 60.0, 0.0, 60.0, h_mm=4.0, label="A1"),
        MeshRefinementBox(240.0, 300.0, 0.0, 60.0, h_mm=4.0, label="A2"),
        MeshRefinementBox(0.0, 60.0, 240.0, 300.0, h_mm=4.0, label="A3"),
        MeshRefinementBox(240.0, 300.0, 240.0, 300.0, h_mm=4.0, label="A4"),
    ]
    options = AnalysisOptions(target_h_mm=8.0, output_dir="outputs/demo_single_case")
    result = solve_anchor_plate(
        plate=plate,
        supports=supports,
        coupled_loads=loads,
        options=options,
        refinement_boxes=refinement,
        name="demo_single_case",
    )
    Path(options.output_dir).mkdir(parents=True, exist_ok=True)
    plot_mesh(result.mesh, plate, supports, loads, [], refinement_boxes=refinement, outpath=Path(options.output_dir) / "mesh.png")
    plot_result(plate, supports, [], loads, result, options)
    plot_result_3d(plate, supports, result, options)
    export_result_npz(result, Path(options.output_dir) / f"{result.name}_result.npz")
    print(f"w_max = {result.max_deflection_mm:.4f} mm")
    print(f"sigma_vm,max = {result.max_von_mises_mpa:.2f} MPa")


if __name__ == "__main__":
    main()
