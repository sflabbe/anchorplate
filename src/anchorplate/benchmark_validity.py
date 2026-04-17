from __future__ import annotations

from dataclasses import dataclass


DEFAULT_EQUILIBRIUM_TOL_KN_ABS = 0.50
DEFAULT_EQUILIBRIUM_TOL_REL = 0.01


@dataclass(frozen=True)
class CaseValidity:
    solve_status: str
    valid_solution: bool
    failure_reason: str
    equilibrium_error_kN: float
    equilibrium_ok: bool
    metrics_comparable: bool
    equilibrium_tol_kN: float


def _equilibrium_tolerance_kN(expected_vertical_load_kN: float, tol_abs_kN: float, tol_rel: float) -> float:
    return max(float(tol_abs_kN), float(tol_rel) * max(abs(float(expected_vertical_load_kN)), 1.0))


def classify_case_validity(
    *,
    initial_solve_status: str,
    solve_error: str,
    expected_vertical_load_kN: float,
    total_reactions_kN: float,
    contact_converged: bool | None,
    requires_contact_convergence: bool,
    has_foundation_patch: bool,
    support_type: str,
    force_n: float,
    mx_nmm: float,
    my_nmm: float,
    equilibrium_tol_abs_kN: float,
    equilibrium_tol_rel: float,
) -> CaseValidity:
    equilibrium_error_kN = float(total_reactions_kN - expected_vertical_load_kN)
    tol_kN = _equilibrium_tolerance_kN(expected_vertical_load_kN, equilibrium_tol_abs_kN, equilibrium_tol_rel)
    equilibrium_ok = abs(equilibrium_error_kN) <= tol_kN

    if initial_solve_status == "failed":
        detail = solve_error.strip() or "solver_exception"
        return CaseValidity(
            solve_status="solver_failed",
            valid_solution=False,
            failure_reason=f"solver_failed: {detail}",
            equilibrium_error_kN=equilibrium_error_kN,
            equilibrium_ok=False,
            metrics_comparable=False,
            equilibrium_tol_kN=tol_kN,
        )

    if requires_contact_convergence and contact_converged is False:
        is_pure_moment = abs(force_n) < 1e-9 and (abs(mx_nmm) > 1e-9 or abs(my_nmm) > 1e-9)
        if has_foundation_patch and support_type == "spring_tension_only" and is_pure_moment:
            return CaseValidity(
                solve_status="inadmissible_or_not_converged",
                valid_solution=False,
                failure_reason=(
                    "contact_not_converged_in_pure_moment_with_compression_only_foundation"
                    "_and_tension_only_anchors"
                ),
                equilibrium_error_kN=equilibrium_error_kN,
                equilibrium_ok=equilibrium_ok,
                metrics_comparable=False,
                equilibrium_tol_kN=tol_kN,
            )
        return CaseValidity(
            solve_status="not_converged",
            valid_solution=False,
            failure_reason="contact_iteration_not_converged",
            equilibrium_error_kN=equilibrium_error_kN,
            equilibrium_ok=equilibrium_ok,
            metrics_comparable=False,
            equilibrium_tol_kN=tol_kN,
        )

    if not equilibrium_ok:
        return CaseValidity(
            solve_status="inadmissible",
            valid_solution=False,
            failure_reason=(
                f"global_vertical_equilibrium_not_closed: error={equilibrium_error_kN:.6f}kN"
                f", tol={tol_kN:.6f}kN"
            ),
            equilibrium_error_kN=equilibrium_error_kN,
            equilibrium_ok=False,
            metrics_comparable=False,
            equilibrium_tol_kN=tol_kN,
        )

    return CaseValidity(
        solve_status="ok",
        valid_solution=True,
        failure_reason="",
        equilibrium_error_kN=equilibrium_error_kN,
        equilibrium_ok=True,
        metrics_comparable=True,
        equilibrium_tol_kN=tol_kN,
    )
