# Hybrid benchmark validity policy

This project now treats hybrid benchmark outputs with an explicit **validity gate**.

## Valid solution criteria

A benchmark case is considered valid (`valid_solution = true`) only if all checks pass:

1. Solver does not raise an exception.
2. Contact iteration converges (`contact_converged = true`) for contact-based hybrid models.
3. Global vertical equilibrium closes:

   \[
   \left|\sum R_z - F_z\right| \le \max\left(\text{equilibrium\_tol\_abs\_kN},\ \text{equilibrium\_tol\_rel}\cdot\max(|F_z|, 1\ \text{kN})\right)
   \]

Default tolerances are defined in `AnalysisOptions`:

- `equilibrium_tol_abs_kN = 0.50`
- `equilibrium_tol_rel = 0.01`

## Status classification

Per-case `solve_status` is reported as one of:

- `ok`
- `solver_failed`
- `not_converged`
- `inadmissible`
- `inadmissible_or_not_converged`

`failure_reason` explains the exact trigger.

## Pure moment + compression-only foundation + tension-only anchors

For some pure-moment cases (for example `LC04_pure_Mx`) the current idealization can end in unresolved active-set states. Those are now flagged as:

- `solve_status = inadmissible_or_not_converged`
- `valid_solution = false`

This is intentional: physical honesty is prioritized over returning a misleading "ok" numeric result.

## Metrics comparability

If a case is invalid:

- `metrics_comparable = false`
- metrics can still be kept for debugging, but they must not be used as benchmark comparisons.
