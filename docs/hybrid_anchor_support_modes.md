# Hybrid benchmark support modes: `spring` vs `spring_tension_only`

This repository now supports two explicit discrete-anchor variants in hybrid benchmarks (`anchors + foundation_patch`):

- `spring`: linear bidirectional spring. The anchor reaction is `R = kz·w` and can be positive or negative.
- `spring_tension_only`: unilateral anchor spring. The spring is active only when the local anchor displacement is in tension (per solver sign convention) and deactivates under compression.

## When to use each mode

- Use **`spring`** when you want a numerically smooth linearized discrete-anchor model, e.g. for fast baseline sweeps or correlation against legacy studies that assumed bidirectional springs.
- Use **`spring_tension_only`** when you want a physically realistic anchor model in uplift-sensitive scenarios (mixed contact, one-sided lift-off, pure moment), where anchors should not provide fictitious compression resistance.

## What is now exported in benchmark outputs

For hybrid runs, CSV/Markdown rows now include:

- `support_type` (`spring` or `spring_tension_only`)
- per-case anchor activity (`anchor_active_count`, `anchor_inactive_count`)
- per-anchor reactions (`anchor_reactions_kN_json`)
- `sum_spring_reactions_kN` and `sum_foundation_reaction_kN`

In `benchmark_material`, the per-case JSON metadata (`*_material_model.json`) now also records `support_type`.

## Convergence and interpretation notes

- `spring_tension_only` may require additional active-set iterations because anchor activity is solved together with foundation contact activity.
- If some cases need more iterations, compare `contact_iterations` and `anchor_iterations` across support modes instead of masking differences.
- Global equilibrium should still close with total reaction:

  `sum_spring_reactions_kN + sum_foundation_reaction_kN ≈ applied Fz`.
