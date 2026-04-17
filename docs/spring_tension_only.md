# Soportes `spring` vs `spring_tension_only`

## Diferencia física

- `spring`:
  - Resorte lineal bidireccional.
  - Siempre activo.
  - Reacción: `R = kz * w` para cualquier signo de `w`.

- `spring_tension_only`:
  - Resorte unilateral (solo tracción).
  - Convención del solver: `w > 0` es desplazamiento hacia abajo.
  - Extensión de anclaje: `delta_tension = w`.
  - Estado:
    - **activo** si `w > +tol`,
    - **inactivo** si `w < -tol`,
    - en `[-tol, +tol]` conserva el estado previo (histeresis anti-chattering).
  - Cuando está inactivo, su contribución de rigidez y reacción se anula.

## Integración con `foundation_patch`

El solver itera con un active-set acoplado mínimo:

1. Resuelve con:
   - placa,
   - springs lineales,
   - springs `tension_only` activos en esa iteración,
   - foundation patch activo en contacto.
2. Actualiza:
   - foundation compresión-only (`w > tol`),
   - springs `tension_only` (regla con histeresis sobre `w`).
3. Repite hasta no tener cambios (o hasta `foundation_iterations_max`).

Esto permite casos híbridos donde el foundation patch toma compresión y los anclajes solo toman tracción.

## Salidas por anclaje

Se agregaron exports por caso:

- `export_support_reactions_json(result, path)`
- `export_support_reactions_csv(result, path)`

Campos por anclaje:
- `label`
- `kind`
- `reaction_n`
- `active`
- `vertex_id`, `dof`

## Limitaciones numéricas

- Método de active-set básico (no Newton semisuave).
- La convergencia usa `foundation_contact_tol_mm` como tolerancia de estado.
- Si toda la estructura queda sin restricciones activas (sin fixed, sin foundation en contacto, sin springs activos), el sistema puede volverse singular.
