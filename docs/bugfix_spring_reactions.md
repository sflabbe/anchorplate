# Bugfix: reacciones de springs siempre cero

**Fecha:** 2026-04-17  
**Archivo afectado:** `src/anchorplate/solver.py` → `solve_anchor_plate()`  
**Tests nuevos:** `tests/test_spring_reactions.py` (17 tests, 4 casos)

---

## El bug era de postproceso, no de ensamblaje

El ensamblaje del resorte en la matriz global era **correcto**. La rigidez nodal `kz` (N/mm) se aplica al DOF `w` del vértice correspondiente mediante `diags(spring_diag)`, y el DOF de `w` se identifica correctamente vía `basis.nodal_dofs[0, vertex_id]` (DOF nodal del elemento Morley). La deformada resultante es físicamente válida.

El problema estaba en cómo se extraían las reacciones después de resolver.

---

## Causa raíz

```python
# ANTES — incorrecto para springs
residual = k_total @ solution - rhs
support_reactions = -residual[support_dofs]
```

Para **soportes fijos** (`kind="fixed"`): `condense()` elimina esos DOFs del sistema. Tras resolver, `solution[fixed_dof] = 0` y `K·u ≠ f` en esos DOFs → el residual da la reacción. ✅

Para **resortes** (`kind="spring"`): los DOFs de resorte son DOFs **libres** en el sistema global. El solver satisface la ecuación de equilibrio en todos los DOFs libres:

```
K_total · u = f  →  residual = K_total · u − f = 0   en TODOS los DOFs libres
```

Esto ocurre siempre, independientemente de si hay otros soportes fijos en el modelo. Los DOFs de resorte nunca se condensa → el residual en ellos es siempre exactamente cero → la reacción extraída es cero. ❌

La reacción de un resorte es directamente la fuerza que ejerce:

```
R_i = kz_i · w_i
```

que ya está disponible como `(k_springs @ solution)[dof_i]`.

---

## Corrección

```python
# DESPUÉS — correcto para ambos tipos
residual = k_total @ solution - rhs
spring_reactions = (k_springs @ solution)[support_dofs]  # R = kz·w; cero en DOFs fijos
fixed_reactions  = -residual[support_dofs]               # correcto en DOFs fijos; cero en springs
support_reactions = spring_reactions + fixed_reactions
```

Los dos términos no se solapan: `spring_diag[fixed_dof] = 0` (la rigidez del resorte no se pone en DOFs fijos), y `residual[spring_dof] = 0` (DOF libre, ya resuelto). La suma es exacta para ambos tipos sin condicionales adicionales.

---

## Verificación numérica

Con `Fz = 50 kN` céntrico y 4 springs de `kz = 150,000 N/mm`:

| Métrica | Antes del fix | Después del fix | Esperado |
|---|---|---|---|
| `sum(reactions)` | ~0 N | 50,000.0 N | 50,000 N |
| `R_i` por anclaje | ~0 N | 12,500 N c/u | 12,500 N |
| `R_i = kz · w_i` | — | ✅ (error < 1e-6 N) | |
| `w_max` (k=100 N/mm) | 125.93 mm | 125.93 mm (sin cambio) | — |
| `w_max` (k=5M N/mm) | 0.928 mm | 0.928 mm (sin cambio) | — |
| Fixed no roto | ✅ | ✅ | |

El campo `w` (deformada de placa) y `sigma_vm` no cambian — estaban correctos desde antes. Solo las reacciones reportadas estaban mal.

---

## Nota sobre resultados de placa idénticos (fixed vs spring)

Con `kz = 150,000 N/mm` y geometría simétrica, la diferencia de `sigma_vm` entre el caso `fixed` y `spring` es efectivamente pequeña — no es un bug. La tensión de placa depende de las curvaturas relativas, no del desplazamiento absoluto. Al variar `kz` la diferencia de `w_max` sí cambia (traslación de cuerpo rígido), pero `sigma_vm` solo cambia si el reparto de reacciones cambia, lo cual ocurre en configuraciones asimétricas o cuando la rigidez del resorte es comparable a la rigidez flexional de la placa. Para confirmar que el reparto de carga cambia con `kz`, usar un caso con momento (`Mx` o `My`) o springs no simétricos.

---

## Tests añadidos

`tests/test_spring_reactions.py` — 17 tests en 4 casos:

| Caso | Qué verifica |
|---|---|
| `TestCase1CentricFz` | Equilibrio global, no-cero, simetría, `R = kz·w` |
| `TestCase2PureMx` | Suma≈0, par de reacciones opuesto, no-cero |
| `TestCase3KzSweep` | Equilibrio para todo kz, monotonía de w_max, `R = kz·w` |
| `TestCase4SpringToFixed` | Convergencia stiff-spring→fixed en w_max, sigma_vm, reactions; fixed no roto |
