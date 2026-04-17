# Guía de interpretación: contacto y lift-off en `foundation_patch`

## Modelo físico

El submodelo `foundation_patch` implementa un apoyo de Winkler con compresión-only (sin tracción). Cada nodo del parche contribuye a la matriz global con una rigidez nodal:

```
K_nodal_i = k_area [N/mm³] × A_tributaria_i [mm²]   →   [N/mm]
```

La condición de contacto se evalúa nodo a nodo tras cada iteración:

| Estado   | Condición         | Significado físico                              |
|----------|-------------------|-------------------------------------------------|
| Activo   | `w_i > tol`       | El nodo está en contacto — la fundación empuja la placa hacia arriba |
| Inactivo | `w_i ≤ tol`       | El nodo se ha levantado — no hay contacto, rigidez de ese nodo = 0 |

`tol = AnalysisOptions.foundation_contact_tol_mm = 1e-10 mm` (prácticamente cero).

## Convención de signos

```
w > 0  →  placa se mueve hacia abajo (positivo = compresión contra la fundación)
w < 0  →  placa se mueve hacia arriba (lift-off, separación de la fundación)
```

La carga se aplica con `force_n > 0` como fuerza vertical descendente. El solver
Kirchhoff produce `w > 0` donde la placa baja hacia la fundación.

## Iteración del active-set

El solver itera hasta que el conjunto activo converge (cero cambios en una iteración):

```
iter 0: todos los nodos del parche → activos
iter 1: solve + check w_i > 0 → eliminar nodos con w_i ≤ 0
iter 2: re-solve + check → …
…
iter N: sin cambios → convergido
```

`FoundationState.history_changes[i]` = número de nodos que cambiaron de estado en la iteración i.
Una historia `[2139, 670, 240, 52, 5, 0]` indica convergencia monótona sana.

## Máscaras en el `.npz`

| Array                       | Tipo   | Descripción |
|-----------------------------|--------|-------------|
| `active_foundation_mask`    | uint8  | 1 si el nodo está en el parche Y en contacto (`w > 0`) |
| `inactive_foundation_mask`  | uint8  | 1 si el nodo está en el parche Y con lift-off (`w ≤ 0`) |
| `in_patch_mask`             | uint8  | 1 si el nodo está dentro de cualquier parche (= active OR inactive) |
| `w_mm`                      | float  | Deflexión nodal [mm] |

**Invariante garantizado:**

```python
assert np.array_equal(active | inactive, in_patch)   # sin gaps ni extras
assert not np.any(active & inactive)                  # disjuntas
assert w[active].min()  > 0                           # física correcta
assert w[inactive].max() <= 0                         # física correcta
```

## Script de verificación post-hoc

```python
import numpy as np

d = np.load("demo_foundation_patch_3d_result.npz")
active   = d["active_foundation_mask"].astype(bool)
inactive = d["inactive_foundation_mask"].astype(bool)
in_patch = d["in_patch_mask"].astype(bool)
w        = d["w_mm"]

print(f"Nodos en contacto : {active.sum()} ({100*active.sum()/in_patch.sum():.1f}%)")
print(f"Nodos con lift-off: {inactive.sum()} ({100*inactive.sum()/in_patch.sum():.1f}%)")
print(f"w activo  [{w[active].min():.4f}, {w[active].max():.4f}] mm")
print(f"w inactivo[{w[inactive].min():.4f}, {w[inactive].max():.4f}] mm")

# Invariantes
assert np.array_equal(active | inactive, in_patch)
assert not np.any(active & inactive)
assert w[active].min() > 0
assert w[inactive].max() <= 0
print("Todos los invariantes OK.")
```

## Interpretación del reparto de contacto

### Caso simétrico (solo `Fz`)
- La placa baja uniformemente → mayor porcentaje de nodos en contacto.
- El lift-off, si existe, aparece en bordes libres donde la placa tiende a levantarse.

### Caso con momento (`Fz + Mx` o `Fz + My`)
- Un lado del parche queda comprimido (en contacto), el otro se levanta.
- El contorno de lift-off (`w = 0`) cruza el parche aproximadamente por la línea neutra del momento.
- La no-linealidad de contacto puede desplazar esa línea respecto a la línea media geométrica.

### Indicadores de advertencia
- Convergencia no monótona en `history_changes` → revisar la rigidez del parche vs la rigidez de placa.
- Porcentaje de contacto < 5% → revisar que la carga no sea de tracción pura.
- `w_inactive.max() >> 0` → la tolerancia de contacto puede ser demasiado grande.

## Relación con el benchmark de materiales

Los tres materiales del benchmark futuro difieren principalmente en `k_area`:

| Material    | `E` o `E_90` | `h_eff` | `k_area` approx. |
|-------------|-------------|---------|-----------------|
| Hormigón    | 32 000 MPa  | 200 mm  | 160 N/mm³       |
| Acero/chapa | 210 000 MPa | 10 mm   | 21 000 N/mm³    |
| Madera      | 370 MPa     | 100 mm  | 3.7 N/mm³       |

A mayor `k_area`, mayor reacción de fundación por mm de deformación → menor lift-off para la misma carga.
