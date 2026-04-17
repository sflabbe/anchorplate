# Actualización técnica: refinamiento monotónico coarse/medium/fine

Fecha de regeneración: **2026-04-17**.

## Qué causaba la no monotonicidad

La malla anterior inyectaba seeds internas de `MeshRefinementBox` con `linspace` dependiente de `h_mm` de cada caja. Como esas seeds se mezclaban con el mallado global por eje, la combinación podía cambiar entre niveles y producir conteos no monotónicos (por ejemplo, `medium` con más nodos que `fine`) aunque la convergencia global siguiera siendo razonable.

## Cambio aplicado

- Las cajas ahora aportan **solo bordes** como breakpoints de partición.
- El tamaño objetivo por segmento se controla explícitamente como `min(target_h_global, h_local_de_caja)` en cada tramo.
- Se respeta `n_div_min` como cota inferior de divisiones efectivas por caja/eje.
- En el demo de convergencia, cada nivel define su `h_mm` local por caja de forma monotónica:
  - coarse: 8.0 mm
  - medium: 6.0 mm
  - fine: 4.0 mm

## Resultados regenerados (`examples/demo_mesh_convergence.py --mode both`)

### with_boxes

| level | n_nodes | n_elements | h_min [mm] | h_max [mm] | w_max [mm] | sigma_vm_max [MPa] | ΣR [kN] |
|---|---:|---:|---:|---:|---:|---:|---:|
| coarse | 1681 | 3200 | 5.77 | 8.94 | 0.9777 | 342.95 | 50.00 |
| medium | 2915 | 5616 | 5.00 | 6.00 | 0.9754 | 348.11 | 50.00 |
| fine | 5929 | 11552 | 3.75 | 5.00 | 0.9739 | 353.05 | 50.00 |

### without_boxes

| level | n_nodes | n_elements | h_min [mm] | h_max [mm] | w_max [mm] | sigma_vm_max [MPa] | ΣR [kN] |
|---|---:|---:|---:|---:|---:|---:|---:|
| coarse | 841 | 1568 | 10.00 | 11.46 | 0.9822 | 323.54 | 50.00 |
| medium | 1681 | 3200 | 7.32 | 7.64 | 0.9775 | 337.33 | 50.00 |
| fine | 2809 | 5408 | 5.59 | 6.00 | 0.9755 | 347.15 | 50.00 |

## Nota de interpretación

Incluso con jerarquía de malla limpia, los máximos locales (por ejemplo `sigma_vm_max`) pueden seguir siendo sensibles a singularidades locales. Para aceptación de convergencia, priorizar métricas globales (`w_max`, `ΣR`) y equilibrio.
