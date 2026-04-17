# Nota técnica: backend experimental `quad_bfs` para placa Kirchhoff-Love

Fecha: 2026-04-17

## 1) Alcance del modelo actual y evaluación de *shear locking*

El solver de `anchorplate` resuelve una **placa Kirchhoff-Love (biharmónica)** con formulación de flexión pura. La ruta de referencia usa:

- malla triangular estructurada (`MeshTri.init_tensor`)
- elemento no conforme `ElementTriMorley`
- forma bilineal basada en curvaturas de segundo orden (`dd(u)`), sin variables de corte transversal explícitas.

Por esta razón, el problema principal del backend actual **no debe describirse como “shear locking” de un elemento shear-flexible** (tipo Reissner-Mindlin), porque en esta formulación el corte transversal no es el campo primario discretizado.

Evaluación honesta en esta fase:

- Si aparecen discrepancias en máximos locales (`w_max`, `sigma_vm_max`) o sensibilidad a refinamiento, hoy se interpretan mejor como:
  - **resolución local insuficiente**,
  - **sensibilidad de máximos puntuales**,
  - **estructura/calidad de malla** (alineación, transición de tamaños, densidad local),
  - y método de recuperación de tensiones basado en ajuste cuadrático local.
- Con la evidencia disponible en este repositorio, **no hay demostración suficiente para atribuir el error dominante a shear locking**.

## 2) Backend experimental añadido: `mesh_backend = "quad_bfs"`

Se añadió una ruta experimental de malla cuadrilateral para el mismo subproblema 2D de placa:

- malla: `MeshQuad`
- elemento: `ElementQuadBFS`
- formulación de rigidez: se mantiene el mismo bilinear form de placa Kirchhoff-Love (`kirchhoff_bending`).

El backend de referencia se mantiene intacto:

- `mesh_backend = "tri_morley"` (default)

### Estado

`quad_bfs` quedó **operativo y usable** para casos base del benchmark interno (Fz y Fz+Mx), pero se mantiene explícitamente como **experimental** para evitar sobreprometer convergencia/mejora universal sin campaña de validación más amplia.

## 3) Por qué `hex` 3D NO es el siguiente paso directo

En esta fase se decidió **no** introducir `MeshHex`/`ElementHex*` como si fuera una simple “optimización de malla” del solver actual, porque eso implica cambiar de modelo físico-numérico:

- `MeshHex` + `ElementHex*` corresponde a un **sólido 3D**, no a una placa KL 2D.
- Cambian los **DOFs** (de `w`, rotaciones implícitas/derivadas en placa, a campos 3D de desplazamiento `u,v,w`).
- Cambian la definición de **cargas** (presiones/volúmenes/superficies 3D),
  **soportes** (restricciones 3D),
  **contacto** (interfaces 3D) y
  **postproceso** (tensiones volumétricas, no solo indicadores de flexión de placa).
- Por tanto, no es un swap directo de `MeshTri` → `MeshHex`; sería un **nuevo solver/familia de formulación**.

Conclusión de roadmap: antes de dar ese salto, tiene más sentido cerrar la comparación 2D entre backends de placa (tri vs quad) y separar claramente cuándo hace falta un modelo 3D completo por alcance físico.

## 4) Benchmark comparativo mínimo (tri vs quad)

Configuración base:

- placa 300x300x12 mm
- 4 apoyos fijos en esquinas internas (50,50), (250,50), (50,250), (250,250)
- `target_h_mm = 12.5`

Resultados (`examples/demo_mesh_backend_benchmark.py`):

| Caso | Backend | Nodos | Elementos | w_max [mm] | sigma_vm_max [MPa] | ΣRz [kN] | t_solve [s] |
|---|---|---:|---:|---:|---:|---:|---:|
| Fz_only | tri_morley | 625 | 1152 | 0.9147 | 251.78 | 50.000 | 0.163 |
| Fz_only | quad_bfs   | 625 | 576  | 0.8971 | 244.36 | 50.000 | 0.867 |
| Fz_plus_Mx | tri_morley | 625 | 1152 | 0.9606 | 415.65 | 50.000 | 0.100 |
| Fz_plus_Mx | quad_bfs   | 625 | 576  | 0.9392 | 380.59 | 50.000 | 0.554 |

Lectura rápida:

- Equilibrio global vertical consistente (ΣRz ≈ Fz) en ambos backends.
- Diferencias en máximos locales moderadas (esperables en ruta experimental y recuperación de tensiones).
- En esta configuración, `quad_bfs` no mostró ventaja de tiempo; es más lento que `tri_morley`.

## 5) Conclusión técnica de esta fase

- Se habilitó una ruta cuadrilateral defendible para el **mismo** modelo de placa 2D.
- Se mantiene `tri_morley` como referencia estable.
- No se observó base para vender `quad_bfs` como mejora universal automática.
- La narrativa técnica correcta en esta fase es:
  - priorizar resolución/calidad de malla y convergencia de métricas,
  - evitar atribuir errores a shear locking sin evidencia,
  - y tratar `hex` 3D como cambio de formulación, no como cambio cosmético de malla.
