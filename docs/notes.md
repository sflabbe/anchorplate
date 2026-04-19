# Notes

## Cómo pensar el soporte

1. El Bettungsmodul se mete como `k_area [N/mm³]`.
2. El solver lo convierte a resortes nodales con áreas tributarias.
3. Si el patch es `compression_only=True`, en cada iteración se apagan los nodos con `w <= tol`.
4. Para acero-acero, usa el modelo por capas solo como arranque. Si la placa receptora es flexible, modela también esa pieza.

## Dónde refinar

- alrededor de anclajes
- alrededor de las dos líneas del perfil
- en esquinas de la placa
- cerca del borde de contacto si usas foundation compresión-solo

## Convergencia de malla (coarse/medium/fine)

- Usa un caso representativo (p.ej. placa 300×300×15 con 4 anchors y `Fz + Mx`) y evalúa mínimo tres niveles de malla.
- Reporta al menos: `n_nodes`, `n_elements`, `h_char`, `w_max`, `sigma_vm_max`, `Rmin/Rmax/ΣR`.
- Para decidir malla por defecto prioriza métricas globales (`w_max`, `ΣR`) y equilibrio.
- `sigma_vm_max` es útil como alerta, pero puede estar sesgado por singularidades nodales/locales; no lo uses como criterio único.
- Incluso con jerarquía de malla limpia, máximos locales (`sigma_vm_max`, picos en nodos puntuales) pueden seguir siendo sensibles al refinamiento local; valida siempre con métricas globales y contexto de detalle constructivo.
- Con `examples/demo_mesh_convergence.py` puedes comparar además la opción con y sin `MeshRefinementBox` para ver sensibilidad local.
