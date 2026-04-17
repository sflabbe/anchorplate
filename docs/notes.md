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
