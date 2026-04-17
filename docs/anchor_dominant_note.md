# Nota técnica corta: benchmark `anchor-dominant`

## Cuándo domina el foundation patch

El `foundation_patch` tiende a dominar cuando se combinan dos factores:

- **Área activa de contacto grande** (poca o nula zona en lift-off).
- **Rigidez distribuida alta** (`k_area` elevado).

En ese régimen, la mayor parte de `Fz` se transmite por apoyo continuo y las reacciones discretas de anclaje quedan en un rol secundario.

## Cuándo gobiernan los anclajes

El sistema pasa a estar gobernado por anclajes cuando:

- **No hay patch**, o
- Hay patch pero es **pequeño y/o blando**, de modo que su contribución global a reacción es baja frente a `ΣR_anchor`.

En ese contexto, la firma mecánica de `Fz + Mx` o `Fz + Mx + My` se ve directamente en el reparto `A1…A4` y en el uplift local.

## Por qué importa en benchmarks

- Evita **mezclar conclusiones** entre modelos patch-dominant y anchor-dominant.
- Permite usar el benchmark como **caso pedagógico** (interpretación física clara).
- Lo vuelve útil como **regression test físico**, porque el camino de carga principal (discreto vs distribuido) queda explícito y verificable.
