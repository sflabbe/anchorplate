# Hoja de ruta técnica — modelación avanzada de apoyo en madera

## Alcance y objetivo
Este documento define una ruta **realista** para evolucionar el soporte en madera desde `timber_simple` (base equivalente) hacia modelos con mayor fidelidad física, sin romper la arquitectura actual de `anchorplate`.

---

## 1) Qué captura y qué no captura `timber_simple`

### Qué sí captura (valor actual)
`timber_simple` usa una rigidez equivalente tipo Winkler:

- `k = spread_factor * E90 / h_eff`
- aplicado como `k_area_n_per_mm3` en `foundation_patch` (compresión-only)

Con esto se obtiene:
- comparación rápida entre materiales en benchmark,
- sensibilidad de flecha/reacción al orden de magnitud de la rigidez de apoyo,
- una no linealidad geométrica mínima por active-set de contacto (nodos activos/inactivos).

### Qué no captura (límites explícitos)
No representa de forma física:
- ortotropía completa de la madera (`E_L`, `E_R`, `E_T`, cortantes y Poisson acoplados),
- comportamiento no lineal en compresión perpendicular a la fibra,
- plastificación local/aplastamiento bajo concentraciones de presión,
- evolución temporal (creep/reología),
- contacto mecánico completo (separación + posible fricción/deslizamiento + área real de contacto dependiente de presión).

Conclusión: `timber_simple` es un **modelo de ingeniería temprana**, útil para cribado y benchmarking, no para predicción local de daño/indentación.

---

## 2) Cuándo seguir con foundation patch equivalente

Mantener `foundation_patch` equivalente es adecuado cuando:
- la pregunta es global (flecha, rigidez global, reparto cualitativo de reacción),
- la geometría/carga no induce gradientes locales extremos,
- no se requieren tensiones de contacto locales para verificación normativa,
- el objetivo es comparar alternativas de diseño de forma rápida y estable.

Regla práctica:
- si las decisiones dependen de diferencias de **orden de magnitud** en rigidez global, seguir con modelo equivalente;
- si dependen de **picos locales** de presión/deformación en madera, el modelo equivalente deja de ser suficiente.

---

## 3) Cuándo pasar a shell-shell, shell-solid y solid-solid

## Opción A — shell-shell
Útil cuando se quiere un incremento moderado de fidelidad con costo controlado.

Recomendado si:
- la madera puede idealizarse como capa delgada,
- interesa capturar acoplamientos de placa/lámina sin resolver espesor 3D local,
- se acepta mantener una ley constitutiva simplificada.

No recomendado para:
- aplastamiento local fuerte,
- gradientes severos a través del espesor,
- contacto local altamente concentrado.

## Opción B — shell-solid
Compromiso más sólido para siguiente fase.

Recomendado si:
- la placa metálica sigue bien representada como shell,
- la madera necesita volumen 3D para capturar compresión ⟂ fibra y distribución local,
- se busca analizar contacto local con mejor realismo sin pagar costo máximo global.

Ventaja clave: permite concentrar DOFs 3D donde importa (madera/contacto) y conservar eficiencia en el resto.

## Opción C — solid-solid
Máxima fidelidad, máximo costo.

Recomendado solo si:
- se requiere trazabilidad local de tensiones/deformaciones en todos los cuerpos,
- hay validación experimental suficiente para calibrar parámetros avanzados,
- existe presupuesto computacional y de mantenimiento para ello.

Riesgo principal: complejidad numérica/paramétrica alta antes de tener datos de calibración robustos.

---

## 4) Fenómenos que sí importan (priorización)

### 4.1 Ortotropía
Impacta la distribución de deformaciones y presión de contacto cuando la dirección de fibra no está alineada favorablemente.

Prioridad: **alta** cuando haya direccionalidad clara de veta o piezas laminadas.

### 4.2 Compresión perpendicular a la fibra
Es el fenómeno más directamente ligado a asentamiento local bajo placa/base.

Prioridad: **muy alta** para diseño de apoyo en madera.

### 4.3 Plasticidad local / aplastamiento
Controla la redistribución de presión en picos de contacto y la deformación residual.

Prioridad: **alta** en cargas elevadas, anclajes cercanos a borde o áreas pequeñas de apoyo.

### 4.4 Creep
Puede dominar desplazamientos de servicio a mediano/largo plazo.

Prioridad: **media-alta**, pero típicamente fase posterior a cerrar el modelo cuasi-estático no lineal local.

### 4.5 Contacto con separación
Ya existe una versión simplificada (compresión-only); falta contacto más fiel para interacción local.

Prioridad: **alta** en la transición a submodelo local.

---

## 5) Arquitectura futura recomendada para este repo

### 5.1 Módulo nuevo (recomendado)
Crear un módulo dedicado, p. ej. `anchorplate/timber_advanced.py`, con:
- APIs explícitas de materiales ortotrópicos y leyes locales en compresión ⟂ fibra,
- interfaces de contacto desacopladas del `foundation_patch` actual,
- outputs específicos para comparación con benchmark global.

Objetivo: no contaminar el camino rápido actual (`timber_simple`) y preservar compatibilidad.

### 5.2 Submodelo local (muy recomendado)
Estrategia de dos escalas:
1. modelo global actual para cargas/reacciones globales,
2. submodelo local (zona anclaje-contacto) para tensiones y aplastamiento en madera.

Ventajas:
- costo computacional controlado,
- trazabilidad técnica clara,
- adopción incremental sin rehacer todo el solver.

### 5.3 Benchmark separado (obligatorio)
Mantener benchmark específico para madera avanzada (casos mínimos reproducibles), separado del benchmark rápido de materiales equivalente.

Debe medir al menos:
- sensibilidad a orientación de fibra,
- presión máxima y área efectiva de contacto,
- asentamiento local y su estabilización con refinamiento de malla.

---

## 6) Qué NO conviene prometer todavía

No prometer en la siguiente fase inmediata:
- “predicción normativa definitiva” sin calibración experimental,
- creep confiable de largo plazo sin campaña de parámetros,
- contacto con fricción robusto para cualquier escenario,
- equivalencia universal entre resultados globales y daño local.

Mensaje técnico correcto: la siguiente fase debe entregar **mejor física local en casos acotados**, no una solución universal cerrada.

---

## Recomendación concreta de siguiente fase (propuesta ejecutable)

Fase recomendada: **shell-solid + submodelo local**, manteniendo `timber_simple` para benchmark global rápido.

Entregables mínimos de esta fase:
1. módulo nuevo de madera avanzada (arquitectura separada),
2. ley ortotrópica elástica inicial + compresión ⟂ fibra con no linealidad local básica,
3. contacto con separación a nivel submodelo,
4. benchmark dedicado con 2–3 casos de referencia y criterio de convergencia de malla.

Criterio de éxito:
- reproducir tendencias globales del modelo rápido,
- mejorar predicción local (presión/asentamiento) de forma estable y trazable,
- mantener tiempos de cómputo compatibles con uso de ingeniería.
