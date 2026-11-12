# Guía para interpretar los resultados

Este análisis cuantifica incertidumbre espacial condicional y utilidad de priorización. No confirma el fenómeno en terreno ni transforma la referencia auxiliar en verdad independiente. Se fijaron las familias de contrastes en el código, pero el análisis se construyó después de conocer los resultados; sus intervalos son exploratorios.

## Lectura de la evidencia

Una diferencia de AP es una mejora del ordenamiento, no una reducción del error físico ni un porcentaje adicional de superficie correctamente detectada. Se presentan intervalos marginales y bandas simultáneas bootstrap max t para todos los contrastes de AP dentro de cada combinación de referencia y soporte. Las bandas no cubren conjuntamente todas las referencias del estudio. Un intervalo que incluye cero indica que la comparación no separa los métodos con este diseño; no demuestra equivalencia ni ausencia de efecto. No se fabrican valores p a partir de la proporción de réplicas positivas.

## 3observaciones

- Soporte 500 m, alineamiento_general menos directo: ΔAP +0.0658; intervalo simultáneo [+0.0188; +0.1129]. Evidencia condicional: ganancia. 5/5 grupos con ganancia, 2000 réplicas válidas.
- Soporte 500 m, alineado_blanco menos alineamiento_general: ΔAP -0.0444; intervalo simultáneo [-0.1046; +0.0159]. Evidencia condicional: sin separación de cero. 0/5 grupos con ganancia, 2000 réplicas válidas.
- Soporte 500 m, directo_blanco menos directo: ΔAP -0.1624; intervalo simultáneo [-0.2421; -0.0827]. Evidencia condicional: deterioro. 0/5 grupos con ganancia, 2000 réplicas válidas.
- Soporte 1000 m, alineamiento_general menos directo: ΔAP +0.0547; intervalo simultáneo [-0.0215; +0.1310]. Evidencia condicional: sin separación de cero. 4/5 grupos con ganancia, 2000 réplicas válidas.
- Soporte 1000 m, alineado_blanco menos alineamiento_general: ΔAP -0.0164; intervalo simultáneo [-0.1132; +0.0803]. Evidencia condicional: sin separación de cero. 2/5 grupos con ganancia, 2000 réplicas válidas.
- Soporte 1000 m, directo_blanco menos directo: ΔAP -0.1482; intervalo simultáneo [-0.2700; -0.0264]. Evidencia condicional: deterioro. 0/5 grupos con ganancia, 2000 réplicas válidas.

## huella_manuscrito

- Soporte 500 m, alineamiento_general menos directo: ΔAP +0.0812; intervalo simultáneo [+0.0059; +0.1564]. Evidencia condicional: ganancia. 5/5 grupos con ganancia, 1366 réplicas válidas.
- Soporte 500 m, alineado_blanco menos alineamiento_general: ΔAP -0.0506; intervalo simultáneo [-0.1459; +0.0448]. Evidencia condicional: sin separación de cero. 2/5 grupos con ganancia, 1366 réplicas válidas.
- Soporte 500 m, directo_blanco menos directo: ΔAP -0.1342; intervalo simultáneo [-0.2230; -0.0454]. Evidencia condicional: deterioro. 0/5 grupos con ganancia, 1366 réplicas válidas.
- Soporte 1000 m, alineamiento_general menos directo: ΔAP +0.1308; intervalo simultáneo [-0.0587; +0.3204]. Evidencia condicional: sin separación de cero. 3/4 grupos con ganancia, 1284 réplicas válidas.
- Soporte 1000 m, alineado_blanco menos alineamiento_general: ΔAP -0.1371; intervalo simultáneo [-0.2744; +0.0003]. Evidencia condicional: sin separación de cero. 1/4 grupos con ganancia, 1284 réplicas válidas.
- Soporte 1000 m, directo_blanco menos directo: ΔAP -0.1230; intervalo simultáneo [-0.3069; +0.0609]. Evidencia condicional: sin separación de cero. 0/4 grupos con ganancia, 1284 réplicas válidas.

## estricta_4meses

- Soporte 500 m, alineamiento_general menos directo: ΔAP -0.0149; intervalo simultáneo [-0.1560; +0.1263]. Evidencia condicional: sin separación de cero. 1/3 grupos con ganancia, 1412 réplicas válidas.
- Soporte 500 m, alineado_blanco menos alineamiento_general: ΔAP -0.0708; intervalo simultáneo [-0.1416; +0.0000]. Evidencia condicional: sin separación de cero. 0/3 grupos con ganancia, 1412 réplicas válidas.
- Soporte 500 m, directo_blanco menos directo: ΔAP -0.1436; intervalo simultáneo [-0.3115; +0.0244]. Evidencia condicional: sin separación de cero. 1/3 grupos con ganancia, 1412 réplicas válidas.

## Qué permite explicar cada salida

- 01 y 09: distribución entre grupos, prevalencia, soporte, AP normalizada respecto de la prevalencia y heterogeneidad. La AP media es macro por grupo; no es una AP global combinada.
- 02: magnitud de las diferencias, intervalos marginales y simultáneos, signos por grupo y sensibilidad a retirar un grupo. La familia incluye las comparaciones entre escalas frente al píxel.
- 03 y 08: precisión, recuperación y lift para presupuestos de revisión; el cálculo reproduce el redondeo y los desempates originales. Los intervalos del presupuesto son marginales y exploratorios.
- 04: influencia de retirar bloques de evaluación, sin reajustar modelos. Examinar si el signo depende de pocas zonas, sin tratar este diagnóstico como nueva validación.
- 05: distribución de puntajes para etiquetas baja (0), intermedia (−1) y alta (1). Las magnitudes de ángulos originales y blanqueados pertenecen a geometrías distintas; no compararlas como severidad física.
- 06: relación por grupo con Jensen–Shannon y variaciones absolutas de construido, vegetación y suelo desnudo. Son componentes de la misma referencia de evaluación, no verificaciones independientes.
- 07: correlación de rankings y coincidencia de su 10 % superior sobre todas las unidades. No confundir este presupuesto con los presupuestos calculados solo sobre el subconjunto etiquetado.
- 10: espectro, regularización, rango efectivo y amplificación máxima de la transformación. Una amplificación alta puede motivar una hipótesis de sensibilidad al ruido, pero no la demuestra.

## Conclusión que debe sostenerse

La recomendación de Procrustes sin blanqueamiento es provisional y depende del contraste auxiliar de mayor cobertura. Las pruebas estrictas tienen soporte reducido. No debe presentarse una mejora universal ni una equivalencia demostrada de 10 y 100 m. La utilidad aplicada se expresa como concentración de candidatos bajo un presupuesto de revisión. Los agrupamientos espaciales de candidatos y cualquier validación primaria siguen siendo decisiones posteriores.

## Controles aún pendientes

Quedan pendientes la sensibilidad con máscaras de subpíxeles exactamente coincidentes, una comparación estacional sobre poblaciones emparejadas, la incertidumbre de reajustar operadores y la evaluación reservada de la configuración seleccionada. Estos controles no se sustituyen con más remuestreos de las predicciones actuales. No se cambió la geometría, regularización ni selección de modelos para optimizar los resultados de este análisis.
