"""Resumen legible de las cuatro variantes, con resultados exclusivamente corridos."""
from pathlib import Path
import pandas as pd
import numpy as np
root=Path(__file__).resolve().parent/'salidas_blanqueamiento'
metrics=pd.read_csv(root/'comparacion_ap.csv')
contrasts=pd.read_csv(root/'contrastes_ap.csv')
names={'directo':'Directo','alineamiento_general':'Procrustes','directo_blanco':'Directo blanqueado','alineado_blanco':'Procrustes blanqueado'}
lines=['# Comparación del cambio directo, Procrustes y coseno blanqueado','',
'Se preservaron las coordenadas de entrenamiento, operadores Procrustes, particiones espaciales, muestras de evaluación y etiquetas de Dynamic World del experimento anterior. El blanqueamiento utiliza la media y la covarianza regularizada de los embeddings de 2025 del entrenamiento de cada grupo. La misma transformación se aplica a los dos años; no se ajusta con Dynamic World. La regularización se fijó en 0,05 veces la mediana de los autovalores positivos antes de evaluar.', '',
'Todos los puntajes son ángulos derivados del coseno, para conservar la media de ángulos del experimento previo. En la geometría blanqueada, un ángulo mayor también significa mayor prioridad. El conjunto de entrenamiento sin etiquetas caracteriza la dispersión de fondo, pero no identifica clases físicas de fondo.', '',
'Las tablas presentan AP media entre grupos espaciales sobre soportes comunes. AP no equivale al porcentaje de área correctamente detectada. Las variantes se comparan con exactamente las mismas etiquetas dentro de cada tabla; distintas referencias y soportes corresponden a poblaciones diferentes.','']
for tag,title in [('3observaciones','Dominio completo con referencia anual de tres observaciones'),('huella_manuscrito','Sensibilidad en la huella histórica'),('estricta_4meses','Control temporal de cuatro meses válidos')]:
 lines.extend(['## '+title,''])
 for target in [500,1000]:
  df=metrics[(metrics.reference==tag)&(metrics.scale_m==target)&np.isclose(metrics.budget,.1)]
  if df.empty:continue
  table=df.pivot(index='source_scale_m',columns='method',values='ap')
  lines.extend([f'### Celdas comunes de {target} m','', '| Escala fuente | Directo | Procrustes | Directo blanqueado | Procrustes blanqueado |','|---|---:|---:|---:|---:|'])
  for s,row in table.iterrows():lines.append('| '+str(s)+' m | '+' | '.join(f'{row[m]:.4f}'.replace('.',',') for m in names)+' |')
  lines.extend(['','Diferencia de AP de Procrustes blanqueado respecto a Procrustes, con intervalo espacial exploratorio del 95 %:',''])
  for _,r in contrasts[(contrasts.reference==tag)&(contrasts.target_m==target)&(contrasts.comparison=='alineado_blanco frente a alineamiento_general')].iterrows():
   if r.status=='evaluado':lines.append(f"- {int(r.source_m)} m: {r.mean_ap_gain:+.4f} [{r.ci_low:+.4f}; {r.ci_high:+.4f}], {int(r.groups)} grupos y {int(r.units)} etiquetas.")
  lines.append('')
lines.extend(['## Interpretación del experimento','',
'Con la referencia anual de mayor cobertura, Procrustes sin blanqueamiento conserva la mayor AP puntual en las cuatro escalas fuente del dominio completo. En las celdas comunes de 1 km, píxeles y ventanas de 100 m permanecen prácticamente empatados. Agregar blanqueamiento reduce la AP puntual respecto a Procrustes en todas las escalas de ese dominio. La diferencia a partir de píxeles sobre soporte de 1 km es −0,0164 [−0,0819; +0,0434], por lo que ese contraste no separa claramente los métodos; en el soporte de 500 m es −0,0444 [−0,0826; −0,0026], compatible con un deterioro.', '',
'En la huella histórica de 500 m, el blanqueamiento reduce la AP de las fuentes de 10 y 100 m. La fuente de 500 m mejora puntualmente de 0,4945 a 0,5125, pero la diferencia +0,0180 tiene un intervalo [−0,0253; +0,0715]; no demuestra una mejora. El filtro temporal estricto tampoco muestra una ventaja puntual del blanqueamiento sobre Procrustes. Por tanto, esta ablación no respalda añadirlo como método preferente de la línea 1. La comparación depende de la geometría global y regularización ensayadas y no descarta todas las formas posibles de blanqueamiento.', '',
'Una explicación posible es que estandarizar direcciones de baja varianza incremente su influencia sin que aporten cambio de cobertura útil, y que el centrado modifique la geometría semántica original. Son hipótesis, no mecanismos demostrados. La decisión provisional es conservar Procrustes sin blanqueamiento, trabajar desde 10 m o mantener 100 m como alternativa próxima, y agregar los puntajes después para su revisión.', '',
'## Alcance','',
'Los intervalos usan 2.000 remuestreos pareados de bloques de 5 km dentro de cada grupo espacial, con predicciones y calibraciones fijas. Son exploratorios, sin corrección por comparaciones múltiples, incertidumbre del producto ni reajuste del modelo. La selección del mejor método requiere una evaluación posterior reservada. El control estricto tiene cobertura reducida; la comparación a 1 km no cuenta con soporte de calibración.', '',
'Los criterios temporales, el dominio y las máscaras internas son los mismos que en el experimento anterior. No se corrigió aquí la diferencia de disponibilidad entre embeddings y Dynamic World dentro de las ventanas. Esta es una ablación de geometría de comparación, no una nueva validación independiente del fenómeno.', '',
'Los modelos y puntajes originales no se modificaron. Los modelos nuevos se congelaron antes de cruzar sus puntajes con las etiquetas auxiliares. Los resultados completos, incluidos precisión y recuperación para presupuestos del 5, 10 y 20 %, se conservan en comparacion_ap.csv y en los subdirectorios de cada referencia.'])
(root/'RESUMEN_COMPARACION.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(root/'RESUMEN_COMPARACION.md')
