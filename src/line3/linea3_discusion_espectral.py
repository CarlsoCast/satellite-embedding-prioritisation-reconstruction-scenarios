"""Presentación actual: NDMI principal y NDVI como contraste negativo."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import linea3_downstream_espectral as spectral
H=Path(__file__).resolve().parent;OUT=spectral.OUT
F=spectral.eng.FAMILIES;M=['desplazamiento_medio','familia_ridge','fondo_residuo_rf']
FN={F[0]:'Pérdida verde',F[1]:'Ganancia leñosa'};MN={'desplazamiento_medio':'Media familiar','familia_ridge':'Ridge contextual','fondo_residuo_rf':'Fondo–residuo'}

def analyse():
    p=pd.read_parquet(OUT/'escenarios_oof.parquet');dc=pd.read_parquet(OUT/'decodificador_oof.parquet');initial=pd.read_parquet(OUT/'validacion2022_oof.parquet')
    assert set(p.variable)=={'ndmi','ndvi'}
    decomposition=p.assign(transformation_error=p.predicted-p.real2025,decoder_error=p.real2025-p.observed_2025)
    assert np.allclose(decomposition.transformation_error+decomposition.decoder_error,p.predicted-p.observed_2025)
    decomposition.groupby(['protocol','variable','cohort','source_family','method'],as_index=False).agg(n=('window_id','size'),signed_transformation_error=('transformation_error','mean'),signed_decoder_error=('decoder_error','mean'),mean_effect=('embedding_effect','mean')).to_csv(OUT/'descomposicion_errores.csv',index=False)
    rows=[];persist_ci=[];background=[]
    for var in ['ndmi','ndvi']:
        a=initial[(initial.protocol=='anual')&(initial.variable==var)]
        rows.append(dict(variable=var,case='Modelo en 2022',n=len(a),real2022=a.observed2022.mean(),real2025=np.nan,prediction=a.predicted2022.mean(),rmse2022=np.sqrt(np.mean((a.predicted2022-a.observed2022)**2)),rmse2025=np.nan,closer='—'))
        a=dc[(dc.protocol=='anual')&(dc.variable==var)]
        rows.append(dict(variable=var,case='Mismo modelo en 2025',n=len(a),real2022=np.nan,real2025=a.observed_2025.mean(),prediction=a.real2025.mean(),rmse2022=np.nan,rmse2025=np.sqrt(np.mean((a.real2025-a.observed_2025)**2)),closer='—'))
        for fam in F:
            a=p[(p.protocol=='anual')&(p.variable==var)&(p.cohort==fam)&(p.source_family==fam)&(p.method=='fondo_residuo_rf')].dropna(subset=['observed_2022'])
            e22=np.sqrt(np.mean((a.predicted-a.observed_2022)**2));e25=np.sqrt(np.mean((a.predicted-a.observed_2025)**2))
            rows.append(dict(variable=var,case='Transformado: '+FN[fam].lower(),n=len(a),real2022=a.observed_2022.mean(),real2025=a.observed_2025.mean(),prediction=a.predicted.mean(),rmse2022=e22,rmse2025=e25,closer='2025' if e25<e22 else '2022'))
    for key,g in p.dropna(subset=['observed_2022']).groupby(['protocol','variable','cohort','source_family','method']):
        g=g.assign(persistence_ae=abs(g.observed_2022-g.observed_2025))
        persist_ci.append(dict(zip(['protocol','variable','cohort','source_family','method'],key),**spectral.intervals(g,'persistence_ae','ae')))
    for (protocol,var,cohort),g in dc.groupby(['protocol','variable','cohort']):
        pair=g.dropna(subset=['observed_2022'])
        background.append(dict(protocol=protocol,variable=var,cohort=cohort,n=len(g),paired_n=len(pair),mean_observed_change=(pair.observed_2025-pair.observed_2022).mean(),mean_real_embedding_effect=(g.real2025-g.unchanged2022).mean()))
    tables=pd.DataFrame(rows);tables.to_csv(OUT/'tablas_directas.csv',index=False)
    pd.DataFrame(persist_ci).to_csv(OUT/'persistencia_intervalos.csv',index=False);pd.DataFrame(background).to_csv(OUT/'cambio_fondo_y_familias.csv',index=False)
    return tables

def figures():
    emb=pd.read_parquet(H/'nueva_corrida_integral/principal/02_embedding_oof.parquet')
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    names=['sin_cambio',*M]
    for ax,fam in zip(axes,F):
        ax.boxplot([emb[(emb.family==fam)&(emb.method==m)].angular_error_deg for m in names],labels=['Sin\ncambio','Media','Ridge','Fondo–\nresiduo'],showfliers=False)
        ax.set(title=FN[fam],ylabel='Error angular (°)');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Reconstrucción geométrica fuera de muestra · menor error es mejor')
    fig.savefig(OUT/'01_geometria.png',dpi=170);plt.close(fig)
    s=pd.read_csv(OUT/'metricas_escenarios.csv')
    for var,number in [('ndmi','02'),('ndvi','03')]:
        fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
        for ax,fam in zip(axes,F):
            g=s[(s.protocol=='anual')&(s.variable==var)&(s.cohort==fam)&(s.source_family==fam)].set_index('method').reindex(M)
            values=[g.unchanged_mae.iloc[0],*g.mae.tolist(),g.real2025_mae.iloc[0]]
            ax.bar(range(5),values,color=['#94a3b8','#64748b','#176b8c','#d97706','#059669']);ax.set_xticks(range(5),['2022\nsin cambio','Media','Ridge','Fondo–\nresiduo','Embedding\nreal 2025'],fontsize=9)
            ax.set(title=FN[fam]+f' · n={int(g.n.iloc[0])}',ylabel='MAE frente al índice observado de 2025');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        fig.suptitle(var.upper()+(' · Aplicación principal' if var=='ndmi' else ' · Contraste: mejora geométrica sin mejora ambiental general')+'\nMismo modelo entrenado solo en 2022',fontsize=12)
        fig.savefig(OUT/f'{number}_{var}.png',dpi=170);plt.close(fig)

def report(tables):
    text='''# Línea 3 — reconstrucción de familias y downstream de NDMI

**Versión actual:** NDMI es la aplicación principal; NDVI se conserva únicamente como contraste de los límites de transferencia. No se incorporan más variables. La búsqueda final de alternativas queda cerrada en los experimentos separados: no produjo otra referencia validada y comparable 2022/2025. Los resultados originales se conservan por trazabilidad, pero no forman parte del downstream activo ni de sus figuras.

## Qué se ejecuta

1. Se conserva la geometría del Word: dos familias, cuatro operadores, cinco pliegues de bloques de 2 km con exclusión espacial de 1 km, y diez configuraciones de sensibilidad geométrica. Los umbrales de familia y las referencias se estiman solo en entrenamiento.
2. Se preparan referencias anuales de Sentinel-2 independientemente para cada año. NDVI=(B08−B04)/(B08+B04), 10 m; NDMI=(B8A−B11)/(B8A+B11), 20 m. La referencia es la media por píxel de medianas mensuales despejadas, con al menos 4 meses y 8 días claros por píxel y 80 % de área válida en cada ventana. Sensibilidades: 6 meses/12 días y 4 meses/4 días. Se conservan los filtros SCL, margen de nube/sombra de 60 m, reflectancias positivas, offset aplicado una vez y composición de productos parciales del mismo día de la extracción auditada. Es una referencia anual de meses disponibles, no doce meses completos.
3. El modelo ambiental Ridge alpha=15, con embedding y terreno, se ajusta solo en **2022**. Se aplica congelado a E2022, E2025 real y Tfamilia(E2022), contra la referencia observada de 2025. No hay recalibración con etiquetas de 2025. Se evalúa también el ajuste retenido en 2022.
4. Se conservan los controles de no transformar, desplazamiento medio, persistencia observada 2022, otra familia y zonas estables/otros cambios. La predicción con E2025 real es un control del decodificador, no un objetivo perfecto ni un límite inferior garantizado.

Los escenarios usan pares de embeddings 2022/2025 de otras ventanas y las familias se estratifican con cambios observados: la cadena completa sigue siendo retrospectiva. No es un pronóstico disponible íntegramente en 2022, una intervención causal ni una validación con sensor independiente. Se trabaja sobre el universo original de 627 ventanas; no se reintegran las ventanas excluidas por la preparación histórica de esa base. El detalle de las referencias no aumenta la resolución de los operadores, que sigue siendo 500 m.

## Resultados: NDMI principal

El modelo de 2022 aplicado al embedding real de 2025 obtiene RMSE **0,02719** y R² **0,874** en 623 ventanas. Fondo–residuo reduce el RMSE frente a no transformar de **0,05666 a 0,03298** en pérdida verde y de **0,03615 a 0,02201** en ganancia leñosa. La reducción del MAE también tiene intervalos exploratorios favorables en ambas familias y se mantiene bajo los filtros de calidad alternativos, con muestras menores en el estricto. Supera la persistencia del índice observado 2022 sobre la submuestra común.

El desplazamiento medio es un control fuerte: MAE 0,02209 frente a 0,02340 de fondo–residuo en pérdida verde. No se demuestra superioridad general de la complejidad, ni especificidad exclusiva de familia; los controles estables/otras familias también captan parte del cambio temporal. Los intervalos son bootstrap por bloques (2000 réplicas), con modelos fijos y sin corrección múltiple. La selección de NDMI como aplicación procede de esta exploración y debe declararse, no presentarse como hipótesis confirmatoria preregistrada desde el inicio.

## NDVI: contraste negativo

La transferencia desde E2025 real conserva R² 0,931 pero tiene sesgo **−0,02180**. Los escenarios de ganancia leñosa empeoran respecto a no transformar; en pérdida verde el RMSE baja ligeramente mientras el MAE aumenta. No se sostiene una mejora ambiental general. Su inclusión muestra que acercarse al embedding observado no garantiza preservar cada propiedad espectral.

## Tablas directas

Valores reales y predichos son medias. La cercanía a cada año se calcula con RMSE por ventana, no restando promedios. Las filas transformadas muestran fondo–residuo aplicado a su familia correcta y usan únicamente ventanas con referencias de ambos años. Las tablas de todos los operadores y controles permanecen en los resultados.

'''
    for var in ['ndmi','ndvi']:
        a=tables[tables.variable==var].drop(columns='variable').copy()
        for col in ['real2022','real2025','prediction','rmse2022','rmse2025']:a[col]=a[col].map(lambda x:'—' if pd.isna(x) else f'{x:.4f}')
        text+='### '+var.upper()+'\n\n'+a.to_markdown(index=False,disable_numparse=True)+'\n\n'
    text+='''## Hipótesis: qué está apoyado y qué falta comprobar

| Hipótesis | Evidencia y límite |
|---|---|
| **Cambio de la relación embedding→índice entre años** | Hay evidencia de sesgo de transferencia: NDVI tiene −0,02180 y NDMI +0,00697 con E2025 real. La descomposición exacta separa error del escenario frente a f2022(E2025) y error de ese decodificador frente al índice observado. Explica cómo un escenario geométricamente mejor puede producir una estimación peor. No identifica por sí sola el origen físico del sesgo. |
| **Los índices responden a propiedades distintas** | NDMI usa NIR/SWIR y responde a contenido de agua de vegetación; NDVI usa rojo/NIR y expresa verdor/contraste vegetal. Es plausible que una actualización temporal conserve mejor la señal de agua/estructura que la respuesta rojo/NIR. No se ha aislado ese mecanismo ni se ha medido humedad de suelo. |
| **Mezcla urbana y fondo espectral** | Una ventana mezcla tejados, vías, sombra y vegetación. La literatura documenta sensibilidad de NDVI a fondo y atmósfera. Puede alterar la relación entre cambio de probabilidades de cobertura y cambio de índice; falta contrastar por composición y geometría de observación. No atribuimos automáticamente el resultado a saturación de NDVI: no se demostró vegetación densa saturada en esta muestra. |
| **Componente compartida de humedad, fenología u observación** | Las referencias estables también muestran respuesta temporal. La diferencia de NDMI puede incluir clima, fechas despejadas, fenología, atmósfera o calibración, no solo cobertura. Se documentan cambios por cohorte; faltan observaciones/forzantes independientes para atribuirlos. |
| **El objetivo geométrico no optimiza la propiedad ambiental** | Reducir ángulo y conservar norma no obliga a reducir el error de una función ambiental. La descomposición firmada y el contraste NDVI/NDMI muestran esa limitación. Los errores pueden reforzarse o compensarse; no se suman sus MAE como si fueran errores independientes. |

Las hipótesis no se convierten en explicaciones demostradas por citarlas. [USGS: NDMI](https://www.usgs.gov/landsat-missions/normalized-difference-moisture-index) sustenta su interpretación espectral; [NASA: índices de vegetación](https://science.nasa.gov/earth/earth-observatory/measuring-vegetation-ndvi-evi/) y [NASA: productos MOD13](https://modis.gsfc.nasa.gov/data/dataprod/mod13.php) documentan influencias de fondo y diferencias entre índices; [NASA HLS: índices y usos](https://science.nasa.gov/mission/landsat/hls-vegetation-indices/) distingue verdor y humedad. [AlphaEarth](https://arxiv.org/abs/2507.22291) describe una representación multifuente, sin garantizar invariancia de cualquier decodificador temporal. La búsqueda final no justifica incorporar otra variable ni adjudicar una causa no probada.

## Conclusión que sí admite esta versión

Las actualizaciones familiares recuperan parcialmente el estado observado del embedding y mejoran la reconstrucción retrospectiva de NDMI respecto a conservar el estado inicial. La mejora no se generaliza a NDVI ni demuestra superioridad universal del operador complejo, causalidad hidrológica o una transición física confirmada de cada ventana. NDVI se conserva como contraejemplo de esa generalización.

## Reproducción

El cuaderno `linea3_pipeline_resultados.ipynb` reúne configuración, código geométrico, preparación anual, ajuste ambiental 2022, evaluación, controles, sensibilidad y discusión. Puede ejecutarse con la caché mensual de Sentinel-2 y la base original sin autenticación ni nuevas descargas. También conserva el código de extracción opcional de reflectancias. La reconstrucción opcional de los operadores comprueba que reproduce los vectores OOF originales; las diez sensibilidades geométricas calculadas se auditan con sus protocolos y tablas. Los resultados activos se escriben únicamente en `downstream_ndmi/`. El archivo ZIP conserva la versión anterior al cambio. No se modificó el Word ni se atribuye al Word este nuevo protocolo ambiental.
'''
    (H/'linea3_analisis.md').write_text(text,encoding='utf-8')
    return text

def main():
    t=analyse();figures();report(t);print('Línea 3: NDMI principal, NDVI contraste; análisis y figuras actualizados')

if __name__=='__main__':main()
