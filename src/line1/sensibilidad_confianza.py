"""Niveles 90/95/99 y convergencia 2.000/5.000 remuestreos, sin reajuste.

El 95 % permanece como nivel principal. No se optimiza el nivel para obtener
separación de cero. La comparación de prefijos evalúa estabilidad numérica,
no constituye una validación independiente ni agrega información territorial.
"""
from pathlib import Path
import hashlib,json,argparse
import numpy as np
import pandas as pd
import analisis_robusto as a
import experimento as e

def run(reps=5000):
    if reps<5000:raise ValueError('Se requieren al menos 5.000 intentos para esta sensibilidad.')
    root=a.ROOT;dest=root/'sensibilidad_confianza';dest.mkdir(exist_ok=True)
    e.dump(dest/'estado.json',{'status':'ejecutando','attempts':reps})
    c=e.config(e.HOME/'config.json');rows=[];convergence=[];audit={}
    original=json.loads((root/'puntajes_congelados.json').read_text(encoding='utf8'))
    for p,digest in original['files'].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=digest:raise ValueError('Predicción o modelo congelado modificado.')
    code_hash=hashlib.sha256(Path(a.__file__).read_bytes()).hexdigest()
    for tag in ['3observaciones','huella_manuscrito','estricta_4meses']:
        for target in [500,1000]:
            frames={};input_hashes={}
            for source in [s for s in c['scales_m'] if s<=target]:
                p=root/tag/f'evaluacion/{source}/predicciones_comun_{target}.parquet'
                if p.exists():frames[source]=pd.read_parquet(p);input_hashes[str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
            if not frames:continue
            signature=dict(inputs=input_hashes,code_hash=code_hash,seed=c['seed'],attempts=reps,block_m=c['block_m'])
            cache=dest/f'{tag}_{target}_bootstrap.npz';meta=cache.with_suffix('.json')
            if not (cache.exists() and meta.exists() and json.loads(meta.read_text(encoding='utf8'))==signature):
                a.analyze_family(frames,c,reps,c['seed'],bootstrap_path=cache)
                e.dump(meta,signature)
            data=np.load(cache,allow_pickle=False);observed=data['observed'];boot=data['bootstrap'];family=data['family']
            for level in [.90,.95,.99]:
                marginal,lo,hi,q=a.simultaneous_ci(observed,boot,confidence=level)
                for j,(s,ma,t,mb) in enumerate(family):
                    rows.append(dict(reference=tag,target_m=target,source_a_m=int(s),method_a=a.METHODS[int(ma)],source_b_m=int(t),method_b=a.METHODS[int(mb)],confidence=level,ap_difference=float(observed[j]),ci_low=float(marginal[0,j]),ci_high=float(marginal[1,j]),simultaneous_low=float(lo[j]),simultaneous_high=float(hi[j]),family_size=len(family),valid_replicates=len(boot),invalid_replicates=reps-len(boot),n_groups=int(data['n_groups']),marginal_excludes_zero=bool(marginal[0,j]>0 or marginal[1,j]<0),simultaneous_excludes_zero=bool(lo[j]>0 or hi[j]<0),max_t_quantile=q))
            for count in [2000,reps]:
                prefix=boot[data['attempt_indices']<count]
                marginal,lo,hi,q=a.simultaneous_ci(observed,prefix)
                for j,(s,ma,t,mb) in enumerate(family):convergence.append(dict(reference=tag,target_m=target,source_a_m=int(s),method_a=a.METHODS[int(ma)],source_b_m=int(t),method_b=a.METHODS[int(mb)],attempts=count,valid_replicates=len(prefix),confidence=.95,ap_difference=float(observed[j]),ci_low=float(marginal[0,j]),ci_high=float(marginal[1,j]),simultaneous_low=float(lo[j]),simultaneous_high=float(hi[j])))
            audit.update(input_hashes);print('Sensibilidad lista',tag,target,'m',flush=True)
    result=pd.DataFrame(rows);stability=pd.DataFrame(convergence)
    result.to_csv(dest/'01_niveles_confianza.csv',index=False,encoding='utf-8-sig')
    stability.to_csv(dest/'02_convergencia_remuestreo.csv',index=False,encoding='utf-8-sig')
    for p,digest in audit.items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=digest:raise ValueError('Cambió una evaluación de entrada.')
    e.dump(dest/'protocolo.json',dict(confidence_levels=[.90,.95,.99],primary_level=.95,attempts=reps,seed=c['seed'],input_hashes=audit,analysis_code_sha256=code_hash,cache_npz_saved=True,interpretation=['El nivel de confianza es una convención de comunicación de incertidumbre, no un hiperparámetro del detector.','No se elige un nivel según su separación de cero.','Se reutilizan las mismas réplicas para todos los niveles y comparaciones pareadas.','Se conservan las mismas familias y el mismo procedimiento espacial del análisis robusto.','El prefijo de 2.000 intentos está incluido en la corrida de 5.000: es convergencia numérica, no repetición independiente.','El soporte espacial, las máscaras y las limitaciones de la referencia no se mejoran aumentando el número de réplicas.','Las réplicas sin ambas clases en cualquier grupo se excluyen; los intervalos son condicionales a esa disponibilidad.','No se incluye incertidumbre de reajustar los operadores ni del producto satelital.']))
    write_summary(dest,result,stability)
    e.dump(dest/'estado.json',dict(status='completo',attempts=reps,confidence_levels=[.9,.95,.99],input_hashes_verified=True))

def write_summary(dest,result,stability):
    lines=['# Sensibilidad del nivel de confianza','',
    'El 95 % es el nivel principal y convencional de comunicación de la incertidumbre. Los extremos del intervalo se calculan a partir de los remuestreos; no se fijan manualmente. El 90 % produce intervalos más estrechos y el 99 % más amplios. Esta sensibilidad no debe usarse para escoger el nivel que hace favorable un resultado.','',
    'La corrida conserva 5.000 intentos de remuestreo de bloques de 5 km dentro de los grupos espaciales originales. Los modelos, puntajes, calibraciones y etiquetas permanecen fijos. Los intervalos marginales describen un contraste; los simultáneos buscan cubrir la familia de contrastes de AP de cada referencia y soporte. Son aproximaciones exploratorias condicionales, no garantías de cobertura exacta.','',
    '## Cambios desde píxeles sobre celdas comunes de 500 m','',
    '| Referencia | Comparación | Confianza | Diferencia AP | Intervalo marginal | Intervalo simultáneo |',
    '|---|---|---:|---:|---|---|']
    selected=result[(result.target_m==500)&(result.source_a_m==10)&(result.source_b_m==10)&result.method_a.isin(['alineamiento_general','alineado_blanco'])]
    for _,r in selected.iterrows():lines.append(f'| {r.reference} | {r.method_a} menos {r.method_b} | {r.confidence:.0%} | {r.ap_difference:+.4f} | [{r.ci_low:+.4f}; {r.ci_high:+.4f}] | [{r.simultaneous_low:+.4f}; {r.simultaneous_high:+.4f}] |')
    lines.extend(['','## Huella histórica','',
    'Es la restricción geográfica a las ubicaciones de las ventanas históricas del manuscrito, no un análisis de agua. Sobre soporte de 500 m quedan 795 celdas, de las cuales 397 tienen etiqueta binaria auxiliar, con cinco grupos evaluables. Se mantienen los operadores aprendidos en el dominio completo; la evaluación no reproduce el ajuste histórico.','',
    '## Justificación para el manuscrito','',
    'Se recomienda reportar estimaciones puntuales e intervalos bootstrap espaciales del 95 %, acompañados de sensibilidad al 90 y 99 %. Para los contrastes de AP se presentan además bandas simultáneas por familia. La interpretación se apoya en el tamaño de la diferencia, su estabilidad espacial, el presupuesto de revisión y la referencia utilizada; no depende únicamente de que un extremo del intervalo cruce cero.', '',
    'Exploratorio significa que las alternativas y los análisis se examinaron durante el desarrollo del estudio, con etiquetas auxiliares, sin una prueba confirmatoria reservada. Los intervalos cuantifican incertidumbre espacial condicional a los modelos y referencias existentes. No representan la incertidumbre total del fenómeno.', '',
    '## Estabilidad del remuestreo',''])
    keys=['reference','target_m','source_a_m','method_a','source_b_m','method_b']
    first=stability[stability.attempts==2000];last=stability[stability.attempts==stability.attempts.max()]
    pair=first.merge(last,on=keys,suffixes=('_2000','_5000'),validate='one_to_one')
    for tag in result.reference.unique():
        z=pair[pair.reference==tag];m=max((z.ci_low_2000-z.ci_low_5000).abs().max(),(z.ci_high_2000-z.ci_high_5000).abs().max());s=max((z.simultaneous_low_2000-z.simultaneous_low_5000).abs().max(),(z.simultaneous_high_2000-z.simultaneous_high_5000).abs().max())
        lines.append(f'- {tag}: cambio máximo de un extremo de intervalo al pasar de 2.000 a 5.000 intentos, {m:.4f} AP marginal y {s:.4f} AP simultáneo. Ver tabla de convergencia para cada contraste; las variaciones mayores en poblaciones pequeñas deben describirse como incertidumbre numérica adicional.')
    (dest/'INTERPRETACION_CONFianza.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--replicas',type=int,default=5000)
    args=parser.parse_args()
    try:run(args.replicas)
    except Exception as exc:
        e.dump(a.ROOT/'sensibilidad_confianza/estado.json',dict(status='fallido',error=str(exc)));raise
