"""Contrastes pareados de AP con bootstrap espacial y resumen de la corrida real."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import experimento as e

def ap_prepared(y,s):
    order=np.argsort(-s,kind='stable');y=y[order];s=s[order]
    ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1]
    def stat(w):
        w=w[order];pos=np.cumsum(w*y);total=np.cumsum(w);n=pos[-1]
        if n<=0:return np.nan
        good=total[ends]>0;pe=pos[ends][good];te=total[ends][good]
        return np.sum(np.diff(np.r_[0,pe])*pe/te)/n
    return stat

def paired(a,b,c,reps=2000):
    z=a.merge(b,on=['x','y'],suffixes=('_a','_b'),validate='one_to_one')
    assert (z.label_a==z.label_b).all(),'Las comparaciones no comparten etiquetas.'
    z=z[z.label_a>=0].copy();z['fold']=e.folds(z.x,z.y,c)
    diffs=[];group=[]
    for fold,g in z.groupby('fold'):
        if g.label_a.nunique()<2:continue
        blocks=(np.floor(g.x/c['block_m']).astype(str)+'_'+np.floor(g.y/c['block_m']).astype(str)).to_numpy()
        _,inverse=np.unique(blocks,return_inverse=True);nb=int(inverse.max()+1)
        aa=ap_prepared(g.label_a.to_numpy(),g.score_a.to_numpy());bb=ap_prepared(g.label_b.to_numpy(),g.score_b.to_numpy())
        group.append((aa,bb,inverse,nb));diffs.append(aa(np.ones(len(g)))-bb(np.ones(len(g))))
    if len(group)<2:return dict(status='sin_soporte',groups=len(group),units=len(z))
    rng=np.random.default_rng(20261015);boot=[]
    for _ in range(reps):
        vals=[]
        for aa,bb,inv,nb in group:
            weights=rng.multinomial(nb,np.full(nb,1/nb))[inv];vals.append(aa(weights)-bb(weights))
        boot.append(np.nanmean(vals))
    lo,hi=np.nanquantile(boot,[.025,.975])
    return dict(status='evaluado',groups=len(group),units=len(z),mean_ap_gain=float(np.mean(diffs)),ci_low=float(lo),ci_high=float(hi),bootstrap_replicates=reps)

def summarize(report,c):
    table=pd.read_csv(report/'04_metricas_medias.csv');rows=[]
    for target in [500,1000]:
        for source in [s for s in c['scales_m'] if s<=target]:
            p=report/f'evaluacion/{source}/predicciones_comun_{target}.parquet'
            if not p.exists():continue
            z=pd.read_parquet(p)
            for method in ['alineamiento_general','referencias_30']:
                a=z[['x','y','label',method]].rename(columns={method:'score'});b=z[['x','y','label','directo']].rename(columns={'directo':'score'})
                if a.score.isna().any():continue
                rows.append(dict(comparison=f'{method} frente a directo',target_m=target,source_m=source,**paired(a,b,c)))
        pixel=report/f'evaluacion/10/predicciones_comun_{target}.parquet'
        if pixel.exists():
            base=pd.read_parquet(pixel)
            for source in [s for s in c['scales_m'] if 10<s<=target]:
                path=report/f'evaluacion/{source}/predicciones_comun_{target}.parquet'
                if not path.exists():continue
                other=pd.read_parquet(path)
                for method in ['directo','alineamiento_general','referencias_30']:
                    a=other[['x','y','label',method]].rename(columns={method:'score'});b=base[['x','y','label',method]].rename(columns={method:'score'})
                    if a.score.isna().any() or b.score.isna().any():continue
                    rows.append(dict(comparison=f'ventana frente a píxel: {method}',target_m=target,source_m=source,**paired(a,b,c)))
    pd.DataFrame(rows).to_csv(report/'07_contrastes_pareados_ap.csv',index=False,encoding='utf-8-sig')
    e.dump(report/'07_protocolo_intervalos.json',dict(replicates=2000,unit='Bloques de 5 km remuestreados dentro de cada grupo espacial.',statistic='Diferencia de AP media entre grupos, sobre celdas y etiquetas comunes.',scope='Predicciones, calibración y modelos fijos; intervalos exploratorios sin corrección múltiple. No incluye incertidumbre del producto DW ni reajuste del modelo.'))
    print('\nREPORTE',report)
    print(table[(table.view=='comun_1000')&np.isclose(table.budget,.1)].pivot(index='source_scale_m',columns='method',values='ap').round(4).to_string())
    print(pd.DataFrame(rows).round(4).to_string(index=False))

if __name__=='__main__':
    c=e.config(e.HOME/'config.json');root=Path(c['out'])
    for report in [root,root/'sensibilidad_referencia/3observaciones']:
        if (report/'04_metricas_medias.csv').exists():summarize(report,c)
