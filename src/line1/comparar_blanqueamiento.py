"""Ablación nueva con modelos y particiones previos fijos y geometría común 2025.

No modifica el experimento congelado. No consulta DW durante ajuste/puntuación.
Los puntajes son ángulos derivados del coseno en las respectivas geometrías;
se conserva así la misma operación de promediar ángulos en soportes comunes.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import experimento as e
from resumir_corrida import paired

METHODS=['directo','alineamiento_general','directo_blanco','alineado_blanco']
OUT=e.HOME/'salidas_blanqueamiento'

def fit_whitening(y,ridge_fraction=.05):
    y=e.unit(y).astype(float);mu=y.mean(0)
    cov=np.cov(y-mu,rowvar=False);vals,vec=np.linalg.eigh(cov)
    ridge=ridge_fraction*np.median(vals[vals>0])
    w=(vec*(1/np.sqrt(np.maximum(vals+ridge,1e-12))))@vec.T
    return mu,w,vals,float(ridge)

def whitened_angles(x,y,r,mu,w):
    x=e.unit(x).astype(float);y=e.unit(y).astype(float)
    wy=(y-mu)@w
    return e.angle((x-mu)@w,wy),e.angle((x@r-mu)@w,wy)

def train_models(scale,c):
    root=Path(c['out']);mp=OUT/f'modelos/{scale}';mp.mkdir(parents=True,exist_ok=True)
    if all((mp/f'fold_{f}.npz').exists() for f in range(c['folds'])):return
    # Exactamente las coordenadas muestreadas del entrenamiento original.
    coords=[]
    for f in range(c['folds']):
        z=pd.read_parquet(root/f'modelos/{scale}/fold_{f}.train.parquet');z['training_fold']=f
        assert e.train_mask(z.x,z.y,f,scale,c).all()
        coords.append(z)
    coords=pd.concat(coords,ignore_index=True)
    coords['tile']=(np.floor(coords.x/1000).astype(int)*1000).astype(str)+'_'+(np.floor(coords.y/1000).astype(int)*1000).astype(str)
    parts=[]
    for i,(tile,g) in enumerate(coords.groupby('tile'),1):
        y=pd.read_parquet(root/f'embeddings/{scale}/{tile}.parquet',columns=['x','y']+e.E25)
        z=g.merge(y,on=['x','y'],validate='many_to_one')
        assert len(z)==len(g)
        parts.append(z)
        if i%300==0:print('Muestra recuperada',scale,i,'teselas',flush=True)
    data=pd.concat(parts,ignore_index=True)
    for f,g in data.groupby('training_fold'):
        mu,w,vals,ridge=fit_whitening(g[e.E25].to_numpy())
        r=np.load(root/f'modelos/{scale}/fold_{f}.npz')['alineamiento_general']
        np.savez_compressed(mp/f'fold_{f}.npz',mu=mu,W=w,eigenvalues=vals,ridge=ridge,R=r,n_train=len(g))
    print('Blanqueamiento ajustado',scale,'m',flush=True)

def score_scale(scale,c):
    root=Path(c['out']);dest=OUT/f'puntajes/{scale}';dest.mkdir(parents=True,exist_ok=True)
    ops={f:dict(np.load(OUT/f'modelos/{scale}/fold_{f}.npz')) for f in range(c['folds'])}
    commons={t:[] for t in [500,1000] if scale<=t}
    for i,path in enumerate(sorted((root/f'embeddings/{scale}').glob('*.parquet')),1):
        target=dest/path.name
        if target.exists():z=pd.read_parquet(target)
        else:
            a=pd.read_parquet(path,columns=['x','y']+e.E22+e.E25)
            z=pd.read_parquet(root/f'puntajes/{scale}/{path.name}',columns=['x','y','fold','directo','alineamiento_general'])
            assert np.array_equal(a[['x','y']].to_numpy(),z[['x','y']].to_numpy())
            z['directo_blanco']=np.nan;z['alineado_blanco']=np.nan
            for f,g in z.groupby('fold'):
                ix=g.index;m=ops[int(f)]
                raw,aligned=whitened_angles(a.loc[ix,e.E22].to_numpy(),a.loc[ix,e.E25].to_numpy(),m['R'],m['mu'],m['W'])
                z.loc[ix,'directo_blanco']=raw;z.loc[ix,'alineado_blanco']=aligned
            assert not z[METHODS].isna().any().any()
            z.to_parquet(target,index=False)
        for t in commons:commons[t].append(e.aggregate_scores(z,scale,t,c))
        if i%250==0:print('Puntuación',scale,'m',i,'teselas',flush=True)
    for t,parts in commons.items():
        p=OUT/f'puntajes_comunes/fuente_{scale}_objetivo_{t}.parquet';p.parent.mkdir(exist_ok=True)
        pd.concat(parts,ignore_index=True).to_parquet(p,index=False)
    print('Escala completada',scale,flush=True)

def evaluate(c):
    root=Path(c['out']);allmeans=[];allcontrasts=[]
    for tag,prior in [('3observaciones',root/'comparacion_comun/3observaciones'),('estricta_4meses',root/'comparacion_comun/estricta_4meses'),('huella_manuscrito',root/'sensibilidad_referencia/huella_manuscrito')]:
        report=OUT/tag;report.mkdir(exist_ok=True);rows=[];contrasts=[]
        for t in [500,1000]:
            for s in [v for v in c['scales_m'] if v<=t]:
                source=prior/f'evaluacion/{s}/predicciones_comun_{t}.parquet'
                if not source.exists():continue
                # Reutiliza exactamente las etiquetas, calibraciones y población previas.
                old=pd.read_parquet(source,columns=['x','y','label','proxy','directo','alineamiento_general'])
                old['fold']=e.folds(old.x,old.y,c)
                scores=pd.read_parquet(OUT/f'puntajes_comunes/fuente_{s}_objetivo_{t}.parquet')
                z=old.merge(scores[['x','y','directo_blanco','alineado_blanco','directo','alineamiento_general']],on=['x','y'],validate='one_to_one',suffixes=('','_check'))
                assert len(z)==len(old)
                for method in METHODS[:2]:assert np.allclose(z[method],z[method+'_check'],atol=1e-7)
                z=z.drop(columns=['directo_check','alineamiento_general_check'])
                dest=report/f'evaluacion/{s}';dest.mkdir(parents=True,exist_ok=True)
                z.to_parquet(dest/f'predicciones_comun_{t}.parquet',index=False)
                for f,g in z.groupby('fold'):
                    rr=e.metric_rows(g,int(f),t,f'comun_{t}',c)
                    for r in rr:r['source_scale_m']=s
                    rows.extend(rr)
                for a,b in [('alineamiento_general','directo'),('directo_blanco','directo'),('alineado_blanco','directo'),('alineado_blanco','alineamiento_general'),('alineado_blanco','directo_blanco')]:
                    left=z[['x','y','label',a]].rename(columns={a:'score'});right=z[['x','y','label',b]].rename(columns={b:'score'})
                    contrasts.append(dict(reference=tag,target_m=t,source_m=s,comparison=f'{a} frente a {b}',**paired(left,right,c)))
        if not rows:continue
        df=pd.DataFrame(rows);df.to_csv(report/'03_metricas_por_grupo.csv',index=False,encoding='utf-8-sig')
        keys=['view','scale_m','source_scale_m','method','budget']
        mean=df.groupby(keys,as_index=False)[['ap','spearman','precision','recall','lift']].mean()
        counts=df.groupby(keys,as_index=False).agg(valid_ap_folds=('ap','count'),reported_folds=('fold','nunique'))
        mean=mean.merge(counts,on=keys);mean.to_csv(report/'04_metricas_medias.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(contrasts).to_csv(report/'07_contrastes_pareados_ap.csv',index=False,encoding='utf-8-sig')
        allmeans.append(mean.assign(reference=tag));allcontrasts.extend(contrasts)
        print('\nRESULTADOS',tag,mean[np.isclose(mean.budget,.1)].pivot(index=['view','source_scale_m'],columns='method',values='ap').round(5).to_string(),flush=True)
    pd.concat(allmeans,ignore_index=True).to_csv(OUT/'comparacion_ap.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(allcontrasts).to_csv(OUT/'contrastes_ap.csv',index=False,encoding='utf-8-sig')

def checks():
    rng=np.random.default_rng(11);y=rng.normal(size=(200,64));mu,w,vals,ridge=fit_whitening(y)
    assert np.allclose(w,w.T) and ridge>0
    normalized=e.unit(y).astype(float);cov=np.cov(normalized-mu,rowvar=False)
    assert np.allclose(w@(cov+ridge*np.eye(64))@w,np.eye(64),atol=1e-8)
    q=np.linalg.qr(rng.normal(size=(64,64)))[0];x=normalized@q.T
    direct,aligned=whitened_angles(x,normalized,q,mu,w)
    assert aligned.max()<.04 and direct.mean()>1
    return dict(symmetric_whitening=True,regularized_covariance_identity=True,exact_alignment_in_shared_geometry=True)

def main():
    OUT.mkdir(exist_ok=True);c=e.config(e.HOME/'config.json');e.METHODS=METHODS
    if (OUT/'puntajes_congelados.json').exists():
        verify_frozen()
        if not (OUT/'comparacion_ap.csv').exists():evaluate(c)
        return
    e.dump(OUT/'protocolo.json',dict(methods=METHODS,configuration=c,whitening='Media y covarianza de 2025 de las mismas coordenadas de entrenamiento del ajuste original. W simétrica regularizada compartida por ambos años y variantes. Sin DW.',ridge_fraction=.05,score='Ángulo en grados del coseno original o blanqueado. Agregación por media de ángulos, igual que el experimento original.',reference='Evaluación con poblaciones, calibraciones y etiquetas congeladas previas.',proposed_before_metrics=True,checks=checks()))
    baseline=Path(c['out'])/'02_puntajes_congelados.json'
    manifest=json.loads(baseline.read_text(encoding='utf8'))['files']
    for path,digest in manifest.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('Cambió entrada congelada '+path)
    e.dump(OUT/'integridad_entrada.json',dict(verified=len(manifest),all_hashes_match=True))
    for s in c['scales_m']:train_models(s,c);score_scale(s,c)
    # Congelar modelos y predicciones nuevos antes de leer referencias DW.
    paths=sorted((OUT/'modelos').rglob('*.npz'))+sorted((OUT/'puntajes_comunes').glob('*.parquet'))
    e.dump(OUT/'puntajes_congelados.json',dict(files=e.manifest(paths),model_selection_without_dw=True))
    evaluate(c)
    e.dump(OUT/'estado.json',dict(status='completo',all_scales=c['scales_m'],four_variants=True))
    print('COMPLETO',OUT,flush=True)

def verify_frozen():
    for manifest in [Path(e.config(e.HOME/'config.json')['out'])/'02_puntajes_congelados.json',OUT/'puntajes_congelados.json']:
        frozen=json.loads(manifest.read_text(encoding='utf8'))
        for path,digest in frozen['files'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
                raise ValueError('Entrada congelada modificada: '+path)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--evaluar',action='store_true',help='Recalcular métricas desde puntajes congelados y ejecutar análisis robusto.')
    mode.add_argument('--analizar',action='store_true',help='Solo análisis robusto de las evaluaciones guardadas; no reentrena.')
    mode.add_argument('--sensibilidad-confianza',action='store_true',help='Comparar niveles 90/95/99 y convergencia del remuestreo sin cambiar el detector.')
    parser.add_argument('--replicas',type=int,default=2000)
    parser.add_argument('--replicas-confianza',type=int,default=5000)
    args=parser.parse_args()
    if args.sensibilidad_confianza:
        verify_frozen()
        from sensibilidad_confianza import run as confidence_run
        confidence_run(args.replicas_confianza)
        raise SystemExit(0)
    if args.evaluar or args.analizar:
        verify_frozen()
        if args.evaluar:
            e.METHODS=METHODS;c=e.config(e.HOME/'config.json');evaluate(c)
            e.dump(OUT/'estado.json',dict(status='completo',all_scales=c['scales_m'],four_variants=True))
    else:main()
    from analisis_robusto import run
    run(args.replicas)
