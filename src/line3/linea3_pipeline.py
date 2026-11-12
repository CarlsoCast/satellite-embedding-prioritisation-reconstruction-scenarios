"""Línea 3 completa desde bases preparadas; réplica y sensibilidad espacial."""
from pathlib import Path
import json,hashlib,os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent
REPO_ROOT=HERE.parents[1]
ROOT=Path(os.environ.get('SATEMB_LINE3_DATA',REPO_ROOT/'data/raw/line3')).resolve()
OUT=HERE/'nueva_corrida_integral';OUT.mkdir(exist_ok=True)
PREP=ROOT/'resultados_60_downstream_unificado_lst_bsa'
BASE=PREP/'03_base_autocontenida.parquet'
RAW=[ROOT/'datos_para_procesar/alphaearth_dynamicworld_residual/ventanas_dynamicworld_alphaearth_residual.parquet']
STATE=['water','trees','grass','flooded_vegetation','crops','shrub_and_scrub','built','bare']
AE22=[f'emb_{i}_2022' for i in range(1,65)];AE25=[f'emb_{i}_2025' for i in range(1,65)]
DW22=[f'{c}_2022' for c in STATE]
HYDRO=['elevacion','pendiente','twi','stream_distance','tan_curvatura','perfil_curvatura','accum_flujo','drenaje','longitud_flujo','elev_range_3x3','elev_std_3x3','elev_diff_mean_3x3']
FAMILIES=['densificacion_perdida_verde','construido_a_vegetacion_lenosa']
SEED=20260920;N_SPLITS=5;BUFFER=1000.;Q=.85;PURITY=.40;PCA_IN=16;PCA_V=12;TREES=300;MAX_ANGLE=30.;STABLE_Q=.30

def prepare_base():
    """Control de la base preparada; reconstrucción local si falta el consolidado."""
    if BASE.exists():
        d=pd.read_parquet(BASE)
    else:
        raw=next((p for p in RAW if p.exists()),None)
        if raw is None:raise FileNotFoundError('Falta la base de ventanas AlphaEarth + Dynamic World')
        d=pd.read_parquet(raw)
        required=['window_id','coverage','xmin','ymin','xmax','ymax','x_centroid','y_centroid']+AE22+AE25
        d=d.dropna(subset=required);d=d[(d.coverage>=.95)&np.isclose(d.xmax-d.xmin,500)&np.isclose(d.ymax-d.ymin,500)].copy()
        for columns in [AE22,AE25]:
            a=d[columns].to_numpy(float);d[columns]=a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)
        d['spatial_block']=np.floor(d.x_centroid/2000).astype(int).astype(str)+'_'+np.floor(d.y_centroid/2000).astype(int).astype(str)
        for filename in ['02_controles_hidrologicos.parquet','01_objetivos_lst_bsa_anuales.parquet']:
            p=PREP/filename
            if not p.exists():raise FileNotFoundError('Falta la extracción persistida de terreno o productos satelitales: '+str(p))
            extra=pd.read_parquet(p);assert extra.window_id.is_unique
            columns=['window_id']+[c for c in extra if c!='window_id' and c not in d]
            d=d.merge(extra[columns],on='window_id',validate='one_to_one')
        d=d[(d.lst_valid_fraction_2022>=.20)&(d.lst_valid_fraction_2025>=.20)&(d.bsa_month_support_2022>=4)&(d.bsa_month_support_2025>=4)].dropna(subset=HYDRO+['lst_c_2022','lst_c_2025','bsa_2022','bsa_2025']).reset_index(drop=True)
        # Escribir solo en el área de resultados nueva.
        d.to_parquet(OUT/'00_base_reconstruida.parquet',index=False)
    assert d.window_id.is_unique and (d.coverage>=.95).all()
    for year in [2022,2025]:
        assert (d[f'lst_valid_fraction_{year}']>=.20).all()
        assert (d[f'bsa_month_support_{year}']>=4).all()
    return d

ENGINE_SHA256='a9bb164cbcb1edb007cff10f6837969a33c28748a9e2b23cc706559e0c653a74'

def unit(a):
 a=np.asarray(a,float); return a/np.clip(np.linalg.norm(a,axis=1,keepdims=True),1e-12,None)

def project(x,v): return v-(v*x).sum(1,keepdims=True)*x

def logmap(x,y):
 x,y=unit(x),unit(y); c=np.clip((x*y).sum(1,keepdims=True),-1+1e-7,1-1e-7); t=np.arccos(c); return (y-c*x)*(t/np.clip(np.sin(t),1e-8,None))

def expmap(x,v):
 x=unit(x); v=project(x,v); t=np.linalg.norm(v,axis=1,keepdims=True); z=np.cos(t)*x+np.sin(t)*v/np.clip(t,1e-12,None); z[t[:,0]<1e-10]=x[t[:,0]<1e-10]; return unit(z)

def angle(x,y): return np.degrees(np.arccos(np.clip((unit(x)*unit(y)).sum(1),-1,1)))

def cap(x,v):
 v=project(unit(x),v); n=np.linalg.norm(v,axis=1,keepdims=True); return v*np.minimum(1,np.radians(MAX_ANGLE)/np.clip(n,1e-12,None))

def downmodel(): return Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),('ridge',Ridge(alpha=15.))])

def getdata():
 d=prepare_base()
 raw=next((pd.read_parquet(p) for p in RAW if p.exists()),None)
 needed=['built_2022','delta_built','delta_vegetation','dw_js_divergence','delta_water','delta_flooded_vegetation']+DW22+[f'{x}_2025' for x in ['trees','grass','flooded_vegetation','crops','shrub_and_scrub']]
 missing=[c for c in needed if c not in d]
 if missing:
  if raw is None: raise FileNotFoundError('Falta la fuente Dynamic World para completar la base.')
  d=d.merge(raw[['window_id']+[c for c in missing if c in raw]].drop_duplicates('window_id'),on='window_id',how='left')
 required=AE22+AE25+DW22+HYDRO+needed+['window_id','spatial_block','x_centroid','y_centroid','lst_c_2025','bsa_2025']
 bad=[c for c in required if c not in d]
 if bad: raise KeyError(f'Columnas ausentes: {bad}')
 d['dominant_2022']=d[DW22].idxmax(axis=1).str.replace('_2022','',regex=False)
 veg25=[f'{x}_2025' for x in ['trees','grass','flooded_vegetation','crops','shrub_and_scrub']]
 d['vegetation_subtype_2025']=d[veg25].idxmax(axis=1).str.replace('_2025','',regex=False)
 return d.dropna(subset=required).reset_index(drop=True)

def masks(d,tr):
 t=d.iloc[tr]; bh=t.delta_built.quantile(Q); bl=t.delta_built.quantile(1-Q); vh=t.delta_vegetation.quantile(Q); vl=t.delta_vegetation.quantile(1-Q)
 dom=d.dominant_2022; subtype25=d.vegetation_subtype_2025
 m={'densificacion_perdida_verde':((d.delta_built>=bh)&(d.delta_vegetation<=vl)&dom.eq('built')).to_numpy(),
    'construido_a_vegetacion_lenosa':((d.delta_built<=bl)&(d.delta_vegetation>=vh)&(d.built_2022>=PURITY)&dom.eq('built')&subtype25.isin(['trees','shrub_and_scrub'])).to_numpy()}
 # El fondo se define con el 30 % de menor deriva auxiliar dentro de cada entrenamiento.
 # Las escalas de deriva se ajustan solo con el entrenamiento de este pliegue.
 # El bloque de prueba no participa en la selección de referencias estables.
 def empirical_rank(values):
  reference=np.sort(np.asarray(values, dtype=float)[tr])
  return np.searchsorted(reference, np.asarray(values, dtype=float), side='right') / len(reference)
 drift=(empirical_rank(d.dw_js_divergence)+empirical_rank(d.delta_built.abs())+empirical_rank(d.delta_vegetation.abs()))
 stable=(drift<=np.quantile(drift[tr],STABLE_Q))
 return m,stable

def features(d,x,fit,ix):
 p=PCA(n_components=min(PCA_IN,len(fit)-1,64),whiten=True,random_state=SEED).fit(x[fit]); s=StandardScaler().fit(d.iloc[fit][DW22]); return np.c_[p.transform(x[ix]),s.transform(d.iloc[ix][DW22])],p,s

def ridge_operator(d,x,v,fam,fit,te):
 ff,p,s=features(d,x,fit,fam); ft=np.c_[p.transform(x[te]),s.transform(d.iloc[te][DW22])]; mean=v[fam].mean(0); model=Ridge(alpha=30.).fit(ff,v[fam]-mean); return expmap(x[te],cap(x[te],mean+model.predict(ft)))

def rf_operator(d,x,v,fam,stable,fit,te):
 fs,p,s=features(d,x,fit,stable); ff=np.c_[p.transform(x[fam]),s.transform(d.iloc[fam][DW22])]; ft=np.c_[p.transform(x[te]),s.transform(d.iloc[te][DW22])]
 bp=PCA(n_components=min(PCA_V,len(stable)-1,64),random_state=SEED).fit(v[stable]); bg=RandomForestRegressor(n_estimators=TREES,min_samples_leaf=3,max_features=.75,n_jobs=1,random_state=SEED).fit(fs,bp.transform(v[stable])); b_f=bp.inverse_transform(bg.predict(ff)); b_t=bp.inverse_transform(bg.predict(ft)); resid=v[fam]-b_f
 rp=PCA(n_components=min(PCA_V,len(fam)-1,64),random_state=SEED).fit(resid); rf=RandomForestRegressor(n_estimators=TREES,min_samples_leaf=3,max_features=.75,n_jobs=1,random_state=SEED+1).fit(ff,rp.transform(resid)); r=rp.inverse_transform(rf.predict(ft)); lim=np.quantile(np.linalg.norm(resid,axis=1),.95); r*=np.minimum(1,lim/np.clip(np.linalg.norm(r,axis=1,keepdims=True),1e-12,None)); return expmap(x[te],cap(x[te],project(x[te],b_t)+project(x[te],r)))

CONFIGS=[
 dict(name='principal',q=.85,buffer=1000.,angle=30.,stable=.30),
 dict(name='cola80',q=.80,buffer=1000.,angle=30.,stable=.30),
 dict(name='cola90',q=.90,buffer=1000.,angle=30.,stable=.30),
 dict(name='buffer0',q=.85,buffer=0.,angle=30.,stable=.30),
 dict(name='buffer500',q=.85,buffer=500.,angle=30.,stable=.30),
 dict(name='buffer1500',q=.85,buffer=1500.,angle=30.,stable=.30),
 dict(name='limite15',q=.85,buffer=1000.,angle=15.,stable=.30),
 dict(name='limite45',q=.85,buffer=1000.,angle=45.,stable=.30),
 dict(name='fondo20',q=.85,buffer=1000.,angle=30.,stable=.20),
 dict(name='fondo40',q=.85,buffer=1000.,angle=30.,stable=.40),
]
METHODS=['sin_cambio','desplazamiento_medio','familia_ridge','fondo_residuo_rf']
LABELS={'sin_cambio':'Sin cambio','desplazamiento_medio':'Desplazamiento medio','familia_ridge':'Ridge contextual','fondo_residuo_rf':'Fondo–residuo'}
FLABELS={'densificacion_perdida_verde':'Densificación con pérdida verde','construido_a_vegetacion_lenosa':'Construido a vegetación leñosa'}
COLORS={'sin_cambio':'#94a3b8','desplazamiento_medio':'#64748b','familia_ridge':'#176b8c','fondo_residuo_rf':'#d97706'}

def set_configuration(config):
    global Q,BUFFER,MAX_ANGLE,STABLE_Q
    Q=config['q'];BUFFER=config['buffer'];MAX_ANGLE=config['angle'];STABLE_Q=config['stable']

def spatial_splits(d):
    xy=d[['x_centroid','y_centroid']].to_numpy(float)
    for fold,(train0,test) in enumerate(GroupKFold(n_splits=5).split(d,groups=d.spatial_block.astype(str)),1):
        distances=np.sqrt(((xy[train0,None]-xy[None,test])**2).sum(2))
        train=train0[distances.min(1)>BUFFER]
        assert not set(d.spatial_block.iloc[train])&set(d.spatial_block.iloc[test])
        if len(train):assert distances.min(1)[distances.min(1)>BUFFER].min()>BUFFER
        yield fold,train,test

def summarize_results(emb,down):
    es=emb.groupby(['family','method'],as_index=False).agg(windows=('window_id','nunique'),blocks=('spatial_block','nunique'),
        median_error_deg=('angular_error_deg','median'),median_paired_improvement_deg=('improvement_deg','median'),improved_fraction=('improvement_deg',lambda z:float((z>0).mean())))
    rows=[]
    for (family,target,method),g in down.groupby(['family','target','method']):
        rows.append(dict(family=family,target=target,method=method,windows=g.window_id.nunique(),blocks=g.spatial_block.nunique(),
            mae_observed=mean_absolute_error(g.observed_2025,g.predicted),mae_reference_model=mean_absolute_error(g.oracle,g.predicted),
            reference_model_mae_observed=mean_absolute_error(g.observed_2025,g.oracle),
            rho_reference_model=float(spearmanr(g.oracle,g.predicted).statistic),r2_reference_model=r2_score(g.oracle,g.predicted)))
    return es,pd.DataFrame(rows)

def run_configuration(d,config):
    set_configuration(config);folder=OUT/config['name'];folder.mkdir(parents=True,exist_ok=True)
    input_path=BASE if BASE.exists() else OUT/'00_base_reconstruida.parquet'
    protocol={'config':config,'engine_sha256':ENGINE_SHA256,'input_sha256':hashlib.sha256(input_path.read_bytes()).hexdigest(),
              'methods':METHODS,'trees':300,'spatial_splits':5,'seed':SEED,'purpose':'Retrospectiva condicionada a una familia conocida; no pronóstico ni efecto causal.'}
    protocol['environment']={name:__import__('importlib.metadata',fromlist=['version']).version(name) for name in ['numpy','pandas','scipy','scikit-learn']}
    marker=folder/'00_protocolo.json'
    if marker.exists():
        old=json.loads(marker.read_text(encoding='utf-8'))
        if old!=protocol:raise ValueError('El código o los datos cambiaron. Usar una carpeta nueva: '+str(folder))
        if all((folder/name).exists() for name in ['01_pliegues.csv','02_embedding_oof.parquet','03_downstream_oof.parquet','04_resumen_embedding.csv','05_resumen_downstream.csv','06_vectores_oof.npz']):
            print('Ajuste verificado y recuperado:',config['name'],flush=True);return
    x=unit(d[AE22].to_numpy(float));y=unit(d[AE25].to_numpy(float));v=logmap(x,y)
    erows=[];drows=[];diagnostics=[];vector_rows=[];vectors=[];starts=[];observed_vectors=[]
    for fold,tr,te0 in spatial_splits(d):
        ms,stable=masks(d,tr);sta=tr[stable[tr]]
        # Modelos ambientales idénticos para todas las familias y operadores del pliegue.
        htr=d.iloc[tr][HYDRO].to_numpy(float)
        environment={target:downmodel().fit(np.c_[y[tr],htr],d.iloc[tr][target+'_2025']) for target in ['lst_c','bsa']}
        for family in FAMILIES:
            fam=tr[ms[family][tr]];te=te0[ms[family][te0]]
            eligible=len(fam)>=20 and len(sta)>=40 and len(te)>0
            diagnostics.append(dict(fold=fold,family=family,train_pixels=len(tr),test_pixels=len(te0),train_family=len(fam),test_family=len(te),train_stable=len(sta),eligible=eligible,
                buffer_m=BUFFER,min_test_train_m=float(cKDTree(d.iloc[tr][['x_centroid','y_centroid']]).query(d.iloc[te0][['x_centroid','y_centroid']])[0].min())))
            if not eligible:continue
            fit=np.unique(np.r_[fam,sta])
            pred={'sin_cambio':x[te],
                  'desplazamiento_medio':expmap(x[te],cap(x[te],np.repeat(v[fam].mean(0)[None,:],len(te),axis=0))),
                  'familia_ridge':ridge_operator(d,x,v,fam,fit,te),
                  'fondo_residuo_rf':rf_operator(d,x,v,fam,sta,fit,te)}
            identity=angle(x[te],y[te]);nearest=np.degrees(np.arccos(np.clip((x[te]@x[fam].T).max(1),-1,1)))
            for method,z in pred.items():
                assert np.isfinite(z).all() and np.allclose(np.linalg.norm(z,axis=1),1,atol=1e-9)
                if method!='sin_cambio':assert (angle(x[te],z)<=MAX_ANGLE+1e-6).all()
                errors=angle(z,y[te])
                for j,pos in enumerate(te):
                    item=dict(fold=fold,family=family,window_id=str(d.window_id.iloc[pos]),spatial_block=str(d.spatial_block.iloc[pos]),method=method,
                              angular_error_deg=float(errors[j]),identity_error_deg=float(identity[j]),improvement_deg=float(identity[j]-errors[j]),initial_support_distance_deg=float(nearest[j]))
                    erows.append(item);vector_rows.append(len(erows)-1);vectors.append(z[j]);starts.append(x[pos]);observed_vectors.append(y[pos])
            hte=d.iloc[te][HYDRO].to_numpy(float)
            for target,f in environment.items():
                obs=d.iloc[te][target+'_2025'].to_numpy(float);oracle=f.predict(np.c_[y[te],hte])
                if target=='bsa':oracle=np.clip(oracle,0,1)
                for method,z in pred.items():
                    p=f.predict(np.c_[z,hte]);p=np.clip(p,0,1) if target=='bsa' else p
                    for j,pos in enumerate(te):drows.append(dict(fold=fold,family=family,target=target,window_id=str(d.window_id.iloc[pos]),spatial_block=str(d.spatial_block.iloc[pos]),method=method,observed_2025=float(obs[j]),predicted=float(p[j]),oracle=float(oracle[j])))
        print(config['name'],'pliegue',fold,'completado',flush=True)
    emb=pd.DataFrame(erows);down=pd.DataFrame(drows)
    if emb.empty:raise RuntimeError('Configuración sin evaluación elegible: '+config['name'])
    assert not emb[['family','method','window_id']].duplicated().any()
    pd.DataFrame(diagnostics).to_csv(folder/'01_pliegues.csv',index=False)
    emb.to_parquet(folder/'02_embedding_oof.parquet',index=False);down.to_parquet(folder/'03_downstream_oof.parquet',index=False)
    es,ds=summarize_results(emb,down);es.to_csv(folder/'04_resumen_embedding.csv',index=False);ds.to_csv(folder/'05_resumen_downstream.csv',index=False)
    np.savez_compressed(folder/'06_vectores_oof.npz',scenario=np.asarray(vectors),initial=np.asarray(starts),observed=np.asarray(observed_vectors),row=np.asarray(vector_rows))
    marker.write_text(json.dumps(protocol,indent=2,ensure_ascii=False),encoding='utf-8')

def eligibility_four_families(d):
    set_configuration(CONFIGS[0]);rows=[]
    for fold,tr,te in spatial_splits(d):
        ms,stable=masks(d,tr)
        ms.update(aumento_relativo_agua=(d.delta_water>=d.delta_water.iloc[tr].quantile(Q)).to_numpy(),
                  aumento_vegetacion_inundada=(d.delta_flooded_vegetation>=d.delta_flooded_vegetation.iloc[tr].quantile(Q)).to_numpy())
        for family,mask in ms.items():
            rows.append(dict(fold=fold,family=family,train_family=int(mask[tr].sum()),test_family=int(mask[te].sum()),train_stable=int(stable[tr].sum()),
                             support_conditions_met=bool(mask[tr].sum()>=20 and stable[tr].sum()>=40 and mask[te].sum()>0),implemented=family in FAMILIES))
    pd.DataFrame(rows).to_csv(OUT/'07_elegibilidad_cuatro_familias.csv',index=False)

def sensitivity_summary():
    principal=pd.read_parquet(OUT/'principal/02_embedding_oof.parquet');rows=[];matched=[]
    key=['family','method','window_id']
    for cfg in CONFIGS:
        s=pd.read_csv(OUT/cfg['name']/'04_resumen_embedding.csv');s['configuration']=cfg['name'];rows.append(s)
        other=pd.read_parquet(OUT/cfg['name']/'02_embedding_oof.parquet')
        pair=principal.merge(other,on=key,suffixes=('_primary','_variant'),validate='one_to_one')
        for (family,method),g in pair.groupby(['family','method']):
            matched.append(dict(configuration=cfg['name'],family=family,method=method,common_windows=len(g),
                primary_error_median=float(g.angular_error_deg_primary.median()),variant_error_median=float(g.angular_error_deg_variant.median()),
                median_error_reduction_same_windows=float((g.angular_error_deg_primary-g.angular_error_deg_variant).median())))
    pd.concat(rows,ignore_index=True).to_csv(OUT/'08_sensibilidad_todas_ventanas.csv',index=False)
    pd.DataFrame(matched).to_csv(OUT/'09_sensibilidad_ventanas_comunes.csv',index=False)

def paired_block_intervals(replicates=2000):
    emb=pd.read_parquet(OUT/'principal/02_embedding_oof.parquet');down=pd.read_parquet(OUT/'principal/03_downstream_oof.parquet');rng=np.random.default_rng(SEED);rows=[]
    for family,g in emb.groupby('family'):
        table=g.pivot(index=['window_id','spatial_block'],columns='method',values='angular_error_deg').dropna()
        blocks=table.index.get_level_values('spatial_block');unique=np.unique(blocks);parts=[np.flatnonzero(blocks==b) for b in unique]
        comparisons=[('familia_ridge','sin_cambio'),('fondo_residuo_rf','sin_cambio'),('familia_ridge','desplazamiento_medio'),('fondo_residuo_rf','desplazamiento_medio'),('fondo_residuo_rf','familia_ridge')]
        for method,control in comparisons:
            difference=(table[control]-table[method]).to_numpy();draw=[]
            for _ in range(replicates):
                ix=np.concatenate([parts[j] for j in rng.integers(len(parts),size=len(parts))]);draw.append(np.median(difference[ix]))
            lo,hi=np.quantile(draw,[.025,.975]);rows.append(dict(family=family,metric='angular_median_improvement',target='embedding',method=method,control=control,estimate=float(np.median(difference)),ci_low=float(lo),ci_high=float(hi),blocks=len(parts),windows=len(table),replicates=replicates))
        for target,t in down[down.family==family].groupby('target'):
            p=t.pivot(index=['window_id','spatial_block'],columns='method',values='predicted').dropna();reference=t.drop_duplicates(['window_id','spatial_block']).set_index(['window_id','spatial_block']).reindex(p.index)
            blocks=p.index.get_level_values('spatial_block');unique=np.unique(blocks);parts=[np.flatnonzero(blocks==b) for b in unique]
            for truth,label in [('observed_2025','mae_observed_improvement'),('oracle','mae_reference_model_improvement')]:
                errors=p.sub(reference[truth],axis=0).abs()
                for method,control in comparisons:
                    difference=(errors[control]-errors[method]).to_numpy();draw=[]
                    for _ in range(replicates):
                        ix=np.concatenate([parts[j] for j in rng.integers(len(parts),size=len(parts))]);draw.append(np.mean(difference[ix]))
                    lo,hi=np.quantile(draw,[.025,.975]);rows.append(dict(family=family,metric=label,target=target,method=method,control=control,estimate=float(np.mean(difference)),ci_low=float(lo),ci_high=float(hi),blocks=len(parts),windows=len(p),replicates=replicates))
    pd.DataFrame(rows).to_csv(OUT/'10_intervalos_pareados_bloques.csv',index=False)

def verify_archived_results():
    current=pd.read_csv(OUT/'principal/04_resumen_embedding.csv');historical=pd.read_csv(ROOT/'resultados_88_linea3_operadores_sin_fuga/04_resumen_embedding.csv')
    pair=current.merge(historical,on=['family','method'],validate='one_to_one');error=float(np.abs(pair.median_error_deg-pair.error_mediano_deg).max())
    report={'maximum_angular_summary_difference_deg':error,'matched_historical_rows':len(pair),'same_window_counts':bool(np.array_equal(pair.windows,pair.ventanas)),
            'note':'El Word asigna 98,3 % de mejora a ambos modelos en densificación; la ejecución histórica y esta réplica dan 100 % para fondo–residuo.'}
    (OUT/'11_comparacion_archivo_historico.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    if error>1e-5:print('Atención: diferencia numérica respecto al archivo histórico:',error,flush=True)

def run_all_models():
    """Entrada actual: geometría OOF existente + NDMI principal y NDVI contraste.

    Las funciones previas de réplica se conservan por trazabilidad histórica;
    este punto de entrada ya no ajusta ni presenta LST o albedo.
    """
    from linea3_validacion_transformaciones import generar_validacion_transformaciones
    generar_validacion_transformaciones()
    from linea3_ejemplo_transformacion import generar_ejemplo_transformacion
    generar_ejemplo_transformacion()
    import linea3_downstream_espectral as spectral
    with threadpool_limits(limits=1):spectral.main()
    import linea3_discusion_espectral as discussion
    discussion.main()
    return getdata()


if __name__=='__main__':run_all_models()
