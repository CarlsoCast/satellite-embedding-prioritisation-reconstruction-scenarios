"""Downstream actual de línea 3: NDMI principal y NDVI como contraste.
Modelo ambiental entrenado solo en 2022, escenarios OOF congelados.
No ejecuta ni selecciona variables ambientales históricas descartadas.
"""
from pathlib import Path
import importlib.util,json,hashlib,os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent
REPO_ROOT=H.parents[1]
ROOT=Path(os.environ.get('SATEMB_LINE3_DATA',REPO_ROOT/'data/raw/line3')).resolve()
PREV=ROOT/'experimento_cuatro_variables_familias_20261014'
import linea3_pipeline as eng
from types import SimpleNamespace
OUT=H/'downstream_ndmi';OUT.mkdir(exist_ok=True)
VARIABLES=('ndmi','ndvi')
REGENERAR_OPERADORES=False  # True reproduce el ajuste antes de la aplicación espectral

def target(d,refs,var,protocol):
    assert var in VARIABLES
    a=refs[(refs.variable==var)&(refs.protocol==protocol)]
    return [a[a.year==year].set_index('window_id').reindex(d.window_id).value.to_numpy() for year in [2022,2025]]

def intervals(g,control_col,test_col):
    diff=(g[control_col]-g[test_col]).to_numpy();blocks=g.spatial_block.to_numpy();unique=np.unique(blocks)
    sums=np.array([diff[blocks==b].sum() for b in unique]);counts=np.array([(blocks==b).sum() for b in unique])
    ix=np.random.default_rng(20261014).integers(len(unique),size=(2000,len(unique)))
    draws=sums[ix].sum(1)/counts[ix].sum(1)
    return dict(improvement=float(diff.mean()),low=float(np.quantile(draws,.025)),high=float(np.quantile(draws,.975)),windows=len(g),blocks=len(unique))

cross=SimpleNamespace(eng=eng,target=target,intervals=intervals)

def geometry_sensitivity(recalculate=False):
    """Diez configuraciones: auditar cache o recalcular solo geometría.

    No ajusta variables ambientales. El recálculo escribe en otra carpeta.
    """
    from scipy.spatial import cKDTree
    d=eng.getdata();x=eng.unit(d[eng.AE22]);y=eng.unit(d[eng.AE25]);v=eng.logmap(x,y)
    all_summaries=[];primary=None
    for config in eng.CONFIGS:
        eng.set_configuration(config)
        folder=OUT/'sensibilidad_recalculada'/config['name'] if recalculate else H/'nueva_corrida_integral'/config['name']
        if recalculate:
            folder.mkdir(parents=True,exist_ok=True);rows=[];foldlog=[];vectors=[];starts=[];observed=[]
            for fold,tr,te0 in eng.spatial_splits(d):
                masks,stable=eng.masks(d,tr);sta=tr[stable[tr]]
                for famname in eng.FAMILIES:
                    fam=tr[masks[famname][tr]];te=te0[masks[famname][te0]]
                    eligible=len(fam)>=20 and len(sta)>=40 and len(te)>0
                    foldlog.append(dict(fold=fold,family=famname,train_family=len(fam),train_stable=len(sta),test_family=len(te),eligible=eligible))
                    if not eligible:continue
                    fit=np.unique(np.r_[fam,sta]);estimates={'sin_cambio':x[te],
                        'desplazamiento_medio':eng.expmap(x[te],eng.cap(x[te],np.repeat(v[fam].mean(0)[None,:],len(te),axis=0))),
                        'familia_ridge':eng.ridge_operator(d,x,v,fam,fit,te),
                        'fondo_residuo_rf':eng.rf_operator(d,x,v,fam,sta,fit,te)}
                    identity=eng.angle(x[te],y[te]);nearest=np.degrees(np.arccos(np.clip((x[te]@x[fam].T).max(1),-1,1)))
                    for method,z in estimates.items():
                        errors=eng.angle(z,y[te])
                        for j,i in enumerate(te):
                            rows.append(dict(fold=fold,family=famname,window_id=d.window_id.iloc[i],spatial_block=d.spatial_block.iloc[i],method=method,angular_error_deg=errors[j],identity_error_deg=identity[j],improvement_deg=identity[j]-errors[j],initial_support_distance_deg=nearest[j]))
                            vectors.append(z[j]);starts.append(x[i]);observed.append(y[i])
            frame=pd.DataFrame(rows);frame.to_parquet(folder/'02_embedding_oof.parquet',index=False);pd.DataFrame(foldlog).to_csv(folder/'01_pliegues.csv',index=False)
            np.savez_compressed(folder/'06_vectores_oof.npz',scenario=vectors,initial=starts,observed=observed,row=np.arange(len(frame)))
        else:
            protocol=json.loads((folder/'00_protocolo.json').read_text(encoding='utf-8'))
            assert protocol['config']==config
            assert protocol['input_sha256']==hashlib.sha256(eng.BASE.read_bytes()).hexdigest()
            frame=pd.read_parquet(folder/'02_embedding_oof.parquet')
        assert not frame.duplicated(['family','method','window_id']).any()
        arrays=np.load(folder/'06_vectores_oof.npz');assert np.allclose(np.linalg.norm(arrays['scenario'],axis=1),1,atol=1e-8)
        if primary is None:primary=frame
        pair=primary.merge(frame,on=['family','method','window_id'],suffixes=('_primary','_variant'),validate='one_to_one')
        for (family,method),g in pair.groupby(['family','method']):
            all_summaries.append(dict(configuration=config['name'],family=family,method=method,common_windows=len(g),mean_angular_improvement=(g.angular_error_deg_primary-g.angular_error_deg_variant).mean(),median_angular_improvement=(g.angular_error_deg_primary-g.angular_error_deg_variant).median()))
    eng.set_configuration(eng.CONFIGS[0]);result=pd.DataFrame(all_summaries);result.to_csv(OUT/'sensibilidad_geometrica.csv',index=False)
    return result

def prepare_annual_references():
    """Referencias por año independiente; no restringir 2025 por calidad 2022."""
    source=ROOT/'experimento_ndvi_ndmi_20261014'
    gr=np.load(source/'grillas.npz');ids=gr['window_ids'];records=[]
    protocols={'anual':(4,8,.8),'estricto':(6,12,.8),'soporte4':(4,4,.8)}
    for year in [2022,2025]:
        for var,labels in [('ndvi',gr['labels10']),('ndmi',gr['labels20'])]:
            total=np.zeros(labels.shape,dtype='float64');months=np.zeros(labels.shape,dtype='uint8');days=np.zeros(labels.shape,dtype='uint16')
            for month in range(1,13):
                with np.load(source/'cache_mensual'/f'{year}_{month:02d}.npz') as cache:
                    a=cache[var];n=cache[var+'_n'];valid=np.isfinite(a)&(n>0)
                    total+=np.where(valid,a,0);months+=valid;days+=n
            annual=np.divide(total,months,out=np.full_like(total,np.nan),where=months>0)
            areas=np.bincount(labels.ravel(),minlength=len(ids)+1)
            for protocol,(minmonths,mindays,minarea) in protocols.items():
                valid=(labels>0)&(months>=minmonths)&(days>=mindays)&np.isfinite(annual)
                count=np.bincount(labels[valid],minlength=len(ids)+1);sums=np.bincount(labels[valid],weights=annual[valid],minlength=len(ids)+1)
                for i,wid in enumerate(ids,1):
                    fraction=count[i]/areas[i]
                    records.append(dict(protocol=protocol,variable=var,year=year,window_id=wid,value=sums[i]/count[i] if fraction>=minarea else np.nan,valid_fraction=fraction))
    frame=pd.DataFrame(records);frame.to_parquet(OUT/'referencias_anuales.parquet',index=False)
    frame.groupby(['protocol','variable','year']).value.count().to_csv(OUT/'cobertura_referencias.csv')
    return frame

def hashes():
    paths=[eng.BASE,PREV/'vectores_cruzados.npz',PREV/'vectores_cruzados.parquet',PREV/'referencias_espectrales_anuales.parquet']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def cohorts(d,tr):
    masks,stable=eng.masks(d,tr)
    main=np.logical_or.reduce(list(masks.values()))
    water=(d.delta_water>=d.delta_water.iloc[tr].quantile(eng.Q)).to_numpy()
    flooded=(d.delta_flooded_vegetation>=d.delta_flooded_vegetation.iloc[tr].quantile(eng.Q)).to_numpy()
    # Cohortes disjuntas, conservando explícitamente el grupo mixto.
    result={**masks,'otro_agua':water&~flooded&~main,'otra_inundada':flooded&~water&~main,
            'otro_agua_inundada':water&flooded&~main,'estable_sin_familias':stable&~water&~flooded&~main}
    seen=np.sum(np.stack(list(result.values())),axis=0)
    assert (seen<=1).all()
    return result,stable

def regenerate_operators(d):
    folder=OUT/'operadores_reproducidos';folder.mkdir(exist_ok=True)
    cache=folder/'vectores_cruzados.npz';metadata=folder/'vectores_cruzados.parquet'
    if cache.exists() and metadata.exists():return pd.read_parquet(metadata),np.load(cache)['vectors']
    x=eng.unit(d[eng.AE22]);y=eng.unit(d[eng.AE25]);v=eng.logmap(x,y)
    L3=H
    old=pd.read_parquet(L3/'nueva_corrida_integral/principal/02_embedding_oof.parquet')
    oldz=np.load(L3/'nueva_corrida_integral/principal/06_vectores_oof.npz')['scenario']
    rows=[];vectors=[];audit=[];support=[];foldrows=[]
    for fold,tr,te in eng.spatial_splits(d):
        groups,stable=cohorts(d,tr);sta=tr[stable[tr]]
        for pos in te:
            group=next((name for name,mask in groups.items() if mask[pos]),'resto')
            foldrows.append(dict(fold=fold,window_id=d.window_id.iloc[pos],cohort=group,spatial_block=d.spatial_block.iloc[pos]))
        for famname in eng.FAMILIES:
            fam=tr[groups[famname][tr]];fit=np.unique(np.r_[fam,sta])
            assert len(fam)>=20 and len(sta)>=40
            estimates={
                'desplazamiento_medio':eng.expmap(x[te],eng.cap(x[te],np.repeat(v[fam].mean(0)[None,:],len(te),axis=0))),
                'familia_ridge':eng.ridge_operator(d,x,v,fam,fit,te),
                'fondo_residuo_rf':eng.rf_operator(d,x,v,fam,sta,fit,te)}
            nearest=np.degrees(np.arccos(np.clip((x[te]@x[fam].T).max(1),-1,1)))
            for method,a in estimates.items():
                assert np.isfinite(a).all() and np.allclose(np.linalg.norm(a,axis=1),1)
                before=old[(old.fold==fold)&(old.family==famname)&(old.method==method)]
                positions={str(d.window_id.iloc[i]):j for j,i in enumerate(te)}
                diff=float(np.max(abs(a[[positions[str(w)] for w in before.window_id]]-oldz[before.index])))
                assert diff<1e-6,(fold,famname,method,diff)
                audit.append(dict(fold=fold,source_family=famname,method=method,max_abs_difference=diff))
                for j,pos in enumerate(te):
                    group=next((name for name,mask in groups.items() if mask[pos]),'resto')
                    rows.append(dict(row=len(rows),fold=fold,window_id=d.window_id.iloc[pos],spatial_block=d.spatial_block.iloc[pos],cohort=group,source_family=famname,method=method,angular_error=float(eng.angle(a[j:j+1],y[pos:pos+1])[0]),support_distance=float(nearest[j])))
                    vectors.append(a[j])
            print('Operador reproducido y cruzado',fold,famname,flush=True)
    meta=pd.DataFrame(rows);meta.to_parquet(metadata,index=False);np.savez_compressed(cache,vectors=np.asarray(vectors))
    pd.DataFrame(audit).to_csv(OUT/'reproduccion_operadores.csv',index=False)
    pd.DataFrame(foldrows).to_csv(OUT/'cohortes_fuera_muestra.csv',index=False)
    return meta,np.asarray(vectors)

def metrics(g,pred='predicted',truth='observed_2025'):
    y=g[truth].to_numpy();p=g[pred].to_numpy();error=p-y;mse=np.mean(error**2);sd=np.std(y);iqr=np.quantile(y,.75)-np.quantile(y,.25)
    result=dict(n=len(g),blocks=g.spatial_block.nunique(),mae=np.mean(abs(error)),mse=mse,rmse=np.sqrt(mse),bias=np.mean(error),r2=r2_score(y,p) if len(y)>2 else np.nan,target_sd=sd,target_iqr=iqr,nrmse_sd=np.sqrt(mse)/sd if sd>1e-10 else np.nan,nrmse_iqr=np.sqrt(mse)/iqr if iqr>1e-10 else np.nan)
    if 'training_mean_2022' in g:
        base=np.mean((g.training_mean_2022-y)**2);result['skill_vs_train_mean']=1-mse/base if base>1e-12 else np.nan
    return result

def main():
    hp=OUT/'huellas_antes.json'
    if not hp.exists():hp.write_text(json.dumps(hashes(),indent=2))
    d=eng.getdata();refs=prepare_annual_references()
    if REGENERAR_OPERADORES:
        meta,z=regenerate_operators(d)
    else:
        meta=pd.read_parquet(PREV/'vectores_cruzados.parquet');z=np.load(PREV/'vectores_cruzados.npz')['vectors']
    cohorts=pd.read_csv(PREV/'resultados/cohortes_fuera_muestra.csv').set_index('window_id').cohort
    x=eng.unit(d[eng.AE22]);y=eng.unit(d[eng.AE25]);terrain=d[eng.HYDRO].to_numpy();lookup=dict(zip(d.window_id,range(len(d))))
    decoder=[];scenarios=[];fitlog=[];initial_rows=[]
    for protocol in ['anual','estricto','soporte4']:
        for var in VARIABLES:
            t22,t25=cross.target(d,refs,var,protocol)
            for fold,tr,te in eng.spatial_splits(d):
                train=tr[np.isfinite(t22[tr])];test=te[np.isfinite(t25[te])]
                fitlog.append(dict(protocol=protocol,variable=var,fold=fold,train_2022=len(train),test_2025=len(test),eligible=bool(len(train)>=80 and len(test)>0)))
                if len(train)<80 or not len(test):continue
                model=eng.downmodel().fit(np.c_[x[train],terrain[train]],t22[train])
                test22=te[np.isfinite(t22[te])]
                initial=model.predict(np.c_[x[test22],terrain[test22]])
                for i,p0 in zip(test22,initial):initial_rows.append(dict(protocol=protocol,variable=var,fold=fold,window_id=d.window_id.iloc[i],spatial_block=d.spatial_block.iloc[i],observed2022=t22[i],predicted2022=p0))
                p22=model.predict(np.c_[x[test],terrain[test]]);p25=model.predict(np.c_[y[test],terrain[test]])
                base=float(np.mean(t22[train]));mapping={i:(p22[j],p25[j]) for j,i in enumerate(test)}
                for j,i in enumerate(test):
                    decoder.append(dict(protocol=protocol,variable=var,fold=fold,window_id=d.window_id.iloc[i],spatial_block=d.spatial_block.iloc[i],cohort=cohorts.loc[d.window_id.iloc[i]],observed_2022=t22[i],observed_2025=t25[i],unchanged2022=p22[j],real2025=p25[j],training_mean_2022=base))
                subset=meta[(meta.fold==fold)&meta.window_id.isin(d.window_id.iloc[test])]
                ix=np.array([lookup[w] for w in subset.window_id]);pred=model.predict(np.c_[z[subset.row],terrain[ix]])
                for (_,r),i,p in zip(subset.iterrows(),ix,pred):
                    initial,real=mapping[i]
                    scenarios.append(dict(protocol=protocol,variable=var,fold=fold,window_id=r.window_id,spatial_block=r.spatial_block,cohort=r.cohort,source_family=r.source_family,method=r.method,observed_2025=t25[i],observed_2022=t22[i],predicted=p,real2025=real,unchanged2022=initial,training_mean_2022=base,ae=abs(p-t25[i]),unchanged_ae=abs(initial-t25[i]),real_ae=abs(real-t25[i]),embedding_effect=p-initial,support_distance=r.support_distance,angular_error=r.angular_error))
            print('Transferencia',protocol,var,flush=True)
    p=pd.DataFrame(scenarios);dc=pd.DataFrame(decoder)
    pd.DataFrame(initial_rows).to_parquet(OUT/'validacion2022_oof.parquet',index=False)
    p.to_parquet(OUT/'escenarios_oof.parquet',index=False);dc.to_parquet(OUT/'decodificador_oof.parquet',index=False);pd.DataFrame(fitlog).to_csv(OUT/'pliegues.csv',index=False)
    quality=[];paired=[]
    for (protocol,var),g in dc.groupby(['protocol','variable']):
        for pred,label,truth in [('unchanged2022','sin_cambio_2022','observed_2025'),('real2025','real2025_transferido','observed_2025'),('training_mean_2022','media_entrenamiento_2022','observed_2025')]:
            quality.append(dict(protocol=protocol,variable=var,comparison=label,**metrics(g,pred,truth)))
        common=g.dropna(subset=['observed_2022'])
        if len(common):
            quality.append(dict(protocol=protocol,variable=var,comparison='reconstruccion2022_mismas_ventanas',**metrics(common,'unchanged2022','observed_2022')))
            quality.append(dict(protocol=protocol,variable=var,comparison='transferencia2025_mismas_ventanas',**metrics(common,'real2025','observed_2025')))
        if len(g)>2:
            q=g.assign(a=abs(g.unchanged2022-g.observed_2025),b=abs(g.real2025-g.observed_2025))
            paired.append(dict(protocol=protocol,variable=var,comparison='real2025_vs_sin_cambio',**cross.intervals(q,'a','b')))
    pd.DataFrame(quality).to_csv(OUT/'calidad_transferencia.csv',index=False);pd.DataFrame(paired).to_csv(OUT/'beneficio_embedding_real2025.csv',index=False)
    keys=['protocol','variable','cohort','source_family','method'];summ=[];ci=[];persistence=[]
    for key,g in p.groupby(keys):
        common=g.dropna(subset=['observed_2022'])
        summ.append(dict(zip(keys,key),**metrics(g),unchanged_mae=g.unchanged_ae.mean(),unchanged_rmse=np.sqrt(np.mean((g.unchanged2022-g.observed_2025)**2)),real2025_mae=g.real_ae.mean(),real2025_rmse=np.sqrt(np.mean((g.real2025-g.observed_2025)**2)),mean_embedding_effect=g.embedding_effect.mean()))
        ci.append(dict(zip(keys,key),**cross.intervals(g,'unchanged_ae','ae')))
        if len(common):
            m=metrics(common);base=np.mean((common.observed_2022-common.observed_2025)**2)
            persistence.append(dict(zip(keys,key),n=len(common),model_mae=m['mae'],model_rmse=m['rmse'],persistence_mae=np.mean(abs(common.observed_2022-common.observed_2025)),persistence_rmse=np.sqrt(base),skill_mse=1-m['mse']/base if base>1e-12 else np.nan))
    pd.DataFrame(summ).to_csv(OUT/'metricas_escenarios.csv',index=False);pd.DataFrame(ci).to_csv(OUT/'mejora_vs_sin_cambio.csv',index=False);pd.DataFrame(persistence).to_csv(OUT/'comparacion_persistencia.csv',index=False)
    specificity=[];complexity=[]
    for (protocol,var,cohort,method),g in p[p.cohort.isin(eng.FAMILIES)].groupby(['protocol','variable','cohort','method']):
        pivot=g.pivot(index=['window_id','spatial_block'],columns='source_family',values='ae').dropna().reset_index();other=next(f for f in eng.FAMILIES if f!=cohort)
        specificity.append(dict(protocol=protocol,variable=var,cohort=cohort,method=method,**cross.intervals(pivot,other,cohort)))
    for (protocol,var,cohort),g in p[p.cohort==p.source_family].groupby(['protocol','variable','cohort']):
        pivot=g.pivot(index=['window_id','spatial_block'],columns='method',values='ae').dropna().reset_index()
        for method in ['familia_ridge','fondo_residuo_rf']:
            complexity.append(dict(protocol=protocol,variable=var,cohort=cohort,method=method,**cross.intervals(pivot,'desplazamiento_medio',method)))
    pd.DataFrame(specificity).to_csv(OUT/'especificidad.csv',index=False);pd.DataFrame(complexity).to_csv(OUT/'complejidad_vs_media.csv',index=False)
    assert hashes()==json.loads(hp.read_text())
    (OUT/'integridad.json').write_text(json.dumps(dict(unchanged=True,files_verified=len(json.loads(hp.read_text()))),indent=2))
    print(pd.DataFrame(quality).query("protocol=='anual'")[['variable','comparison','n','mae','rmse','bias','r2','nrmse_sd']].to_string(index=False))
    s=pd.DataFrame(summ);print(s[(s.protocol=='anual')&(s.cohort==s.source_family)][['variable','cohort','method','n','mae','rmse','bias','unchanged_rmse','real2025_rmse']].to_string(index=False))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
