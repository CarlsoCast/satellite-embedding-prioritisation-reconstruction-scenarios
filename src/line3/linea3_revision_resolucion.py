"""Revisión del soporte espacial y preparación de contrastes NDVI/NDMI."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import linea3_pipeline as engine

HERE=Path(__file__).resolve().parent;OUT=engine.OUT
REVIEW=HERE/'revision_resolucion';REVIEW.mkdir(exist_ok=True)
S2_SCALE={'ndvi':10,'ndmi':20}
S2_MIN_OBS=3;S2_MIN_FRACTION=.80;S2_MIN_COMMON_MONTHS=4;S2_CLOUD_MAX=40

def diagnose_existing_targets():
    d=engine.getdata();rows=[]
    for target in ['lst_c','bsa']:
        y=d[target+'_2025'];delta=y-d[target+'_2022'];iqr=float(y.quantile(.75)-y.quantile(.25))
        rows.append(dict(target=target,units='°C' if target=='lst_c' else 'fracción',windows=len(d),minimum=float(y.min()),maximum=float(y.max()),standard_deviation=float(y.std()),iqr=iqr,
            interannual_change_median=float(delta.median()),interannual_absolute_change_median=float(delta.abs().median()),
            native_support_m=100 if target=='lst_c' else 500,delivered_grid_m=30 if target=='lst_c' else 500,analysis_window_m=500))
    stats=pd.DataFrame(rows);stats.to_csv(REVIEW/'01_variacion_y_soporte.csv',index=False)
    ci=pd.read_csv(OUT/'10_intervalos_pareados_bloques.csv')
    ci=ci[(ci.metric=='mae_observed_improvement')&(ci.control=='sin_cambio')].copy()
    ci['interval_includes_zero']=(ci.ci_low<=0)&(ci.ci_high>=0)
    metrics=pd.read_csv(OUT/'principal/05_resumen_downstream.csv')
    baseline=metrics[metrics.method=='sin_cambio'][['family','target','mae_observed']].rename(columns={'mae_observed':'baseline_mae'})
    ci=ci.merge(baseline,on=['family','target'],validate='many_to_one')
    ci['relative_mae_reduction_pct']=100*ci.estimate/ci.baseline_mae
    ci.to_csv(REVIEW/'02_mejoras_ambientales_con_incertidumbre.csv',index=False)
    return stats,ci

def multiscale_frames():
    """Mallas geométricas; no inventa embeddings ni referencias a mayor resolución."""
    d=engine.getdata();summary=[]
    for size in [100,250,500]:
        rows=[];n=500//size
        for r in d.itertuples():
            for i in range(n):
                for j in range(n):
                    xmin=float(r.xmin+j*size);ymin=float(r.ymin+i*size);x=xmin+size/2;y=ymin+size/2
                    rows.append(dict(window_id=str(r.window_id) if size==500 else f'{r.window_id}_{size}_{i}_{j}',parent_window_id=str(r.window_id),size_m=size,
                        xmin=xmin,ymin=ymin,xmax=xmin+size,ymax=ymin+size,x_centroid=x,y_centroid=y,
                        spatial_block=str(r.spatial_block)))
        frame=pd.DataFrame(rows);assert frame.window_id.is_unique
        assert np.isclose((frame.xmax-frame.xmin)*(frame.ymax-frame.ymin),size*size).all()
        frame.to_parquet(REVIEW/f'03_malla_{size}m.parquet',index=False)
        summary.append(dict(window_m=size,windows=len(frame),embedding_pixels_per_full_window=(size//10)**2,status='Solo malla geométrica' if size<500 else 'Escala del ajuste actual'))
    summary=pd.DataFrame(summary);summary.to_csv(REVIEW/'03_mallas_preparadas.csv',index=False);return summary

def paired_spectral_annual(monthly):
    """Promedio de los mismos meses de 2022/2025; calidad por variable y ventana."""
    key=['window_id','variable','month'];assert not monthly[key+['year']].duplicated().any()
    a=monthly[monthly.year==2022];b=monthly[monthly.year==2025]
    pair=a.merge(b,on=key,suffixes=('_2022','_2025'),validate='one_to_one')
    quality=(pair.valid_fraction_2022>=S2_MIN_FRACTION)&(pair.valid_fraction_2025>=S2_MIN_FRACTION)&pair.value_2022.notna()&pair.value_2025.notna()
    pair=pair[quality].copy();rows=[]
    for (wid,var),g in pair.groupby(['window_id','variable']):
        if len(g)<S2_MIN_COMMON_MONTHS:continue
        rows.append(dict(window_id=wid,variable=var,common_months=len(g),months=';'.join(map(str,sorted(g.month))),
            value_2022=float(g.value_2022.mean()),value_2025=float(g.value_2025.mean()),
            within_window_std_2022=float(g.spatial_std_2022.mean()),within_window_std_2025=float(g.spatial_std_2025.mean())))
    return pd.DataFrame(rows,columns=['window_id','variable','common_months','months','value_2022','value_2025','within_window_std_2022','within_window_std_2025'])

def sentinel2_indices(image):
    """NDVI con B8/B4 de 10 m; NDMI con B8A/B11 de 20 m."""
    import ee
    scl=image.select('SCL');keep=scl.neq(0).And(scl.neq(1)).And(scl.neq(3)).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    probability=ee.Image(image.get('cloud_probability')).select('probability');keep=keep.And(probability.lt(S2_CLOUD_MAX))
    reflectance=image.select(['B4','B8','B8A','B11']).multiply(.0001).updateMask(keep)
    ndvi=reflectance.normalizedDifference(['B8','B4']).rename('ndvi')
    ndmi=reflectance.normalizedDifference(['B8A','B11']).rename('ndmi')
    return ndvi.addBands(ndmi).copyProperties(image,['system:time_start','system:index'])

def extract_sentinel2_references(grid_m=500,project='eloquent-petal-472121-d5'):
    """Extracción real opcional. No inicia autenticación interactiva automáticamente."""
    import ee
    from pyproj import Transformer,CRS
    ee.Initialize(project=project)
    frame=pd.read_parquet(REVIEW/f'03_malla_{grid_m}m.parquet');tr=Transformer.from_crs('EPSG:9377','EPSG:4326',always_xy=True)
    features=[]
    for r in frame.itertuples():
        xx,yy=tr.transform([r.xmin,r.xmax,r.xmax,r.xmin,r.xmin],[r.ymin,r.ymin,r.ymax,r.ymax,r.ymin])
        features.append(ee.Feature(ee.Geometry.Polygon([list(map(list,zip(xx,yy)))],None,False),{'window_id':str(r.window_id)}))
    fc=ee.FeatureCollection(features);bounds=fc.geometry().bounds(1);wkt=CRS.from_epsg(9377).to_wkt(version='WKT1_GDAL');records=[];provenance={};collections={}
    for year in [2022,2025]:
        sr=ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(bounds).filterDate(f'{year}-01-01',f'{year+1}-01-01')
        clouds=ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY').filterBounds(bounds).filterDate(f'{year}-01-01',f'{year+1}-01-01')
        joined=ee.ImageCollection(ee.Join.saveFirst('cloud_probability').apply(primary=sr,secondary=clouds,condition=ee.Filter.equals(leftField='system:index',rightField='system:index'))).filter(ee.Filter.notNull(['cloud_probability']))
        provenance[str(year)]=joined.aggregate_array('system:index').getInfo();collections[year]=joined.map(sentinel2_indices)
    for month in range(1,13):
        monthly={year:collection.filter(ee.Filter.calendarRange(month,month,'month')) for year,collection in collections.items()}
        if any(collection.size().getInfo()==0 for collection in monthly.values()):continue
        for var,scale in S2_SCALE.items():
            clear={year:collection.select(var).count().gte(S2_MIN_OBS) for year,collection in monthly.items()}
            common=clear[2022].And(clear[2025])
            for year,collection in monthly.items():
                value=collection.select(var).median().updateMask(common).rename('value')
                image=value.addBands(common.unmask(0).rename('valid_fraction'))
                reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(),sharedInputs=True)
                for start in range(0,len(features),100):
                    batch=ee.FeatureCollection(features[start:start+100]);output=image.reduceRegions(collection=batch,reducer=reducer,scale=scale,crs=wkt,tileScale=4).getInfo()['features']
                    for f in output:
                        p=f['properties'];records.append(dict(window_id=p['window_id'],variable=var,year=year,month=month,
                            value=p.get('value_mean',np.nan),spatial_std=p.get('value_stdDev',np.nan),valid_fraction=p.get('valid_fraction_mean',0)))
        pd.DataFrame(records).to_parquet(REVIEW/f'04_sentinel2_mensual_{grid_m}m.parquet',index=False)
        print('Sentinel-2:',grid_m,'mes común',month,flush=True)
    monthly=pd.DataFrame(records);annual=paired_spectral_annual(monthly)
    annual.to_parquet(REVIEW/f'05_sentinel2_referencias_{grid_m}m.parquet',index=False)
    (REVIEW/f'05_protocolo_sentinel2_{grid_m}m.json').write_text(json.dumps(dict(collection='COPERNICUS/S2_SR_HARMONIZED',scales=S2_SCALE,cloud_probability_max=S2_CLOUD_MAX,min_clear_observations=S2_MIN_OBS,min_valid_fraction=S2_MIN_FRACTION,min_common_months=S2_MIN_COMMON_MONTHS,scene_ids=provenance,status='Extracted actual reflectance; not an independent sensor validation'),indent=2),encoding='utf-8')
    return annual

def evaluate_spectral_references():
    """Contraste a 500 m con las mismas transformaciones OOF; sin ajustar con prueba."""
    path=REVIEW/'05_sentinel2_referencias_500m.parquet'
    if not path.exists():return 'Pendiente: extraer reflectancias Sentinel-2 y referencias NDVI/NDMI.'
    targets=pd.read_parquet(path)
    if targets.empty:return 'Sin ventanas que cumplan los meses comunes y la cobertura exigida.'
    d=engine.getdata();engine.set_configuration(engine.CONFIGS[0]);y=engine.unit(d[engine.AE25].to_numpy(float));h=d[engine.HYDRO].to_numpy(float)
    emb=pd.read_parquet(OUT/'principal/02_embedding_oof.parquet');vectors=np.load(OUT/'principal/06_vectores_oof.npz');positions={str(w):i for i,w in enumerate(d.window_id)};rows=[]
    for var in S2_SCALE:
        t=targets[targets.variable==var].set_index('window_id').reindex(d.window_id.astype(str));observed=t.value_2025.to_numpy(float)
        for fold,tr,te in engine.spatial_splits(d):
            fit=tr[np.isfinite(observed[tr])]
            if len(fit)<50:continue
            model=engine.downmodel().fit(np.c_[y[fit],h[fit]],observed[fit])
            for i,r in emb[emb.fold==fold].iterrows():
                pos=positions[str(r.window_id)]
                if not np.isfinite(observed[pos]):continue
                assert pos in te and pos not in tr
                p=float(np.clip(model.predict(np.c_[vectors['scenario'][i:i+1],h[pos:pos+1]])[0],-1,1))
                reference=float(np.clip(model.predict(np.c_[y[pos:pos+1],h[pos:pos+1]])[0],-1,1))
                rows.append(dict(variable=var,fold=fold,family=r.family,method=r.method,window_id=r.window_id,spatial_block=r.spatial_block,observed=observed[pos],predicted=p,reference_model=reference))
    if not rows:raise ValueError('No hay referencias espectrales elegibles para evaluar')
    result=pd.DataFrame(rows);result.to_parquet(REVIEW/'06_validacion_espectral_oof.parquet',index=False)
    summary=[]
    for (var,family,method),g in result.groupby(['variable','family','method']):
        summary.append(dict(variable=var,family=family,method=method,windows=len(g),mae_observed=float(np.abs(g.observed-g.predicted).mean()),mae_reference_model=float(np.abs(g.reference_model-g.predicted).mean())))
    pd.DataFrame(summary).to_csv(REVIEW/'06_resumen_espectral.csv',index=False)
    return pd.DataFrame(summary)

if __name__=='__main__':
    stats,ci=diagnose_existing_targets();print(stats.to_string(index=False));print(ci[['family','target','method','estimate','ci_low','ci_high']].to_string(index=False))
    print(multiscale_frames().to_string(index=False));print(evaluate_spectral_references())
