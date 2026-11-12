"""Experimento separado: escalas, Procrustes sin Dynamic World y evaluación auxiliar.

Etapas: auditar -> preparar -> puntuar -> evaluar. Ver LEEME.md.
Las dos primeras etapas de modelación NO abren archivos Dynamic World.
"""
from pathlib import Path
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse, hashlib, json, math, sys
import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from sklearn.metrics import average_precision_score
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

CLASSES=['water','trees','grass','flooded_vegetation','crops','shrub_and_scrub','built','bare','snow_and_ice']
VEG=[1,2,3,4,5]
E22=[f'e22_{i:02}' for i in range(64)];E25=[f'e25_{i:02}' for i in range(64)]
METHODS=['directo','alineamiento_general','referencias_20','referencias_30','referencias_40']
HOME=Path(__file__).resolve().parent
BOUNDS_CACHE={}

def dump(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf8')

def config(path):
    c=json.loads(Path(path).read_text(encoding='utf8'))
    c['out']=str((Path(path).resolve().parent/c.get('out','salidas')).resolve())
    assert c['scales_m']==[10,100,500,1000], 'Conservar las cuatro escalas declaradas.'
    assert c['block_m']%1000==0 and c['tile_m']==1000
    assert c['buffer_m']<=c['block_m'], 'El control espacial admite buffers de hasta un bloque.'
    assert not any('dw' in k.lower() or 'dynamic' in k.lower() for k in c['model']), 'No admitir referencia auxiliar en configuración del modelo.'
    return c

def unit(a):
    a=np.asarray(a,dtype=np.float64);n=np.linalg.norm(a,axis=1)
    if np.any(~np.isfinite(a)) or np.any(n<=1e-12):raise ValueError('Embedding inválido o nulo.')
    return (a/n[:,None]).astype(np.float32)

def angle(x,y):
    return np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i',unit(x),unit(y)),-1,1)))

def folds(x,y,c):
    return ((3*np.floor(np.asarray(x)/c['block_m']).astype(np.int64)+2*np.floor(np.asarray(y)/c['block_m']).astype(np.int64))%c['folds']).astype(np.int8)

def train_mask(x,y,fold,scale,c):
    """Distancia entre la huella cuadrada y los bloques excluidos completos.

    Es más conservador que retirar solo vecinos de puntos observados. La misma
    cuadrícula se utiliza en todas las escalas; ninguna ventana cruza un bloque.
    """
    x=np.asarray(x);y=np.asarray(y);b=c['block_m'];bc=np.floor(x/b).astype(np.int64);br=np.floor(y/b).astype(np.int64)
    dist=np.full(len(x),np.inf)
    radius=math.ceil((c['buffer_m']+scale/2)/b)+1
    for dc in range(-radius,radius+1):
        for dr in range(-radius,radius+1):
            cc=bc+dc;rr=br+dr;held=(3*cc+2*rr)%c['folds']==fold
            dx=np.maximum(np.abs(x-(cc+.5)*b)-b/2-scale/2,0)
            dy=np.maximum(np.abs(y-(rr+.5)*b)-b/2-scale/2,0)
            dist=np.minimum(dist,np.where(held,np.hypot(dx,dy),np.inf))
    return (folds(x,y,c)!=fold)&(dist>=c['buffer_m'])

def aggregate(frame,scale,coverage=.9):
    """Promediar píxeles originales sobre malla anclada en (0,0); nunca subdividir ventanas."""
    if scale==10:return frame.copy()
    z=frame.copy();z['col']=np.floor(z.x/scale).astype(np.int64);z['row']=np.floor(z.y/scale).astype(np.int64)
    cols=[k for k in E22+E25 if k in z]
    cols += [k for k in ['dw_js','delta_built','delta_vegetation','delta_bare'] if k in z]
    a=z.groupby(['col','row'],sort=False)[cols].mean()
    a['valid_pixels']=z.groupby(['col','row'],sort=False).size()
    a['coverage']=a.valid_pixels/(scale//10)**2
    a=a[a.coverage>=coverage].reset_index()
    a['x']=(a.col+.5)*scale;a['y']=(a.row+.5)*scale
    if E22[0] in a:
        good=(np.linalg.norm(a[E22].to_numpy(),axis=1)>1e-8)&(np.linalg.norm(a[E25].to_numpy(),axis=1)>1e-8)
        a=a.loc[good].copy()
        if len(a):a[E22]=unit(a[E22].to_numpy());a[E25]=unit(a[E25].to_numpy())
    return a

def learn(x,y,model):
    """API deliberadamente limitada a dos matrices de embeddings. Sin etiquetas."""
    x=unit(x);y=unit(y);raw=angle(x,y)
    ops={};audit=[]
    for name,q in [('alineamiento_general',1.),('referencias_20',.2),('referencias_30',.3),('referencias_40',.4)]:
        cut=float(np.quantile(raw,q)) if len(raw) else np.nan;keep=raw<=cut
        if keep.sum()<model['min_references']:
            audit.append(dict(method=name,status='sin_soporte',reference_fraction=q,reference_count=int(keep.sum()),required=model['min_references']))
            continue
        rotation,_=orthogonal_procrustes(x[keep].astype(float),y[keep].astype(float))
        ops[name]=rotation
        rank=int(np.linalg.matrix_rank(x[keep].astype(float).T@y[keep].astype(float)))
        audit.append(dict(method=name,status='rango_completo' if rank==64 else 'rango_incompleto',reference_fraction=q,raw_angle_threshold_deg=cut,reference_count=int(keep.sum()),cross_covariance_rank=rank))
    return ops,audit

def predict(x,y,ops):
    return {'directo':angle(x,y),**{name:angle(x@r,y) for name,r in ops.items()}}

def raster_info(path):
    with rasterio.open(path) as s:return dict(path=str(path),bytes=Path(path).stat().st_size,bands=s.count,dtype=s.dtypes[0],crs=str(s.crs),resolution=s.res,bounds=tuple(s.bounds),descriptions=s.descriptions,nodata=str(s.nodata))

def auditar(c):
    rows=[]
    for name,paths in [('ae22',c['ae22']),('ae25',c['ae25']),('dw22',c.get('dw22',[])),('dw25',c.get('dw25',[]))]:
        for p in paths:
            if Path(p).is_file():rows.append({'role':name,**raster_info(p)})
            else:rows.append({'role':name,'path':p,'missing':True})
    for year in ['dw22','dw25']:
        if not c.get(year):rows.append({'role':year,'missing':True,'reason':'Falta ráster anual completo de nueve probabilidades; no usar casos seleccionados.'})
    dump(Path(c['out'])/'00_inventario.json',rows)
    print(json.dumps(rows,ensure_ascii=False,indent=2))

def grid(c):
    if c.get('bounds_m'):bounds=c['bounds_m']
    else:
        with rasterio.open(c['ae22'][0]) as s:bounds=transform_bounds(s.crs,c['crs'],*s.bounds,densify_pts=21)
    x0,y0,x1,y1=bounds;t=c['tile_m']
    for x in range(math.floor(x0/t)*t,math.ceil(x1/t)*t,t):
        for y in range(math.floor(y0/t)*t,math.ceil(y1/t)*t,t):yield x,y

def preparation_signature(c):
    names=['ae22','ae25','crs','bounds_m','tile_m','scales_m','coverage']
    return dict(config={k:c.get(k) for k in names},sources=[dict(path=p,bytes=Path(p).stat().st_size,mtime=Path(p).stat().st_mtime_ns) for p in c['ae22']+c['ae25']])

def read_tile(paths,bands,x,y,c):
    n=c['tile_m']//10;dst=np.full((bands,n,n),np.nan,dtype=np.float32)
    for path in paths:
        # Copia local con almacenamiento por bloques, comprobada sin cambiar valores.
        # Evita releer franjas completas del ráster original de 2022 en cada tesela.
        verification=Path(c['out'])/'cache_fuentes/verificacion.json'
        if bands==64 and verification.exists():
            cache=json.loads(verification.read_text(encoding='utf8'))
            original=Path(path)
            if str(original.resolve())==cache['source'] and original.stat().st_size==cache['bytes'] and original.stat().st_mtime_ns==cache['mtime'] and Path(cache['cache']).stat().st_size==cache['cache_bytes']:
                path=cache['cache']
        key=(str(path),c['crs'])
        if key not in BOUNDS_CACHE:
            with rasterio.open(path) as source:
                BOUNDS_CACHE[key]=transform_bounds(source.crs,c['crs'],*source.bounds,densify_pts=21)
        left,bottom,right,top=BOUNDS_CACHE[key]
        if right<=x or left>=x+c['tile_m'] or top<=y or bottom>=y+c['tile_m']:continue
        with rasterio.open(path) as s:
            if s.count!=bands:raise ValueError(f'{path}: requiere {bands} bandas, tiene {s.count}.')
            with WarpedVRT(s,crs=c['crs'],transform=from_origin(x,y+c['tile_m'],10,10),width=n,height=n,resampling=Resampling.nearest,nodata=np.nan,dtype='float32') as v:
                a=v.read(masked=True).filled(np.nan)
        ok=np.isfinite(a).all(0)
        if bands==64:ok&=np.linalg.norm(a,axis=0)>1e-8
        else:ok&=(a>=0).all(0)&(a<=1.00001).all(0)&(a.sum(0)>.5)
        vacant=~np.isfinite(dst).all(0);dst[:,ok&vacant]=a[:,ok&vacant]
    return dst.reshape(bands,-1).T

def tile_xy(x,y,c):
    n=c['tile_m']//10;xx,yy=np.meshgrid(x+(np.arange(n)+.5)*10,y+c['tile_m']-(np.arange(n)+.5)*10)
    return xx.ravel(),yy.ravel()

def preparar(c):
    root=Path(c['out']);audit=[]
    if (root/'02_puntajes_congelados.json').exists():raise FileExistsError('La preparación ya tiene puntajes congelados. Utilizar otra carpeta para cambiarla.')
    sig=preparation_signature(c);signature_file=root/'01_preparacion_entrada.json'
    if signature_file.exists() and json.loads(signature_file.read_text(encoding='utf8'))!=sig:raise ValueError('Cambió la entrada o la malla. Utilizar otra carpeta de salida.')
    dump(signature_file,sig)
    def prepare_tile(x,y):
        token=f'{x}_{y}';paths=[root/f'embeddings/{s}/{token}.parquet' for s in c['scales_m']]
        if all(p.exists() for p in paths):
            return [dict(tile=token,scale_m=s,units=None,status='cache') for s in c['scales_m']]
        a=read_tile(c['ae22'],64,x,y,c);b=read_tile(c['ae25'],64,x,y,c)
        good=np.isfinite(a).all(1)&np.isfinite(b).all(1)
        if not good.any():return []
        xx,yy=tile_xy(x,y,c);z=pd.DataFrame(np.hstack([unit(a[good]),unit(b[good])]),columns=E22+E25)
        z['x']=xx[good];z['y']=yy[good];z['coverage']=1.;z['valid_pixels']=1
        rows=[]
        for s,p in zip(c['scales_m'],paths):
            d=aggregate(z,s,c['coverage']);p.parent.mkdir(parents=True,exist_ok=True);d.to_parquet(p,index=False)
            rows.append(dict(tile=token,scale_m=s,units=len(d),status='preparado'))
        print('Preparado',token,len(z),'píxeles',flush=True)
        return rows
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks=[pool.submit(prepare_tile,x,y) for x,y in grid(c)]
        for task in as_completed(tasks):audit.extend(task.result())
    dump(root/'01_preparacion.json',dict(crs=c['crs'],reprojection='nearest, 10 m; exportaciones fuente en EPSG:4326',grid_origin=[0,0],coverage=c['coverage'],tiles=audit))

def manifest(paths):
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def sample_train(paths,fold,scale,c):
    rng=np.random.default_rng(c['seed']+fold);pool=None;count=0;cap=c['model']['max_train']
    for path in paths:
        z=pd.read_parquet(path);z=z.loc[train_mask(z.x,z.y,fold,scale,c)]
        count+=len(z)
        if not len(z):continue
        z=z.copy();z['_key']=rng.random(len(z))
        pool=z if pool is None else pd.concat([pool,z],ignore_index=True)
        if len(pool)>cap:pool=pool.iloc[np.argpartition(pool._key.to_numpy(),cap-1)[:cap]].copy()
    if pool is None:pool=pd.DataFrame(columns=E22+E25+['x','y'])
    if len(pool)<c['model']['min_train']:return pool.iloc[:0],count
    return pool,count

def sample_all_folds(paths,scale,c):
    """Mismo muestreo reproducible por grupo, leyendo cada archivo una sola vez."""
    rngs=[np.random.default_rng(c['seed']+f) for f in range(c['folds'])]
    pools=[None]*c['folds'];counts=[0]*c['folds'];cap=c['model']['max_train']
    for i,path in enumerate(paths,1):
        base=pd.read_parquet(path)
        for fold in range(c['folds']):
            z=base.loc[train_mask(base.x,base.y,fold,scale,c)];counts[fold]+=len(z)
            if not len(z):continue
            z=z.copy();z['_key']=rngs[fold].random(len(z))
            pool=z if pools[fold] is None else pd.concat([pools[fold],z],ignore_index=True)
            if len(pool)>cap:pool=pool.iloc[np.argpartition(pool._key.to_numpy(),cap-1)[:cap]].copy()
            pools[fold]=pool
        if i%250==0:print('Entrenamiento preparado',scale,'m:',i,'/',len(paths),'teselas',flush=True)
    result=[]
    for pool,count in zip(pools,counts):
        if pool is None:pool=pd.DataFrame(columns=E22+E25+['x','y'])
        if len(pool)<c['model']['min_train']:pool=pool.iloc[:0]
        result.append((pool,count))
    return result

def puntuar(c):
    root=Path(c['out']);audit=[]
    if json.loads((root/'01_preparacion_entrada.json').read_text(encoding='utf8'))!=preparation_signature(c):raise ValueError('Preparación incompatible; revisar configuración y fuentes.')
    if (root/'02_puntajes_congelados.json').exists():raise FileExistsError('Los puntajes ya están congelados. Use otra carpeta de salida para cambiar configuración.')
    # Solo se buscan archivos en embeddings/. No se abre ni consulta Dynamic World.
    for scale in c['scales_m']:
        paths=sorted((root/f'embeddings/{scale}').glob('*.parquet'))
        if not paths:raise FileNotFoundError(f'Primero preparar embeddings de {scale} m.')
        ops={}
        samples=sample_all_folds(paths,scale,c)
        for fold in range(c['folds']):
            z,count=samples[fold]
            ops[fold],details=learn(z[E22].to_numpy(),z[E25].to_numpy(),c['model'])
            mp=root/f'modelos/{scale}/fold_{fold}.npz';mp.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(mp,**ops[fold])
            z[['x','y']].to_parquet(mp.with_suffix('.train.parquet'),index=False)
            audit.append(dict(scale_m=scale,fold=fold,eligible_training=count,sampled_training=len(z),operators=details))
        for path in paths:
            z=pd.read_parquet(path);f=folds(z.x,z.y,c)
            out=z[['x','y','coverage','valid_pixels']].copy();out['fold']=f
            for method in METHODS:out[method]=np.nan
            for fold in np.unique(f):
                ix=f==fold
                for method,scores in predict(z.loc[ix,E22].to_numpy(),z.loc[ix,E25].to_numpy(),ops[int(fold)]).items():out.loc[ix,method]=scores
            op=root/f'puntajes/{scale}/{path.name}';op.parent.mkdir(parents=True,exist_ok=True);out.to_parquet(op,index=False)
        print('Puntajes fuera de entrenamiento:',scale,'m',flush=True)
    files=sorted((root/'puntajes').rglob('*.parquet'))+sorted((root/'modelos').rglob('*.npz'))
    frozen={k:v for k,v in c.items() if k not in ['dw22','dw25','dw_band_order']}
    dump(root/'02_ajustes_sin_dw.json',audit)
    dump(root/'02_puntajes_congelados.json',dict(config=frozen,files=manifest(files),selection='Baja variación por ángulo directo de entrenamiento; sin consultar Dynamic World.'))

def js(a,b):
    a=np.maximum(a,1e-12);b=np.maximum(b,1e-12);a=a/a.sum(1,keepdims=True);b=b/b.sum(1,keepdims=True);m=(a+b)/2
    return .5*np.sum(a*np.log(a/m),axis=1)+.5*np.sum(b*np.log(b/m),axis=1)

def dw_features(a,b):
    return pd.DataFrame({'dw_js':js(a,b),'delta_built':b[:,6]-a[:,6],'delta_vegetation':b[:,VEG].sum(1)-a[:,VEG].sum(1),'delta_bare':b[:,7]-a[:,7]})

def reference_tile(x,y,c,native_xy):
    a=read_tile(c['dw22'],9,x,y,c);b=read_tile(c['dw25'],9,x,y,c)
    good=np.isfinite(a).all(1)&np.isfinite(b).all(1)
    xx,yy=tile_xy(x,y,c);domain=pd.MultiIndex.from_frame(native_xy[['x','y']])
    good&=pd.MultiIndex.from_arrays([xx,yy]).isin(domain)
    a=a[good];b=b[good];a/=a.sum(1,keepdims=True);b/=b.sum(1,keepdims=True)
    z=pd.DataFrame(np.hstack([a,b]),columns=[f'{y}_{i}' for y in ['p22','p25'] for i in range(9)])
    z['x']=xx[good];z['y']=yy[good]
    return z

def reference_scale(native,scale,c):
    if scale==10:z=native.copy()
    else:
        z=native.copy();z['col']=np.floor(z.x/scale).astype(int);z['row']=np.floor(z.y/scale).astype(int)
        g=z.groupby(['col','row']);cols=[f'{y}_{i}' for y in ['p22','p25'] for i in range(9)]
        z=g[cols].mean();z['dw_coverage']=g.size()/(scale//10)**2;z=z[z.dw_coverage>=c['coverage']].reset_index()
        z['x']=(z.col+.5)*scale;z['y']=(z.row+.5)*scale
    if z.empty:return pd.DataFrame({name:pd.Series(dtype='float64') for name in ['x','y','dw_js','delta_built','delta_vegetation','delta_bare']})
    f=dw_features(z[[f'p22_{i}' for i in range(9)]].to_numpy(),z[[f'p25_{i}' for i in range(9)]].to_numpy())
    f[['x','y']]=z[['x','y']].to_numpy();return f

def calibrate(train,test):
    cols=['dw_js','delta_built','delta_vegetation','delta_bare']
    a=np.abs(train[cols].to_numpy());b=np.abs(test[cols].to_numpy())
    scales=np.maximum(np.quantile(a,.9,axis=0),1e-8);u=(a/scales).mean(1);v=(b/scales).mean(1)
    low,high=np.quantile(u,[.3,.8]);drivers=np.quantile(a[:,1:],.8,axis=0)
    label=np.full(len(test),-1,dtype=np.int8);label[v<=low]=0;label[(v>=high)&(b[:,1:]>=drivers).any(1)]=1
    return label,v,dict(scales=scales.tolist(),low=float(low),high=float(high),drivers=drivers.tolist())

def metric_rows(test,fold,scale,view,c):
    z=test[test.label>=0];rows=[]
    for method in METHODS:
        if test[method].isna().any():
            for budget in c['budgets']:
                rows.append(dict(scale_m=scale,view=view,fold=fold,method=method,status='puntajes_incompletos_o_sin_soporte',n_units=len(test),n_labelled=len(z),n_positive=int((z.label==1).sum()),ap=np.nan,spearman=np.nan,budget=budget,reviewed=0,precision=np.nan,recall=np.nan,lift=np.nan))
            continue
        ap=float(average_precision_score(z.label,z[method])) if z.label.nunique()==2 else np.nan
        rho=float(test[method].corr(test.proxy,method='spearman')) if len(test)>2 and test[method].nunique()>1 and test.proxy.nunique()>1 else np.nan
        order=z.sort_values([method,'x','y'],ascending=[False,True,True]);npos=int((order.label==1).sum());prevalence=npos/len(order) if len(order) else np.nan
        for budget in c['budgets']:
            k=min(len(order),max(1,math.ceil(budget*len(order))));hits=int(order.head(k).label.sum())
            rows.append(dict(scale_m=scale,view=view,fold=fold,method=method,status='evaluado' if z.label.nunique()==2 else 'sin_ambas_etiquetas',n_units=len(test),n_labelled=len(z),n_positive=npos,ap=ap,spearman=rho,budget=budget,reviewed=k,precision=hits/k if k else np.nan,recall=hits/npos if npos else np.nan,lift=(hits/k)/prevalence if k and prevalence>0 else np.nan))
    return rows

def evaluate_frame(z,scale,view,c,out):
    rows=[];thresholds=[];parts=[]
    for fold in range(c['folds']):
        test=z[folds(z.x,z.y,c)==fold].copy();train=z.loc[train_mask(z.x,z.y,fold,scale,c)]
        if len(train)<30 or len(test)<2:
            thresholds.append(dict(fold=fold,status='sin_soporte',n_train=len(train),n_test=len(test)));continue
        test['label'],test['proxy'],cal=calibrate(train,test)
        rows.extend(metric_rows(test,fold,scale,view,c));parts.append(test)
        thresholds.append(dict(fold=fold,status='evaluado',n_train=len(train),n_test=len(test),**cal))
    if parts:pd.concat(parts,ignore_index=True).to_parquet(out/f'predicciones_{view}.parquet',index=False)
    dump(out/f'umbrales_{view}.json',thresholds)
    return rows

def aggregate_scores(z,source,target,c):
    """Media de ángulos locales, distinta del ángulo entre embeddings promediados."""
    assert source<=target and target%source==0
    a=z.copy();a['col']=np.floor(a.x/target).astype(int);a['row']=np.floor(a.y/target).astype(int)
    g=a.groupby(['col','row']);v=g[METHODS].mean();v['coverage']=g.size()/(target//source)**2
    v=v[v.coverage>=c['coverage']].reset_index();v['x']=(v.col+.5)*target;v['y']=(v.row+.5)*target
    return v

def evaluar(c,tag=None):
    root=Path(c['out']);freeze=json.loads((root/'02_puntajes_congelados.json').read_text(encoding='utf8'))
    if tag is not None and (not tag.replace('_','').isalnum()):raise ValueError('Identificador de evaluación no válido.')
    report=root if tag is None else root/'sensibilidad_referencia'/tag
    report.mkdir(parents=True,exist_ok=True)
    refroot=root/('referencias' if tag is None else f'referencias_{tag}')
    frozen={k:v for k,v in c.items() if k not in ['dw22','dw25','dw_band_order']}
    if frozen!=freeze['config']:raise ValueError('La configuración difiere de la usada para congelar puntajes.')
    for p,digest in freeze['files'].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=digest:raise ValueError('Se modificaron puntajes/modelos congelados: '+p)
    if not c.get('dw22') or not c.get('dw25'):raise FileNotFoundError('Faltan probabilidades anuales completas de Dynamic World; definir dw22 y dw25 en config.json. No usar las cachés de casos seleccionados.')
    if c.get('dw_band_order')!=CLASSES:raise ValueError('Declarar y verificar el orden de las nueve clases de Dynamic World.')
    # DW se abre por primera vez aquí, después de verificar el congelamiento.
    def prepare_reference(path):
        x,y=map(int,path.stem.split('_'));xy=pd.read_parquet(path,columns=['x','y'])
        native=reference_tile(x,y,c,xy)
        for scale in c['scales_m']:
            ref=reference_scale(native,scale,c);p=refroot/f'{scale}/{path.name}';p.parent.mkdir(parents=True,exist_ok=True);ref.to_parquet(p,index=False)
        return path.name
    reference_paths=sorted((root/'puntajes/10').glob('*.parquet'))
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks=[pool.submit(prepare_reference,path) for path in reference_paths]
        for i,task in enumerate(as_completed(tasks),1):
            task.result()
            if i%100==0 or i==len(tasks):print('Referencia DW preparada',i,'/',len(tasks),flush=True)
    rows=[];support=[]
    for scale in c['scales_m']:
        parts=[];commons={500:[],1000:[]};count=0
        for p in sorted((root/f'puntajes/{scale}').glob('*.parquet')):
            score=pd.read_parquet(p);count+=len(score)
            ref=pd.read_parquet(refroot/f'{scale}/{p.name}')
            if len(ref) and len(score):parts.append(score.merge(ref,on=['x','y'],validate='one_to_one'))
            for target in commons:
                if scale<=target:commons[target].append(aggregate_scores(score,scale,target,c))
        z=pd.concat(parts,ignore_index=True);support.append(dict(scale_m=scale,n_embedding_units=count,n_with_dw=len(z),fraction_with_dw=len(z)/count if count else 0))
        out=report/f'evaluacion/{scale}';out.mkdir(parents=True,exist_ok=True)
        rows.extend(evaluate_frame(z,scale,'nativa',c,out))
        for target,ss in commons.items():
            if not ss:continue
            v=pd.concat(ss,ignore_index=True)
            refs=pd.concat([pd.read_parquet(p) for p in sorted((refroot/f'{target}').glob('*.parquet'))],ignore_index=True)
            native_target=pd.concat([pd.read_parquet(p,columns=['x','y']) for p in sorted((root/f'puntajes/{target}').glob('*.parquet'))],ignore_index=True)
            # Soporte común: referencia válida, embedding agregado válido y cobertura
            # suficiente de los puntajes a cada escala <= target, independientemente del método.
            eligible=refs.merge(native_target,on=['x','y'],validate='one_to_one')
            for other in [s for s in c['scales_m'] if s<=target]:
                vv=pd.concat([aggregate_scores(pd.read_parquet(p),other,target,c)[['x','y']] for p in sorted((root/f'puntajes/{other}').glob('*.parquet'))],ignore_index=True)
                eligible=eligible.merge(vv,on=['x','y'],validate='one_to_one')
            v=v.merge(eligible,on=['x','y'],validate='one_to_one')
            # La escala fuente identifica la alternativa; la calibración y la partición
            # espacial usan la geometría objetivo común, no la geometría de origen.
            rr=evaluate_frame(v,target,f'comun_{target}',c,out)
            for row in rr:row['source_scale_m']=scale
            rows.extend(rr)
    if not rows:raise ValueError('No hay grupos con soporte de evaluación. Consultar los archivos de umbrales.')
    result=pd.DataFrame(rows);result['source_scale_m']=result.source_scale_m.fillna(result.scale_m).astype(int) if 'source_scale_m' in result else result.scale_m
    result.to_csv(report/'03_metricas_por_grupo.csv',index=False,encoding='utf-8-sig')
    keys=['view','scale_m','source_scale_m','method','budget']
    mean=result.groupby(keys,as_index=False)[['ap','spearman','precision','recall','lift']].mean()
    counts=result.groupby(keys,as_index=False).agg(valid_ap_folds=('ap','count'),reported_folds=('fold','nunique'))
    mean.merge(counts,on=keys).to_csv(report/'04_metricas_medias.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(support).to_csv(report/'05_soporte_por_escala.csv',index=False)
    dump(report/'03_evaluacion_procedencia.json',dict(dynamic_world=[raster_info(p) for p in c['dw22']+c['dw25']],notes=['Evaluación auxiliar, no campo ni fuente sensorial independiente.','Comparaciones de AP entre escalas solo en soportes comunes; nativa describe tareas distintas.','Selección de escala o método después de esta exploración requiere evaluación posterior reservada.','Sin intervalos inferenciales en esta primera exploración; medias de grupos, no píxeles independientes.']))
    print('Evaluación terminada. Consultar 04_metricas_medias.csv y soporte por escala.')
    figures(report,result)

def figures(root,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    names=['Directo','General','Referencias 20 %','Referencias 30 %','Referencias 40 %']
    for metric,title in [('ap','AP media por grupo espacial'),('lift','Lift al revisar el 10 % etiquetado')]:
        fig,axs=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
        for ax,target in zip(axs,[500,1000]):
            z=result[(result.view==f'comun_{target}')&np.isclose(result.budget,.1)]
            scales=[s for s in [10,100,500,1000] if s<=target]
            mean=z.groupby(['method','source_scale_m'])[metric].mean().unstack('source_scale_m').reindex(index=METHODS,columns=scales)
            a=mean.to_numpy();masked=np.ma.masked_invalid(a)
            im=ax.imshow(masked,aspect='auto',cmap='YlGnBu',vmin=0,vmax=1 if metric=='ap' else max(2,float(np.nanmax(a))) if np.isfinite(a).any() else 2)
            for i in range(5):
                for j in range(len(scales)):
                    vals=z[(z.method==METHODS[i])&(z.source_scale_m==scales[j])][metric].dropna()
                    s=f'{a[i,j]:.2f}\n({len(vals)}/5 grupos)' if np.isfinite(a[i,j]) else 'Sin soporte'
                    ax.text(j,i,s.replace('.',','),ha='center',va='center',fontsize=8,color='white' if np.isfinite(a[i,j]) and a[i,j]>.6*im.norm.vmax else 'black')
            ax.set_xticks(range(len(scales)),[f'{s} m' for s in scales]);ax.set_yticks(range(5),names)
            ax.set(xlabel='Escala fuente de los puntajes',title=f'Celdas comunes de {target} m');fig.colorbar(im,ax=ax,shrink=.8)
        fig.suptitle(title+' — exploración, no validación de campo')
        for suffix in ['png','svg','pdf']:fig.savefig(root/f'06_comparacion_comun_{metric}.{suffix}',dpi=300,bbox_inches='tight')
        plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['auditar','preparar','puntuar','evaluar']);p.add_argument('--config',default=str(HOME/'config.json'))
    a=p.parse_args();c=config(a.config);Path(c['out']).mkdir(parents=True,exist_ok=True)
    globals()[a.stage](c)

if __name__=='__main__':main()
