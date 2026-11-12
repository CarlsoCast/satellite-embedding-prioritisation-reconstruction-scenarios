"""Ejemplo visual de un escenario OOF; no vuelve a ajustar los operadores.

El panel espacial del escenario ilustra la dirección del vector de ventana
mediante una rotación rígida de los embeddings de sus píxeles. El operador
se evalúa a 500 m; esa ilustración no estima cambios individuales a 10 m.
"""
from pathlib import Path
import json, hashlib, re, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import from_bounds


def _unit(a):
    a=np.asarray(a,dtype=float)
    return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)


def _rotation(pixels,initial,scenario):
    x,y=_unit(initial),_unit(scenario)
    c=np.clip(x@y,-1,1);theta=np.arccos(c)
    u=y-c*x;length=np.linalg.norm(u)
    if length<1e-12:return _unit(pixels),float(np.degrees(theta))
    u=u/length
    p=_unit(pixels)
    px=np.sum(p*x,axis=-1,keepdims=True)
    pu=np.sum(p*u,axis=-1,keepdims=True)
    rest=p-px*x-pu*u
    return _unit((px*np.cos(theta)-pu*np.sin(theta))*x+
                 (px*np.sin(theta)+pu*np.cos(theta))*u+rest),float(np.degrees(theta))


def _read_pixels(folder,year,row):
    files=sorted(Path(folder).glob(f'AEF_{year}_Bogota_*.tif'),
                 key=lambda p:int(re.search(r'_(\d+)\.tif$',p.name).group(1)))
    assert len(files)==64,(year,len(files))
    arrays=[];transform=None
    for p in files:
        with rasterio.open(p) as src:
            assert src.crs.to_epsg()==9377
            win=from_bounds(row.xmin,row.ymin,row.xmax,row.ymax,src.transform)
            arrays.append(src.read(1,window=win,boundless=True,masked=True).astype('float32').filled(np.nan))
            transform=src.window_transform(win)
    result=np.stack(arrays,axis=-1)
    assert result.shape[:2]==(50,50),result.shape
    return _unit(result),transform,files


def generar_ejemplo_transformacion(carpeta_linea3=None,carpeta_rasters=None):
    home=Path(carpeta_linea3) if carpeta_linea3 else Path(__file__).resolve().parent
    repo_root=home.parents[1]
    root=Path(os.environ.get('SATEMB_LINE3_DATA',repo_root/'data/raw/line3')).resolve()
    rasters=Path(carpeta_rasters) if carpeta_rasters else Path(
        os.environ.get('SATEMB_RASTER_DIR',root/'DATOS_ALINEADOS'))
    source=home/'nueva_corrida_integral/principal'
    out=home/'ejemplo_transformacion';out.mkdir(exist_ok=True)
    mp=source/'02_embedding_oof.parquet';vp=source/'06_vectores_oof.npz'
    frame=pd.read_parquet(mp);vectors=np.load(vp)
    frame=frame.iloc[vectors['row']].reset_index(drop=True)
    selected=frame.index[(frame.window_id=='36_19')&(frame.method=='familia_ridge')&
                         (frame.family=='construido_a_vegetacion_lenosa')]
    assert len(selected)==1
    i=int(selected[0]);record=frame.iloc[i]
    x,z,y=(vectors[k][i] for k in ['initial','scenario','observed'])
    angle=lambda a:float(np.degrees(np.arccos(np.clip(_unit(a)@_unit(y),-1,1))))
    before,after=angle(x),angle(z)
    assert np.isclose(before,record.identity_error_deg,atol=1e-6)
    assert np.isclose(after,record.angular_error_deg,atol=1e-6)
    base=root/'resultados_60_downstream_unificado_lst_bsa/03_base_autocontenida.parquet'
    b=pd.read_parquet(base)
    row=b.loc[b.window_id==record.window_id].iloc[0]
    pixels22,transform,files22=_read_pixels(rasters,2022,row)
    pixels25,_,files25=_read_pixels(rasters,2025,row)
    mean22=_unit(np.nanmean(pixels22,axis=(0,1)))
    spatial,rot=_rotation(pixels22,mean22,z)
    # Tres componentes originales, elegidas por dispersión conjunta, no por
    # cercanía visual al escenario. El estiramiento es idéntico en los tres paneles.
    joined=np.concatenate([a.reshape(-1,64) for a in [pixels22,spatial,pixels25]])
    ok=np.isfinite(joined).all(axis=1)
    dims=np.argsort(np.std(joined[ok],axis=0))[-3:][::-1]
    lo,hi=np.percentile(joined[ok][:,dims],[2,98],axis=0)
    rgb=lambda a:np.clip((a[...,dims]-lo)/np.maximum(hi-lo,1e-12),0,1)
    style={'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
           'axes.spines.right':False,'axes.titlesize':10,'axes.titleweight':'bold'}
    with plt.rc_context(style):
        fig=plt.figure(figsize=(7.2,6.0))
        grid=fig.add_gridspec(2,3,height_ratios=[1,1.2],hspace=.46,wspace=.08,
                             left=.11,right=.985,bottom=.065,top=.86)
        ax=fig.add_subplot(grid[0,:]);comp=np.arange(1,65)
        for vals,color,label in [(x,'#778ba3','2022'),(z,'#dc682a','Escenario'),(y,'#173f60','2025 observado')]:
            ax.plot(comp,vals,lw=1.35,color=color,label=label)
        ax.set(xlim=(1,64),xlabel='Componente del embedding (1–64)',ylabel='Valor del vector unitario',
               title='a. Transformación del embedding de una ventana de 500 × 500 m')
        ax.set_title('a. Transformación del embedding de una ventana de 500 × 500 m',pad=34)
        ax.grid(axis='y',alpha=.18)
        ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),ncol=3,frameon=False,fontsize=9)
        ax.text(.99,1.055,f'Error: {before:.1f}° → {after:.1f}°'.replace('.',','),
                transform=ax.transAxes,ha='right',fontsize=9,color='#173f60')
        for j,(a,title) in enumerate([(pixels22,'b. Estado 2022'),(spatial,'c. Escenario ilustrado'),(pixels25,'d. Estado 2025')]):
            panel=fig.add_subplot(grid[1,j]);panel.imshow(rgb(a),interpolation='nearest')
            panel.set_title(title,loc='left',fontsize=8.8);panel.set_xticks([]);panel.set_yticks([])
            panel.plot([3,13],[45,45],color='white',lw=3,solid_capstyle='butt')
            panel.text(3,43,'100 m',color='white',fontsize=8,fontweight='bold')
            for s in panel.spines.values():s.set_visible(True);s.set_linewidth(.6);s.set_color('#7f8994')
        fig.text(.5,.018,'Color ilustrativo: mismas tres componentes y misma escala en los tres paneles.',
                 ha='center',fontsize=8,color='#495866')
        for ext in ['png','pdf','svg']:fig.savefig(out/f'01_ejemplo_transformacion.{ext}',dpi=300,facecolor='white')
        plt.close(fig)
    pd.DataFrame({'componente':np.arange(1,65),'embedding_2022':x,'escenario_ridge':z,'embedding_2025':y}).to_csv(out/'02_componentes.csv',index=False)
    audit={'window_id':str(record.window_id),'family':str(record.family),'method':'Ridge contextual',
           'fold':int(record.fold),'evaluation':'Fuera del entrenamiento del operador',
           'selection':'Ventana fija 36_19 de la figura anterior; ejemplo ilustrativo, no estimador del desempeño agregado.',
           'identity_angle_deg':before,'scenario_angle_deg':after,
           'mse_initial':float(np.mean((x-y)**2)),'mse_scenario':float(np.mean((z-y)**2)),
           'rgb_dimensions_1_based':(dims+1).tolist(),'rgb_limits_p2_p98':[lo.tolist(),hi.tolist()],
           'spatial_illustration':'Rotación rígida de los píxeles de 2022 que lleva su vector medio normalizado al escenario de ventana; no es la salida de un operador ajustado a 10 m.',
           'rotation_deg':rot,'raster_files':[str(p) for p in files22+files25],
           'sources':[{'file':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [mp,vp,base]],
           'figure':str(out/'01_ejemplo_transformacion.png')}
    (out/'03_auditoria.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Ejemplo OOF:',record.window_id,'ángulo:',before,'->',after,flush=True)
    return audit,out


if __name__=='__main__':generar_ejemplo_transformacion()
