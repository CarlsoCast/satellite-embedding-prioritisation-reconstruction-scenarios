"""Mapa y seis ampliaciones: factor5, tres semillas, consenso3/3 sin filtros."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
from scipy.ndimage import label,find_objects
import rasterio
from rasterio.transform import from_origin
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'linea2_transferencia_y_zonas_20261028'
sys.path.insert(0,str(SOURCE))
import engine as e
import figure_style as style
OUT=ROOT/'outputs/figura_final_factor5'

def main():
    OUT.mkdir(exist_ok=True)
    _,row,col,risk,core,shape=e.load_data()
    masks=[];records=[]
    for i in range(1,4):
        path=ROOT/'outputs'/f'factor_5_semilla_{i}.npz'
        with np.load(path) as d:
            m=np.unpackbits(d['active'])[:len(risk)].astype(bool);meta=json.loads(str(d['metadata']))
        assert meta['finished'] and meta['penalty']==5 and len(meta['trace'])==200
        t=meta['trace'][-1]
        assert int((m&core).sum())==t['hit_core'] and int((m&~risk).sum())==t['activated_outside']
        records.append(dict(seed=meta['seed'],rules=len(meta['trace']),recovery=t['core_recovery'],outside_km2=t['outside_km2'],sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        masks.append(m)
    votes=np.sum(masks,axis=0).astype(np.uint8)
    selected=(votes==3)&~risk
    candidate=np.zeros(shape,bool);candidate[row,col]=selected
    components,num=label(candidate,structure=np.ones((3,3),int))
    sizes=np.bincount(components.ravel());ids=np.arange(1,len(sizes));order=np.lexsort((ids,-sizes[1:]))
    biggest=ids[order[:6]];objects=find_objects(components)
    with np.load(SOURCE/'outputs/mapa_categorias.npz') as ref:x0=float(ref['x0']);y0=float(ref['y0'])
    areas=[]
    for cid in biggest:
        rs,cs=objects[cid-1];rr,cc=np.nonzero(components[rs,cs]==cid);rr=rr+rs.start;cc=cc+cs.start
        areas.append(dict(component_id=int(cid),pixel_count=int(sizes[cid]),area_m2=int(sizes[cid])*100,area_ha=int(sizes[cid])*.01,row_min=rs.start,row_max=rs.stop-1,col_min=cs.start,col_max=cs.stop-1,x_centroid=x0+cc.mean()*10,y_centroid=y0+rr.mean()*10))
    frame=pd.DataFrame(areas);frame.insert(0,'figure_number',np.arange(1,len(frame)+1));frame.to_csv(OUT/'seis_areas.csv',index=False)
    category=np.zeros(shape,np.uint8);category[row,col]=1;category[row[risk],col[risk]]=2;category[candidate]=4
    assert int(candidate.sum())==int(selected.sum()) and not selected[risk].any()
    assert all((votes[selected]==3))
    style.MANUSCRIPT=e.OLD.parent/'resultados_manuscrito_v6'
    style.FIGURE_DIR=OUT;style.COMPLETE_OUTPUT=OUT/'candidatas_factor5_consenso3';style.N_AREAS=6
    style.plot_complete_figure(frame,category,components,x0,y0)
    transform=from_origin(x0-5,y0+(shape[0]-.5)*10,10,10)
    values=np.full(shape,255,np.uint8);values[row,col]=votes;values[row[risk],col[risk]]=0
    for name,data in [('frecuencia_exterior',values),('candidatas_consenso3',np.where(category==0,255,candidate.astype(np.uint8)).astype(np.uint8))]:
        with rasterio.open(OUT/f'{name}.tif','w',driver='GTiff',height=shape[0],width=shape[1],count=1,dtype='uint8',crs='EPSG:9377',transform=transform,nodata=255,compress='lzw') as dst:
            dst.write(np.flipud(data),1);dst.update_tags(penalty='5',rules_per_seed='200',seeds='30261028,40261028,50261028',criterion='3/3 seed agreement outside reference',spatial_filter='none')
        with rasterio.open(OUT/f'{name}.tif') as src:assert np.array_equal(np.flipud(src.read(1)),data)
    np.savez_compressed(OUT/'datos_figura.npz',category=category,components=components,votes=votes,x0=x0,y0=y0)
    summary=dict(penalty=5,geometry='full whitening intensity1 ridge0.05',runs=records,total_pixels=int(selected.sum()),total_ha=float(selected.sum()*.01),components=int(num),criterion='intersection of three runs outside positive reference',filters='none',display_selection='six largest 8-connected components by area; no smoothing or minimum size; component id breaks area ties',six_largest=areas)
    (OUT/'procedencia.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    caption=f'Ubicación de las áreas propuestas para revisión con coseno blanqueado y penalización 5. El naranja indica píxeles exteriores al registro activados en las tres corridas, cada una con 200 reglas. Superficie coincidente total: {selected.sum()*.01:.2f} ha. Las ampliaciones muestran las seis áreas contiguas de mayor superficie (vecindad de ocho), numeradas en orden descendente; las superficies corresponden al componente señalado, no a toda la ventana. No se aplicaron filtros de vecindad, suavizado ni tamaño mínimo. La coincidencia expresa repetibilidad de búsqueda, no amenaza confirmada.'
    (OUT/'pie_de_figura.txt').write_text(caption,encoding='utf-8')
    (OUT/'resultados.html').write_text(f'<!doctype html><html lang="es"><meta charset="utf-8"><style>body{{font:17px system-ui;max-width:1300px;margin:30px auto}}img{{width:100%}}</style><h1>Figura final con penalización 5</h1><img src="candidatas_factor5_consenso3.png"><p>{caption}</p><p><a href="candidatas_factor5_consenso3.pdf">PDF</a> · <a href="seis_areas.csv">Áreas ampliadas</a> · <a href="candidatas_consenso3.tif">GeoTIFF</a></p></html>',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
