"""Postproceso reproducible de máscaras guardadas; no ajusta modelos."""
from pathlib import Path
import sys,json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Rectangle,Patch
from shapely.geometry import shape,box
import rasterio
from rasterio.transform import from_origin

ROOT=Path(__file__).resolve().parent; DATA=ROOT/'outputs'; OUT=DATA/'figuras_finales'; OUT.mkdir(exist_ok=True)
SRC=ROOT.parent/'linea2_transferencia_y_zonas_20261028'
sys.path.insert(0,str(SRC))
import engine as e
import figure_style as sty
from informe_estabilidad import report

with np.load(DATA/'frecuencia_20_semillas.npz') as z:
    votes=z['votes'];row=z['row'];col=z['col'];risk=z['risk'];sh=tuple(z['shape']);seeds=z['seeds']
masks=[];provenance=[]
for i,seed in enumerate(seeds,1):
    path=DATA/f'factor_5_semilla_{i}.npz'
    with np.load(path) as z:
        m=np.unpackbits(z['active'])[:len(votes)].astype(bool);meta=json.loads(str(z['metadata']))
    assert meta['finished'] and meta['seed']==int(seed) and meta['penalty']==5 and len(meta['trace'])==400
    masks.append(m);provenance.append(dict(seed=int(seed),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
masks=np.array(masks)
assert np.array_equal((masks & ~risk).sum(0),votes)
with np.load(SRC/'outputs/mapa_categorias.npz') as z:x0=float(z['x0']);y0=float(z['y0'])
base=np.zeros(sh,np.uint8);base[row,col]=1;base[row[risk],col[risk]]=2
v=np.zeros(sh,np.uint8);v[row,col]=votes
# Ranking de ventanas por suma de frecuencia al cuadrado: todos los votos contribuyen.
# Misma superficie; no se exige continuidad. Umbral 15/20 solo para descripción posterior.
side=200; step=10
score=(v.astype(np.float64)/20)**2
integral=np.pad(score,((1,0),(1,0))).cumsum(0).cumsum(1)
rr,cc=np.meshgrid(np.arange(0,sh[0]-side+1,step),np.arange(0,sh[1]-side+1,step),indexing='ij')
scores=integral[rr+side,cc+side]-integral[rr,cc+side]-integral[rr+side,cc]+integral[rr,cc]
order=np.argsort(-scores.ravel(),kind='stable'); chosen=[]
for k in order:
    r=int(rr.ravel()[k]);c=int(cc.ravel()[k])
    if all(r+side<=a or a+side<=r or c+side<=b or b+side<=c for a,b in chosen):chosen.append((r,c))
    if len(chosen)==6:break
admin=e.OLD.parent/'resultados_manuscrito_v6'
boundary=shape(json.loads((admin/'municipio_oficial.geojson').read_text(encoding='utf-8'))['features'][0]['geometry'])
localities=[(f['properties']['LOCNOMBRE'],shape(f['geometry'])) for f in json.loads((admin/'localidad_oficial.geojson').read_text(encoding='utf-8'))['features']]
ext=[x0-5,x0+(sh[1]-.5)*10,y0-5,y0+(sh[0]-.5)*10]
extent=lambda r,c:[x0+(c-.5)*10,x0+(c+side-.5)*10,y0+(r-.5)*10,y0+(r+side-.5)*10]
records=[]
for i,(r,c) in enumerate(chosen,1):
    sub=v[r:r+side,c:c+side];b=extent(r,c);poly=box(b[0],b[2],b[1],b[3])
    intersections=sorted([(name,poly.intersection(g).area) for name,g in localities if poly.intersects(g)],key=lambda t:-t[1])
    records.append(dict(window=i,row=r,col=c,bounds=b,area_window_ha=400,active_ha=float((sub>0).sum()*.01),high_frequency_ha=float((sub>=15).sum()*.01),unanimity_ha=float((sub==20).sum()*.01),score=float(((sub.astype(float)/20)**2).sum()),localities=[name for name,a in intersections if a>10000]))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
cmap=plt.get_cmap('YlOrRd',20);norm=BoundaryNorm(np.arange(.5,21.5),20)
fig=plt.figure(figsize=(11.7,8.6));gs=fig.add_gridspec(3,3,width_ratios=[1.45,1,1],hspace=.27,wspace=.17)
ax=fig.add_subplot(gs[:,0]);bounds=[4862500,4889000,2050800,2094000]
def draw(ax,b,r=None,c=None,labels=False):
    bg=base if r is None else base[r:r+side,c:c+side]
    vv=v if r is None else v[r:r+side,c:c+side]
    ex=ext if r is None else extent(r,c)
    ax.imshow(bg,origin='lower',extent=ex,cmap=ListedColormap(['white','#eceff1','#8499b4']),vmin=0,vmax=2,interpolation='nearest',zorder=1)
    im=ax.imshow(np.ma.masked_equal(vv,0),origin='lower',extent=ex,cmap=cmap,norm=norm,interpolation='nearest',zorder=2)
    sty.base_map(ax,b,boundary,localities,labels=labels)
    return im
im=draw(ax,bounds,labels=True);sty.map_orientation(ax,bounds);ax.set_title('Ubicación de las áreas propuestas',fontsize=13,fontweight='bold',pad=10)
for rec in records:
    i=rec['window'];r=rec['row'];c=rec['col'];b=rec['bounds']
    ax.add_patch(Rectangle((b[0],b[2]),2000,2000,fill=False,ec='#3b1723',lw=.9,zorder=9))
    ax.text(b[1]+100,b[3],str(i),fontsize=11,fontweight='bold',color='#84202b',zorder=12,bbox=dict(fc='white',ec='none',alpha=.9,pad=.5))
    subax=fig.add_subplot(gs[(i-1)//2,1+(i-1)%2]);draw(subax,b,r,c)
    subax.set_title(f"{i} · {sty.comma(rec['high_frequency_ha'])} ha con frecuencia ≥75 %",fontsize=9.2,fontweight='bold',pad=6)
    subax.plot([b[0]+100,b[0]+600],[b[2]+130]*2,color='#15232c',lw=2,zorder=10)
    subax.text(b[0]+100,b[2]+220,'500 m',fontsize=8,zorder=11,bbox=dict(fc='white',ec='none',alpha=.7,pad=.5))
fig.subplots_adjust(left=.015,right=.985,top=.94,bottom=.18)
cax=fig.add_axes([.50,.092,.41,.019]);cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[1,5,10,15,20]);cb.ax.set_xticklabels(['5 %','25 %','50 %','75 %','100 %']);cb.set_label('Frecuencia de activación exterior en 20 corridas',fontsize=9)
fig.legend(handles=[Patch(color='#8499b4',label='Referencia positiva conocida'),Patch(color='#eceff1',label='Resto del dominio')],loc='lower left',bbox_to_anchor=(.025,.07),frameon=False,fontsize=8)
fig.text(.5,.023,'Penalización 5 · 400 reglas por corrida · Ventanas de 2 × 2 km sin solapamiento · Mapa completo sin filtros espaciales',ha='center',fontsize=9)
fig.savefig(OUT/'mapa_final_frecuencias.png',dpi=300,bbox_inches='tight');fig.savefig(OUT/'mapa_final_frecuencias.pdf',bbox_inches='tight');plt.close(fig)
profile=dict(driver='GTiff',height=sh[0],width=sh[1],count=1,dtype='uint8',crs='EPSG:9377',transform=from_origin(x0-5,y0+(sh[0]-.5)*10,10,10),nodata=255,compress='deflate')
raster=np.full(sh,255,np.uint8);raster[row[~risk],col[~risk]]=votes[~risk]
with rasterio.open(OUT/'frecuencia_exterior_20.tif','w',**profile) as dst:dst.write(np.flipud(raster),1)
with rasterio.open(OUT/'frecuencia_exterior_20.tif') as dst:assert np.array_equal(np.flipud(dst.read(1)),raster)
features=[dict(type='Feature',properties={k:val for k,val in rec.items() if k!='bounds'},geometry=box(rec['bounds'][0],rec['bounds'][2],rec['bounds'][1],rec['bounds'][3]).__geo_interface__) for rec in records]
(OUT/'ventanas.geojson').write_text(json.dumps(dict(type='FeatureCollection',crs=dict(type='name',properties=dict(name='EPSG:9377')),features=features),ensure_ascii=False),encoding='utf-8')
report(masks,risk,OUT,20)
q=np.load(OUT/'permutaciones_20.npz')['quantiles'];summary=json.loads((OUT/'resumen_20.json').read_text())
summary.update(last_frequency_change=float(q[1,-1,0]),last_majority_jaccard=float(q[1,-1,1]),last_enter_ha=float(q[1,-1,2]),last_leave_ha=float(q[1,-1,3]),union_ha=float((votes>0).sum()*.01),high_frequency_ha=float((votes>=15).sum()*.01),windows=records,map_ranking='sum((votes/20)**2) over 200x200 pixel windows; 10-pixel stride; greedy non-overlap; no map filtering',model_provenance=provenance)
(OUT/'resumen_integrado.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({k:val for k,val in summary.items() if k not in ['windows','model_provenance','notes']},ensure_ascii=False),flush=True)
