"""Five hypothesis-driven geometries, three paired random seeds, one regional task."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
import csv,io,json,time,hashlib
from pathlib import Path
import numpy as np
import engine as e

VARIANTS={
 'centrado':(0.,0.,'Solo centrado'),
 'parcial_r005':(.5,.05,'Parcial · r=0,05'),
 'completo_r005':(1.,.05,'Completo · r=0,05'),
 'parcial_r05':(.5,.5,'Parcial · r=0,5'),
 'completo_r05':(1.,.5,'Completo · r=0,5'),
}
SEEDS=(30261028,40261028,50261028)
JOBS={f'{v}__s{s+1}':(v,s) for s in range(3) for v in VARIANTS}
e.METHODS=tuple(JOBS)
e.LABELS={m:f'{VARIANTS[v][2]} · S{s+1}' for m,(v,s) in JOBS.items()}
e.EXPERIMENT_GEOMETRIES=dict(variants={k:dict(intensity=a,ridge_fraction=r,label=l) for k,(a,r,l) in VARIANTS.items()},seeds=list(SEEDS),reference='All regional pixels; same mean/covariance; 64 dimensions',formula='normalize((x-mu) @ V @ diag((eigenvalues + r*median_positive_eigenvalue)^(-alpha/2)) @ V.T)',design='2 intensities x 2 ridge levels + centered control; 3 paired seeds',interpretation='Experimental contrasts, not established optimal parameter values')
e.seed_for_method=lambda method,iteration,attempt: SEEDS[JOBS[method][1]]+iteration*10+attempt


def matrix(cov,alpha,r):
    eig,v=np.linalg.eigh(cov)
    ridge=r*np.median(eig[eig>0])
    if alpha==0: return np.eye(len(eig),dtype=np.float32),ridge
    return ((v*np.maximum(eig+ridge,1e-12)**(-alpha/2))@v.T).astype(np.float32),ridge


def prepare_geometry(x,risk):
    path=e.ROOT/'geometria.npz'
    if path.exists():
        with np.load(path) as z:return {k:z[k] for k in z.files}
    # Reuse the exact regional moments, not approximate reconstructed covariance.
    source=e.ROOT.parent/'linea2_variantes_blanqueamiento_20261028/geometria.npz'
    payload=source.read_bytes()
    with np.load(io.BytesIO(payload)) as z:
        mu=z['region_mu'].copy();cov=z['region_cov'].copy();n=int(z['region_count'])
    assert n==len(x)
    geom=dict(mu=mu,cov=cov,n=np.asarray(n))
    for key,(alpha,r,_) in VARIANTS.items():
        w,ridge=matrix(cov,alpha,r)
        geom[key+'_W']=w;geom[key+'_ridge']=np.asarray(ridge)
    with path.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,**geom)
    os.replace(path.with_suffix('.tmp'),path)
    e.dump(e.ROOT/'procedencia_geometria.json',dict(source=str(source),sha256=hashlib.sha256(payload).hexdigest(),n=n))
    return geom


class Lazy:
    def __init__(self,shape):self.shape=shape
    def __getitem__(self,method):
        variant=JOBS[method][0]
        return np.memmap(e.ROOT/'cache'/f'{variant}.dat',dtype='float32',mode='r',shape=self.shape)


def representations(x,geom):
    for key,(alpha,_,_) in VARIANTS.items():
        path=e.ROOT/'cache'/f'{key}.dat'
        if not path.exists():
            e.log('Preparando '+key)
            tmp=path.with_suffix('.tmp');z=np.memmap(tmp,dtype='float32',mode='w+',shape=x.shape)
            for start in range(0,len(x),50000):
                a=np.asarray(x[start:start+50000])-geom['mu']
                if alpha:a=a@geom[key+'_W']
                z[start:start+len(a)]=e.unit(a)
            z.flush();del z;os.replace(tmp,path)
        assert path.stat().st_size==x.size*4
    return Lazy(x.shape)


def exports(states,totals,args):
    allrows=[];status={}
    for method,s in states.items():
        allrows.extend(s['trace'])
        status[method]=dict(status=s['status'],reason=s['reason'],iterations=len(s['trace']),last=s['trace'][-1] if s['trace'] else None)
    e.dump(e.ROOT/'estado.json',dict(updated=time.strftime('%Y-%m-%d %H:%M:%S'),pid=os.getpid(),methods=status,totals=totals))
    if allrows:
        b=io.StringIO();w=csv.DictWriter(b,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
        e.atomic_bytes(e.ROOT/'trayectoria.csv',b.getvalue().encode('utf-8-sig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig,ax=plt.subplots(1,2,figsize=(13,5.4));colors=plt.get_cmap('tab10').colors
    for method,s in states.items():
        key,seed=JOBS[method];color=colors[list(VARIANTS).index(key)]
        t=s['trace'];style=('-', '--', ':')[seed]
        ax[0].plot([0]+[r['outside_rate']*100 for r in t],[0]+[r['core_recovery']*100 for r in t],linestyle=style,color=color,lw=1.4)
        ax[1].plot([0]+[r['iteration'] for r in t],[0]+[r['objective'] for r in t],linestyle=style,color=color,lw=1.4)
    ax[0].set(xlabel='Territorio sin registro activado (%)',ylabel='Núcleos recuperados (%)',title='Recuperación frente a extensión exterior')
    ax[1].set(xlabel='Número de reglas',ylabel='Recuperación − 3 × extensión exterior',title='Beneficio acumulado por semilla')
    handles=[Line2D([0],[0],color=colors[i],label=v[2]) for i,v in enumerate(VARIANTS.values())]
    for a in ax:a.grid(alpha=.2)
    fig.legend(handles=handles,loc='lower center',ncol=3,fontsize=9)
    fig.suptitle('Intensidad y regularización · líneas continua/discontinua/punteada = semillas 1/2/3')
    fig.tight_layout(rect=(0,.12,1,.94));buf=io.BytesIO();fig.savefig(buf,format='png',dpi=130);plt.close(fig)
    e.atomic_bytes(e.ROOT/'curvas.png',buf.getvalue())
    rows=''
    for key in VARIANTS:
        for seed in range(3):
            method=f'{key}__s{seed+1}';s=states[method];q=s['trace'][-1] if s['trace'] else None
            rows+=f'<tr><td>{VARIANTS[key][2]}</td><td>{seed+1}</td><td>{len(s["trace"])}</td><td>'+ (f'{100*q["core_recovery"]:.2f}%' if q else '—')+'</td><td>'+ (f'{q["outside_km2"]:.3f}' if q else '—')+f'</td><td>{s["status"]}</td></tr>'
    page='''<!doctype html><html lang="es"><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>Intensidad y regularización · Línea 2</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:0 18px;color:#213547}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}img{width:100%}</style><h1>¿Cuánto blanquear y cuánto regularizar?</h1><p>Cinco configuraciones, tres semillas por configuración. Misma media y covarianza regionales, mismos datos, penalización exterior 3 y búsqueda iterativa.</p><p><b>Hipótesis:</b> el blanqueamiento parcial conserva información de la geometría; la regularización mayor modera la amplificación de direcciones de baja variabilidad. El control separa ambos efectos del simple centrado.</p><p>Comparar a igual extensión exterior y considerar el número de reglas. Son curvas de ajuste regional: la activación exterior no equivale a falsos positivos confirmados. Las semillas miden variación de la búsqueda, no incertidumbre del registro.</p>'''+f'<img src="curvas.png?v={time.time_ns()}"><table><tr><th>Configuración</th><th>Semilla</th><th>Reglas</th><th>Núcleos recuperados</th><th>Exterior (km²)</th><th>Estado</th></tr>'+rows+'</table><p>Actualización: '+time.strftime('%Y-%m-%d %H:%M:%S')+'</p><p><a href="trayectoria.csv">Trayectoria CSV</a> · <a href="LEEME.md">Diseño</a></p></html>'
    e.atomic_bytes(e.ROOT/'panel.html',page.encode('utf-8'))


original_checks=e.checks
def checks():
    original_checks()
    rng=np.random.default_rng(72);a=rng.normal(size=(90,6));cov=np.cov(a,rowvar=False)
    w0,_=matrix(cov,0,.05);assert np.array_equal(w0,np.eye(6))
    wf,r=matrix(cov,1,.05);wp,_=matrix(cov,.5,.05)
    assert np.allclose(wf@wf.T,np.linalg.inv(cov+r*np.eye(6)),atol=1e-6)
    assert np.allclose(wp@wp,wf,atol=1e-6)
    assert len({e.seed_for_method(f'{v}__s1',8,0) for v in VARIANTS})==1
    assert len({e.seed_for_method(f'centrado__s{s+1}',8,0) for s in range(3)})==3
    e.log('Pruebas correctas: intensidad cero, potencia parcial, blanqueamiento completo y semillas emparejadas.')

e.checks=checks;e.prepare_geometry=prepare_geometry;e.representations=representations;e.exports=exports
if __name__=='__main__':e.main()
