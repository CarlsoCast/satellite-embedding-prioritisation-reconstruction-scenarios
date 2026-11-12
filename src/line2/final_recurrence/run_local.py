"""Sensibilidad pareada de la penalización exterior; una tarea y un hilo a la vez."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import json,time,hashlib,traceback,csv,msvcrt
from argparse import Namespace
import numpy as np
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import engine as e

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'linea2_transferencia_y_zonas_20261028'
OUT=ROOT/'outputs'
PENALTIES=[5]
SEEDS=[30261028 + i*10000000 for i in range(20)]
MAX_RULES=400

def save(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8');os.replace(tmp,path)

def measure(active,core,outside,penalty):
    hc=int((active&core).sum());hu=int((active&outside).sum())
    cr=hc/int(core.sum());ur=hu/int(outside.sum())
    return dict(hit_core=hc,activated_outside=hu,core_recovery=cr,outside_rate=ur,outside_km2=hu*.0001,objective=cr-penalty*ur)

def export(results,current=None):
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    colors={5:'#ff7f00'}
    rows=[]
    for result in results+([current] if current else []):
        t=result['trace'];p=result['penalty'];s=result['seed_index']
        axes[0].plot([0]+[r['outside_km2'] for r in t],[0]+[100*r['core_recovery'] for r in t],color=colors[p],ls=['-','--',':'][(s-1)%3],alpha=.8)
        axes[1].plot([0]+list(range(1,len(t)+1)),[0]+[100*r['core_recovery'] for r in t],color=colors[p],ls=['-','--',':'][(s-1)%3],alpha=.8)
        m=t[-1] if t else dict(core_recovery=0,outside_km2=0)
        rows.append(dict(penalty=p,seed=result['seed'],rules=len(t),core_recovery=m['core_recovery'],outside_km2=m['outside_km2'],status='partial' if result is current else result['reason']))
    axes[0].set(xlabel='Exterior activado (km²)',ylabel='Núcleos recuperados (%)',title='Recuperación frente a extensión exterior')
    axes[1].set(xlabel='Número de reglas',ylabel='Núcleos recuperados (%)',title='Recuperación frente a complejidad')
    from matplotlib.lines import Line2D
    handles=[Line2D([0],[0],color=c,label=f'Penalización {p}') for p,c in colors.items()]
    for ax in axes:ax.grid(alpha=.2)
    fig.legend(handles=handles,loc='lower center',ncol=5);fig.suptitle('Sensibilidad al costo de activación exterior · penalización 5 · veinte semillas')
    fig.tight_layout(rect=(0,.08,1,.94));fig.savefig(OUT/'comparacion.png',dpi=170);plt.close(fig)
    with (OUT/'sintesis.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    table=''.join(f'<tr><td>{r["penalty"]}</td><td>{r["seed"]}</td><td>{r["rules"]}</td><td>{r["core_recovery"]*100:.2f}%</td><td>{r["outside_km2"]:.4f}</td><td>{r["status"]}</td></tr>' for r in rows)
    page=f'''<!doctype html><html lang="es"><meta charset="utf-8"><meta http-equiv="refresh" content="120"><style>body{{font:16px system-ui;max-width:1200px;margin:30px auto}}img{{width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd}}table{{border-collapse:collapse}}</style><h1>Sensibilidad a la penalización exterior</h1><p>Penalización 5; veinte semillas fijas. Misma geometría y búsqueda. Se reutilizan los checkpoints disponibles. Los factores multiplican la proporción exterior, no su área absoluta. Máximo de 400 reglas y 15 km² exteriores por corrida; se detiene al alcanzar primero cualquiera de los límites. No hay filtros espaciales.</p><img src="comparacion.png"><p>Cada línea corresponde a una semilla. Las frecuencias finales indican repetibilidad de selección, no probabilidad de amenaza. Comparar las curvas a una misma extensión exterior. El último punto puede quedar por debajo del presupuesto; no es una comparación a igualdad exacta de área. El beneficio de factores distintos no se compara directamente.</p><table><tr><th>Factor</th><th>Semilla</th><th>Reglas</th><th>Recuperación</th><th>Exterior km²</th><th>Estado</th></tr>{table}</table><p>Esta prueba estudia el compromiso entre recuperación y extensión, no confirma amenaza fuera del registro ni demuestra que exista un factor universalmente correcto.</p><a href="estado.json">Estado</a> · <a href="sintesis.csv">Tabla</a></html>'''
    (OUT/'panel.html').write_text(page,encoding='utf-8')

def main():
    OUT.mkdir(exist_ok=True)
    cfg=json.loads((SOURCE/'seleccion_congelada.json').read_text(encoding='utf-8'))
    protocol=dict(penalties=PENALTIES,baseline=None,seeds=SEEDS,variant='completo_r005',intensity=1,regularization=.05,penalty_definition='new_core/total_core - lambda * new_outside/total_outside',budget_km2=15,search=dict(discovery_sample=20000,clusters=48,random_candidates=16,screen_per_class=10000,finalists=4),stop='last prefix before next best rule exceeds exterior budget; or all cores recovered; or three searches without positive gain',comparison='paired seeds and raw recovery/exterior curves, not objectives across different lambdas',threads=1,engine_sha256=hashlib.sha256((ROOT/'engine.py').read_bytes()).hexdigest())
    protocol['max_rules']=MAX_RULES
    protocol['stop']+='; maximum 400 accepted rules'
    save(OUT/'protocolo.json',protocol)
    results=[]
    with threadpool_limits(limits=1):
        x,row,col,risk,core,shape=e.load_data();outside=~risk
        z=np.memmap(Path(cfg['source'])/'cache/completo_r005.dat',dtype='float32',mode='r',shape=x.shape)
        # First seed across all penalties gives an early overview; then paired replication.
        for i,seed in enumerate(SEEDS,1):
            for penalty in PENALTIES:
                path=OUT/f'factor_{penalty}_semilla_{i}.npz'
                active=np.zeros(len(x),bool);trace=[];finished=False;reason='running'
                if path.exists():
                    with np.load(path) as d:
                        active=np.unpackbits(d['active'])[:len(x)].astype(bool);meta=json.loads(str(d['metadata']));trace=meta['trace'];finished=meta['finished'];reason=meta['reason']
                        assert meta['penalty']==penalty and meta['seed']==seed
                args=Namespace(**protocol['search'],penalty=float(penalty))
                if len(trace)>=MAX_RULES:
                    if len(trace)>MAX_RULES:
                        import shutil
                        archive=OUT/'historico_sin_limite';archive.mkdir(exist_ok=True)
                        backup=archive/path.name
                        if not backup.exists():shutil.copy2(path,backup)
                        trace=trace[:MAX_RULES];active=np.zeros(len(x),bool)
                        for rule in trace:active|=e.full_scores(z,rule['prototype_id'])>=rule['threshold']
                        verified=measure(active,core,outside,penalty)
                        assert verified['hit_core']==trace[-1]['hit_core'] and verified['activated_outside']==trace[-1]['activated_outside']
                    finished=True;reason='rule_limit_400'
                    meta=dict(penalty=penalty,seed=seed,seed_index=i,trace=trace,finished=True,reason=reason)
                    temp=path.with_suffix('.tmp')
                    with temp.open('wb') as f:np.savez_compressed(f,active=np.packbits(active),metadata=json.dumps(meta))
                    os.replace(temp,path)
                while not finished:
                    if (ROOT/'STOP').exists():
                        save(OUT/'estado.json',dict(status='paused',penalty=penalty,seed=seed,rules=len(trace)));return
                    step=len(trace)+1
                    save(OUT/'estado.json',dict(status='running',pid=os.getpid(),penalty=penalty,seed=seed,iteration=step,completed_new_runs=len(results),total_new_runs=20))
                    before=measure(active,core,outside,penalty);rule=None
                    if (core&~active).any():
                        for attempt in range(3):
                            rule,_=e.find_rule(z,core,outside,active,args,seed+step*10+attempt)
                            if rule is not None:break
                    if rule is None:finished=True;reason='no_gain_or_all_core'
                    else:
                        newactive=active|rule['mask'];m=measure(newactive,core,outside,penalty)
                        assert np.isclose(m['objective']-before['objective'],rule['gain'],atol=1e-10)
                        if m['activated_outside']>150000:finished=True;reason='budget'
                        else:
                            active=newactive;trace.append(dict(iteration=step,prototype_id=int(rule['prototype_id']),threshold=float(rule['threshold']),gain=float(rule['gain']),**m))
                            print(f'Factor {penalty}; semilla {i}; regla {step}; recuperacion {m["core_recovery"]:.4f}; exterior {m["outside_km2"]:.4f}',flush=True)
                    if len(trace)>=MAX_RULES:finished=True;reason='rule_limit_400'
                    meta=dict(penalty=penalty,seed=seed,seed_index=i,trace=trace,finished=finished,reason=reason)
                    temp=path.with_suffix('.tmp')
                    with temp.open('wb') as f:np.savez_compressed(f,active=np.packbits(active),metadata=json.dumps(meta))
                    os.replace(temp,path)
                    if step%25==0 or finished:export(results,dict(penalty=penalty,seed=seed,seed_index=i,trace=trace,reason=reason))
                result=dict(penalty=penalty,seed=seed,seed_index=i,trace=trace,reason=reason)
                save(OUT/f'factor_{penalty}_semilla_{i}.json',result);results.append(result);export(results)
    # Preserve spatial votes directly from saved activation masks; no rescoring.
    votes=np.zeros(len(x),dtype=np.uint8)
    finals=[]
    for i,seed in enumerate(SEEDS,1):
        with np.load(OUT/f'factor_5_semilla_{i}.npz') as d:
            active=np.unpackbits(d['active'])[:len(x)].astype(bool)
            meta=json.loads(str(d['metadata']))
        assert meta['finished'] and meta['seed']==seed and meta['penalty']==5
        assert len(meta['trace'])<=MAX_RULES
        m=measure(active,core,outside,5)
        assert m['activated_outside']<=150000
        if meta['trace']:
            assert m['hit_core']==meta['trace'][-1]['hit_core']
            assert m['activated_outside']==meta['trace'][-1]['activated_outside']
        votes += active & outside
        finals.append(dict(seed=seed,rules=len(meta['trace']),**m))
    np.savez_compressed(OUT/'frecuencia_20_semillas.npz',votes=votes,row=row,col=col,risk=risk,shape=np.array(shape),seeds=np.array(SEEDS))
    save(OUT/'resumen_final.json',dict(runs=20,penalty=5,max_rules=400,budget_km2=15,median_recovery=float(np.median([v['core_recovery'] for v in finals])),median_outside_km2=float(np.median([v['outside_km2'] for v in finals])),area_ha_by_exact_votes={str(k):int((votes==k).sum())*.01 for k in range(1,21)},runs_detail=finals))
    save(OUT/'estado.json',dict(status='complete',completed_new_runs=20,baseline_runs=0,frequency_saved=True))

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    lock=(ROOT/'worker.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    try:main()
    except Exception:
        save(OUT/'error.json',dict(traceback=traceback.format_exc()));save(OUT/'estado.json',dict(status='error'));raise
