"""Regional descriptive prototype covering. Original data are always read-only."""
from pathlib import Path
import argparse
import csv
import hashlib
import html
import io
import json
import os
import time
import traceback
import warnings

os.environ.setdefault('LOKY_MAX_CPU_COUNT', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
from scipy.ndimage import convolve
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
OLD = Path(os.environ.get('SATEMB_LINE2_DATA', REPO_ROOT/'data/raw/line2')).resolve()
CACHE = OLD/'nueva_corrida/resultados_linea2_validacion_espacial_folds/_cache'
METHODS = ('normal', 'blanqueado', 'centrado_fondo')
LABELS = {'normal': 'Coseno original', 'blanqueado': 'Coseno blanqueado', 'centrado_fondo': 'Coseno centrado respecto al fondo'}
SEED = 20261028


def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)


def atomic_bytes(path, payload):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def dump(path, value):
    atomic_bytes(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode('utf-8'))


def unit(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def sample(ids, maximum, rng):
    return ids if len(ids) <= maximum else rng.choice(ids, maximum, replace=False)


def load_data():
    cfg = json.loads((CACHE/'config.json').read_text())
    n, shape = cfg['n'], tuple(cfg['shape'])
    x = np.memmap(OLD/'comparacion_alineada/X_unit.dat', dtype='float32', mode='r', shape=(n,64))
    row = np.memmap(CACHE/'row.dat', dtype='int32', mode='r', shape=(n,))
    col = np.memmap(CACHE/'col.dat', dtype='int32', mode='r', shape=(n,))
    risk = np.asarray(np.memmap(CACHE/'risk.dat', dtype='bool', mode='r', shape=(n,)))
    assert np.allclose(np.linalg.norm(x[::997], axis=1), 1, atol=1e-5)
    grid = np.zeros(shape, np.uint8)
    grid[row,col] = risk
    count = convolve(grid, np.ones((3,3), np.uint8), mode='constant')-grid
    core = risk & (count[row,col] >= 5)
    assert core.any() and (~risk).any()
    return x, row, col, risk, core, shape


def prepare_geometry(x, risk):
    """Exact regional moments (chunked); regularized whitening, fixed in all rounds."""
    path = ROOT/'geometria.npz'
    if path.exists():
        with np.load(path) as z:
            return {k:z[k] for k in z.files}
    n, d = x.shape
    sx = np.zeros(d); cross = np.zeros((d,d)); bg = np.zeros(d)
    for start in range(0,n,50000):
        a = np.asarray(x[start:start+50000], dtype=np.float64)
        sx += a.sum(0); cross += a.T@a
        bg += a[~risk[start:start+len(a)]].sum(0)
    mu = sx/n
    cov = (cross-n*np.outer(mu,mu))/(n-1)
    eig, vec = np.linalg.eigh(cov)
    ridge = .05*np.median(eig[eig>0])
    w = ((vec*(1/np.sqrt(np.maximum(eig+ridge,1e-12))))@vec.T).astype(np.float32)
    geom = dict(mu_region=mu.astype(np.float32), mu_fondo=(bg/(~risk).sum()).astype(np.float32), W=w, ridge=np.asarray(ridge), eigenvalues=eig)
    with path.with_suffix('.tmp').open('wb') as f:
        np.savez_compressed(f, **geom)
    os.replace(path.with_suffix('.tmp'),path)
    return geom


def representations(x, geom):
    out = {'normal':x}
    for method in METHODS[1:]:
        path = ROOT/'cache'/f'{method}.dat'
        if not path.exists():
            log('Preparando representación '+method)
            tmp = path.with_suffix('.tmp')
            z = np.memmap(tmp, dtype='float32', mode='w+', shape=x.shape)
            for start in range(0,len(x),50000):
                a = np.asarray(x[start:start+50000])
                a = (a-geom['mu_region'])@geom['W'] if method=='blanqueado' else a-geom['mu_fondo']
                z[start:start+len(a)] = unit(a)
            z.flush(); del z
            os.replace(tmp,path)
        assert path.stat().st_size == x.size*4
        out[method] = np.memmap(path, dtype='float32', mode='r', shape=x.shape)
    return out


def best_threshold(scores, core, outside, active, penalty, ncore=None, noutside=None):
    """Exact best observed threshold; ties must enter together, >= comparator.

    Denominators remain fixed. Passing sample denominators gives a stratified
    screening estimate, never the final regional objective.
    """
    nc = int(core.sum()) if ncore is None else ncore
    nu = int(outside.sum()) if noutside is None else noutside
    order = np.argsort(scores, kind='stable')[::-1]
    ss = scores[order]
    fresh = ~active[order]
    cp = np.cumsum(core[order]&fresh, dtype=np.int64)
    up = np.cumsum(outside[order]&fresh, dtype=np.int64)
    ends = np.r_[ss[:-1]!=ss[1:], True]
    pos = np.flatnonzero(ends)
    gain = cp[pos]/max(1,nc)-penalty*up[pos]/max(1,nu)
    j = int(np.argmax(gain))
    k = int(pos[j])
    return float(gain[j]), float(ss[k]), int(cp[k]), int(up[k])


def candidates(z, residual, rng, args):
    ids = sample(np.flatnonzero(residual), args.discovery_sample, rng)
    # Equal candidate effort in each geometry; representatives are real pixels.
    a = np.asarray(z[ids]); valid = np.linalg.norm(a,axis=1)>1e-8
    ids = ids[valid]; a = a[valid]
    if len(ids) <= args.clusters:
        return ids
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        km = MiniBatchKMeans(n_clusters=args.clusters, batch_size=2048, n_init=1,
                            max_iter=60, random_state=int(rng.integers(2**31-1))).fit(a)
    centers = unit(km.cluster_centers_)
    lab = km.predict(a)
    selected = []
    for j in range(args.clusters):
        members = np.flatnonzero(lab==j)
        if len(members):
            selected.append(ids[members[np.argmax(a[members]@centers[j])]])
    selected.extend(sample(ids,args.random_candidates,rng))
    return np.unique(np.asarray(selected,dtype=np.int64))


def full_scores(z, pid):
    # Fixed block size/matvec path for both optimization and model replay.
    out = np.empty(len(z), np.float32)
    p = np.asarray(z[pid])
    for start in range(0,len(z),50000):
        out[start:start+50000] = z[start:start+50000]@p
    return out


def find_rule(z, core, outside, active, args, seed):
    rng = np.random.default_rng(seed)
    ids = candidates(z,core&~active,rng,args)
    if not len(ids):
        return None, {'candidates':0,'exact_evaluations':0}
    ci = sample(np.flatnonzero(core),args.screen_per_class,rng)
    ui = sample(np.flatnonzero(outside),args.screen_per_class,rng)
    si = np.r_[ci,ui]
    sc = core[si]; su = outside[si]; sa = active[si]
    # Only screening is sampled; every accepted rule is optimized on all pixels.
    vals = np.asarray(z[si])@np.asarray(z[ids]).T
    estimated = [best_threshold(vals[:,j],sc,su,sa,args.penalty)[0] for j in range(len(ids))]
    ranked = np.argsort(-np.asarray(estimated),kind='stable')
    best = None; checked = 0
    for index in ranked:
        pid = int(ids[index])
        scores = full_scores(z,pid)
        gain,t,cp,up = best_threshold(scores,core,outside,active,args.penalty)
        checked += 1
        if gain>1e-12 and (best is None or gain>best['gain']):
            best = dict(prototype_id=pid, threshold=t, gain=gain,
                        new_core=cp, new_outside=up, mask=(scores>=t))
        # If no shortlisted candidate improves, expand to ALL generated candidates.
        if checked>=args.finalists and best is not None:
            break
    return best, {'candidates':len(ids),'exact_evaluations':checked}


def save_checkpoint(method, active, trace, status, reason, z, x):
    metadata = json.dumps(dict(trace=trace,status=status,reason=reason),ensure_ascii=False)
    path = ROOT/'models'/f'{method}.npz'
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        ids=np.asarray([r['prototype_id'] for r in trace],dtype=np.int64)
        np.savez_compressed(f,active_bits=np.packbits(active,bitorder='little'),n=np.asarray(len(active)),metadata=np.asarray(metadata),prototype_ids=ids,raw_prototypes=np.asarray(x[ids]),transformed_prototypes=np.asarray(z[ids]))
    os.replace(tmp,path)


def read_checkpoint(method, n):
    path = ROOT/'models'/f'{method}.npz'
    if not path.exists():
        return dict(active=np.zeros(n,bool),trace=[],status='running',reason='')
    with np.load(path,allow_pickle=False) as z:
        assert int(z['n'])==n
        meta = json.loads(str(z['metadata']))
        meta['active'] = np.unpackbits(z['active_bits'],bitorder='little')[:n].astype(bool)
    return meta


def exports(states, totals, args):
    allrows = []
    status = {}
    for method,s in states.items():
        allrows.extend(s['trace'])
        status[method] = dict(status=s['status'],reason=s['reason'],iterations=len(s['trace']),last=s['trace'][-1] if s['trace'] else None)
    dump(ROOT/'estado.json',dict(updated=time.strftime('%Y-%m-%d %H:%M:%S'),pid=os.getpid(),methods=status,totals=totals))
    if allrows:
        buf = io.StringIO(); writer = csv.DictWriter(buf,fieldnames=list(allrows[0]))
        writer.writeheader(); writer.writerows(allrows)
        atomic_bytes(ROOT/'trayectoria.csv',buf.getvalue().encode('utf-8-sig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(1,2,figsize=(12,4.8))
    for method,s in states.items():
        trace=s['trace']
        ax[0].plot([0]+[v['outside_rate']*100 for v in trace],[0]+[v['core_recovery']*100 for v in trace],'-o',ms=3,label=LABELS[method])
        ax[1].plot([0]+[v['iteration'] for v in trace],[0]+[v['objective'] for v in trace],'-o',ms=3,label=LABELS[method])
        if trace:
            q=trace[-1]; ax[0].annotate(str(q['iteration']),(q['outside_rate']*100,q['core_recovery']*100),xytext=(4,4),textcoords='offset points')
    ax[0].set(xlabel='Territorio sin registro activado (%)',ylabel='Núcleos recuperados (%)',title='Un punto por regla añadida')
    ax[1].set(xlabel='Número de reglas',ylabel='Recuperación − penalización × extensión',title=f'Objetivo acumulado; penalización = {args.penalty:g}')
    for a in ax:
        a.grid(alpha=.2); a.legend(fontsize=8)
    fig.suptitle('Reconstrucción regional · ajuste descriptivo, no ROC de validación')
    fig.tight_layout()
    out=io.BytesIO(); fig.savefig(out,format='png',dpi=140); plt.close(fig)
    atomic_bytes(ROOT/'curvas.png',out.getvalue())
    rows=''
    for method,s in states.items():
        q=s['trace'][-1] if s['trace'] else None
        rows += '<tr><td>'+LABELS[method]+'</td><td>'+str(len(s['trace']))+'</td><td>'+ (f'{100*q["core_recovery"]:.2f}%' if q else '—')+'</td><td>'+ (f'{q["outside_km2"]:.3f} km²' if q else '—')+'</td><td>'+html.escape(s['status']+' '+s['reason'])+'</td></tr>'
    page='''<!doctype html><html lang="es"><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>Recuperación iterativa · Línea 2</title><style>body{font:16px system-ui;max-width:1150px;margin:36px auto;padding:0 20px;color:#213547}table{width:100%;border-collapse:collapse}td,th{padding:12px;text-align:left;border-bottom:1px solid #ddd}img{width:100%}small{color:#526475}</style><h1>Recuperación iterativa de la señal regional</h1><p>Resultados acumulados por regla. Actualización automática cada 30 segundos.</p><p><b>Lectura:</b> recuperar más núcleos con menos activación exterior es mejor. La activación exterior no equivale a falsos positivos confirmados. Estas curvas describen el ajuste en la misma región.</p><table><tr><th>Representación</th><th>Reglas</th><th>Núcleos recuperados</th><th>Exterior activado</th><th>Estado</th></tr>'''+rows+'</table><img src="curvas.png?v='+str(time.time_ns())+'"><p><a href="trayectoria.csv">Registro completo por iteración (CSV)</a> · <a href="protocolo.json">Configuración</a></p><small>Última actualización: '+time.strftime('%Y-%m-%d %H:%M:%S')+'</small></html>'
    atomic_bytes(ROOT/'panel.html',page.encode('utf-8'))


def checks():
    # Compare exact optimization against exhaustive threshold enumeration, including ties.
    rng=np.random.default_rng(18)
    for _ in range(50):
        scores=rng.integers(-4,5,100).astype(np.float32)
        labels=rng.integers(0,3,100); c=labels==0; u=labels==1; a=rng.random(100)<.4
        g,t,cp,up=best_threshold(scores,c,u,a,3)
        brute=[(((scores>=v)&~a&c).sum()/c.sum()-3*((scores>=v)&~a&u).sum()/u.sum(),v) for v in np.unique(scores)]
        assert np.isclose(g,max(q[0] for q in brute))
        assert cp==((scores>=t)&~a&c).sum() and up==((scores>=t)&~a&u).sum()
    # Whole synthetic loop, residual discovery, replay and monotone objective.
    args=argparse.Namespace(discovery_sample=200,clusters=6,random_candidates=3,screen_per_class=200,finalists=9,penalty=3.)
    z=unit(rng.normal(size=(150,8))); core=np.zeros(150,bool); core[:30]=True; outside=np.zeros(150,bool); outside[50:]=True
    active=np.zeros(150,bool); previous=0.; rules=[]
    for step in range(30):
        if not (core&~active).any(): break
        rule,_=find_rule(z,core,outside,active,args,step)
        if rule is None: break
        assert core[rule['prototype_id']] and not active[rule['prototype_id']]
        active|=rule['mask']; rules.append((rule['prototype_id'],rule['threshold']))
        objective=(active&core).sum()/core.sum()-3*(active&outside).sum()/outside.sum()
        assert np.isclose(objective-previous,rule['gain']) and objective>previous
        previous=objective
    replay=np.zeros(150,bool)
    for p,t in rules: replay|=full_scores(z,p)>=t
    assert rules and np.array_equal(replay,active)
    tied=np.zeros(5,np.float32); c=np.array([1,1,0,0,0],bool)
    assert best_threshold(tied,c,~c,np.zeros(5,bool),3)[0]<0
    log('Pruebas correctas: umbral exhaustivo, empates, ganancia marginal, residuales y reproducción del modelo.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--penalty',type=float,default=3.)
    parser.add_argument('--clusters',type=int,default=48)
    parser.add_argument('--random-candidates',type=int,default=16)
    parser.add_argument('--discovery-sample',type=int,default=20000)
    parser.add_argument('--screen-per-class',type=int,default=10000)
    parser.add_argument('--finalists',type=int,default=4)
    parser.add_argument('--threads',type=int,default=1)
    args=parser.parse_args()
    assert args.penalty>=0 and min(args.clusters,args.discovery_sample,args.screen_per_class,args.finalists,args.threads)>0
    if args.check:
        with threadpool_limits(limits=args.threads): checks()
        return
    for name in ('cache','models'): (ROOT/name).mkdir(exist_ok=True)
    lock=ROOT/'run.lock'
    import psutil
    if lock.exists():
        try: oldpid=int(lock.read_text())
        except ValueError: raise RuntimeError('Lock ilegible; inspeccionar antes de reiniciar.')
        if psutil.pid_exists(oldpid): raise RuntimeError(f'Ya existe un proceso con PID {oldpid}; no se inicia otra copia.')
        lock.unlink()
    with lock.open('x') as f: f.write(str(os.getpid()))
    try:
        protocol=dict(algorithm='cobertura_iterativa_factorial_intensidad_regularizacion_v1',settings=vars(args),seed=SEED,
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()+(ROOT/'run_factorial.py').read_bytes()).hexdigest(),
                      source=str(OLD/'comparacion_alineada/X_unit.dat'),cache=str(CACHE),
                      objective='new_core / total_core - penalty * new_unregistered / total_unregistered',
                      threshold='All distinct observed regional scores; ties enter together; >= comparator',
                      stopping='No remaining core, or no positive exact gain among generated candidates in three independent searches; no rule count cap',
                      geometries=EXPERIMENT_GEOMETRIES,
                      evaluation='Descriptive fit on same region; no independent validation or spatial postfilter; fixed denominators',
                      search='Residual candidates refreshed each iteration, approximate stratified screening, exact thresholds for finalists; expand pool evaluation if none improves')
        pp=ROOT/'protocolo.json'
        if pp.exists():
            assert json.loads(pp.read_text(encoding='utf-8'))==protocol,'Configuration/source changed: use a new folder.'
        else: dump(pp,protocol)
        with threadpool_limits(limits=args.threads):
            x,row,col,risk,core,shape=load_data(); outside=~risk
            totals=dict(pixels=len(x),cores=int(core.sum()),registered=int(risk.sum()),unregistered=int(outside.sum()),pixel_area_m2=100,shape=list(shape))
            log('Datos cargados '+json.dumps(totals))
            geom=prepare_geometry(x,risk); zz=representations(x,geom)
            states={m:read_checkpoint(m,len(x)) for m in METHODS}
            exports(states,totals,args)
            while any(s['status']=='running' for s in states.values()):
                if (ROOT/'STOP').exists():
                    log('Pausa solicitada mediante archivo STOP. Se conserva el último checkpoint.'); break
                for mi,method in enumerate(METHODS):
                    if (ROOT/'STOP').exists(): break
                    s=states[method]
                    if s['status']!='running': continue
                    active=s['active']; iteration=len(s['trace'])+1
                    if not (core&~active).any():
                        s['status']='complete'; s['reason']='Todos los núcleos recuperados'
                    else:
                        started=time.time(); log(f'{method}: buscando regla {iteration}')
                        attempts=[]; rule=None
                        for attempt in range(3):
                            rule,detail=find_rule(zz[method],core,outside,active,args,seed_for_method(method,iteration,attempt))
                            attempts.append(detail)
                            if rule is not None: break
                        if rule is None:
                            s['status']='search_exhausted'; s['reason']='Tres búsquedas sin mejora; no garantiza óptimo global'
                            dump(ROOT/f'parada_{method}.json',dict(iteration=iteration,searches=attempts))
                        else:
                            old_objective=s['trace'][-1]['objective'] if s['trace'] else 0.
                            active|=rule.pop('mask')
                            hitc=int((active&core).sum()); hitu=int((active&outside).sum()); hitr=int((active&risk).sum())
                            objective=hitc/totals['cores']-args.penalty*hitu/totals['unregistered']
                            assert np.isclose(objective-old_objective,rule['gain'],atol=1e-10)
                            pid=rule['prototype_id']
                            record=dict(method=method,iteration=iteration,prototype_id=pid,row=int(row[pid]),col=int(col[pid]),threshold=rule['threshold'],gain=rule['gain'],new_core=rule['new_core'],new_outside=rule['new_outside'],hit_core=hitc,hit_registered=hitr,activated_outside=hitu,activated=int(active.sum()),core_recovery=hitc/totals['cores'],record_recovery=hitr/totals['registered'],outside_rate=hitu/totals['unregistered'],outside_km2=hitu*.0001,objective=objective,candidates=sum(t['candidates'] for t in attempts),exact_evaluations=sum(t['exact_evaluations'] for t in attempts),seconds=time.time()-started)
                            s['trace'].append(record)
                            log(f'{method} regla {iteration}: núcleo={record["core_recovery"]:.2%}, exterior={record["outside_km2"]:.4f} km², ganancia={rule["gain"]:.6f}, umbral={rule["threshold"]:.8f}, {record["seconds"]:.1f}s')
                    save_checkpoint(method,active,s['trace'],s['status'],s['reason'],zz[method],x)
                    exports(states,totals,args)
            dump(ROOT/'sesion.json',dict(status='paused' if (ROOT/'STOP').exists() else 'finished',pid=os.getpid(),updated=time.strftime('%Y-%m-%d %H:%M:%S')))
    except Exception:
        dump(ROOT/'error.json',dict(traceback=traceback.format_exc(),time=time.strftime('%Y-%m-%d %H:%M:%S')))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__': main()
