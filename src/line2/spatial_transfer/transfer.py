"""Conditional N/S transfer; region-selected hyperparameters, train-only fitting."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
import json,time,argparse,hashlib,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import engine as e
ROOT=Path(__file__).resolve().parent

def save(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8');os.replace(temp,path)

def metrics(active,risk,core):
    ncore=int(core.sum());nu=int((~risk).sum());nr=int(risk.sum())
    hc=int((active&core).sum());hr=int((active&risk).sum());hu=int((active&~risk).sum())
    return dict(pixels=len(risk),registered=nr,cores=ncore,unregistered=nu,hit_core=hc,hit_registered=hr,outside_pixels=hu,core_recovery=hc/ncore,record_recovery=hr/nr,outside_rate=hu/nu,outside_km2=hu*.0001,objective=hc/ncore-3*hu/nu)

def balanced_split(row,risk,core):
    maxrow=int(row.max());counts=np.bincount(row[risk],minlength=maxrow+1);cum=np.r_[0,np.cumsum(counts)]
    candidates=np.arange(51,maxrow-50)
    south=cum[candidates-50];north=cum[-1]-cum[candidates+50]
    valid=(south>counts.sum()*.2)&(north>counts.sum()*.2)
    candidates=candidates[valid];south=south[valid];north=north[valid]
    # Label-count balance after 1 km gap; tie: retain more positives.
    order=np.lexsort((-(south+north),np.abs(south-north)))
    cut=int(candidates[order[0]])
    sm=row<cut-50;nm=row>=cut+50;gap=~(sm|nm)
    assert not (sm&nm).any()
    summary=dict(cut_row=cut,gap_rows=100,gap_width_m=1000,rule='minimize absolute difference in registered pixel counts after symmetric 1km exclusion; tie retain more positives',south=metrics(np.zeros(sm.sum(),bool),risk[sm],core[sm]),north=metrics(np.zeros(nm.sum(),bool),risk[nm],core[nm]),excluded_pixels=int(gap.sum()),excluded_positive=int((gap&risk).sum()),excluded_core=int((gap&core).sum()))
    return sm,nm,gap,summary

def prepare_z(x,trainids,name):
    modelpath=ROOT/'transferencia'/f'{name}_geometria.npz'
    if modelpath.exists():
        with np.load(modelpath) as d:mu=d['mu'];w=d['W']
    else:
        s=np.zeros(64);cross=np.zeros((64,64));n=len(trainids)
        for st in range(0,n,50000):
            a=np.asarray(x[trainids[st:st+50000]],dtype=np.float64);s+=a.sum(0);cross+=a.T@a
        mu=s/n;cov=(cross-n*np.outer(mu,mu))/(n-1);eig,v=np.linalg.eigh(cov);ridge=.05*np.median(eig[eig>0])
        w=((v/np.sqrt(np.maximum(eig+ridge,1e-12)))@v.T).astype(np.float32);mu=mu.astype(np.float32)
        np.savez_compressed(modelpath,mu=mu,W=w,cov=cov,ridge=ridge,train_ids=trainids)
    p=ROOT/'cache'/f'{name}_train.dat'
    if not p.exists():
        z=np.memmap(p,dtype='float32',mode='w+',shape=(len(trainids),64))
        for st in range(0,len(trainids),50000):z[st:st+50000]=e.unit((x[trainids[st:st+50000]]-mu)@w)
        z.flush();del z
    return np.memmap(p,dtype='float32',mode='r',shape=(len(trainids),64)),mu,w

def test_score(x,ids,mu,w,p):
    s=np.empty(len(ids),np.float32)
    for st in range(0,len(ids),50000):s[st:st+50000]=e.unit((x[ids[st:st+50000]]-mu)@w)@p
    return s

def main():
    config=json.loads((ROOT/'seleccion_congelada.json').read_text(encoding='utf-8'))
    args=argparse.Namespace(discovery_sample=20000,clusters=48,random_candidates=16,screen_per_class=10000,finalists=4,penalty=3.)
    with threadpool_limits(limits=1):
        x,row,col,risk,core,shape=e.load_data();sm,nm,gap,split=balanced_split(row,risk,core)
        geo=np.load(e.OLD/'revision_integral/10_mapa_categorias.npz');y0=float(geo['y0'])
        split['cut_northing_epsg9377']=y0+(split['cut_row']-.5)*10
        save(ROOT/'transferencia/division.json',split)
        np.savez_compressed(ROOT/'transferencia/division_masks.npz',south=np.packbits(sm),north=np.packbits(nm),gap=np.packbits(gap),n=len(x),bitorder='big')
        budget_rate=config['budget_km2']/(int((~risk).sum())*.0001)
        protocol=dict(selection=config,split=split,train_budget_rate=budget_rate,stopping='last prefix before next selected rule would exceed train exterior rate; otherwise no gain or full recovery; no test-based stopping',fit='mean/covariance/prototypes/thresholds train only; one km excluded from both sides',seeds=config['seeds'],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        save(ROOT/'transferencia/protocolo.json',protocol)
        finals=[]
        for name,train,test in [('Sur_a_Norte',sm,nm),('Norte_a_Sur',nm,sm)]:
            ti=np.flatnonzero(train);vi=np.flatnonzero(test);z,mu,w=prepare_z(x,ti,name)
            for rep,seed in enumerate(config['seeds'],1):
                tag=f'{name}__s{rep}';out=ROOT/'transferencia'/f'{tag}.json'
                if out.exists():finals.append(json.loads(out.read_text(encoding='utf-8')));continue
                e.log('INICIO '+tag);active=np.zeros(len(ti),bool);va=np.zeros(len(vi),bool);trace=[];protos=[];step=0
                while True:
                    step+=1;rule=None
                    if not (core[ti]&~active).any():reason='all_core';break
                    for attempt in range(3):
                        rule,detail=e.find_rule(z,core[ti],~risk[ti],active,args,seed+step*10+attempt)
                        if rule is not None:break
                    if rule is None:reason='no_gain_in_three_searches';break
                    newactive=active|rule['mask'];newmet=metrics(newactive,risk[ti],core[ti])
                    if newmet['outside_rate']>budget_rate:
                        reason='train_budget';break
                    active=newactive;pid=rule['prototype_id'];proto=np.asarray(z[pid]).copy();protos.append(proto)
                    va|=test_score(x,vi,mu,w,proto)>=rule['threshold'];vm=metrics(va,risk[vi],core[vi])
                    trace.append(dict(iteration=step,prototype_id=int(ti[pid]),threshold=rule['threshold'],gain=rule['gain'],train=newmet,test=vm))
                    save(ROOT/'transferencia/estado.json',dict(status='running',current=tag,iteration=step,completed=len(finals),expected=6,train=newmet,test=vm))
                    e.log(f'{tag} {step}: train core={newmet["core_recovery"]:.3f}; test core={vm["core_recovery"]:.3f}')
                result=dict(job=tag,direction=name,seed=seed,rules=len(trace),stop_reason=reason,train=metrics(active,risk[ti],core[ti]),test=metrics(va,risk[vi],core[vi]),trace=trace)
                np.savez_compressed(ROOT/'transferencia'/f'{tag}_modelo.npz',prototypes=np.asarray(protos),thresholds=np.asarray([r['threshold'] for r in trace]),prototype_ids=np.asarray([r['prototype_id'] for r in trace]),mu=mu,W=w,train_activation=np.packbits(active),test_activation=np.packbits(va),train_ids=ti,test_ids=vi)
                save(out,result);finals.append(result)
                pd.DataFrame([dict(direction=r['direction'],seed=r['seed'],rules=r['rules'],stop_reason=r['stop_reason'],**r['test']) for r in finals]).to_csv(ROOT/'outputs/transferencia_resultados.csv',index=False)
            del z
        save(ROOT/'transferencia/estado.json',dict(status='complete',completed=6,expected=6))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(11,4.5))
        for name,ax in zip(['Sur_a_Norte','Norte_a_Sur'],axes):
            for result in finals:
                if result['direction']!=name:continue
                t=result['trace'];ax.plot([0]+[r['test']['outside_rate']*100 for r in t],[0]+[r['test']['core_recovery']*100 for r in t],marker='.',label=f'Semilla {config["seeds"].index(result["seed"])+1}')
            ax.set(title=name.replace('_',' '),xlabel='Exterior activado en evaluación (%)',ylabel='Núcleos recuperados en evaluación (%)');ax.grid(alpha=.2);ax.legend()
        fig.suptitle('Transferencia territorial condicionada a selección regional previa');fig.tight_layout();fig.savefig(ROOT/'outputs/transferencia_curvas.png',dpi=180);plt.close(fig)
        e.log('COMPLETADAS seis pruebas de transferencia')
if __name__=='__main__':
    try:main()
    except Exception:
        save(ROOT/'transferencia/error.json',dict(traceback=traceback.format_exc()));raise
