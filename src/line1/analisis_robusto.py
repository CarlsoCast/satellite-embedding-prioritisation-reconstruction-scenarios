"""Análisis condicional de resultados congelados; no entrena ni selecciona modelos.

Intervalos marginales y simultáneos mediante remuestreo pareado de bloques.
Las familias de contrastes se fijan aquí, antes del cálculo de los intervalos.
No es un protocolo prerregistrado ni una prueba confirmatoria independiente.
"""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import experimento as e

METHODS=['directo','alineamiento_general','directo_blanco','alineado_blanco']
METRICS=['ap']+[f'{m}_{b:g}' for b in [.05,.1,.2] for m in ['precision','recall','lift']]
ROOT=e.HOME/'salidas_blanqueamiento'
NAMES=['Directo','Procrustes','Directo blanqueado','Procrustes blanqueado']

def prepared_metrics(y,score,x=None,yy=None):
    """AP con empates y revisión ponderada equivalente a duplicar bloques.

    Los pesos son multiplicidades enteras de bloques. Dentro de empates la
    revisión usa x e y, igual que la corrida original; AP agrupa los empates.
    """
    y=np.asarray(y,dtype=float);score=np.asarray(score,dtype=float)
    if not len(y) or not np.isfinite(score).all() or not np.isin(y,[0,1]).all():
        raise ValueError('Etiquetas binarias y puntajes finitos obligatorios.')
    x=np.arange(len(y)) if x is None else np.asarray(x)
    yy=np.zeros(len(y)) if yy is None else np.asarray(yy)
    order=np.lexsort((yy,x,-score));ys=y[order];ss=score[order]
    ends=np.r_[np.flatnonzero(ss[1:]!=ss[:-1]),len(ss)-1]
    def stat(weights):
        w=np.asarray(weights,dtype=float)[order]
        if not np.isfinite(w).all() or (w<0).any():raise ValueError('Pesos inválidos.')
        tp=np.cumsum(w*ys);total=np.cumsum(w);pos=tp[-1];n=total[-1]
        if pos<=0 or n-pos<=0:return np.full(len(METRICS),np.nan)
        good=total[ends]>0;p=tp[ends][good];t=total[ends][good]
        ap=np.sum(np.diff(np.r_[0,p])*p/t)/pos;result=[ap]
        for budget in [.05,.1,.2]:
            k=min(n,max(1,np.ceil(n*budget)));j=np.searchsorted(total,k,side='left')
            before=total[j-1] if j else 0;hits=(tp[j-1] if j else 0)+(k-before)*ys[j]
            precision=hits/k;result.extend([precision,hits/pos,precision/(pos/n)])
        return np.asarray(result)
    return stat

def align_sources(frames):
    aligned={};reference=None
    for s,z in sorted(frames.items()):
        if z[['x','y']].duplicated().any():raise ValueError('Coordenadas duplicadas.')
        z=z.sort_values(['x','y']).reset_index(drop=True)
        if reference is None:reference=z
        else:
            for col in ['x','y','fold','label']:
                if not np.array_equal(z[col],reference[col]):raise ValueError('Soportes o etiquetas diferentes entre escalas.')
        if not np.isfinite(z[METHODS]).all().all():raise ValueError('Puntajes incompletos.')
        aligned[s]=z
    return aligned

def contrast_family(scales):
    # Comparaciones dentro de escala y contraste espacial frente al píxel.
    family=[]
    for s in scales:
        for a,b in [(1,0),(3,1),(2,0)]:family.append((s,a,s,b))
    for s in scales:
        if s!=10:family.append((s,1,10,1))
    return family

def simultaneous_ci(observed,boot,confidence=.95):
    """Banda bootstrap max t centrada, conjunta para AP de cada familia."""
    if not 0<confidence<1:raise ValueError('Nivel de confianza fuera de (0,1).')
    observed=np.asarray(observed);boot=np.asarray(boot)
    if len(boot)<100:raise ValueError('Menos de cien réplicas válidas para la familia.')
    sd=np.std(boot,axis=0,ddof=1);active=sd>1e-12
    q=float(np.quantile(np.max(np.abs((boot[:,active]-observed[active])/sd[active]),axis=1),confidence)) if active.any() else 0.
    tail=(1-confidence)/2
    marginal=np.quantile(boot,[tail,1-tail],axis=0)
    return marginal,observed-q*sd,observed+q*sd,q

def analyze_family(frames,c,reps,seed,bootstrap_path=None):
    frames=align_sources(frames);scales=list(frames);family=contrast_family(scales)
    groups=[];fold_rows=[];base=frames[scales[0]]
    for fold in sorted(base.fold.unique()):
        ix=(base.fold==fold)&(base.label>=0);z=base.loc[ix]
        if z.label.nunique()!=2:continue
        blocks=np.column_stack([np.floor(z.x/c['block_m']),np.floor(z.y/c['block_m'])])
        _,inv=np.unique(blocks,axis=0,return_inverse=True);nb=int(inv.max()+1)
        funcs=[]
        for s in scales:
            local=frames[s].loc[ix]
            fs=[prepared_metrics(z.label,local[m],z.x,z.y) for m in METHODS];funcs.append(fs)
            for m,f in zip(METHODS,fs):
                values=f(np.ones(len(z)));prevalence=float(z.label.mean())
                fold_rows.append(dict(source_m=s,fold=int(fold),method=m,n_units=int((base.fold==fold).sum()),n_labelled=len(z),n_positive=int(z.label.sum()),n_blocks=nb,prevalence=prevalence,normalized_ap=(values[0]-prevalence)/(1-prevalence),**dict(zip(METRICS,values))))
        groups.append((int(fold),funcs,inv,nb))
    if len(groups)<2:raise ValueError('Insuficiente soporte espacial.')
    def evaluate_weights(group,weights):return np.array([[f(weights) for f in fs] for fs in group[1]])
    original=np.array([evaluate_weights(g,np.ones(len(g[2]))) for g in groups])
    observed=original.mean(0);index={s:i for i,s in enumerate(scales)}
    def diffs(arr):return np.array([arr[index[s],a]-arr[index[t],b] for s,a,t,b in family])
    obs_diff=diffs(observed);rng=np.random.default_rng(seed);boot=[];skipped=0;attempts=[]
    for _ in range(reps):
        parts=[evaluate_weights(g,rng.multinomial(g[3],np.full(g[3],1/g[3]))[g[2]]) for g in groups]
        arr=np.array(parts)
        # Mantener exactamente los mismos grupos en todas las réplicas válidas.
        if not np.isfinite(arr).all():skipped+=1;continue
        boot.append(diffs(arr.mean(0)));attempts.append(_)
    boot=np.asarray(boot);marg,lo,hi,q=simultaneous_ci(obs_diff[:,0],boot[:,:,0])
    contrast_rows=[];budget_rows=[];jackknife=[]
    for k,(s,a,t,b) in enumerate(family):
        fd=np.array([diffs(v)[k,0] for v in original]);leave=[float(np.delete(fd,i).mean()) for i in range(len(fd))]
        evidence='ganancia' if lo[k]>0 else ('deterioro' if hi[k]<0 else 'sin separación de cero')
        contrast_rows.append(dict(source_a_m=s,method_a=METHODS[a],source_b_m=t,method_b=METHODS[b],ap_difference=obs_diff[k,0],ci_low=marg[0,k],ci_high=marg[1,k],simultaneous_low=lo[k],simultaneous_high=hi[k],family_size=len(family),max_t_quantile=q,valid_replicates=len(boot),invalid_replicates=skipped,n_groups=len(groups),positive_groups=int((fd>0).sum()),negative_groups=int((fd<0).sum()),leave_one_group_min=min(leave),leave_one_group_max=max(leave),conditional_evidence=evidence))
        for j,metric in enumerate(METRICS[1:],1):
            low,high=np.quantile(boot[:,k,j],[.025,.975])
            budget_rows.append(dict(source_a_m=s,method_a=METHODS[a],source_b_m=t,method_b=METHODS[b],metric=metric,difference=obs_diff[k,j],ci_low=low,ci_high=high,interval='marginal exploratorio sin ajuste múltiple',valid_replicates=len(boot)))
    # Diagnóstico de influencia espacial, sin volver a ajustar ni recalibrar.
    for gi,g in enumerate(groups):
        for block in range(g[3]):
            changed=evaluate_weights(g,(g[2]!=block).astype(float))
            if not np.isfinite(changed).all():continue
            arr=original.copy();arr[gi]=changed;delta=diffs(arr.mean(0))
            for k,(s,a,t,b) in enumerate(family):jackknife.append(dict(excluded_group=g[0],excluded_block_within_group=block,source_a_m=s,method_a=METHODS[a],source_b_m=t,method_b=METHODS[b],ap_difference=float(delta[k,0])))
    if bootstrap_path is not None:
        np.savez_compressed(bootstrap_path,observed=obs_diff[:,0],bootstrap=boot[:,:,0],family=np.asarray(family,dtype=str),attempt_indices=np.asarray(attempts),attempts=reps,seed=seed,n_groups=len(groups))
    return pd.DataFrame(fold_rows),pd.DataFrame(contrast_rows),pd.DataFrame(budget_rows),pd.DataFrame(jackknife)

def descriptive(frames,prior,c):
    distributions=[];relationships=[];rank=[];curves=[]
    for s,z in frames.items():
        # Diagnósticos por grupo: los ángulos de operadores distintos no se mezclan.
        original=pd.read_parquet(prior/f'evaluacion/{s}/predicciones_comun_{int(c["target"])}.parquet')
        cols=['dw_js','delta_built','delta_vegetation','delta_bare']
        features=original[['x','y']+cols]
        data=z.merge(features,on=['x','y'],validate='one_to_one')
        for fold,g in data.groupby('fold'):
            for m in METHODS:
                for label,q in g.groupby('label'):
                    values=q[m].quantile([.1,.25,.5,.75,.9]).to_numpy()
                    distributions.append(dict(source_m=s,fold=int(fold),method=m,label=int(label),n=len(q),mean=q[m].mean(),sd=q[m].std(),**dict(zip(['p10','p25','median','p75','p90'],values))))
                for feature in cols:
                    values=g[feature] if feature=='dw_js' else g[feature].abs()
                    relationships.append(dict(source_m=s,fold=int(fold),method=m,feature=feature,n=len(g),spearman=g[m].corr(values,method='spearman')))
                labelled=g[g.label>=0]
                if labelled.label.nunique()==2:
                    for budget in [.01,.02,.05,.1,.2,.3,.5,1.]:
                        order=labelled.sort_values([m,'x','y'],ascending=[False,True,True]);k=max(1,int(np.ceil(budget*len(order))));hits=order.head(k).label.sum()
                        curves.append(dict(source_m=s,fold=int(fold),method=m,budget=budget,n_labelled=len(order),reviewed=k,precision=hits/k,recall=hits/order.label.sum(),lift=(hits/k)/order.label.mean()))
            for a,b in [(1,0),(3,1),(2,0)]:
                k=max(1,int(np.ceil(.1*len(g))))
                sets=[set(g.sort_values([METHODS[j],'x','y'],ascending=[False,True,True]).head(k).index) for j in [a,b]]
                rank.append(dict(source_m=s,fold=int(fold),method_a=METHODS[a],method_b=METHODS[b],n_all_units=len(g),spearman=g[METHODS[a]].corr(g[METHODS[b]],method='spearman'),top_fraction=.1,jaccard=len(sets[0]&sets[1])/len(sets[0]|sets[1]),scope='todas las unidades incluida etiqueta intermedia'))
    return [pd.DataFrame(x) for x in [distributions,relationships,rank,curves]]

def model_diagnostics():
    rows=[]
    for p in sorted((ROOT/'modelos').rglob('*.npz')):
        m=dict(np.load(p));v=m['eigenvalues'];ridge=float(m['ridge']);trace=np.maximum(v,0).sum()
        shares=np.maximum(v,0)/trace;nonzero=shares>0
        rows.append(dict(source_m=int(p.parent.name),fold=int(p.stem.split('_')[1]),n_train=int(m['n_train']),ridge=ridge,eigen_min=float(v.min()),eigen_max=float(v.max()),regularized_condition=(v.max()+ridge)/(v.min()+ridge),whitening_max_amplification=float(1/np.sqrt(v.min()+ridge)),effective_rank=float(np.exp(-(shares[nonzero]*np.log(shares[nonzero])).sum())),R_determinant=float(np.linalg.det(m['R'])),R_orthogonality_error=float(np.linalg.norm(m['R'].T@m['R']-np.eye(64)))))
    return pd.DataFrame(rows)

def figures(dest,means,contrasts,curves):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    colors=['#6E7684','#086A78','#B18A52','#99495C']
    primary=means[(means.reference=='3observaciones')&(means.target_m==1000)]
    fig,ax=plt.subplots(figsize=(7.5,3.8))
    scales=sorted(primary.source_m.unique());x=np.arange(len(scales))
    for m,label,color in zip(METHODS,NAMES,colors):
        z=primary[primary.method==m].groupby('source_m').ap.mean().reindex(scales)
        ax.plot(x,z,marker='o',color=color,label=label)
    ax.set(xticks=x,xticklabels=scales,xlabel='Escala fuente (m)',ylabel='AP media entre grupos',ylim=(.35,.85),title='Referencia anual · celdas comunes de 1 km')
    ax.grid(axis='y',alpha=.2);ax.legend(frameon=False,ncol=2,fontsize=9);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(dest/f'01_ap_por_escala.{ext}',dpi=240)
    plt.close(fig)
    con=contrasts[(contrasts.reference=='3observaciones')&(contrasts.target_m==500)&(contrasts.source_a_m==contrasts.source_b_m)]
    fig,axs=plt.subplots(1,2,figsize=(9.5,3.7),sharey=True)
    for ax,aa,bb,title in [(axs[0],'alineamiento_general','directo','Procrustes menos directo'),(axs[1],'alineado_blanco','alineamiento_general','Procrustes blanqueado menos Procrustes')]:
        z=con[(con.method_a==aa)&(con.method_b==bb)].sort_values('source_a_m');y=np.arange(len(z))
        ax.errorbar(z.ap_difference,y,xerr=np.vstack([z.ap_difference-z.simultaneous_low,z.simultaneous_high-z.ap_difference]),fmt='o',color='#086A78',capsize=4)
        ax.axvline(0,color='#666',ls='--',lw=1);ax.set_yticks(y,[f'{s} m' for s in z.source_a_m]);ax.set_xlabel('Diferencia de AP');ax.set_title(title,fontsize=11);ax.grid(axis='x',alpha=.2)
    axs[0].set_ylabel('Escala fuente');fig.suptitle('Celdas comunes de 500 m · intervalos simultáneos por familia',fontsize=11);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(dest/f'02_contrastes_simultaneos.{ext}',dpi=240)
    plt.close(fig)
    z=curves[(curves.reference=='3observaciones')&(curves.target_m==500)&(curves.source_m==10)]
    fig,axs=plt.subplots(1,2,figsize=(9,3.6))
    for method,label,color in zip(METHODS,NAMES,colors):
        a=z[z.method==method].groupby('budget')[['precision','recall']].mean()
        for ax,metric in zip(axs,['precision','recall']):ax.plot(a.index*100,a[metric],marker='o',label=label,color=color,markersize=3)
    for ax,lab in zip(axs,['Precisión auxiliar','Recuperación auxiliar']):ax.set(xlabel='Subconjunto etiquetado revisado (%)',ylabel=lab,ylim=(0,1.03));ax.grid(alpha=.2)
    axs[0].legend(frameon=False,fontsize=8);fig.suptitle('Cambio desde 10 m · celdas comunes de 500 m · referencia anual',fontsize=11);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(dest/f'03_utilidad_revision.{ext}',dpi=240)
    plt.close(fig)

def verify_point_metrics(group):
    checks=[]
    for (tag,target),part in group.groupby(['reference','target_m']):
        prior=pd.read_csv(ROOT/tag/'03_metricas_por_grupo.csv')
        prior=prior[(prior.scale_m==target)&prior.ap.notna()].rename(columns={'source_scale_m':'source_m'})
        n=0
        for budget in [.05,.1,.2]:
            a=prior[np.isclose(prior.budget,budget)]
            joined=part.merge(a,on=['source_m','fold','method'],suffixes=('_new','_old'),validate='one_to_one')
            if len(joined)!=len(a) or not np.allclose(joined.ap_new,joined.ap_old):raise ValueError('No se reproducen las AP previas.')
            for metric in ['precision','recall','lift']:
                if not np.allclose(joined[f'{metric}_{budget:g}'],joined[metric]):raise ValueError('No se reproduce el presupuesto previo.')
            n+=len(joined)
        checks.append(dict(reference=tag,target_m=int(target),identical_metric_rows=n))
    return checks

def run(reps=2000):
    dest=ROOT/'analisis_robusto';dest.mkdir(exist_ok=True)
    e.dump(dest/'estado.json',dict(status='ejecutando',replicates=reps))
    try:_run(reps)
    except Exception as exc:
        e.dump(dest/'estado.json',dict(status='fallido',error=str(exc),replicates=reps))
        raise

def _run(reps=2000):
    if reps<200:raise ValueError('Utilizar al menos 200 réplicas; recomendadas 2.000.')
    c=e.config(e.HOME/'config.json');dest=ROOT/'analisis_robusto';dest.mkdir(exist_ok=True)
    frozen=json.loads((ROOT/'puntajes_congelados.json').read_text(encoding='utf8'))
    for p,digest in frozen['files'].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=digest:raise ValueError('Entrada congelada modificada: '+p)
    audit={};folds=[];contrasts=[];budgets=[];influences=[];distributions=[];relationships=[];ranks=[];curves=[]
    for tag,prior in [('3observaciones',Path(c['out'])/'comparacion_comun/3observaciones'),('huella_manuscrito',Path(c['out'])/'sensibilidad_referencia/huella_manuscrito'),('estricta_4meses',Path(c['out'])/'comparacion_comun/estricta_4meses')]:
        for target in [500,1000]:
            frames={}
            for s in [v for v in c['scales_m'] if v<=target]:
                p=ROOT/tag/f'evaluacion/{s}/predicciones_comun_{target}.parquet'
                if p.exists():
                    frames[s]=pd.read_parquet(p);audit[str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
                    original=prior/f'evaluacion/{s}/predicciones_comun_{target}.parquet';audit[str(original.resolve())]=hashlib.sha256(original.read_bytes()).hexdigest()
            if not frames:continue
            frames=align_sources(frames)
            rr=analyze_family(frames,c,reps,c['seed'])
            for collection,result in zip([folds,contrasts,budgets,influences],rr):collection.append(result.assign(reference=tag,target_m=target))
            extra=descriptive(frames,prior,{**c,'target':target})
            for collection,result in zip([distributions,relationships,ranks,curves],extra):collection.append(result.assign(reference=tag,target_m=target))
            print('Análisis terminado',tag,target,'m',flush=True)
    tables={name:pd.concat(items,ignore_index=True) for name,items in [('01_metricas_por_grupo',folds),('02_contrastes_ap',contrasts),('03_contrastes_presupuestos',budgets),('04_influencia_bloques',influences),('05_distribuciones_puntajes',distributions),('06_asociacion_componentes_dw',relationships),('07_estabilidad_rankings',ranks),('08_curvas_revision',curves)]}
    group=tables['01_metricas_por_grupo'];summary=group.groupby(['reference','target_m','source_m','method'],as_index=False).agg(ap_mean=('ap','mean'),ap_sd=('ap','std'),ap_min=('ap','min'),ap_max=('ap','max'),n_groups=('fold','nunique'),prevalence_mean=('prevalence','mean'),normalized_ap_mean=('normalized_ap','mean'),n_units=('n_units','sum'),n_labelled=('n_labelled','sum'),n_positive=('n_positive','sum'),n_blocks=('n_blocks','sum'))
    tables['09_soporte_y_heterogeneidad']=summary;tables['10_geometria_modelos']=model_diagnostics()
    for name,table in tables.items():table.to_csv(dest/f'{name}.csv',index=False,encoding='utf-8-sig')
    e.dump(dest/'verificacion_metricas.json',verify_point_metrics(group))
    figures(dest,group,tables['02_contrastes_ap'],tables['08_curvas_revision'])
    e.dump(dest/'protocolo_y_auditoria.json',dict(replicates=reps,seed=c['seed'],input_hashes=audit,frozen_model_score_hashes=frozen['files'],families='Una familia por referencia y soporte objetivo: Procrustes/directo, Procrustes blanqueado/Procrustes, directo blanqueado/directo en cada fuente, más Procrustes de ventanas/Procrustes de píxeles.',simultaneous='Bootstrap max t centrado para diferencias de AP, aproximación condicional; presupuestos con intervalos marginales.',resampling='Bloques de 5 km dentro de cada grupo espacial; mismos pesos en todas las variantes y escalas. Réplicas sin ambas clases en cualquier grupo se excluyen y se cuentan.',scope=['Exploratorio y posterior a observar resultados; no confirmatorio ni prerregistrado.','Operadores, puntajes, máscaras y calibraciones fijos; no incluye incertidumbre del ajuste o productos.','Grupos comparten entrenamiento; no son réplicas independientes de entrenamiento.','Sensibilidades geográficas y temporales son dependientes y usan poblaciones distintas.','La AP excluye etiquetas intermedias; el diagnóstico de rankings incluye todas las unidades.','Los intervalos de presupuestos no tienen ajuste por multiplicidad.','La correlación con componentes DW es descriptiva; no asigna causas o tipos físicos de cambio.','No se probó equivalencia entre 10 y 100 m: intervalo con cero no demuestra igualdad.']))
    # Comprobar que el análisis no alteró ninguna predicción de entrada.
    for p,digest in {**audit,**frozen['files']}.items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=digest:raise ValueError('El análisis alteró una entrada: '+p)
    write_interpretation(dest,tables)
    import scipy,sklearn
    e.dump(dest/'entorno_y_codigo.json',dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,code_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),e.HOME/'comparar_blanqueamiento.py',e.HOME/'test_analisis_robusto.py']},configuration=c))
    e.dump(dest/'estado.json',dict(status='completo',replicates=reps,tables=len(tables),input_hashes_verified_before_and_after=True))
    print('Análisis robusto completo',dest,flush=True)

def write_interpretation(dest,tables):
    d=tables['02_contrastes_ap'];lines=['# Guía para interpretar los resultados','',
    'Este análisis cuantifica incertidumbre espacial condicional y utilidad de priorización. No confirma el fenómeno en terreno ni transforma la referencia auxiliar en verdad independiente. Se fijaron las familias de contrastes en el código, pero el análisis se construyó después de conocer los resultados; sus intervalos son exploratorios.','',
    '## Lectura de la evidencia','',
    'Una diferencia de AP es una mejora del ordenamiento, no una reducción del error físico ni un porcentaje adicional de superficie correctamente detectada. Se presentan intervalos marginales y bandas simultáneas bootstrap max t para todos los contrastes de AP dentro de cada combinación de referencia y soporte. Las bandas no cubren conjuntamente todas las referencias del estudio. Un intervalo que incluye cero indica que la comparación no separa los métodos con este diseño; no demuestra equivalencia ni ausencia de efecto. No se fabrican valores p a partir de la proporción de réplicas positivas.','']
    for tag in ['3observaciones','huella_manuscrito','estricta_4meses']:
        lines.extend(['## '+tag,''])
        for _,r in d[(d.reference==tag)&(d.source_a_m==10)&(d.source_b_m==10)].iterrows():
            lines.append(f"- Soporte {r.target_m} m, {r.method_a} menos {r.method_b}: ΔAP {r.ap_difference:+.4f}; intervalo simultáneo [{r.simultaneous_low:+.4f}; {r.simultaneous_high:+.4f}]. Evidencia condicional: {r.conditional_evidence}. {r.positive_groups}/{r.n_groups} grupos con ganancia, {r.valid_replicates} réplicas válidas.")
        lines.append('')
    lines.extend(['## Qué permite explicar cada salida','',
    '- 01 y 09: distribución entre grupos, prevalencia, soporte, AP normalizada respecto de la prevalencia y heterogeneidad. La AP media es macro por grupo; no es una AP global combinada.',
    '- 02: magnitud de las diferencias, intervalos marginales y simultáneos, signos por grupo y sensibilidad a retirar un grupo. La familia incluye las comparaciones entre escalas frente al píxel.',
    '- 03 y 08: precisión, recuperación y lift para presupuestos de revisión; el cálculo reproduce el redondeo y los desempates originales. Los intervalos del presupuesto son marginales y exploratorios.',
    '- 04: influencia de retirar bloques de evaluación, sin reajustar modelos. Examinar si el signo depende de pocas zonas, sin tratar este diagnóstico como nueva validación.',
    '- 05: distribución de puntajes para etiquetas baja (0), intermedia (−1) y alta (1). Las magnitudes de ángulos originales y blanqueados pertenecen a geometrías distintas; no compararlas como severidad física.',
    '- 06: relación por grupo con Jensen–Shannon y variaciones absolutas de construido, vegetación y suelo desnudo. Son componentes de la misma referencia de evaluación, no verificaciones independientes.',
    '- 07: correlación de rankings y coincidencia de su 10 % superior sobre todas las unidades. No confundir este presupuesto con los presupuestos calculados solo sobre el subconjunto etiquetado.',
    '- 10: espectro, regularización, rango efectivo y amplificación máxima de la transformación. Una amplificación alta puede motivar una hipótesis de sensibilidad al ruido, pero no la demuestra.', '',
    '## Conclusión que debe sostenerse','',
    'La recomendación de Procrustes sin blanqueamiento es provisional y depende del contraste auxiliar de mayor cobertura. Las pruebas estrictas tienen soporte reducido. No debe presentarse una mejora universal ni una equivalencia demostrada de 10 y 100 m. La utilidad aplicada se expresa como concentración de candidatos bajo un presupuesto de revisión. Los agrupamientos espaciales de candidatos y cualquier validación primaria siguen siendo decisiones posteriores.', '',
    '## Controles aún pendientes','',
    'Quedan pendientes la sensibilidad con máscaras de subpíxeles exactamente coincidentes, una comparación estacional sobre poblaciones emparejadas, la incertidumbre de reajustar operadores y la evaluación reservada de la configuración seleccionada. Estos controles no se sustituyen con más remuestreos de las predicciones actuales. No se cambió la geometría, regularización ni selección de modelos para optimizar los resultados de este análisis.'])
    (dest/'INTERPRETACION.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--replicas',type=int,default=2000)
    run(parser.parse_args().replicas)
