"""Valida los embeddings de escenario antes de cualquier aplicación ambiental.

Reconstruye MSE y RMSE directamente de los vectores de evaluación guardados;
verifica sus ángulos y calcula intervalos pareados por bloques. No entrena ni
selecciona modelos nuevos y no utiliza NDMI, NDVI u otro objetivo ambiental.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generar_validacion_transformaciones(carpeta_linea3=None,replicates=2000):
    home=Path(carpeta_linea3) if carpeta_linea3 is not None else Path(__file__).resolve().parent
    source=home/'nueva_corrida_integral/principal'
    output=home/'validacion_transformaciones';output.mkdir(exist_ok=True)
    metadata=source/'02_embedding_oof.parquet';vectors=source/'06_vectores_oof.npz'
    data=pd.read_parquet(metadata);z=np.load(vectors)
    assert np.array_equal(np.sort(z['row']),np.arange(len(data)))
    d=data.iloc[z['row']].reset_index(drop=True).copy()
    initial,observed,scenario=(z[k] for k in ['initial','observed','scenario'])
    for array in [initial,observed,scenario]:
        assert array.shape==(len(d),64) and np.isfinite(array).all()
        assert np.allclose(np.linalg.norm(array,axis=1),1,atol=1e-9)
    # Cada error es el promedio de los 64 errores de componente al cuadrado.
    d['mse_inicial']=np.mean((initial-observed)**2,axis=1)
    d['mse_escenario']=np.mean((scenario-observed)**2,axis=1)
    angle=lambda a:np.degrees(np.arccos(np.clip(np.sum(a*observed,axis=1),-1,1)))
    assert np.allclose(angle(scenario),d.angular_error_deg,atol=1e-6)
    assert np.allclose(angle(initial),d.identity_error_deg,atol=1e-6)
    # En vectores unitarios, MSE y ángulo describen la misma discrepancia.
    assert np.allclose(d.mse_escenario,2*(1-np.cos(np.radians(d.angular_error_deg)))/64,atol=1e-12)
    families=['densificacion_perdida_verde','construido_a_vegetacion_lenosa']
    methods=['desplazamiento_medio','familia_ridge','fondo_residuo_rf']
    rng=np.random.default_rng(20261015);rows=[]
    for family in families:
        g=d[d.family==family];blocks=np.sort(g.spatial_block.unique());b=len(blocks)
        weights=rng.multinomial(b,np.full(b,1/b),size=replicates)
        control=g[g.method=='desplazamiento_medio'].set_index('window_id').mse_escenario
        ids=None
        for method in methods:
            q=g[g.method==method].copy();assert q.window_id.is_unique
            if ids is None:ids=set(q.window_id)
            assert set(q.window_id)==ids
            old=float(q.mse_inicial.mean());new=float(q.mse_escenario.mean())
            groups=q.groupby('spatial_block')[['mse_inicial','mse_escenario']].sum().reindex(blocks)
            ratio=(weights@groups.mse_escenario.to_numpy())/(weights@groups.mse_inicial.to_numpy())
            bootstrap=100*(1-ratio)
            rows.append(dict(family=family,method=method,windows=len(q),blocks=b,
                mse_initial=old,mse_scenario=new,mse_reduction_pct=100*(1-new/old),
                mse_reduction_low=float(np.quantile(bootstrap,.025)),mse_reduction_high=float(np.quantile(bootstrap,.975)),
                rmse_initial=float(np.sqrt(old)),rmse_scenario=float(np.sqrt(new)),rmse_reduction_pct=100*(1-np.sqrt(new/old)),
                median_initial_deg=float(q.identity_error_deg.median()),median_scenario_deg=float(q.angular_error_deg.median()),
                median_angle_reduction_pct=100*(1-q.angular_error_deg.median()/q.identity_error_deg.median()),
                improved_windows=int((q.mse_escenario<q.mse_inicial).sum()),
                improvement_vs_family_mean_pct=100*(1-new/control.reindex(q.window_id).mean())))
    result=pd.DataFrame(rows)
    result.to_csv(output/'01_error_embeddings_mse_rmse.csv',index=False)
    d.to_parquet(output/'02_errores_pareados_por_ventana.parquet',index=False)
    # Figuras anteriores de la línea 3 se conservan; esta añade porcentajes e intervalos.
    names=['Desplazamiento medio','Ridge contextual','Fondo–residuo']
    colors=['#ae7127','#974e79','#007d8c']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(8.0,2.9),sharex=True,sharey=True)
    for ax,family,title in zip(axes,families,['Densificación con pérdida verde','Construido a vegetación leñosa']):
        for i,(method,color) in enumerate(zip(methods,colors)):
            r=result[(result.family==family)&(result.method==method)].iloc[0];y=2-i
            ax.errorbar(r.mse_reduction_pct,y,xerr=[[r.mse_reduction_pct-r.mse_reduction_low],[r.mse_reduction_high-r.mse_reduction_pct]],fmt='o',color=color,capsize=3,ms=5)
            ax.annotate(f'{r.mse_reduction_pct:.1f} %'.replace('.',','),(r.mse_reduction_pct,y),xytext=(0,11),textcoords='offset points',ha='center',fontsize=10,fontweight='bold')
        ax.set(title=title,yticks=[2,1,0],yticklabels=names,xlim=(0,100),ylim=(-.6,2.6),xlabel='Reducción del MSE del embedding (%)')
        ax.title.set_fontsize(10);ax.axvline(0,color='#526575',lw=.7);ax.grid(axis='x',alpha=.15)
    fig.tight_layout(w_pad=1.8)
    fig.savefig(output/'03_reduccion_mse_embeddings.png',dpi=300,bbox_inches='tight',pad_inches=.12)
    fig.savefig(output/'03_reduccion_mse_embeddings.pdf',bbox_inches='tight',pad_inches=.12);plt.close(fig)
    # Comparación ventana a ventana: la mejora debe ocurrir fuera del ajuste.
    fig,axes=plt.subplots(1,2,figsize=(7.8,3.15),sharex=True,sharey=True)
    for ax,family,title in zip(axes,families,['Densificación con pérdida verde','Construido a vegetación leñosa']):
        q=d[(d.family==family)&(d.method=='fondo_residuo_rf')]
        ax.scatter(q.identity_error_deg,q.angular_error_deg,s=20,color='#007d8c',alpha=.75)
        ax.plot([0,32],[0,32],'--',color='#74808a',lw=.9)
        ax.set(xlim=(0,32),ylim=(0,32),xlabel='Error sin transformar (grados)',title=title)
        ax.title.set_fontsize(10);ax.grid(alpha=.12)
        ax.text(.05,.94,f'{int((q.angular_error_deg<q.identity_error_deg).sum())} de {len(q)} ventanas\ncon menor error',transform=ax.transAxes,va='top',fontsize=9)
    axes[0].set_ylabel('Error del escenario (grados)')
    fig.tight_layout(w_pad=1.8)
    fig.savefig(output/'04_mejora_angular_por_ventana.png',dpi=300,bbox_inches='tight',pad_inches=.12)
    fig.savefig(output/'04_mejora_angular_por_ventana.pdf',bbox_inches='tight',pad_inches=.12);plt.close(fig)
    audit={'purpose':'Validación del embedding antes del downstream','sources':[{'file':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [metadata,vectors]],
        'dimensions':64,'reference':'Embedding observado de 2025','baseline':'Embedding de 2022 sin transformar','replicates':replicates,
        'seed':20261015,'uncertainty':'Bloques espaciales pareados, modelos fijos, intervalos exploratorios del 95 % sin corrección múltiple.',
        'checks':['Vectores unitarios y finitos.','Ángulos del archivo reproducidos desde los vectores.','MSE = 2(1−cos(ángulo))/64.','Mismas ventanas y referencia entre métodos.'],
        'note':'MSE y ángulo son medidas relacionadas de la misma discrepancia geométrica; no son validaciones independientes. No se utilizan índices ambientales en estos cálculos.'}
    (output/'05_auditoria.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
    return result,output

if __name__=='__main__':
    summary,folder=generar_validacion_transformaciones()
    print(summary.to_string(index=False))
    print('Resultados y figuras:',folder)
