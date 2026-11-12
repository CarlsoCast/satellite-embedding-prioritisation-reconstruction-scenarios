"""Tablas y cuatro figuras legibles desde predicciones espaciales independientes."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from scipy.stats import spearmanr
import linea3_pipeline as engine

HERE=Path(__file__).resolve().parent;OUT=engine.OUT
FAMILIES=engine.FAMILIES;METHODS=engine.METHODS;LABELS=engine.LABELS;FLABELS=engine.FLABELS;COLORS=engine.COLORS

def analysis_tables():
    d=engine.getdata();emb=pd.read_parquet(OUT/'principal/02_embedding_oof.parquet');down=pd.read_parquet(OUT/'principal/03_downstream_oof.parquet')
    # La media complementa la mediana cuando el límite modifica solo algunas ventanas.
    additions=[]
    for config in ['principal','cola80','cola90','buffer0','buffer500','buffer1500','limite15','limite45','fondo20','fondo40']:
        variant=pd.read_parquet(OUT/config/'02_embedding_oof.parquet')
        pair=emb.merge(variant,on=['family','method','window_id'],suffixes=('_primary','_variant'),validate='one_to_one')
        for (family,method),g in pair.groupby(['family','method']):
            additions.append(dict(configuration=config,family=family,method=method,mean_error_reduction_same_windows=float((g.angular_error_deg_primary-g.angular_error_deg_variant).mean())))
    sensitivity=pd.read_csv(OUT/'09_sensibilidad_ventanas_comunes.csv').drop(columns='mean_error_reduction_same_windows',errors='ignore')
    sensitivity.merge(pd.DataFrame(additions),on=['configuration','family','method'],validate='one_to_one').to_csv(OUT/'09_sensibilidad_ventanas_comunes.csv',index=False)
    # Soporte continuo en el estado inicial; no se aplica un umbral arbitrario de exclusión.
    rows=[]
    for (family,method),g in emb.groupby(['family','method']):
        r=spearmanr(g.initial_support_distance_deg,g.angular_error_deg)
        rows.append(dict(family=family,method=method,windows=len(g),median_nearest_train_angle=float(g.initial_support_distance_deg.median()),
                         rho_support_distance_error=float(r.statistic),scope='Diagnóstico de extrapolación; sin umbral ni AOA formal.'))
    pd.DataFrame(rows).to_csv(OUT/'12_soporte_estado_inicial.csv',index=False)
    # Descomposición exacta del error firmado del escenario respecto del producto.
    assert np.allclose((down.predicted-down.oracle)+(down.oracle-down.observed_2025),down.predicted-down.observed_2025)
    decomposition=[]
    for (family,target,method),g in down.groupby(['family','target','method']):
        reconstruction=(g.predicted-g.oracle).to_numpy();reference=(g.oracle-g.observed_2025).to_numpy()
        decomposition.append(dict(family=family,target=target,method=method,
            scenario_mae=float(np.abs(g.predicted-g.observed_2025).mean()),reference_model_mae=float(np.abs(reference).mean()),transformation_mae=float(np.abs(reconstruction).mean()),
            signed_error_correlation=float(np.corrcoef(reconstruction,reference)[0,1]),
            note='Los errores firmados se suman; los MAE no se suman y pueden compensarse.'))
    pd.DataFrame(decomposition).to_csv(OUT/'13_descomposicion_error_ambiental.csv',index=False)
    # Caso sin escoger el mayor éxito: ventana con error Ridge más cercano a su mediana.
    examples=[]
    for family,g in emb[emb.method=='familia_ridge'].groupby('family'):
        chosen=g.iloc[np.argmin(np.abs(g.angular_error_deg-g.angular_error_deg.median()))]
        examples.append(chosen.to_dict())
    pd.DataFrame(examples).to_csv(OUT/'14_casos_representativos.csv',index=False)
    return d,emb,down

def figures(d,emb,down):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'axes.titlesize':12,'axes.spines.top':False,'axes.spines.right':False})
    short=['Sin\ncambio','Desplazamiento\nmedio','Ridge\ncontextual','Fondo–\nresiduo']
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    for ax,family in zip(axes,FAMILIES):
        g=emb[emb.family==family];values=[g[g.method==m].angular_error_deg.to_numpy() for m in METHODS]
        b=ax.boxplot(values,labels=short,patch_artist=True,showfliers=False,medianprops=dict(color='black',lw=1.4))
        for box,method in zip(b['boxes'],METHODS):box.set_facecolor(COLORS[method]);box.set_alpha(.75)
        ax.set(title=FLABELS[family]+f'\n{g.window_id.nunique()} ventanas fuera de muestra',ylabel='Error angular frente a E2025 (°)');ax.grid(axis='y',alpha=.15)
    fig.savefig(OUT/'15_errores_angulares.png',dpi=210);plt.close(fig)
    summary=pd.read_csv(OUT/'principal/05_resumen_downstream.csv');fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for j,family in enumerate(FAMILIES):
        for i,(target,unit) in enumerate([('lst_c','°C'),('bsa','fracción de albedo')]):
            ax=axes[i,j];g=summary[(summary.family==family)&(summary.target==target)].set_index('method').loc[METHODS]
            bars=ax.bar(np.arange(4),g.mae_observed,color=[COLORS[m] for m in METHODS],width=.65)
            ax.axhline(g.reference_model_mae_observed.iloc[0],color='black',ls='--',lw=1.2,label='Modelo con E2025 observado')
            ax.set_xticks(np.arange(4),short);ax.set(title=('LST' if target=='lst_c' else 'BSA')+' · '+FLABELS[family],ylabel=f'MAE frente al producto ({unit})');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
            ax.yaxis.set_major_locator(plt.MaxNLocator(5));ax.set_ylim(0,float(max(g.mae_observed.max(),g.reference_model_mae_observed.max()))*1.15)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',frameon=False)
    fig.savefig(OUT/'16_errores_ambientales_unidades_separadas.png',dpi=210);plt.close(fig)
    sensitivity=pd.read_csv(OUT/'09_sensibilidad_ventanas_comunes.csv');fig,axes=plt.subplots(2,2,figsize=(11,7.6),layout='constrained')
    configs=[(['cola80','principal','cola90'],[80,85,90],'Percentil de las colas de familia'),
             (['buffer0','buffer500','principal','buffer1500'],[0,500,1000,1500],'Exclusión espacial (m)'),
             (['limite15','principal','limite45'],[15,30,45],'Límite de transformación (°)'),
             (['fondo20','principal','fondo40'],[20,30,40],'Fracción de referencias estables (%)')]
    for ax,(names,x,title) in zip(axes.flat,configs):
        for family,style in zip(FAMILIES,['-','--']):
            for method in ['familia_ridge','fondo_residuo_rf']:
                g=sensitivity[(sensitivity.family==family)&(sensitivity.method==method)].set_index('configuration').reindex(names)
                ax.plot(x,g.mean_error_reduction_same_windows,marker='o',ls=style,color=COLORS[method],label=LABELS[method]+' · '+('densificación' if family==FAMILIES[0] else 'vegetación leñosa'))
        ax.axhline(0,color='gray',lw=.8);ax.set(xlabel=title,ylabel='Reducción media del error frente\na la configuración principal (°)',xticks=x);ax.grid(alpha=.15)
    fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False,fontsize=9)
    fig.savefig(OUT/'17_sensibilidad_cohortes_comunes.png',dpi=210);plt.close(fig)
    # Cada cuadrado corresponde a una ventana real de 500 m; no se representan círculos.
    fig,axes=plt.subplots(1,2,figsize=(10,8),layout='constrained')
    coord=d.set_index(d.window_id.astype(str));all_improvement=emb[emb.method=='familia_ridge'].improvement_deg;limit=max(float(all_improvement.abs().max()),1)
    for ax,family in zip(axes,FAMILIES):
        background=[Rectangle(((r.x_centroid-250)/1000,(r.y_centroid-250)/1000),.5,.5) for r in d.itertuples()]
        ax.add_collection(PatchCollection(background,facecolor='#e5e7eb',edgecolor='none'))
        g=emb[(emb.family==family)&(emb.method=='familia_ridge')];patches=[]
        for wid in g.window_id:
            r=coord.loc[str(wid)];patches.append(Rectangle(((r.x_centroid-250)/1000,(r.y_centroid-250)/1000),.5,.5))
        collection=PatchCollection(patches,cmap='RdBu',norm=plt.Normalize(-limit,limit),edgecolor='white',linewidth=.15);collection.set_array(g.improvement_deg.to_numpy());ax.add_collection(collection)
        ax.autoscale_view();ax.set_aspect('equal');ax.set(title=FLABELS[family]+f'\n{len(g)} ventanas evaluadas con Ridge',xlabel='Este (km)',ylabel='Norte (km)');ax.ticklabel_format(useOffset=False,style='plain');ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    fig.colorbar(collection,ax=axes,label='Mejora angular frente a no cambiar (°)',shrink=.7)
    fig.savefig(OUT/'18_dominio_evaluado_ventanas.png',dpi=220);plt.close(fig)

def main():
    d,emb,down=analysis_tables();figures(d,emb,down)
    print('Cuatro figuras y tablas adicionales de Línea 3 terminadas')

if __name__=='__main__':main()
