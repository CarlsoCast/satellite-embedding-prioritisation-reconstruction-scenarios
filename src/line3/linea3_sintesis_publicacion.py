"""Evaluación angular principal y NDMI como aplicación adicional de los escenarios."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sintesis_utilidades import *

def sintetizar_linea3():
    h=Path(__file__).resolve().parent;out=h/'sintesis_publicacion';out.mkdir(exist_ok=True)
    eligibility_path=h/'nueva_corrida_integral/07_elegibilidad_cuatro_familias.csv'
    eligibility=pd.read_csv(eligibility_path)
    family_support=eligibility.groupby('family',as_index=False).agg(train_min=('train_family','min'),train_max=('train_family','max'),test_windows=('test_family','sum'),eligible_folds=('support_conditions_met','sum'),folds=('fold','nunique'),implemented=('implemented','all'))
    family_support['selection_basis']=family_support.implemented.map({True:'Operador implementado y evaluación OOF disponible',False:'Soporte mínimo cumplido; operador no implementado en esta versión'})
    pub_table(family_support,out,'04_cuatro_familias_y_alcance')
    path=h/'nueva_corrida_integral/principal/02_embedding_oof.parquet';p=pd.read_parquet(path)
    families=['densificacion_perdida_verde','construido_a_vegetacion_lenosa'];fn=['Pérdida verde','Ganancia leñosa'];methods=['desplazamiento_medio','familia_ridge','fondo_residuo_rf'];mn=['Media familiar','Ridge contextual','Fondo–residuo'];rows=[]
    for fam in families:
        for method in methods:
            g=p[(p.family==fam)&(p.method==method)];values=g[['identity_error_deg','angular_error_deg']].to_numpy()
            stat=lambda x:100*(1-np.median(x[:,1])/np.median(x[:,0]))
            rel,lo,hi=pub_interval(values,g.spatial_block.to_numpy(),stat)
            paired,pl,ph=pub_interval(values,g.spatial_block.to_numpy(),lambda x:np.median(x[:,0]-x[:,1]))
            rows.append(dict(family=fam,method=method,windows=len(g),blocks=g.spatial_block.nunique(),median_initial_deg=np.median(values[:,0]),median_transformed_deg=np.median(values[:,1]),relative_median_error_reduction_pct=rel,reduction_low_pct=lo,reduction_high_pct=hi,paired_improvement_deg=paired,paired_low_deg=pl,paired_high_deg=ph,improved_windows=int((values[:,1]<values[:,0]).sum()),improved_pct=100*np.mean(values[:,1]<values[:,0])))
    summary=pub_table(pd.DataFrame(rows),out,'01_reduccion_angular')
    # Todos los métodos y diez sensibilidades; cada fila conserva su muestra.
    sr=[]
    for sp in (h/'nueva_corrida_integral').glob('*/02_embedding_oof.parquet'):
        s=pd.read_parquet(sp)
        for (fam,method),g in s[s.method!='sin_cambio'].groupby(['family','method']):
            sr.append(dict(configuration=sp.parent.name,family=fam,method=method,windows=len(g),median_initial_deg=g.identity_error_deg.median(),median_transformed_deg=g.angular_error_deg.median(),relative_median_error_reduction_pct=100*(1-g.angular_error_deg.median()/g.identity_error_deg.median()),improved_pct=100*(g.angular_error_deg<g.identity_error_deg).mean()))
    pub_table(pd.DataFrame(sr),out,'02_sensibilidad_angular')
    downstream=h/'downstream_ndmi/escenarios_oof.parquet';env=pd.read_parquet(downstream);erows=[]
    for var in ['ndmi','ndvi']:
        for fam in families:
            for method in methods:
                g=env[(env.protocol=='anual')&(env.variable==var)&(env.cohort==fam)&(env.source_family==fam)&(env.method==method)]
                rmse=lambda a,b:float(np.sqrt(np.mean((a-b)**2)))
                base=rmse(g.unchanged2022,g.observed_2025);trans=rmse(g.predicted,g.observed_2025)
                pair=g.dropna(subset=['observed_2022']);pers=rmse(pair.observed_2022,pair.observed_2025);new=rmse(pair.predicted,pair.observed_2025)
                est,lo,hi=pub_interval(g[['unchanged_ae','ae']].to_numpy(),g.spatial_block.to_numpy(),lambda x:np.mean(x[:,0]-x[:,1]))
                erows.append(dict(variable=var,family=fam,method=method,windows=len(g),rmse_unchanged=base,rmse_transformed=trans,relative_rmse_reduction_pct=100*(1-trans/base),mae_unchanged=g.unchanged_ae.mean(),mae_transformed=g.ae.mean(),mae_improvement=est,mae_improvement_low=lo,mae_improvement_high=hi,rmse_real_embedding_2025=rmse(g.real2025,g.observed_2025),persistence_common_windows=len(pair),rmse_observed_persistence=pers,rmse_transformed_common=new,relative_rmse_reduction_vs_persistence_pct=100*(1-new/pers)))
    e=pub_table(pd.DataFrame(erows),out,'03_aplicacion_downstream')
    pub_style();fig,axes=plt.subplots(1,2,figsize=(9.2,3.3),layout='constrained')
    for ax,fam,label in zip(axes,families,fn):
        g=summary[summary.family==fam].set_index('method').loc[methods]
        ax.errorbar(g.relative_median_error_reduction_pct,range(3),xerr=[g.relative_median_error_reduction_pct-g.reduction_low_pct,g.reduction_high_pct-g.relative_median_error_reduction_pct],fmt='o',color=PUB_COLORS[2],capsize=3,ms=6)
        ax.set_yticks(range(3),mn);ax.set_ylim(2.45,-.65);ax.set_xlim(0,75);ax.set_xlabel('Reducción del error angular mediano (%)');ax.set_title(label);ax.grid(axis='x',alpha=.16)
        for j,v in enumerate(g.relative_median_error_reduction_pct):ax.annotate(f'{v:.1f} %',(v,j),xytext=(0,11),textcoords='offset points',ha='center',fontsize=9)
    pub_save(fig,out,'figura_4_reduccion_angular')
    fig,axes=plt.subplots(1,2,figsize=(9.2,3.2),sharey=True,layout='constrained')
    for ax,var in zip(axes,['ndmi','ndvi']):
        for j,method in enumerate(methods):
            g=e[(e.variable==var)&(e.method==method)].set_index('family').loc[families]
            ax.bar(np.arange(2)+(j-1)*.23,g.relative_rmse_reduction_pct,width=.21,color=PUB_COLORS[j],label=mn[j])
        ax.axhline(0,color='#5B6570',lw=.8);ax.set_xticks(range(2),fn,fontsize=9);ax.set_ylim(-45,55);ax.set_title(var.upper());ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
    axes[0].set_ylabel('Reducción del RMSE (%)');axes[0].legend(fontsize=8,loc='lower left');pub_save(fig,out,'figura_5_downstream')
    notes=['La evaluación principal es angular, espacial y retrospectiva; NDMI es una aplicación adicional y NDVI un contraste.','Reducción angular = 100 × (1 − mediana del error transformado / mediana del error sin transformar). No es porcentaje de exactitud ni reducción del error de pronóstico futuro.','Los intervalos se recalculan con 2000 remuestreos de bloques completos, modelos fijos y sin corrección múltiple; son exploratorios.','Mejorar frente a no transformar no establece superioridad universal frente al desplazamiento medio.','Los operadores se ajustan con pares 2022/2025 de otras ventanas; el decodificador ambiental se ajusta solo en 2022.','El NDMI utiliza información espectral también relacionada con entradas de AlphaEarth; es referencia externa al ajuste del operador, no sensor independiente.']
    notes.append('Las cuatro familias cumplen el soporte mínimo en las cinco particiones; agua y vegetación inundada no se evaluaron con operadores porque no están implementados en esta versión, no por insuficiencia de datos ni por resultados desfavorables.')
    (out/'implicaciones.md').write_text('\n\n'.join(notes),encoding='utf8');pub_manifest([path,downstream,eligibility_path],out,notes)
    print('Soporte y alcance de las cuatro familias:');print(family_support.to_string(index=False))
    return summary

if __name__=='__main__':sintetizar_linea3()
