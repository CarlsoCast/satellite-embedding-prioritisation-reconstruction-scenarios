from pathlib import Path
import os
import json,hashlib,shutil,io
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'linea2_intensidad_regularizacion_20261028'
REPO_ROOT=Path(__file__).resolve().parents[3]
OLD=Path(os.environ.get('SATEMB_LINE2_DATA',REPO_ROOT/'data/raw/line2')).resolve()
def main():
    assert not (ROOT/'seleccion_congelada.json').exists(),'Already frozen'
    for sub in ('snapshot','outputs','transferencia','cache'):(ROOT/sub).mkdir(exist_ok=True)
    shutil.copy2(SOURCE/'engine.py',ROOT/'engine.py')
    shutil.copy2(SOURCE/'geometria.npz',ROOT/'snapshot/geometria.npz')
    rows=[];manifest={}
    for p in sorted((SOURCE/'models').glob('*.npz')):
        payload=p.read_bytes();(ROOT/'snapshot'/p.name).write_bytes(payload)
        manifest[p.name]=hashlib.sha256(payload).hexdigest()
        with np.load(io.BytesIO(payload),allow_pickle=False) as z:
            trace=json.loads(str(z['metadata']))['trace']
        assert trace[-1]['outside_km2']>=15, 'Budget not reached'
        available=[r for r in trace if r['outside_km2']<=15]
        r=available[-1];rows.append(r|dict(variant=p.stem.split('__')[0]))
    df=pd.DataFrame(rows);assert len(df)==15
    df.to_csv(ROOT/'outputs/seleccion_regional_por_semilla.csv',index=False)
    rank=df.groupby('variant').agg(median_core=('core_recovery','median'),min_core=('core_recovery','min'),max_core=('core_recovery','max'),median_rules=('iteration','median')).sort_values(['median_core','median_rules'],ascending=[False,True])
    rank.to_csv(ROOT/'outputs/seleccion_regional_resumen.csv')
    winner=rank.index[0]
    assert winner=='completo_r005', 'Review unexpected selection before transfer'
    selected=df[df.variant==winner]
    info=dict(variant=winner,alpha=1.,ridge_fraction=.05,penalty=3.,budget_km2=15.,selection='Highest median core recovery across three seeds at last prefix <= 15 km2 unregistered activation; regional sensitivity only',prefixes={r.method:int(r.iteration) for r in selected.itertuples()},seeds=[30261028,40261028,50261028],median_core=float(rank.iloc[0].median_core),source=str(SOURCE),snapshot_sha256=manifest,transfer_caveat='Configuration chosen using full regional labels: conditional within-region transfer check, not nested independent model-selection validation',map='3/3 seed consensus at frozen regional prefixes; K=5 active neighbors then exterior exclusion then 8-connected components >=9 pixels; six largest by area, not a confidence ranking')
    (ROOT/'seleccion_congelada.json').write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf-8')
    plot_source=OLD/'modificar_figura_areas_k5.py'
    code=plot_source.read_text(encoding='utf-8').replace('import linea2_pipeline as legacy','').replace('import linea2_revision_integral as revision','')
    (ROOT/'figure_style.py').write_text(code,encoding='utf-8')
    (ROOT/'snapshot/origen_figura.json').write_text(json.dumps(dict(path=str(plot_source),sha256=hashlib.sha256(plot_source.read_bytes()).hexdigest(),adaptation='Removed unused legacy imports; outputs redirected by make_map.py'),indent=2),encoding='utf-8')
    print(json.dumps(info,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
