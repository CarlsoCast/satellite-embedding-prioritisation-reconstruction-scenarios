"""Auditoría de versión vigente: NDMI principal y NDVI contraste."""
from pathlib import Path
import ast,json,hashlib,zipfile
import numpy as np
import pandas as pd
import linea3_pipeline as engine
H=Path(__file__).resolve().parent;ROOT=H.parents[1];OUT=H/'downstream_ndmi'
nb=json.loads((H/'linea3_pipeline_resultados.ipynb').read_text(encoding='utf-8'))
cells=[c for c in nb['cells'] if c['cell_type']=='code']
assert all(c['execution_count'] is not None for c in cells)
for cell in cells:ast.parse(''.join(cell['source']))
images=sum('image/png' in o.get('data',{}) for c in cells for o in c['outputs'])
core_cells=[c for c in cells if not c.get('metadata',{}).get('publication_synthesis')]
assert sum('image/png' in o.get('data',{}) for c in core_cells for o in c['outputs'])==3
assert nb['metadata']['downstream_actual']==dict(principal='NDMI',contrast='NDVI',training_year=2022,evaluation_year=2025)
with zipfile.ZipFile(H/'respaldo_antes_integracion_ndmi_20261014.zip') as z:old=z.read('linea3_pipeline.py').decode('utf-8')
new=(H/'linea3_pipeline.py').read_text(encoding='utf-8')
names=['unit','project','logmap','expmap','angle','cap','masks','features','ridge_operator','rf_operator','spatial_splits','downmodel']
def bodies(src):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name in names}
assert bodies(old)==bodies(new)
d=engine.getdata();assert len(d)==627 and d.spatial_block.nunique()==73
total=0
for config in engine.CONFIGS:
    folder=H/'nueva_corrida_integral'/config['name'];protocol=json.loads((folder/'00_protocolo.json').read_text())
    assert protocol['config']==config and protocol['input_sha256']==hashlib.sha256(engine.BASE.read_bytes()).hexdigest()
    frame=pd.read_parquet(folder/'02_embedding_oof.parquet');vectors=np.load(folder/'06_vectores_oof.npz')
    assert np.allclose(engine.angle(vectors['scenario'],vectors['observed']),frame.angular_error_deg,atol=1e-6)
    assert np.allclose(np.linalg.norm(vectors['scenario'],axis=1),1,atol=1e-8)
    assert engine.angle(vectors['scenario'],vectors['initial']).max()<=config['angle']+1e-6
    total+=len(frame)
p=pd.read_parquet(OUT/'escenarios_oof.parquet');assert set(p.variable)=={'ndmi','ndvi'}
assert not p.duplicated(['protocol','variable','window_id','source_family','method']).any()
assert p.groupby(['protocol','variable','window_id']).size().eq(6).all()
oldp=pd.read_parquet(ROOT/'experimento_transferencia_2022_2025_20261014/resultados/escenarios_oof.parquet')
keys=['protocol','variable','window_id','source_family','method']
pair=p.merge(oldp,on=keys,suffixes=('_new','_old'),validate='one_to_one')
assert len(pair)==len(p)
max_difference=max(float(abs(pair[c+'_new']-pair[c+'_old']).max()) for c in ['predicted','real2025','unchanged2022','observed_2025'])
assert max_difference<1e-10
s=pd.read_csv(OUT/'metricas_escenarios.csv');assert np.allclose(s.rmse**2,s.mse)
audit=pd.read_csv(OUT/'reproduccion_operadores.csv');assert len(audit)==30 and audit.max_abs_difference.max()<1e-6
pl=pd.read_csv(OUT/'pliegues.csv');assert (pl[pl.eligible].train_2022>=80).all()
report=dict(status='passed',code_cells_executed=len(cells),inline_figures=images,downstream_primary='NDMI',negative_contrast='NDVI',environment_training_year=2022,environment_evaluation_year=2025,geometry_functions_unchanged=True,geometry_configurations=10,geometry_oof_rows=total,new_oof_rows=len(p),max_change_from_separate_experiment=max_difference,operator_reproduction_max_component_error=float(audit.max_abs_difference.max()),new_raw_extraction_executed=False,selection_exploratory=True)
(OUT/'auditoria_actual.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
rows=[]
for path in H.rglob('*'):
    if not path.is_file() or '__pycache__' in path.parts or path.name=='MANIFIESTO_LINEA3.csv':continue
    rows.append(dict(archivo=str(path.relative_to(H)),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
pd.DataFrame(rows).to_csv(H/'MANIFIESTO_LINEA3.csv',index=False)
print(json.dumps(report,ensure_ascii=False,indent=2))
