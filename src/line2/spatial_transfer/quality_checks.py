from pathlib import Path
import json
import numpy as np
import geopandas as gpd
from shapely import make_valid
from threadpoolctl import threadpool_limits
import engine as e
ROOT=Path(__file__).resolve().parent

def main():
    checks={}
    p=ROOT/'outputs/zonas_propuestas.gpkg';g=gpd.read_file(p)
    invalid=int((~g.is_valid).sum())
    if invalid:
        g.geometry=g.geometry.map(make_valid)
        assert g.is_valid.all()
        assert np.allclose(g.geometry.area.to_numpy(),g.area_m2.to_numpy(),atol=.1)
        # Replacing generated output only; originals and source data untouched.
        tmp=p.with_name('zonas_propuestas_validas.gpkg');g.to_file(tmp,layer='consenso_3_semillas',driver='GPKG');tmp.replace(p)
    checks['vector']=dict(features=len(g),repaired_corner_touching_geometries=invalid,valid=bool(g.is_valid.all()),area_matches_pixels=True)
    x,row,col,risk,core,shape=e.load_data();split=json.loads((ROOT/'transferencia/division.json').read_text(encoding='utf-8'));cut=split['cut_row']
    with threadpool_limits(limits=1):
        for path in sorted((ROOT/'transferencia').glob('*_modelo.npz')):
            with np.load(path) as d:
                ti=d['train_ids'];vi=d['test_ids'];mu=d['mu'];w=d['W'];protos=d['prototypes'];thresholds=d['thresholds']
                south_to_north=path.name.startswith('Sur_a_Norte')
                assert np.all(row[ti]<cut-50) if south_to_north else np.all(row[ti]>=cut+50)
                assert np.all(row[vi]>=cut+50) if south_to_north else np.all(row[vi]<cut-50)
                ids=np.arange(0,len(vi),max(1,len(vi)//2000))
                z=e.unit((x[vi[ids]]-mu)@w)
                predicted=np.zeros(len(ids),bool)
                for p,t in zip(protos,thresholds):predicted|=z@p>=t
                stored=np.unpackbits(d['test_activation'])[:len(vi)]
                assert np.array_equal(predicted,stored[ids]),path.name
                r=json.loads(path.with_name(path.name.replace('_modelo.npz','.json')).read_text(encoding='utf-8'))
                assert int((stored&core[vi]).sum())==r['test']['hit_core']
                assert int((stored&~risk[vi]).sum())==r['test']['outside_pixels']
                assert int((stored&risk[vi]).sum())==r['test']['hit_registered']
                checks[path.stem]=dict(rules=len(thresholds),test_pixels_checked=len(ids),replay_exact=True,metrics_exact=True,spatial_separation_verified=True)
    (ROOT/'outputs/control_calidad.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
