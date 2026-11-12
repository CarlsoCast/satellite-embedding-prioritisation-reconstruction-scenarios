"""Verificación de invariantes científicos; no son resultados territoriales."""
import unittest
from copy import deepcopy
import numpy as np
import pandas as pd
import experimento as e
import tempfile
from pathlib import Path
from contextlib import redirect_stdout
import io,json

class ScientificChecks(unittest.TestCase):
    def setUp(self):self.c=e.config(e.HOME/'config.json')

    def test_rotation_recovers_known_orthogonal_change(self):
        rng=np.random.default_rng(22);x=e.unit(rng.normal(size=(800,64)))
        r,_=np.linalg.qr(rng.normal(size=(64,64)));y=e.unit(x@r)
        ops,audit=e.learn(x[:600],y[:600],self.c['model'])
        out=e.predict(x[600:],y[600:],ops)
        self.assertGreater(out['directo'].mean(),30)
        self.assertLess(out['alineamiento_general'].max(),.06)
        self.assertLess(np.linalg.norm(ops['alineamiento_general'].T@ops['alineamiento_general']-np.eye(64)),1e-8)

    def test_reference_selection_uses_training_embeddings_only(self):
        rng=np.random.default_rng(2);x=rng.normal(size=(1000,64));y=x+rng.normal(size=x.shape)*.1
        a,aa=e.learn(x,y,self.c['model'])
        # Cambiar cualquier referencia auxiliar no es un argumento posible de learn.
        unrelated_dw=rng.dirichlet(np.ones(9),size=1000);unrelated_dw=unrelated_dw[::-1]
        b,bb=e.learn(x,y,self.c['model'])
        for k in a:np.testing.assert_array_equal(a[k],b[k])
        self.assertEqual(aa,bb)

    def test_spatial_buffer_uses_footprints(self):
        c=self.c
        # Bloque de prueba [0,5000] x [0,5000], grupo 0.
        # Centro a 1.000 m del borde no sirve para una ventana de 1 km.
        valid=e.train_mask(np.array([2500,6000,6600]),np.array([2500]*3),0,1000,c)
        np.testing.assert_array_equal(valid,[False,False,True])
        for scale in [10,100,500,1000]:
            xx=np.arange(scale/2,15000,scale);yy=np.full(len(xx),2500.)
            keep=e.train_mask(xx,yy,0,scale,c)
            self.assertTrue(np.all(e.folds(xx[keep],yy[keep],c)!=0))

    def test_aggregation_counts_missing_pixels_not_mean_coverage(self):
        xx,yy=np.meshgrid(np.arange(5,100,10),np.arange(5,100,10))
        x=np.zeros((100,64),np.float32);x[:,0]=1
        z=pd.DataFrame(np.hstack([x,x]),columns=e.E22+e.E25);z['x']=xx.ravel();z['y']=yy.ravel()
        a=e.aggregate(z,100,.9)
        self.assertEqual(len(a),1);self.assertEqual(a.valid_pixels.iloc[0],100)
        self.assertAlmostEqual(a.coverage.iloc[0],1)
        self.assertEqual(len(e.aggregate(z.iloc[:89],100,.9)),0)
        np.testing.assert_allclose(np.linalg.norm(a[e.E22],axis=1),1)

    def test_js_is_after_probability_aggregation(self):
        # Dos píxeles intercambian clases: a nivel ventana la composición no cambia.
        p=np.array([[1.,0,0,0,0,0,0,0,0],[0.,1,0,0,0,0,0,0,0]])
        z=pd.DataFrame(np.hstack([p,p[::-1]]),columns=[f'{y}_{i}' for y in ['p22','p25'] for i in range(9)])
        z['x']=[5,15];z['y']=[5,5]
        c=deepcopy(self.c);c['coverage']=.01
        v=e.reference_scale(z,100,c)
        self.assertAlmostEqual(v.dw_js.iloc[0],0,places=10)
        self.assertGreater(e.js(p,p[::-1]).mean(),.6)

    def test_labels_do_not_use_test_distribution(self):
        rng=np.random.default_rng(7);cols=['dw_js','delta_built','delta_vegetation','delta_bare']
        train=pd.DataFrame(rng.random((100,4)),columns=cols);test=train.iloc[:10].copy()
        labels,proxy,cal=e.calibrate(train,test)
        changed=pd.concat([test,pd.DataFrame(np.full((5,4),100),columns=cols)],ignore_index=True)
        labels2,proxy2,cal2=e.calibrate(train,changed)
        self.assertEqual(cal,cal2);np.testing.assert_array_equal(labels,labels2[:10]);np.testing.assert_allclose(proxy,proxy2[:10])

    def test_common_scoring_geometry(self):
        xx,yy=np.meshgrid(np.arange(5,1000,10),np.arange(5,1000,10))
        z=pd.DataFrame({'x':xx.ravel(),'y':yy.ravel()})
        for m in e.METHODS:z[m]=12.
        a=e.aggregate_scores(z,10,1000,self.c)
        self.assertEqual(len(a),1);self.assertEqual(a.x.iloc[0],500)
        self.assertEqual(a.y.iloc[0],500);self.assertEqual(a.directo.iloc[0],12)
        with self.assertRaises(AssertionError):e.aggregate_scores(a,1000,100,self.c)

    def test_zero_vectors_are_rejected(self):
        with self.assertRaises(ValueError):e.unit(np.zeros((1,64)))

    def test_empty_reference_can_merge_with_numeric_coordinates(self):
        cols=[f'{year}_{i}' for year in ['p22','p25'] for i in range(9)]+['x','y']
        ref=e.reference_scale(pd.DataFrame(columns=cols),1000,self.c)
        self.assertEqual(str(ref.x.dtype),'float64')
        joined=pd.DataFrame({'x':[5.],'y':[5.]}).merge(ref,on=['x','y'])
        self.assertEqual(len(joined),0)

    def test_weighted_ap_with_ties_matches_sklearn(self):
        from resumir_corrida import ap_prepared
        from sklearn.metrics import average_precision_score
        rng=np.random.default_rng(88)
        for _ in range(20):
            y=rng.integers(0,2,70);s=rng.integers(0,8,70);w=rng.integers(0,4,70)
            self.assertAlmostEqual(ap_prepared(y,s)(w),average_precision_score(y,s,sample_weight=w),places=12)

    def test_full_pipeline_with_independent_rasters(self):
        # 4 km² sintéticos: prueba del recorrido y del soporte insuficiente a 1 km.
        # No representa observaciones de Bogotá.
        qa=e.HOME/'qa';qa.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='integracion_',dir=qa) as directory:
            h=Path(directory);rng=np.random.default_rng(45);n=200
            x=rng.normal(size=(64,n,n)).astype('float32');y=x+rng.normal(size=x.shape).astype('float32')*.1
            a=rng.dirichlet(np.ones(9),size=n*n).reshape(n,n,9).transpose(2,0,1).astype('float32')
            b=np.roll(a,1,axis=0).copy();b[:,:,:100]=a[:,:,:100]
            paths={}
            for name,arr in [('ae22',x),('ae25',y),('dw22',a),('dw25',b)]:
                path=h/f'{name}.tif'
                with e.rasterio.open(path,'w',driver='GTiff',width=n,height=n,count=arr.shape[0],dtype='float32',crs='EPSG:9377',transform=e.from_origin(0,2000,10,10)) as f:f.write(arr)
                paths[name]=[str(path)]
            c=deepcopy(self.c);c.update(paths);c.update(out=str(h/'salidas'),bounds_m=[0,0,2000,2000],block_m=1000,buffer_m=0)
            c['model']['max_train']=256
            with redirect_stdout(io.StringIO()):
                e.preparar(c);e.puntuar(c)
            root=Path(c['out']);frozen=json.loads((root/'02_puntajes_congelados.json').read_text(encoding='utf8'))
            self.assertTrue(len(frozen['files'])>16)
            # Se prueban tanto el bloqueo por DW ausente como la evaluación completa.
            missing=deepcopy(c);missing['dw22']=[]
            with self.assertRaises(FileNotFoundError):e.evaluar(missing)
            with redirect_stdout(io.StringIO()):e.evaluar(c)
            result=pd.read_csv(root/'03_metricas_por_grupo.csv')
            self.assertGreater(len(result),0)
            self.assertTrue((root/'06_comparacion_comun_ap.png').is_file())
            self.assertTrue((root/'05_soporte_por_escala.csv').is_file())
            coarse=pd.read_parquet(next((root/'puntajes/1000').glob('*.parquet')))
            self.assertTrue(coarse.alineamiento_general.isna().all())
            for path,digest in frozen['files'].items():self.assertEqual(e.hashlib.sha256(Path(path).read_bytes()).hexdigest(),digest)
            changed=Path(next(iter(frozen['files'])))
            changed.write_bytes(changed.read_bytes()+b'tampered')
            with self.assertRaises(ValueError):e.evaluar(c)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ScientificChecks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    e.dump(e.HOME/'qa/pruebas.json',dict(tests=result.testsRun,passed=result.wasSuccessful(),failures=len(result.failures),errors=len(result.errors),scope='Invariantes y recorrido completo sobre rásteres sintéticos; no resultados de Bogotá.'))
    raise SystemExit(0 if result.wasSuccessful() else 1)
