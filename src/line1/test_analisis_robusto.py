"""Controles de inferencia, soporte y revisión por presupuesto."""
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from analisis_robusto import prepared_metrics,align_sources,simultaneous_ci,METHODS,analyze_family

class Checks(unittest.TestCase):
    def test_weighted_review_matches_explicit_duplicate_blocks(self):
        rng=np.random.default_rng(2);y=rng.integers(0,2,35);score=np.round(rng.random(35),1)
        w=rng.integers(0,4,35);xx=np.arange(35)
        result=prepared_metrics(y,score,xx)(w)
        ids=np.repeat(xx,w);order=np.lexsort((ids,-score[ids]));ids=ids[order]
        self.assertAlmostEqual(result[0],average_precision_score(y[ids],score[ids]))
        for j,budget in enumerate([.05,.1,.2]):
            k=max(1,int(np.ceil(budget*len(ids))));hits=y[ids[:k]].sum()
            np.testing.assert_allclose(result[1+j*3:4+j*3],[hits/k,hits/y[ids].sum(),(hits/k)/y[ids].mean()])

    def test_zero_weight_and_missing_classes_not_false_certainty(self):
        stat=prepared_metrics([0,0,1,1],[4,3,2,1])
        self.assertTrue(np.isnan(stat([0,0,1,1])).all())
        self.assertTrue(np.isnan(stat([0,0,0,0])).all())
        with self.assertRaises(ValueError):stat([-1,1,1,1])

    def test_mismatched_support_and_labels_are_rejected(self):
        a=pd.DataFrame({'x':[0.,1.],'y':[0.,0.],'fold':[0,1],'label':[0,1],**{m:[1.,2.] for m in METHODS}})
        b=a.copy();b.loc[0,'label']=1
        with self.assertRaises(ValueError):align_sources({10:a,100:b})
        b=a.copy();b.loc[0,'x']=2
        with self.assertRaises(ValueError):align_sources({10:a,100:b})

    def test_zero_differences_have_zero_simultaneous_bands(self):
        marginal,lo,hi,q=simultaneous_ci(np.zeros(4),np.zeros((200,4)))
        np.testing.assert_array_equal(lo,0);np.testing.assert_array_equal(hi,0)
        self.assertEqual(q,0)

    def test_simultaneous_critical_value_is_joint_maximum(self):
        rng=np.random.default_rng(7);boot=rng.normal(size=(2000,8))
        obs=np.zeros(8);_,lo,hi,q=simultaneous_ci(obs,boot)
        sd=boot.std(0,ddof=1);coverage=(np.abs(boot/sd).max(1)<=q).mean()
        self.assertAlmostEqual(coverage,.95,places=2)
        self.assertTrue((hi>1.96*sd).all())

    def test_confidence_sensitivity_uses_nested_intervals(self):
        rng=np.random.default_rng(19);boot=rng.normal(size=(5000,4));obs=np.zeros(4)
        intervals=[simultaneous_ci(obs,boot,confidence=p) for p in [.90,.95,.99]]
        for small,large in zip(intervals,intervals[1:]):
            self.assertTrue((large[0][0]<=small[0][0]).all())
            self.assertTrue((large[0][1]>=small[0][1]).all())
            self.assertTrue((large[1]<=small[1]).all())
            self.assertTrue((large[2]>=small[2]).all())
        with self.assertRaises(ValueError):simultaneous_ci(obs,boot,confidence=1.)

    def test_paired_bootstrap_identical_methods_cannot_create_gain(self):
        # Tres grupos con varios bloques y ambas etiquetas en cada bloque.
        rows=[]
        for f in range(3):
            for b in range(3):
                for i in range(8):rows.append({'x':(f*3+b)*5000+i*10.,'y':10.,'fold':f,'label':i%2,**{m:float(i) for m in METHODS}})
        frames={10:pd.DataFrame(rows),100:pd.DataFrame(rows)}
        _,contrasts,_,_=analyze_family(frames,{'block_m':5000},200,10)
        np.testing.assert_array_equal(contrasts.ap_difference,0)
        np.testing.assert_array_equal(contrasts.simultaneous_low,0)
        np.testing.assert_array_equal(contrasts.simultaneous_high,0)

if __name__=='__main__':unittest.main()
