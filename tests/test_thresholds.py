import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import analyze_thresholds as module
from audit_near_duplicates import structural_similarity, hamming_distances
import numpy as np


class ThresholdTests(unittest.TestCase):
    def test_macro_selection_is_not_dominated_by_common_class(self):
        sweep=[{'threshold':.25,'macro_f1':.3,'f1':.7},{'threshold':.15,'macro_f1':.4,'f1':.6}]
        self.assertEqual(module.select_from_sweep(sweep)['threshold'],.15)

    def test_threshold_counts_background_false_alarms_and_missed_targets(self):
        target={'class_name':'D40','xyxy':[0,0,.5,.5]}
        rows=[{'image_id':'a.jpg','targets':[target],'predictions':[{**target,'confidence':.2}]},
              {'image_id':'b.jpg','targets':[],'predictions':[{**target,'confidence':.3}]}]
        low=module.summarize(rows,.15)
        high=module.summarize(rows,.25)
        self.assertEqual((low['TP'],low['FP'],low['FN']),(1,1,0))
        self.assertEqual((high['TP'],high['FP'],high['FN']),(0,1,1))
        self.assertEqual(high['background_false_alarm_image_rate'],1.)
        filtered=module.summarize(rows,.15,{'b.jpg'})
        self.assertEqual((filtered['images'],filtered['FP']),(1,0))

    def test_test_split_is_rejected_for_threshold_selection(self):
        local=ROOT/'.local/test-temporary'
        local.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=local) as folder:
            args=['analyze_thresholds.py','select','--evaluations','test/metrics.json','--output',str(Path(folder)/'selection')]
            with patch.object(sys,'argv',args),patch.object(module,'load_cache',return_value=({'split':'test'},[])):
                with self.assertRaisesRegex(ValueError,'validation evaluations only'):
                    module.main()
            self.assertFalse((Path(folder)/'selection').exists())

    def test_similarity_identical_image_is_one_and_changed_structure_reduces_it(self):
        a=np.arange(64*64,dtype=np.uint8).reshape(64,64)
        self.assertAlmostEqual(structural_similarity(a,a),1.)
        self.assertLess(structural_similarity(a,np.flip(a,axis=0)),.85)

    def test_perceptual_hash_distance_counts_bits_in_each_byte(self):
        a=np.array([0,255],dtype=np.uint8)
        b=np.array([[0,255],[1,254],[255,0]],dtype=np.uint8)
        self.assertEqual(hamming_distances(a,b).tolist(),[0,2,16])


if __name__=='__main__':
    unittest.main()
