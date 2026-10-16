import contextlib
import csv
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_dataset import make_labels, prepare
from rdd_quality import read_target_annotation


def xml(width=30):
    return f'<annotation><size><width>{width}</width><height>20</height></size><object><name>D40</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>15</xmax><ymax>19</ymax></bndbox></object></annotation>'.encode()


class PreparationTests(unittest.TestCase):
    def setUp(self):
        location=ROOT/'.local/test-temporary'
        location.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=location)
        self.folder=Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        manifests=self.folder/'manifests'
        manifests.mkdir()
        assignments=[('train','Czech','a'),('val','India','b'),('test','India','c'),('domain_test','United_States','d')]
        for split,country,name in assignments:
            with (manifests/f'{split}.csv').open('w',newline='') as h:
                writer=csv.DictWriter(h,fieldnames=['image_path','xml_path','country','image_width','image_height','has_object'])
                writer.writeheader()
                writer.writerow({'image_path':f'nonexistent/old/computer/{name}.jpg','xml_path':'missing.xml','country':country,'image_width':30,'image_height':20,'has_object':False})
        for country in ('Czech','India','United_States'):
            with zipfile.ZipFile(self.folder/f'{country}.zip','w') as z:
                for split,c,n in assignments:
                    if c==country:
                        ok,data=cv2.imencode('.jpg',np.full((20,30,3),127,dtype=np.uint8))
                        self.assertTrue(ok)
                        z.writestr(f'{country}/train/images/{n}.jpg',data.tobytes())
                        z.writestr(f'{country}/train/annotations/xmls/{n}.xml',xml())

    def test_old_paths_and_false_presence_flags_do_not_change_targets_or_splits(self):
        with contextlib.redirect_stdout(io.StringIO()):
            report=prepare(self.folder,self.folder/'manifests',self.folder/'dataset')
        self.assertEqual(report['images'],4)
        for split,stem in [('train','a'),('val','b'),('test','c'),('domain_test','d')]:
            self.assertEqual(report['split_stats'][split]['positive'],1)
            self.assertTrue((self.folder/f'dataset/labels/{split}/{stem}.txt').read_text().startswith('3 '))

    def test_actual_dimensions_must_match_annotation(self):
        with zipfile.ZipFile(self.folder/'Czech.zip','w') as z:
            ok,data=cv2.imencode('.jpg',np.zeros((20,31,3),dtype=np.uint8))
            z.writestr('Czech/train/images/a.jpg',data.tobytes())
            z.writestr('Czech/train/annotations/xmls/a.xml',xml())
        with self.assertRaisesRegex(ValueError,'size differs from XML'):
            prepare(self.folder,self.folder/'manifests',self.folder/'dataset')
        self.assertFalse((self.folder/'dataset/preparation.json').exists())

    def test_object_without_class_name_cannot_become_background(self):
        malformed=b'<annotation><size><width>20</width><height>20</height></size><object><bndbox/></object></annotation>'
        with self.assertRaisesRegex(ValueError,'missing its class name'):
            read_target_annotation(malformed)

    def test_label_roundtrip_is_verified_after_decimal_serialization(self):
        labels,error=make_labels(read_target_annotation(xml(1280)))
        self.assertEqual(labels.split()[0],'3')
        self.assertLess(error,1.)


if __name__=='__main__':
    unittest.main()
