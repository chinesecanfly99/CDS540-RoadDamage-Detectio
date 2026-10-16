import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from rdd_quality import read_target_annotation, match_predictions, validation_class_rows
from scripts.priority import RoadDamagePrioritizer
from scripts import voc_to_yolo


def item(name='D40', box=None, conf=None):
    record = {'class_name': name, 'xyxy': box or [0., 0., 10., 10.]}
    if conf is not None:
        record['confidence'] = conf
    return record


class DiagnosticTests(unittest.TestCase):
    def test_duplicate_predictions_do_not_inflate_true_positives(self):
        result = match_predictions([item(conf=.9), item(conf=.8)], [item()])
        self.assertEqual((result['TP'], result['FP'], result['FN']), (1, 1, 0))

    def test_wrong_class_is_false_positive_and_false_negative(self):
        result = match_predictions([item(conf=.9)], [item('D00')])
        self.assertEqual((result['TP'], result['FP'], result['FN']), (0, 1, 1))

    def test_low_overlap_is_not_a_correct_localization(self):
        result = match_predictions([item(box=[8., 8., 18., 18.], conf=.9)], [item()])
        self.assertEqual((result['TP'], result['FP'], result['FN']), (0, 1, 1))

    def test_empty_predictions_preserve_missed_targets(self):
        result = match_predictions([], [item()])
        self.assertEqual(result['FN'], 1)
        self.assertIsNone(result['by_class'][0]['recall'])
        self.assertEqual(result['by_class'][3]['recall'], 0)

    def test_absent_metric_classes_are_not_shifted(self):
        box = SimpleNamespace(ap_class_index=[3, 0], p=[.2, .8], r=[.3, .7], ap50=[.4, .9], ap=[.1, .6])
        rows = validation_class_rows(SimpleNamespace(box=box, nt_per_class=[5, 0, 0, 8]))
        self.assertEqual(rows[3]['ap50'], .4)
        self.assertEqual(rows[0]['ap50'], .9)
        self.assertIsNone(rows[1]['ap50'])

    def test_voc_typo_mapping_and_non_target_exclusion(self):
        xml = b'<annotation><size><width>20</width><height>20</height></size><object><name>D0w0</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>10</xmax><ymax>10</ymax></bndbox></object><object><name>D44</name></object></annotation>'
        record = read_target_annotation(xml)
        self.assertEqual(record['targets'][0]['class_name'], 'D00')
        self.assertEqual(record['skipped'], {'D44': 1})

    def test_voc_invalid_target_coordinates_are_rejected(self):
        xml = b'<annotation><size><width>20</width><height>20</height></size><object><name>D40</name><bndbox><xmin>12</xmin><ymin>1</ymin><xmax>10</xmax><ymax>10</ymax></bndbox></object></annotation>'
        with self.assertRaises(ValueError):
            read_target_annotation(xml)


class ReviewRuleTests(unittest.TestCase):
    def test_no_detection_is_not_a_road_condition_diagnosis(self):
        result = RoadDamagePrioritizer().evaluate_detections([], [], [], 100, 100)
        self.assertEqual(result['detection_status'], 'NO_TARGET_DETECTED')
        self.assertEqual(result['total_distress_count'], 0)

    def test_unknown_class_is_not_silently_a_longitudinal_crack(self):
        with self.assertRaises(ValueError):
            RoadDamagePrioritizer().evaluate_detections([[1, 1, 10, 10]], [99], [.9], 100, 100)

    def test_confidence_boundary(self):
        result = RoadDamagePrioritizer().evaluate_detections([[1, 1, 10, 10], [20, 20, 30, 30]], [0, 0], [.2499, .25], 100, 100)
        self.assertEqual(result['total_distress_count'], 1)

    def test_invalid_geometry_is_rejected(self):
        with self.assertRaises(ValueError):
            RoadDamagePrioritizer().evaluate_detections([[10, 10, 1, 1]], [0], [.9], 100, 100)


class ConverterTests(unittest.TestCase):
    def test_missing_xml_cannot_become_a_background_label(self):
        local = ROOT / '.local/test-temporary'
        local.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=local) as temp:
            folder = Path(temp)
            (folder / 'data/manifests').mkdir(parents=True)
            image = folder / 'sample.jpg'
            image.write_bytes(b'sample')
            import pandas as pd
            pd.DataFrame([{'image_path': str(image), 'xml_path': str(folder / 'missing.xml'), 'has_object': False}]).to_csv(folder / 'data/manifests/train.csv', index=False)
            old = Path.cwd()
            try:
                os.chdir(folder)
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileNotFoundError):
                    voc_to_yolo.convert_voc_to_yolo()
                self.assertFalse((folder / 'data/yolo/labels/train/sample.txt').exists())
            finally:
                os.chdir(old)

    def test_generated_domain_config_keeps_source_train_and_validation(self):
        local = ROOT / '.local/test-temporary'
        local.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=local) as temp:
            old = Path.cwd()
            try:
                os.chdir(temp)
                with contextlib.redirect_stdout(io.StringIO()):
                    voc_to_yolo.create_yolo_configs()
                import yaml
                config = yaml.safe_load(Path('configs/data_domain_test.yaml').read_text())
                self.assertEqual(config['test'], 'images/domain_test')
                self.assertEqual(config['val'], 'images/val')
                self.assertEqual(config['train'], 'images/train')
            finally:
                os.chdir(old)


if __name__ == '__main__':
    unittest.main()
