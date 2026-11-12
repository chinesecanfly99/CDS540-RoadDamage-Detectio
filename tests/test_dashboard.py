from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo.presentation import compare_detections, render_boxes, summary_cards


def item(code='D00', confidence=.9, box=None):
    return {'class_code': code, 'confidence': confidence, 'box': box or [10, 10, 35, 35]}


def report(items, name='YOLO11n (baseline)'):
    return {'model': name, 'device': 'cpu', 'review': {
        'priority_level': 'LOW', 'total_distress_count': len(items), 'items': items,
        'defect_summary': {code: sum(value['class_code'] == code for value in items)
                           for code in ('D00', 'D10', 'D20', 'D40')}}}


class DashboardPresentationTests(unittest.TestCase):
    def test_model_comparison_uses_one_to_one_same_class_matching(self):
        record = {'reports': [report([item(confidence=.8), item(confidence=.95), item('D40')]),
                              report([item(), item('D10')])]}
        rows = compare_detections(record)
        self.assertEqual([row[1] for row in rows].count('两模型共同检出'), 1)
        self.assertEqual([row[1] for row in rows].count('仅 YOLO11n 检出'), 2)
        self.assertEqual([row[1] for row in rows].count('仅 YOLO11s 检出'), 1)
        self.assertEqual(rows[0][2], .95)

    def test_no_detections_produce_no_difference_rows(self):
        self.assertEqual(compare_detections({'reports': [report([]), report([])]}), [])

    def test_layer_filter_does_not_change_input_or_prediction_statistics(self):
        original = np.zeros((64, 64, 3), dtype=np.uint8)
        record = report([item()])
        hidden = render_boxes(original, record, [])
        shown = render_boxes(original, record, ['D00'])
        self.assertTrue(np.array_equal(hidden, original))
        self.assertGreater(np.count_nonzero(shown), 0)
        self.assertEqual(np.count_nonzero(original), 0)
        self.assertEqual(record['review']['total_distress_count'], 1)

    def test_summary_escapes_model_label(self):
        html = summary_cards(report([], name='<script>bad()</script>'))
        self.assertNotIn('<script>', html)
        self.assertIn('&lt;script&gt;', html)


if __name__ == '__main__':
    unittest.main()
