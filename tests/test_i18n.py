"""Language switching must preserve predictions, source images and API contracts."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo.i18n import localized_record
from demo.presentation import compare_detections, empty_summary, summary_cards
from demo.web_app import build_app, refresh_record


def sample_record():
    report = {
        'model': 'YOLO11n (baseline)', 'device': 'cpu', 'image_sha256': 'source-image-hash',
        'model_sha256': 'weights-hash', 'predict_call_ms_including_first_call': 12.3,
        'note': '复核优先级是启发式建议；无检出不代表道路完好。单次调用耗时包含首次初始化，不能作为正式速度排名。',
        'review': {'priority_level': 'LOW', 'total_distress_count': 1,
                   'defect_summary': {'D00': 1, 'D10': 0, 'D20': 0, 'D40': 0},
                   'items': [{'class_code': 'D00', 'confidence': .8, 'box': [1, 2, 25, 35]}]},
    }
    second = deepcopy(report)
    second['model'] = 'YOLO11s (comparison)'
    return {'note': '现有两份权重训练预算不同；检出数量和个例不能替代统一条件下的正式评估。',
            'reports': [report, second]}


class BilingualPresentationTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        patcher = patch('demo.inference.EXPORT_DIR', Path(folder.name))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_notes_round_trip_without_mutating_predictions_or_original(self):
        original = sample_record()
        before = deepcopy(original)
        english = localized_record(original, 'en')
        self.assertNotEqual(english['note'], original['note'])
        for first, second in zip(original['reports'], english['reports']):
            for key in ('review', 'device', 'image_sha256', 'model_sha256', 'predict_call_ms_including_first_call'):
                self.assertEqual(first[key], second[key])
        self.assertEqual(localized_record(english, 'zh'), original)
        self.assertEqual(original, before)

    def test_cards_and_matching_translate_with_identical_counts(self):
        record = sample_record()
        self.assertIn('Longitudinal crack', summary_cards(record, 'en'))
        self.assertIn('纵向裂缝', summary_cards(record, 'zh'))
        zh, en = compare_detections(record, 'zh'), compare_detections(record, 'en')
        self.assertEqual(zh[0][1], '两模型共同检出')
        self.assertEqual(en[0][1], 'Detected by both')
        self.assertEqual(zh[0][2:], en[0][2:])

    def test_switch_renders_saved_result_without_inference(self):
        record = sample_record()
        with patch('demo.web_app.analyze_image', side_effect=AssertionError('must not infer')), \
             patch('demo.web_app.compare_images', side_effect=AssertionError('must not infer')):
            en = refresh_record(record, 'en')
            zh = refresh_record(en[4], 'zh')
        self.assertIn('Comparison complete', en[5])
        self.assertIn('对比完成', zh[5])
        self.assertEqual(zh[4], record)
        self.assertEqual(refresh_record(None, 'en')[0], empty_summary('en'))

    def test_default_english_and_legacy_api_input_shapes(self):
        app = build_app('cpu')
        components = app.config['components']
        switch = next(c for c in components if c['props'].get('elem_id') == 'language-switch')
        self.assertEqual(switch['props']['value'], 'en')
        for value in app.review_i18n.translations['en'].values():
            self.assertFalse(any('\u4e00' <= c <= '\u9fff' for c in value), value)
        shapes = {d['api_name']: len(d['inputs']) for d in app.config['dependencies']}
        for name, count in {'analyze': 6, 'run_comparison': 3, 'run_web_prediction': 4,
                            'load_example': 1, 'filter_cases': 2, 'analyze_view': 7}.items():
            self.assertEqual(shapes[name], count)

    def test_switch_callback_preserves_source_and_current_case(self):
        app = build_app('cpu')
        switch = next(fn.fn for fn in app.fns.values() if getattr(fn.fn, '__name__', '') == 'switch_language')
        image = np.zeros((5, 6, 3), dtype=np.uint8)
        result = switch('en', sample_record(), None, image, None, '全部', '全部')
        # The callback has no image output and does not alter its input array.
        self.assertEqual(np.count_nonzero(image), 0)
        self.assertIn('Uploaded image', result[5])
        self.assertEqual(result[6]['value'], None)
        self.assertEqual(result[9]['headers'][0], 'Class')
        self.assertEqual(result[9]['value'][0][1], 'Detected by both')


if __name__ == '__main__':
    unittest.main()
