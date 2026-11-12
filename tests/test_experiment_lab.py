from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo import lab
from demo.lab_views import adaptation, export_record, country_plot
from demo.media import image_output
from PIL import Image


class DiagnosticLabTests(unittest.TestCase):
    def test_bundled_cases_cover_three_countries_and_retain_original_hashes(self):
        catalog = lab.cases()
        self.assertEqual(len(catalog), 18)
        for country in ('Czech', 'India', 'United_States'):
            self.assertEqual(sum(i['country'] == country for i in catalog), 6)
        for item in catalog:
            self.assertEqual(hashlib.sha256((lab.ASSETS / item['image']).read_bytes()).hexdigest(), item['sha256'])
            self.assertIn(item['split'], ('test', 'domain_test'))

    def test_image_challenge_identity_and_change_do_not_mutate_original(self):
        image = np.full((21, 21, 3), 160, dtype=np.uint8)
        unchanged = lab.transform(image)
        self.assertTrue(np.array_equal(unchanged, image))
        self.assertFalse(np.shares_memory(unchanged, image))
        darker = lab.transform(image, .5, 3)
        self.assertEqual(darker.dtype, np.uint8)
        self.assertEqual(float(darker.mean()), 80.)
        self.assertEqual(float(image.mean()), 160.)

    def test_match_counts_duplicate_wrong_class_and_missed_targets(self):
        targets = [{'class_name': 'D00', 'xyxy': [0, 0, 20, 20]}, {'class_name': 'D40', 'xyxy': [30, 30, 50, 50]}]
        items = [{'class_code': 'D00', 'box': [0, 0, 20, 20], 'confidence': .9},
                 {'class_code': 'D00', 'box': [0, 0, 20, 20], 'confidence': .8},
                 {'class_code': 'D10', 'box': [30, 30, 50, 50], 'confidence': .7}]
        self.assertEqual(lab.match_targets(items, targets), (1, 2, 1))
        self.assertEqual(lab.match_targets([], targets), (0, 0, 2))

    def test_adaptation_controls_change_actual_records_and_metric(self):
        chart50, rows50, _ = adaptation(50, 'mAP@50')
        chart100, rows100, _ = adaptation(100, 'mAP@50')
        _, potholes, _ = adaptation(100, 'D40_Recall')
        self.assertEqual(rows100[0][1:3], [.0863, .1957])
        self.assertNotEqual(rows50, rows100)
        self.assertNotEqual(chart50.to_dict(), chart100.to_dict())
        self.assertEqual(potholes[0][1:3], [.1835, .1193])

    def test_comparison_bars_use_distinct_positions_not_stacked_rates(self):
        record = {'rows': [[c, 6, 1, 0, 2, 1., .3333] for c in ('Czech', 'India', 'United_States')]}
        self.assertEqual(str(country_plot(None)['Rate'].dtype), 'float64')
        frame = country_plot(record)
        self.assertEqual(frame['Country'].nunique(), 6)
        adapted, _, _ = adaptation(100, 'mAP@50')
        self.assertEqual(adapted['Dataset'].nunique(), 4)
        self.assertTrue(all(0 <= v <= 1 for v in frame['Rate']))

    def test_export_formats_show_different_parity_failures(self):
        good, rows32 = export_record('fp32')
        bad, rows8 = export_record('int8_dynamic')
        self.assertIn('PASS', good)
        self.assertIn('FAIL', bad)
        self.assertGreater(rows8[0][-2], rows32[0][-2])

    def test_benchmark_rejects_unbounded_runs_before_loading_model(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        with patch('demo.lab.analyze_image', side_effect=AssertionError('should validate first')):
            for count in (0, 21, 3.5):
                with self.assertRaises(ValueError):
                    lab.benchmark(image, list(lab.MODELS), ['cpu'], .25, count)

    def test_identical_concurrent_image_outputs_have_unique_valid_pngs(self):
        image = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
        with tempfile.TemporaryDirectory() as folder, patch('demo.media.IMAGE_DIR', Path(folder)):
            with ThreadPoolExecutor(max_workers=8) as pool:
                paths = list(pool.map(lambda _: image_output(image), range(24)))
            self.assertEqual(len(set(paths)), 24)
            for path in paths:
                with Image.open(path) as loaded:
                    self.assertTrue(np.array_equal(np.array(loaded), image))

    def test_late_invalidation_does_not_erase_matching_new_result(self):
        from demo.experiments import fingerprint
        from demo.web_app import build_app
        app = build_app('cpu')
        clear = next(f.fn for f in app.fns.values() if getattr(f.fn, '__name__', '') == 'clear_ipm')
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        args = (image, 'YOLO11n (baseline)', .25, 'cpu', 3.5, 1., 0)
        context = {'token': fingerprint(*args)}
        self.assertTrue(all(r == {'__type__': 'update'} for r in clear(*args, 'en', context)))
        changed = (*args[:4], 4., *args[5:])
        self.assertIsNone(clear(*changed, 'en', context)[0])

    def test_obsolete_computation_does_not_replace_newer_input(self):
        from demo.web_app import build_app
        app = build_app('cpu')
        run = next(f.fn for f in app.fns.values() if getattr(f.fn, '__name__', '') == 'run_perspective')
        context = {'token': None}
        def changed_during_run(*_):
            context['token'] = 'newer-input'
            return None, None, {}
        with patch('demo.lab.perspective', side_effect=changed_during_run):
            result = run(np.zeros((10, 10, 3), dtype=np.uint8), 'YOLO11n (baseline)', .25, 'cpu', 3.5, 1., 0, 'en', context)
        self.assertTrue(all(item == {'__type__': 'update'} for item in result))


if __name__ == '__main__':
    unittest.main()
