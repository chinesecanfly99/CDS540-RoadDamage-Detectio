"""Regression checks for independently rendered experiment results."""
from pathlib import Path
import sys
import unittest
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo.lab_charts import grouped_bars, live_chart, adaptation_chart


def bars(html):
    start, end = html.index('<svg '), html.index('</svg>') + 6
    return [e.attrib for e in ET.fromstring(html[start:end]).iter() if 'data-value' in e.attrib]


class LabChartTests(unittest.TestCase):
    def test_country_chart_contains_six_colored_bars_with_actual_rates(self):
        record = {'kind': 'countries', 'rows': [[c, 6, 1, 0, 2, 1., r] for c, r in [('Czech', .3), ('India', .1333), ('United_States', .0588)]]}
        drawn = bars(live_chart(record))
        self.assertEqual([float(b['data-value']) for b in drawn], [100., 30., 100., 13.33, 100., 5.88])
        self.assertTrue(all(b['fill'].startswith('#') for b in drawn))
        self.assertEqual(len({b['x'] for b in drawn}), 6)

    def test_zero_detections_remain_a_valid_score_comparison(self):
        record = {'kind': 'perspective', 'prediction': {'review': {'total_score': 0}}, 'ipm_review': {'total_score': 0}}
        drawn = bars(live_chart(record))
        self.assertEqual([b['height'] for b in drawn], ['0.00', '0.00'])
        self.assertTrue(all(float(b['x']) > 0 for b in drawn))
        self.assertIn('当前无预测框', live_chart(record, 'zh'))

    def test_current_feature_and_speed_records_produce_distinct_charts(self):
        features = {'kind': 'features', 'rows': [['D00', .8, 10., 30., 60.], ['D00', .7, 30., 30., 40.]]}
        self.assertEqual([float(b['data-value']) for b in bars(live_chart(features))], [20., 30., 50.])
        bench = {'kind': 'benchmark', 'rows': [['YOLO11n (baseline)', 'cpu', 3, 25., 30., 40., 25.]]}
        self.assertEqual([float(b['data-value']) for b in bars(live_chart(bench))], [25., 40.])
        self.assertEqual(live_chart(None), '')

    def test_archived_selection_changes_its_chart_without_live_inference(self):
        first, second = adaptation_chart(50, 'mAP@50'), adaptation_chart(100, 'mAP@50')
        self.assertNotEqual([b['data-value'] for b in bars(first)], [b['data-value'] for b in bars(second)])
        self.assertIn('Recorded', first)

    def test_labels_are_escaped_and_all_charts_are_bound_to_their_own_record(self):
        chart = grouped_bars('<script>', [('<img onerror="x">', [1.])], ['A&B'])
        self.assertNotIn('<script>', chart)
        self.assertNotIn('<img ', chart)
        self.assertEqual(bars(chart)[0]['data-group'], '<img onerror="x">')
        from demo.web_app import build_app
        app = build_app('cpu')
        renderers = [f for f in app.fns.values() if f.fn is live_chart]
        self.assertEqual(len(renderers), 8)  # own record + display-language update for each chart
        self.assertEqual({f.outputs[0].elem_id for f in renderers}, {'perspective-chart', 'feature-chart', 'country-chart', 'benchmark-chart'})


if __name__ == '__main__':
    unittest.main()
