"""Self-contained SVG charts: no browser chart runtime or other tab is required."""
from html import escape
import math

from demo.lab_views import words, adaptation

COLORS = ('#2563eb', '#0d9488', '#ea8a23')


def grouped_bars(title, groups, series, *, percent=False, caption='', zero_message=''):
    """Render the numbers with their axes, labels and accessible data in one update."""
    if not groups:
        return ''
    width, height = 960, 350
    left, right, top, bottom = 64, 22, 54, 72
    plot_width, plot_height = width - left - right, height - top - bottom
    values = [float(value) for _, row in groups for value in row]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError('Chart values must be finite and non-negative.')
    maximum = 100. if percent else max(1., max(values, default=0.) * 1.2)
    fmt = lambda value: f'{value:.1f}%' if percent else f'{value:.2f}'.rstrip('0').rstrip('.')
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title, quote=True)}">',
           f'<title>{escape(title)}</title><rect width="{width}" height="{height}" rx="10" fill="#fff"/>']
    for tick in range(5):
        value = maximum * tick / 4
        y = top + plot_height * (1 - tick / 4)
        svg.append(f'<line x1="{left}" y1="{y}" x2="{width-right}" y2="{y}" stroke="#dce5ef"/>')
        svg.append(f'<text x="{left-9}" y="{y+4}" text-anchor="end" font-size="13" fill="#54677e">{fmt(value)}</text>')
    for index, name in enumerate(series):
        x = left + index * 250
        svg.append(f'<rect x="{x}" y="15" width="12" height="12" rx="3" fill="{COLORS[index % len(COLORS)]}"/><text x="{x+19}" y="26" font-size="14" fill="#29435f">{escape(name)}</text>')
    if not any(values) and zero_message:
        svg.append(f'<text x="{left+plot_width/2}" y="{top+plot_height/2}" text-anchor="middle" font-size="18" font-weight="600" fill="#315b83">{escape(zero_message)}</text>')
    slot = plot_width / len(groups)
    bar_width = min(100., slot * .72 / len(series))
    for group_index, (label, row) in enumerate(groups):
        center = left + slot * (group_index + .5)
        for series_index, value in enumerate(row):
            value = float(value)
            bar_height = plot_height * value / maximum
            x = center + (series_index - len(series) / 2) * bar_width + 3
            y = top + plot_height - bar_height
            svg.append(f'<rect data-series="{escape(series[series_index], quote=True)}" data-group="{escape(label, quote=True)}" data-value="{value:g}" x="{x:.2f}" y="{y:.2f}" width="{bar_width-6:.2f}" height="{bar_height:.2f}" fill="{COLORS[series_index % len(COLORS)]}"><title>{escape(label)} · {escape(series[series_index])}: {fmt(value)}</title></rect>')
            svg.append(f'<text x="{x+(bar_width-6)/2:.2f}" y="{y-7:.2f}" text-anchor="middle" font-size="13" fill="#29435f">{fmt(value)}</text>')
        for line_index, line in enumerate(label.split(' / ')):
            svg.append(f'<text x="{center:.2f}" y="{top+plot_height+23+line_index*19}" text-anchor="middle" font-size="14" fill="#29435f">{escape(line)}</text>')
    svg.append('</svg>')
    return '<section class="lab-chart"><h4>' + escape(title) + '</h4>' + ''.join(svg) + '<p>' + escape(caption) + '</p></section>'


def live_chart(record, language='en'):
    if not record:
        return ''
    w = lambda en, zh: words(en, zh, language)
    kind = record['kind']
    if kind == 'countries':
        groups = [(('US' if r[0] == 'United_States' else r[0]), [r[5]*100, r[6]*100]) for r in record['rows']]
        return grouped_bars(w('Current run · Precision and Recall', '本次运行 · 精确率与召回率'), groups,
                            [w('Precision', '精确率'), w('Recall', '召回率')], percent=True,
                            caption=w('6 reference images per country; descriptive diagnostic, not overall mAP. Zero-denominator rates are displayed as 0 and should be read with TP/FP/FN below.',
                                      '每国 6 张参考图，仅用于小样本诊断，不代表总体 mAP。分母为零时显示为 0，请结合下方 TP/FP/FN 解读。'))
    if kind == 'perspective':
        no_boxes = not record.get('rows')
        zero_message = w('No predicted boxes · Both scores are 0', '当前无预测框 · 两项分数均为 0') if no_boxes else w('Both review scores are 0', '两项复核分数均为 0')
        return grouped_bars(w('Current image · Review score comparison', '当前图片 · 复核分数对比'),
                            [(w('Same predictions', '相同预测框'), [record['prediction']['review']['total_score'], record['ipm_review']['total_score']])],
                            [w('2D score', '二维分数'), w('Perspective score', '透视分数')], zero_message=zero_message,
                            caption=w('Heuristic review scores under two area models, not probabilities. No detections do not establish an undamaged road; verify against the source and annotations.',
                                      '两种面积方法下的启发式复核分数，不是概率。无检出不代表道路完好，请结合原图与标注复核。'))
    if kind == 'features':
        rows = record['rows']
        means = [sum(r[column] for r in rows) / len(rows) for column in (2, 3, 4)] if rows else [0., 0., 0.]
        return grouped_bars(w('Current image · Feature-layer shares', '当前图片 · 特征层响应占比'),
                            [(w('Mean over current regions', '当前区域均值'), means)], ['P3', 'P4', 'P5'], percent=True,
                            caption=w(f'Mean of {len(rows)} current region(s); FRAME means the whole image when there are no predictions. These normalized shares are not class probabilities.',
                                      f'对当前 {len(rows)} 个区域取均值；无预测框时 FRAME 表示整图。归一化响应占比不是类别概率。'))
    groups = [(r[0].split(' (')[0] + ' / ' + r[1], [r[3], r[5]]) for r in record['rows']]
    return grouped_bars(w('This computer · Mean call time (ms)', '本机实测 · 平均调用耗时（毫秒）'), groups,
                        [w('Prediction', '预测调用'), w('Full call', '完整调用')],
                        caption=w('Lower is faster. Both include preprocessing and NMS; full call also includes review and rendering. Warmup excluded. P95 and repetitions are in the table.',
                                  '越低越快。均包含预处理和 NMS；完整调用还包含复核与绘制。已排除预热，P95 与测量次数见下表。'))


def adaptation_chart(budget, metric, language='en'):
    _, rows, _ = adaptation(budget, metric, language)
    groups = [(r[0], [r[1]*100, r[2]*100]) for r in rows]
    return grouped_bars(words('Recorded US fine-tuning · ', '已有美国微调记录 · ', language) + metric, groups,
                        [words('Baseline', '基线', language), f'{budget}-shot'], percent=True,
                        caption=words('Recorded checkpoints selected above; separate from the live diagnostic.', '上方所选已有权重实验记录，与实时小样本验证分开。', language))
