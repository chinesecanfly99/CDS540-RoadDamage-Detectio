"""Presentation of existing predictions without changing inference or review rules."""
from html import escape

import cv2
import numpy as np

from demo.cases import CLASS_LABELS
from demo.i18n import text
from scripts.rdd_quality import box_iou

CLASS_COLORS = {'D00': '#3b82f6', 'D10': '#06b6d4', 'D20': '#a855f7', 'D40': '#f97316'}
PRIORITIES = {'HIGH': '优先复核', 'MEDIUM': '常规复核', 'LOW': '低优先级', 'NONE': '当前阈值无检出'}


def empty_summary(language='zh'):
    t = lambda message: text(message, language)
    steps = ''.join(f'<span>{t(step)}</span>' for step in ('01　选择图片', '02　运行检测', '03　对照复核'))
    return (f'<div class="welcome"><span class="eyebrow">{t("ROAD REVIEW / 道路巡检")}</span>'
            f'<h2>{t("从一张道路图片开始")}</h2><p>{t("上传图片或从案例库选择，运行后在这里查看检测统计与复核建议。")}</p>'
            f'<div class="steps">{steps}</div></div>')


def summary_cards(record, language='zh'):
    t = lambda message, **values: text(message, language, **values)
    if 'reports' in record:
        first, second = record['reports']
        cards = ''.join(
            f'<div class="metric" style="--accent:{CLASS_COLORS[code]}"><span>{code} · {label}</span>'
            f'<strong>{first["review"]["defect_summary"][code]} <em>/</em> {second["review"]["defect_summary"][code]}</strong><small>YOLO11n / YOLO11s</small></div>'
            for code, label in ((code, t(label)) for code, label in CLASS_LABELS.items()))
        levels = ''.join(f'<span class="priority {report["review"]["priority_level"].lower()}">{name} · {t(PRIORITIES[report["review"]["priority_level"]])}</span>'
                         for name, report in zip(('11n', '11s'), (first, second)))
        return (f'<section class="result-summary"><div class="result-heading"><b>{t("同图双模型统计")}</b>{levels}'
                f'<span class="result-device">{escape(first["device"])} / {escape(second["device"])}</span></div><div class="metrics">{cards}</div></section>')
    reports = record.get('reports', [record])
    groups = []
    for report in reports:
        review = report['review']
        level = review['priority_level']
        cards = ''.join(
            f'<div class="metric" style="--accent:{CLASS_COLORS[code]}"><span>{code} · {label}</span><strong>{review["defect_summary"][code]}</strong></div>'
            for code, label in ((code, t(label)) for code, label in CLASS_LABELS.items()))
        groups.append(
            f'<section class="result-summary"><div class="result-heading"><b>{escape(report["model"])}</b>'
            f'<span class="priority {level.lower()}">{t(PRIORITIES[level])}</span>'
            f'<span class="result-device">{escape(report["device"])} · {t("{count} 个检测框", count=review["total_distress_count"])}</span></div>'
            f'<div class="metrics">{cards}</div></section>')
    return ''.join(groups)


def render_boxes(image_rgb, report, categories=None):
    canvas = np.ascontiguousarray(image_rgb).copy()
    categories = set(CLASS_LABELS if categories is None else categories)
    thickness = max(2, round(min(canvas.shape[:2]) / 350))
    for item in report['review']['items']:
        code = item['class_code']
        if code not in categories:
            continue
        color = tuple(bytes.fromhex(CLASS_COLORS[code][1:]))
        x1, y1, x2, y2 = map(round, item['box'])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        label = f'{code} {item["confidence"]:.2f}'
        (width, height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .5, 1)
        top = max(0, y1 - height - 9)
        cv2.rectangle(canvas, (x1, top), (x1 + width + 8, top + height + 8), color, -1)
        cv2.putText(canvas, label, (x1 + 4, top + height + 3), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def compare_detections(record, language='zh'):
    """Confidence-ordered same-class matching between models, not ground-truth scoring."""
    first, second = [report['review']['items'] for report in record['reports']]
    used = set()
    rows = []
    for index in sorted(range(len(first)), key=lambda i: -first[i]['confidence']):
        item = first[index]
        candidates = [(box_iou(item['box'], candidate['box']), j) for j, candidate in enumerate(second)
                      if j not in used and item['class_code'] == candidate['class_code']]
        overlap, match = max(candidates, default=(0., -1))
        if overlap >= .5:
            used.add(match)
            rows.append([item['class_code'], '两模型共同检出', round(item['confidence'], 3),
                         round(second[match]['confidence'], 3), round(overlap, 3)])
        else:
            rows.append([item['class_code'], '仅 YOLO11n 检出', round(item['confidence'], 3), None, None])
    rows.extend([[item['class_code'], '仅 YOLO11s 检出', None, round(item['confidence'], 3), None]
                 for index, item in enumerate(second) if index not in used])
    return [[row[0], text(row[1], language), *row[2:]] for row in rows]
