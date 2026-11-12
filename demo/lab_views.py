"""Render live diagnostics and selected historical comparisons in either language."""
from html import escape
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def words(en, zh, language='en'):
    return zh if language.startswith('zh') else en


def note(en, zh, language='en'):
    return '<div class="lab-explanation">' + escape(words(en, zh, language)) + '</div>'


def ready(language='en'):
    return note('Ready. Choose a question and run the experiment. Results are computed on this machine.',
                '等待运行。选择要验证的问题并运行实验，结果将在本机实际计算。', language)


def cards(values):
    return '<div class="lab-cards">' + ''.join('<div><span>' + escape(str(label)) + '</span><strong>'
               + escape(str(value)) + '</strong></div>' for label, value in values) + '</div>'


def live_summary(record, language='en'):
    if not record:
        return ready(language)
    w = lambda en, zh: words(en, zh, language)
    if record['kind'] == 'perspective':
        report = record['prediction']
        old, new = report['review'], record['ipm_review']
        return cards([(w('2D review score', '二维复核分数'), old['total_score']),
                      (w('IPM review score', '鸟瞰复核分数'), new['total_score']),
                      (w('Priority change', '优先级变化'), old['priority_level'] + ' → ' + new['priority_level'])]) + note(
            f"{report['model']} · {report['device']} · {len(record['rows'])} predicted boxes. Changing assumed width changes estimated area; priority can remain unchanged. Yellow polygon: assumed road plane. Boxes: model predictions.",
            f"{report['model']} · {report['device']} · {len(record['rows'])} 个预测框。修改假定道路宽度会改变估计面积，优先级可能保持不变。黄色多边形为假定道路区域，框为模型本次预测。", language)
    if record['kind'] == 'features':
        report = record['prediction']
        n = report['review']['total_distress_count']
        return cards([(w('Prediction boxes', '本次预测框'), n), (w('Layers', '特征层'), 'P3 / P4 / P5'),
                      (w('Actual device', '实际设备'), report['device'])]) + note(
            'Heatmaps are live activation energy, not class probabilities or proof of causation. The table summarizes current predicted-box regions; FRAME means the whole image when no boxes were detected.',
            '热图为本次计算的激活能量，不是类别概率或因果证明。表格统计当前预测框区域；无检出时用 FRAME 表示整图，不能据此认为道路完好。', language)
    if record['kind'] == 'countries':
        return note(f"Completed {len(record['cases'])} images with {record['model']}. TP/FP/FN use same-class IoU ≥ 0.5. This small stratified diagnostic set does not measure overall accuracy or mAP. Select a result image to investigate its features.",
                    f"已用 {record['model']} 完成 {len(record['cases'])} 张图。同类别 IoU ≥ 0.5 统计 TP/FP/FN。该小规模分层样例不代表总体精度或 mAP。点击结果图片可进一步分析特征响应。", language)
    return note('Measured on this computer, with two warmup calls per model/device. Prediction time includes preprocessing and NMS; full-call time also includes review rules and rendering. Derived FPS is a sequential estimate, not a video throughput guarantee.',
                '结果来自本机实测，每个模型/设备预热两次。预测调用包含预处理和 NMS；完整调用还包含复核规则和绘制。换算 FPS 为逐图估计，不代表视频吞吐保证。', language)


def country_plot(record):
    return pd.DataFrame([{'Country': f"{('US' if row[0] == 'United_States' else row[0])} / {metric}", 'Rate': row[column], 'Metric': metric}
                         for row in (record or {}).get('rows', [])
                         for column, metric in ((5, 'Precision'), (6, 'Recall'))], columns=['Country', 'Rate', 'Metric']).astype({'Rate': 'float64'})


def adaptation(budget, metric, language='en'):
    path = ROOT / 'results/v2/domain_adapt/domain_adapt_summary.json'
    if not path.is_file():
        return pd.DataFrame(columns=['Dataset', 'Value', 'Model']), [], note('Record unavailable.', '当前没有该实验记录。', language)
    data = json.loads(path.read_text(encoding='utf-8'))
    filename = f'yolo11s_fewshot_{int(budget)}.pt'
    records = data['records']
    rows, bars, changes = [], [], []
    for split, title in [('adapt_holdout (4405)', 'US holdout'), ('in_domain_test (1579)', 'Czech + India')]:
        base = next(r for r in records if r['eval_split'] == split and r['weights_version'] == 'compare_yolo11s_best.pt')
        tuned = next(r for r in records if r['eval_split'] == split and r['weights_version'] == filename)
        delta = tuned[metric] - base[metric]
        rows.append([title, base[metric], tuned[metric], round(delta, 4)])
        bars.extend([{'Dataset': f'{title} / {name}', 'Value': r[metric], 'Model': name} for r, name in [(base, 'Baseline'), (tuned, f'{budget}-shot')]])
        changes.append(f'{title}: {base[metric]:.4f} → {tuned[metric]:.4f} ({delta:+.4f})')
    message = '; '.join(changes)
    conclusion = note(f'Recorded {budget}-image US fine-tuning, metric {metric}. {message}. These checkpoints are not registered in the live workspace; this selection reads experiment records and does not train or switch its model.',
                      f'美国 {budget} 张样本微调记录，指标 {metric}。{message}。这些微调权重尚未登记到实时工作台；这里切换已有实验记录，不进行训练，也不更换工作台模型。', language)
    return pd.DataFrame(bars), rows, conclusion


def export_record(variant, language='en'):
    path = ROOT / 'results/v2/runtime/onnx_parity.json'
    if not path.is_file():
        return note('Record unavailable.', '当前没有导出记录。', language), []
    record = json.loads(path.read_text(encoding='utf-8'))['variants'][variant]
    rows = [[record['execution_provider'], record['n_images'], record['total_pytorch_detections'], record['total_ort_detections'],
             record['unmatched_pt'], record['unmatched_ort'], record['max_box_delta_px'], record['max_score_delta']]]
    reasons = '; '.join(record['failure_reasons']) or words('Within the recorded tolerances.', '在记录的容差范围内。', language)
    return note(f"Recorded {variant.upper()} parity: {record['status']}. {reasons} The original test used CPUExecutionProvider. This is a historical format-conversion check, separate from the live benchmark above.",
                f"已有 {variant.upper()} 一致性测试：{record['status']}。{reasons}。原实验使用 CPUExecutionProvider，这是历史格式转换检验，与上面的本机测速分开。", language), rows
