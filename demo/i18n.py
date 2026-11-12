"""English-first presentation text; prediction data and identifiers stay unchanged."""
from copy import deepcopy
from pathlib import Path
import re

import gradio as gr

ENGLISH = {
    '道路损伤检测与人工复核': 'Road Damage Detection & Review',
    '观察道路细节，比较模型判断，追踪实验依据。': 'Inspect road details, compare predictions and explore the evidence.',
    'ROAD INSIGHT / 道路巡检': 'ROAD INSIGHT / DETECTION & REVIEW',
    '{device} 可用': '{device} available', '{count} 张原始样例': '{count} original cases', '4 类目标': '4 damage classes',
    '检测工作台': 'Detection workspace', '实验分析': 'Experiments', '案例库': 'Case library',
    '### 选择道路图片': '### Choose a road image', '上传或拖入原图': 'Upload an original image',
    '快速选择样例': 'Quick case selection', '载入样例': 'Load case', '浏览案例库': 'Browse cases',
    '拖入道路图片\n或点击上传': 'Drop a road image\nor click to upload',
    '### 设置检测方式': '### Configure analysis', '双模型对比': 'Compare both models', '单模型检测': 'Single model',
    '检测模式': 'Analysis mode', '选择模型': 'Model', '置信度阈值': 'Confidence threshold',
    '设备与图层': 'Device & display layers', '运行设备': 'Inference device', '自动选择': 'Automatic',
    '显示的检测类别': 'Visible damage classes',
    '图层筛选只影响显示，统计保留完整检测结果。': 'Layer filters change the display only. Counts include all detections.',
    '开始分析': 'Run analysis', '等待运行': 'Ready to analyze',
    'YOLO11n · 基线模型': 'YOLO11n · Baseline', 'YOLO11s · 对照模型': 'YOLO11s · Comparison',
    '检测结果': 'Detection result', '双图同步缩放 · 放大后可拖动，双击图像归位': 'Synchronized zoom · Drag to pan; double-click to reset',
    '原图与标注参考': 'Source & annotation reference', '选择样例后可查看原始标注。': 'Load a case to view its original annotations.',
    '青色框为原始标注': 'Original annotations in cyan', '两模型的检出差异': 'Differences between models',
    '同类检测框按 IoU ≥ 0.5 配对。单侧检出可作为人工核对线索。': 'Same-class boxes are matched at IoU ≥ 0.5. One-sided detections are clues for manual review.',
    '逐框比较': 'Box-by-box comparison', '类别': 'Class', '检出情况': 'Detection match',
    'YOLO11n 置信度': 'YOLO11n confidence', 'YOLO11s 置信度': 'YOLO11s confidence', '框 IoU': 'Box IoU',
    '统计、运行条件与完整记录': 'Statistics, settings & full record', '本次调用记录': 'Current inference calls',
    '模型': 'Model', '实际设备': 'Actual device', '框数': 'Boxes', '复核优先级': 'Review priority', '单次调用 ms': 'Call time (ms)',
    '首次调用包含初始化；现有两份权重的训练预算不同。正式比较请结合统一协议的实验记录。': 'The first call includes initialization. These weights used different training budgets; consult experiments with a common protocol for formal comparisons.',
    '完整记录': 'Full record', '下载本次分析 JSON': 'Download analysis JSON',
    '下载双模型 JSON': 'Download comparison JSON', '下载单模型 JSON': 'Download prediction JSON',
    '复核优先级用于人工排序。请结合原图和标注检查检出与漏检情况。': 'Priority supports human review. Check detections and missed damage against the source image and annotations.',
    'CASE LIBRARY / 原始样例': 'CASE LIBRARY / ORIGINAL IMAGES', '从不同国家与病害出发': 'Explore countries and damage classes',
    '图片来自组内提供的原始数据包，保留原标注、原划分和来源哈希。': 'Images come from the original datasets provided by the group, with source annotations, splits and hashes retained.',
    '全部国家': 'All countries', '捷克': 'Czech Republic', '印度': 'India', '美国': 'United States', '国家': 'Country',
    '全部类别': 'All classes', '无四类目标标注': 'No annotations in the four target classes', '原标注类别': 'Annotated damage class',
    '当前共 **{count} 张**，点击缩略图载入检测工作台。': '**{count} cases**. Select a thumbnail to load it into the workspace.',
    '当前筛选共 **{count} 张**，点击缩略图可载入检测工作台。': '**{count} matching cases**. Select a thumbnail to load it into the workspace.',
    '暂未载入案例，可以先在工作台上传自己的原始道路图片。': 'No case library is installed. Upload an original road image in the workspace.',
    '原始道路样例': 'Original road images',
    '案例按原始标注筛选。该集合经过分层选样，总体指标请查看完整评估记录。': 'Filters use source annotations. These cases were selected by stratified sampling; use the full evaluation for overall metrics.',
    'CDS540 · Road Damage Detection　/　当前页面用于检测、实验观察与人工复核。': 'CDS540 · Road Damage Detection / Inference, recorded experiments and human review.',
    '自选图片 · 当前没有关联原始标注。': 'Uploaded image · No source annotations are linked.',
    '样例已变化，请重新选择图片。': 'The case list has changed. Please select an image again.',
    '对比完成 · 下方可查看模型分歧与完整记录。': 'Comparison complete · Explore model differences and the full record below.',
    '复核优先级 {priority} · 检测框 {count} · 设备 {device} · 预测调用 {ms} ms': 'Review priority {priority} · {count} boxes · {device} · Prediction call {ms} ms',
    'ROAD REVIEW / 道路巡检': 'ROAD REVIEW / GET STARTED', '从一张道路图片开始': 'Start with a road image',
    '上传图片或从案例库选择，运行后在这里查看检测统计与复核建议。': 'Upload an image or choose a case, then run analysis to see detections and review suggestions.',
    '01　选择图片': '01  Choose an image', '02　运行检测': '02  Run analysis', '03　对照复核': '03  Compare & review',
    '纵向裂缝': 'Longitudinal crack', '横向裂缝': 'Transverse crack', '网状裂缝': 'Alligator crack', '坑洞': 'Pothole',
    '优先复核': 'High priority', '常规复核': 'Medium priority', '低优先级': 'Low priority', '当前阈值无检出': 'No detections at this threshold',
    '同图双模型统计': 'Two-model counts on the same image', '{count} 个检测框': '{count} detection boxes',
    '两模型共同检出': 'Detected by both', '仅 YOLO11n 检出': 'YOLO11n only', '仅 YOLO11s 检出': 'YOLO11s only',
    '**{id}** · {country} · 原划分 `{split}` · 四类目标标注 {count} 个。\n\n标注仅供个例核对；已知原数据标注仍有待复核项。此样例库经过分层挑选，不能据此推断总体精度。': '**{id}** · {country} · Source split `{split}` · {count} target annotations.\n\nAnnotations support case review and still contain known issues to check. This stratified case library cannot establish overall accuracy.',
    'EXPERIMENTS / 技术分析': 'EXPERIMENTS / EVIDENCE', '看见实验过程与结果': 'Explore the process and results',
    '以下内容读取仓库已记录的 v2 实验。每项结果保留样本范围、测量条件与局限。': 'These are recorded v2 experiments from the repository, with sample scope, measurement conditions and limitations.',
    '道路鸟瞰': 'Bird’s-eye view', '特征可视化': 'Feature responses', '跨国适应': 'Domain adaptation', '部署与导出': 'Deployment & export',
    '### 透视变化会如何影响复核排序？\n\n记录样本 **{images} 张**，标注框 **{boxes} 个**；优先级变化 **{changes} 张**。': '### How does perspective affect review priorities?\n\n**{images} images**, **{boxes} annotated boxes**; priority changed for **{changes} images**.',
    '面积与复核排序分析': 'Area & review priority analysis',
    '采用假定道路几何进行鸟瞰变换，估计面积依赖道路平面与尺度假设。可结合原图观察远近目标的变化。': 'The transformation assumes road geometry. Area estimates depend on the road-plane and scale assumptions; compare near and distant damage with the source image.',
    '查看逐图鸟瞰案例': 'Select a bird’s-eye case', '原图与鸟瞰对照': 'Source & bird’s-eye comparison',
    '当前仓库版本尚未提供该实验记录。': 'This experiment is not included in the current repository version.',
    '### 不同尺度特征层响应哪些区域？\n\n共 **{images} 张图片**，含 **{negative} 张背景对照**。': '### Which regions activate different feature scales?\n\n**{images} images**, including **{negative} background controls**.',
    '选择特征案例': 'Select a feature case', '原图与 P3 / P4 / P5 响应': 'Source & P3 / P4 / P5 responses',
    '色彩展示归一化激活能量。区域响应与正确检出需要分别核对，原始标注和预测框供逐例参考。': 'Colors show normalized activation energy. Check activations and detection correctness separately using source annotations and predicted boxes.',
    '各类别的特征层响应汇总': 'Feature responses by damage class', '分层响应统计': 'Feature-scale response statistics',
    '### 美国样本微调带来哪些收益与代价？\n\n50 / 100 / 200 张微调方案；美国留出评估 **{holdout} 张**，验证 **{validation} 张**。': '### What are the benefits and costs of US fine-tuning?\n\n50 / 100 / 200-image fine-tuning; **{holdout} US holdout images** and **{validation} validation images**.',
    '跨国收益与原领域遗忘': 'Adaptation gains & source-domain forgetting', '模型版本': 'Weight version', '评估集合': 'Evaluation split', '坑洞召回率': 'Pothole recall', '已记录的实验指标': 'Recorded experiment metrics',
    '美国适应收益与原领域遗忘需要共同衡量。图表数值来自仓库记录，检测工作台显示当前可用权重的实时预测。': 'Consider US adaptation gains alongside source-domain forgetting. Charts show repository records; the workspace runs the currently available weights.',
    '### 导出后，检测结果还能保持一致吗？\n\n一致性检验使用 32 张图片，ONNX 使用 CPUExecutionProvider。': '### Are detections consistent after export?\n\nParity checks use 32 images, with ONNX CPUExecutionProvider.',
    '通过': 'PASS', '未通过': 'FAIL', '模型体积、运行速度与一致性': 'Model size, speed & parity',
    '格式': 'Precision', '执行后端': 'Execution provider', '体积 MB': 'Size (MB)', '平均 ms': 'Mean (ms)', '记录的运行条件': 'Recorded runtime conditions',
    '表中包含 CPU 与 GPU 两种执行后端，耗时需要结合各自行的运行条件解读。': 'Rows use CPU and GPU execution providers. Interpret timing with each row’s runtime conditions.',
    '该实验图片不存在，请更新完整仓库。': 'This experiment image is missing. Update the complete repository.',
    '推理未完成。可选择 CPU 重试，详细原因已记录在运行窗口。': 'Inference could not finish. Try CPU; details are logged in the run console.',
    '请先上传一张原始道路图片。': 'Upload an original road image first.', '图片必须是 RGB 彩色图片。': 'The image must be RGB.',
    '无法读取上传的图片。': 'The uploaded image could not be read.', '请选择列表中的模型。': 'Select a model from the list.',
    '置信度阈值必须在 0 到 1 之间。': 'Confidence must be between 0 and 1.', '请选择有效的运行设备。': 'Select a valid inference device.',
    '请先选择一个有效样例。': 'Select a valid case first.',
    '样例清单包含不存在或越界的图片，请重新生成样例库。': 'The case manifest includes a missing or invalid image path. Rebuild the case library.',
    '复核优先级是启发式建议；无检出不代表道路完好。单次调用耗时包含首次初始化，不能作为正式速度排名。': 'Review priority is heuristic; no detections do not establish an undamaged road. Call time includes first-call initialization and is not a formal speed ranking.',
    '现有两份权重训练预算不同；检出数量和个例不能替代统一条件下的正式评估。': 'These weights used different training budgets. Detection counts and individual cases do not replace a formal evaluation under common conditions.',
}


def text(message, language='en', **values):
    return (message if language.startswith('zh') else ENGLISH.get(message, message)).format(**values)


class Localizer:
    """Register native frontend translations once per app, never a shared session locale."""
    def __init__(self):
        self.i18n = gr.I18n(en={}, **{'zh-CN': {}, 'zh': {}})
        self.tables = []
        self.tabs = []

    def __call__(self, message, **values):
        return self.pair(text(message, 'en', **values), text(message, 'zh', **values))

    def pair(self, english, chinese):
        key = f'road_{len(self.i18n.translations["en"])}'
        for language in self.i18n.translations:
            self.i18n.translations[language][key] = english if language == 'en' else chinese
        return self.i18n(key)

    def html(self, renderer):
        return self.pair(renderer('en'), renderer('zh'))

    def resolve(self, message, language='en'):
        if isinstance(message, (tuple, list)):
            return message[1] if language.startswith('zh') else message[0]
        return text(message, language)

    def dataframe(self, *args, headers, **kwargs):
        component = gr.Dataframe(*args, headers=[self.resolve(h) for h in headers], **kwargs)
        self.tables.append((component, headers))
        return component

    def tab(self, message, **kwargs):
        # Gradio's lazy tab labels can retain the browser locale on first mount.
        # Use explicit defaults and updates so tab navigation always matches the switch.
        component = gr.Tab(self.resolve(message), **kwargs)
        self.tabs.append((component, message))
        return component


def localized_record(record, language):
    """Copy display notes only; preserve all predictions, hashes, settings and timing."""
    result = deepcopy(record)
    reverse = {english: chinese for chinese, english in ENGLISH.items()}
    for node in [result, *result.get('reports', [])]:
        if 'note' in node:
            node['note'] = text(reverse.get(node['note'], node['note']), language)
    return result


def locale_module_url():
    # Resolve the installed Gradio frontend rather than pinning a hashed asset name.
    # This is the same locale store used by Gradio's built-in Settings control.
    assets = Path(gr.__file__).parent / 'templates/frontend/assets'
    for path in assets.glob('core-*.js'):
        if re.search(r'\bas changeLocale\b', path.read_text(encoding='utf-8')):
            return f'./assets/{path.name}'
    raise RuntimeError('Gradio locale support is unavailable. Install the pinned project requirements.')
