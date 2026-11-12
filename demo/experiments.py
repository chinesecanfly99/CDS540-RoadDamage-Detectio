"""Read and display the repository's recorded v2 experiments."""
import csv
import json
from pathlib import Path

import gradio as gr
from demo.i18n import Localizer, text

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/v2'
CHARTS = {
    'ipm': RESULTS / 'ipm/11_ipm_birdseye_and_area_comparison.png',
    'cam': RESULTS / 'cam/12_layer_cam_response.png',
    'adapt': RESULTS / 'domain_adapt/13_fewshot_domain_recovery.png',
    'runtime': RESULTS / 'runtime/14_export_accuracy_speed.png',
}
CAM_PANELS = sorted((RESULTS / 'cam/sample_panels').glob('*.png'))
IPM_PANELS = sorted((RESULTS / 'ipm/sample_cards').glob('*.jpg'))


def experiment_files():
    return [str(path.resolve()) for path in [*CHARTS.values(), *CAM_PANELS, *IPM_PANELS] if path.is_file()]


def read_json(relative):
    path = RESULTS / relative
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}


def select_panel(name, panels, language='zh'):
    path = next((path for path in panels if path.name == name), None)
    if path is None:
        raise gr.Error(text('该实验图片不存在，请更新完整仓库。', language))
    return str(path)


def experiment_intro(language):
    return ('<div class="section-intro"><span class="eyebrow">' + text('EXPERIMENTS / 技术分析', language)
            + '</span><h2>' + text('看见实验过程与结果', language) + '</h2><p>'
            + text('以下内容读取仓库已记录的 v2 实验。每项结果保留样本范围、测量条件与局限。', language) + '</p></div>')


def export_badges(data, language):
    badges = ''.join(f'<div class="export-badge {"passed" if value["status"] == "PASS" else "failed"}"><span>{name.upper()}</span><strong>{text("通过" if value["status"] == "PASS" else "未通过", language)}</strong></div>' for name, value in data['variants'].items())
    return f'<div class="export-status">{badges}</div>'


def build_experiments(t=None, language=None):
    t = t or Localizer()
    language = language or gr.State('en')
    gr.HTML(t.html(experiment_intro))
    with gr.Tabs(elem_id='experiment-tabs'):
        with t.tab('道路鸟瞰'):
            data = read_json('ipm/ipm_summary.json')
            if data:
                gr.Markdown(t('### 透视变化会如何影响复核排序？\n\n记录样本 **{images} 张**，标注框 **{boxes} 个**；优先级变化 **{changes} 张**。', images=data['n_sampled_images'], boxes=data['n_gt_boxes_total'], changes=data['priority_level_changes_v1_vs_v2']), elem_classes='experiment-lead')
                gr.Image(str(CHARTS['ipm']), label=t('面积与复核排序分析'), interactive=False, height=370, buttons=['fullscreen', 'download'])
                gr.Markdown(t('采用假定道路几何进行鸟瞰变换，估计面积依赖道路平面与尺度假设。可结合原图观察远近目标的变化。'), elem_classes='notice')
                if IPM_PANELS:
                    selection = gr.Dropdown([p.name for p in IPM_PANELS], value=IPM_PANELS[0].name, label=t('查看逐图鸟瞰案例'))
                    panel = gr.Image(str(IPM_PANELS[0]), label=t('原图与鸟瞰对照'), interactive=False, height=440)
                    selection.change(lambda name, lang: select_panel(name, IPM_PANELS, lang), [selection, language], panel, api_name=False)
            else:
                gr.Markdown(t('当前仓库版本尚未提供该实验记录。'))
        with t.tab('特征可视化'):
            data = read_json('cam/cam_summary.json')
            if data:
                gr.Markdown(t('### 不同尺度特征层响应哪些区域？\n\n共 **{images} 张图片**，含 **{negative} 张背景对照**。', images=data['n_images_sampled'], negative=data['n_negative_images']), elem_classes='experiment-lead')
                if CAM_PANELS:
                    selection = gr.Dropdown([p.name for p in CAM_PANELS], value=CAM_PANELS[0].name, label=t('选择特征案例'))
                    panel = gr.Image(str(CAM_PANELS[0]), label=t('原图与 P3 / P4 / P5 响应'), interactive=False, height=430)
                    selection.change(lambda name, lang: select_panel(name, CAM_PANELS, lang), [selection, language], panel, api_name=False)
                gr.Markdown(t('色彩展示归一化激活能量。区域响应与正确检出需要分别核对，原始标注和预测框供逐例参考。'), elem_classes='notice')
                with gr.Accordion(t('各类别的特征层响应汇总'), open=False):
                    gr.Image(str(CHARTS['cam']), label=t('分层响应统计'), interactive=False)
            else:
                gr.Markdown(t('当前仓库版本尚未提供该实验记录。'))
        with t.tab('跨国适应'):
            data = read_json('domain_adapt/domain_adapt_summary.json')
            if data:
                counts = data['integrity']['counts']
                gr.Markdown(t('### 美国样本微调带来哪些收益与代价？\n\n50 / 100 / 200 张微调方案；美国留出评估 **{holdout} 张**，验证 **{validation} 张**。', holdout=f"{counts['adapt_holdout']:,}", validation=counts['fewshot_val']), elem_classes='experiment-lead')
                gr.Image(str(CHARTS['adapt']), label=t('跨国收益与原领域遗忘'), interactive=False, height=400)
                rows = [[r['weights_version'], r['eval_split'], r['mAP@50'], r['D40_Recall']] for r in data['records']]
                t.dataframe(rows, headers=['模型版本', '评估集合', 'mAP50', '坑洞召回率'], interactive=False, label=t('已记录的实验指标'))
                gr.Markdown(t('美国适应收益与原领域遗忘需要共同衡量。图表数值来自仓库记录，检测工作台显示当前可用权重的实时预测。'), elem_classes='notice')
            else:
                gr.Markdown(t('当前仓库版本尚未提供该实验记录。'))
        with t.tab('部署与导出'):
            data = read_json('runtime/onnx_parity.json')
            if data:
                gr.Markdown(t('### 导出后，检测结果还能保持一致吗？\n\n一致性检验使用 32 张图片，ONNX 使用 CPUExecutionProvider。'), elem_classes='experiment-lead')
                gr.HTML(t.html(lambda language: export_badges(data, language)))
                gr.Image(str(CHARTS['runtime']), label=t('模型体积、运行速度与一致性'), interactive=False, height=360)
                path = RESULTS / 'runtime/export_benchmark.csv'
                if path.is_file():
                    with path.open(encoding='utf-8-sig', newline='') as handle:
                        rows = [[r['precision'], r['ep'], r['model_size_mb'], r['mean_ms'], r['fps']] for r in csv.DictReader(handle)]
                    t.dataframe(rows, headers=['格式', '执行后端', '体积 MB', '平均 ms', 'FPS'], interactive=False, label=t('记录的运行条件'))
                gr.Markdown(t('表中包含 CPU 与 GPU 两种执行后端，耗时需要结合各自行的运行条件解读。'), elem_classes='notice')
            else:
                gr.Markdown(t('当前仓库版本尚未提供该实验记录。'))
