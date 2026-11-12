"""Read and display the repository's recorded v2 experiments."""
import csv
import json
from pathlib import Path

import gradio as gr

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


def select_panel(name, panels):
    path = next((path for path in panels if path.name == name), None)
    if path is None:
        raise gr.Error('该实验图片不存在，请更新完整仓库。')
    return str(path)


def build_experiments():
    gr.HTML('<div class="section-intro"><span class="eyebrow">EXPERIMENTS / 技术分析</span><h2>看见实验过程与结果</h2><p>以下内容读取仓库已记录的 v2 实验。每项结果保留样本范围、测量条件与局限。</p></div>')
    with gr.Tabs(elem_id='experiment-tabs'):
        with gr.Tab('道路鸟瞰'):
            data = read_json('ipm/ipm_summary.json')
            if data:
                gr.Markdown(f"### 透视变化会如何影响复核排序？\n\n记录样本 **{data['n_sampled_images']} 张**，标注框 **{data['n_gt_boxes_total']} 个**；优先级变化 **{data['priority_level_changes_v1_vs_v2']} 张**。", elem_classes='experiment-lead')
                gr.Image(str(CHARTS['ipm']), label='面积与复核排序分析', interactive=False, height=370, buttons=['fullscreen', 'download'])
                gr.Markdown('采用假定道路几何进行鸟瞰变换，估计面积依赖道路平面与尺度假设。可结合原图观察远近目标的变化。', elem_classes='notice')
                if IPM_PANELS:
                    selection = gr.Dropdown([p.name for p in IPM_PANELS], value=IPM_PANELS[0].name, label='查看逐图鸟瞰案例')
                    panel = gr.Image(str(IPM_PANELS[0]), label='原图与鸟瞰对照', interactive=False, height=440)
                    selection.change(lambda name: select_panel(name, IPM_PANELS), selection, panel, api_name=False)
            else:
                gr.Markdown('当前仓库版本尚未提供该实验记录。')
        with gr.Tab('特征可视化'):
            data = read_json('cam/cam_summary.json')
            if data:
                gr.Markdown(f"### 不同尺度特征层响应哪些区域？\n\n共 **{data['n_images_sampled']} 张图片**，含 **{data['n_negative_images']} 张背景对照**。", elem_classes='experiment-lead')
                if CAM_PANELS:
                    selection = gr.Dropdown([p.name for p in CAM_PANELS], value=CAM_PANELS[0].name, label='选择特征案例')
                    panel = gr.Image(str(CAM_PANELS[0]), label='原图与 P3 / P4 / P5 响应', interactive=False, height=430)
                    selection.change(lambda name: select_panel(name, CAM_PANELS), selection, panel, api_name=False)
                gr.Markdown('色彩展示归一化激活能量。区域响应与正确检出需要分别核对，原始标注和预测框供逐例参考。', elem_classes='notice')
                with gr.Accordion('各类别的特征层响应汇总', open=False):
                    gr.Image(str(CHARTS['cam']), label='分层响应统计', interactive=False)
            else:
                gr.Markdown('当前仓库版本尚未提供该实验记录。')
        with gr.Tab('跨国适应'):
            data = read_json('domain_adapt/domain_adapt_summary.json')
            if data:
                counts = data['integrity']['counts']
                gr.Markdown(f"### 美国样本微调带来哪些收益与代价？\n\n50 / 100 / 200 张微调方案；美国留出评估 **{counts['adapt_holdout']:,} 张**，验证 **{counts['fewshot_val']} 张**。", elem_classes='experiment-lead')
                gr.Image(str(CHARTS['adapt']), label='跨国收益与原领域遗忘', interactive=False, height=400)
                rows = [[r['weights_version'], r['eval_split'], r['mAP@50'], r['D40_Recall']] for r in data['records']]
                gr.Dataframe(rows, headers=['模型版本', '评估集合', 'mAP50', '坑洞召回率'], interactive=False, label='已记录的实验指标')
                gr.Markdown('美国适应收益与原领域遗忘需要共同衡量。图表数值来自仓库记录，检测工作台显示当前可用权重的实时预测。', elem_classes='notice')
            else:
                gr.Markdown('当前仓库版本尚未提供该实验记录。')
        with gr.Tab('部署与导出'):
            data = read_json('runtime/onnx_parity.json')
            if data:
                gr.Markdown('### 导出后，检测结果还能保持一致吗？\n\n一致性检验使用 32 张图片，ONNX 使用 CPUExecutionProvider。', elem_classes='experiment-lead')
                badges = ''.join(f'<div class="export-badge {"passed" if value["status"] == "PASS" else "failed"}"><span>{name.upper()}</span><strong>{"通过" if value["status"] == "PASS" else "未通过"}</strong></div>' for name, value in data['variants'].items())
                gr.HTML(f'<div class="export-status">{badges}</div>')
                gr.Image(str(CHARTS['runtime']), label='模型体积、运行速度与一致性', interactive=False, height=360)
                path = RESULTS / 'runtime/export_benchmark.csv'
                if path.is_file():
                    with path.open(encoding='utf-8-sig', newline='') as handle:
                        rows = [[r['precision'], r['ep'], r['model_size_mb'], r['mean_ms'], r['fps']] for r in csv.DictReader(handle)]
                    gr.Dataframe(rows, headers=['格式', '执行后端', '体积 MB', '平均 ms', 'FPS'], interactive=False, label='记录的运行条件')
                gr.Markdown('表中包含 CPU 与 GPU 两种执行后端，耗时需要结合各自行的运行条件解读。', elem_classes='notice')
            else:
                gr.Markdown('当前仓库版本尚未提供该实验记录。')
