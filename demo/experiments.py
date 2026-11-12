"""Question-driven live diagnostics, with historical studies kept as references."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path

import gradio as gr
import torch

from demo import lab
from demo.lab_charts import live_chart, adaptation_chart
from demo.i18n import Localizer, localized_record
from demo.inference import MODELS, export_json
from demo.media import image_output
from demo.lab_views import adaptation, export_record, live_summary, note, ready, words
from demo.presentation import render_boxes
from scripts.demo_runtime import available_devices, resolve_device

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/v2'
CHARTS = {'ipm': RESULTS / 'ipm/11_ipm_birdseye_and_area_comparison.png',
          'cam': RESULTS / 'cam/12_layer_cam_response.png',
          'adapt': RESULTS / 'domain_adapt/13_fewshot_domain_recovery.png',
          'runtime': RESULTS / 'runtime/14_export_accuracy_speed.png'}


def experiment_files():
    return [str(p.resolve()) for p in [*CHARTS.values(), *lab.ASSETS.glob('*.jpg')] if p.is_file()]


def read_json(relative):
    path = RESULTS / relative
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}


def save_record(record, language):
    result = deepcopy(record)
    if 'prediction' in result:
        result['prediction'] = localized_record(result['prediction'], language)
    for item in result.get('cases', []):
        item['report'] = localized_record(item['report'], language)
    result['display_language'] = language
    return export_json(result, 'road-experiment-')


def case_note(item, language):
    return note(f"Source: {item['id']} · {item['country']} · original {item['split']} split. Original bytes and annotations are retained in the diagnostic pack.",
                f"来源：{item['id']} · {item['country']} · 原 {item['split']} 划分。诊断样例包保留原始图片字节、标注和来源哈希。", language)


def guarded(function, language, *args):
    try:
        return function(*args)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        raise gr.Error(words('Experiment could not finish: ', '实验未完成：', language) + str(error)) from error


def fingerprint(*values):
    return hashlib.sha256(json.dumps([hashlib.sha256(v.tobytes()).hexdigest() if hasattr(v, 'tobytes') else v
                                     for v in values], sort_keys=True).encode()).hexdigest()


def build_experiments(t=None, language=None):
    t = t or Localizer()
    language = language or gr.State('en')
    tr = lambda en, zh: t.pair(en, zh)
    catalog = lab.cases()
    initial = catalog[0] if catalog else None
    raw = lab.load_case(initial['id'])[0] if initial else None
    gr.HTML(t.html(lambda lang: '<div class="section-intro"><span class="eyebrow">LIVE DIAGNOSTICS</span><h2>'
            + words('Explain, challenge and verify a detection', '解释检测结果，改变条件，验证表现', lang)
            + '</h2><p>' + words('The workspace finds damage. This lab tests perspective, feature responses, country transfer and runtime. Send an image from the workspace or load a diagnostic reference.',
              '工作台负责检测与复核；这里验证透视、模型响应、跨国家表现和运行性能。可从工作台带入图片，也可载入诊断参考图。', lang) + '</p></div>'))
    with gr.Accordion(tr('Analysis input & controlled image challenge', '分析输入与图像条件'), open=True):
        with gr.Row():
            selection = gr.Dropdown([(f"{i['country']} · {i['id']}", i['id']) for i in catalog],
                                    value=initial['id'] if initial else None, label=tr('Diagnostic reference (6 per country)', '诊断参考图（每国 6 张）'), elem_id='lab-case')
            load = gr.Button(tr('Load reference', '载入参考图'), interactive=bool(catalog))
        source = gr.HTML(case_note(initial, 'en') if initial else ready())
        with gr.Row():
            image = gr.Image(image_output(raw), type='numpy', format='png', interactive=False, height=220,
                             label=tr('Source image from workspace / reference', '来自工作台或参考库的原图'), elem_id='lab-source')
            preview = gr.Image(image_output(raw), type='numpy', format='png', interactive=False, height=220,
                               label=tr('Image used by the experiments', '实验实际使用的图片'), elem_id='lab-preview')
        with gr.Row():
            brightness = gr.Slider(.2, 1.5, value=1., step=.1, label=tr('Brightness multiplier', '亮度倍数'))
            blur = gr.Radio([(tr('None', '无'), 0), ('3×3', 3), ('7×7', 7), ('11×11', 11)], value=0,
                            label=tr('Gaussian blur kernel', '高斯模糊核'))
        with gr.Row():
            model = gr.Dropdown(list(MODELS), value=list(MODELS)[1], label=tr('Model under investigation', '要研究的模型'))
            device = gr.Dropdown([(tr('Automatic', '自动'), 'auto'), *available_devices(torch)], value='auto', label=tr('Experiment device', '实验设备'))
            confidence = gr.Slider(.05, .6, value=.25, step=.05, label=tr('Common confidence threshold', '共用置信度阈值'))
        gr.Markdown(tr('Keep settings fixed for comparisons. Image challenges are synthetic tests, not measurements of real weather.',
                       '比较时应固定其余条件。亮度与模糊属于合成条件测试，不代表真实天气实测。'), elem_classes='small-note')
    source_state = gr.State(initial)
    ipm_state, cam_state, country_state, bench_state = (gr.State(None) for _ in range(4))
    ipm_context, cam_context, country_context, bench_context = (gr.State(lambda: {'token': None}) for _ in range(4))
    with gr.Tabs(elem_id='experiment-tabs') as tabs:
        with t.tab(('Perspective & area', '透视与面积'), id='perspective'):
            gr.Markdown(tr('### Does perspective change review priority?\nRun on the analysis image. Change the assumed road width and compare its area estimates and review scores.',
                           '### 透视会改变复核排序吗？\n使用上方分析图实际计算。修改假定道路宽度，比较面积估计与复核分数。'), elem_classes='experiment-lead')
            width = gr.Slider(3., 4., value=3.5, step=.5, label=tr('Assumed road width (m)', '假定道路宽度（米）'))
            run_ipm = gr.Button(tr('Compute perspective analysis', '计算透视与面积'), variant='primary', elem_id='run-perspective')
            ipm_summary = gr.HTML(ready(), elem_id='perspective-summary')
            with gr.Row():
                ipm_original = gr.Image(label=tr('Current predictions & assumed road polygon', '当前预测与假定道路区域'), format='png', height=360, interactive=False, elem_id='perspective-original')
                bird = gr.Image(label=tr('Computed bird’s-eye projection', '本次计算的鸟瞰投影'), format='png', height=360, interactive=False, elem_id='perspective-bird')
            ipm_chart = gr.HTML('', elem_id='perspective-chart')
            ipm_table = t.dataframe(headers=[('Class', '类别'), ('Model confidence', '模型置信度'), ('2D image area %', '二维图像面积 %'),
                                            ('Estimated ground area m²', '估计地面面积 m²'), ('Fallback to 2D', '回退二维估计')],
                                    interactive=False, label=tr('Current predicted-box measurements', '当前预测框的测量值'))
            ipm_download = gr.File(label=tr('Download perspective record', '下载透视分析记录'), visible=False)
            gr.Markdown(tr('Area assumes a flat road, width selected above and length 20 m. It is not measured physical damage area. The displayed boxes are live predictions, not source annotations.',
                           '面积基于平坦道路、所选宽度和 20 米长度的假设，不是实测病害面积。这里绘制的是本次模型预测框，不是原始标注。'), elem_classes='notice')
            with gr.Accordion(tr('Recorded 40-image study · fixed overall figure', '已有 40 图研究 · 固定总体图'), open=False):
                gr.Markdown(tr('28 India + 12 Czech images; 97 annotation boxes (91 inside the assumed region). The figure below contains its original fixed illustration and does not follow the current analysis image.',
                               '印度 28 张、捷克 12 张；97 个标注框，其中 91 个在假定区域内。下图包含报告原本的固定示例，不随当前分析图片变化。'))
                if CHARTS['ipm'].is_file():
                    gr.Image(str(CHARTS['ipm']), interactive=False, label=tr('Archived overall study', '归档总体研究'))
        with t.tab(('Feature response', '模型响应'), id='features'):
            gr.Markdown(tr('### Where do the model’s feature layers respond?\nCompare the current image at P3, P4 and P5. Change brightness or blur, then recompute to investigate sensitivity.',
                           '### 模型的特征层对哪些区域响应？\n比较当前图片的 P3、P4、P5。修改亮度或模糊后重新计算，观察响应与检出的变化。'), elem_classes='experiment-lead')
            run_cam = gr.Button(tr('Compute live feature responses', '计算当前特征响应'), variant='primary', elem_id='run-features')
            cam_summary = gr.HTML(ready(), elem_id='feature-summary')
            with gr.Row():
                cam_original = gr.Image(label=tr('Current predictions', '当前预测'), format='png', height=300, interactive=False, elem_id='feature-predictions')
                p3 = gr.Image(label='P3 · stride 8', format='png', height=300, interactive=False, elem_id='feature-p3')
            with gr.Row():
                p4 = gr.Image(label='P4 · stride 16', format='png', height=300, interactive=False, elem_id='feature-p4')
                p5 = gr.Image(label='P5 · stride 32', format='png', height=300, interactive=False, elem_id='feature-p5')
            cam_chart = gr.HTML('', elem_id='feature-chart')
            cam_table = t.dataframe(headers=[('Region / class', '区域或类别'), ('Model confidence', '模型置信度'), ('P3 share %', 'P3 占比 %'), ('P4 share %', 'P4 占比 %'), ('P5 share %', 'P5 占比 %')], interactive=False, label=tr('This image only · predicted regions', '仅当前图片 · 预测区域'))
            cam_download = gr.File(label=tr('Download feature record', '下载特征分析记录'), visible=False)
            with gr.Accordion(tr('Recorded 25-image class summary · fixed population', '已有 25 图类别汇总 · 固定样本集'), open=False):
                gr.Markdown(tr('This is a separate recorded experiment on 25 images, including 4 backgrounds, using annotated regions. It does not change when the current image changes. The live table above uses predicted regions.',
                               '这是另一项对 25 张图片（含 4 张背景图）的已有实验，按标注区域统计。它不随当前图片变化；上面的实时表格则统计当前预测区域。'))
                if CHARTS['cam'].is_file():
                    gr.Image(str(CHARTS['cam']), interactive=False, label=tr('Archived class aggregation', '归档类别总体统计'))
        with t.tab(('Country diagnostic', '跨国家验证'), id='countries'):
            gr.Markdown(tr('### How does this model behave on three countries?\nRun the same model and settings on 18 bundled reference images: 6 Czech, 6 India and 6 US. Their annotations let us count matched, extra and missed boxes.',
                           '### 当前模型在三个国家的样例上表现怎样？\n用相同模型和设置实际处理 18 张参考图：捷克、印度、美国各 6 张。用原标注核对正确、额外与漏检框。'), elem_classes='experiment-lead')
            run_country = gr.Button(tr('Run three-country diagnostic', '运行三国小样本验证'), variant='primary', elem_id='run-countries')
            country_summary = gr.HTML(ready(), elem_id='country-summary')
            country_chart = gr.HTML('', elem_id='country-chart')
            country_table = t.dataframe(headers=[('Country', '国家'), ('Images', '图片数'), ('TP', '正确框 TP'), ('FP', '额外框 FP'), ('FN', '漏检框 FN'), ('Precision', '精确率'), ('Recall', '召回率')], interactive=False)
            gallery = gr.Gallery(label=tr('Select a result to inspect its feature response', '点击结果图进一步分析其特征响应'), columns=3, height=400, format='png', preview=False, allow_preview=False, interactive=False, elem_id='country-results')
            country_download = gr.File(label=tr('Download country diagnostic record', '下载跨国家诊断记录'), visible=False)
            with gr.Accordion(tr('Related research: US fine-tuning · recorded checkpoints', '相关研究：美国微调 · 已有权重实验记录'), open=False):
                gr.Markdown(tr('The live check above tests current weights. This recorded study asks whether adding US training examples improves US performance while hurting Czech/India performance. Only the US was studied as the adaptation target.',
                               '上面验证当前权重。这里的已有研究考察：补充美国训练样本，能否改善美国表现，又是否损害捷克/印度表现。适应目标只研究了美国。'))
                with gr.Row():
                    budget = gr.Radio([50, 100, 200], value=100, label=tr('Labelled US training images', '美国有标注训练图片数'))
                    metric = gr.Radio([('mAP50', 'mAP@50'), (tr('Pothole recall', '坑洞召回率'), 'D40_Recall')], value='mAP@50', label=tr('Recorded metric', '查看已有指标'))
                adaptation_initial = adaptation(100, 'mAP@50')
                adapt_plot = gr.HTML(adaptation_chart(100, 'mAP@50'), elem_id='adaptation-chart')
                adapt_table = t.dataframe(adaptation_initial[1], headers=[('Evaluation dataset', '评估集合'), ('Baseline', '基线'), ('Adapted', '微调后'), ('Change', '变化')], interactive=False)
                adapt_summary = gr.HTML(adaptation_initial[2])
        with t.tab(('Runtime & export evidence', '运行验证与导出记录'), id='runtime'):
            gr.Markdown(tr('### How fast is inference on this computer?\nMeasure both models on the same analysis image, settings and selected devices. Warmup is excluded.',
                           '### 本机推理到底多快？\n对同一张分析图片、相同条件和所选设备实际测量两个模型，排除预热。'), elem_classes='experiment-lead')
            with gr.Row():
                bench_models = gr.CheckboxGroup(list(MODELS), value=list(MODELS), label=tr('Models to measure', '要测量的模型'))
                bench_devices = gr.CheckboxGroup(available_devices(torch), value=[resolve_device('auto', torch)], label=tr('Devices on this computer', '本机可用设备'))
                repeats = gr.Slider(3, 20, value=5, step=1, label=tr('Measured repetitions per model/device', '每个模型与设备的测量次数'))
            run_bench = gr.Button(tr('Measure on this computer', '开始本机实测'), variant='primary', elem_id='run-benchmark')
            bench_summary = gr.HTML(ready(), elem_id='benchmark-summary')
            bench_chart = gr.HTML('', elem_id='benchmark-chart')
            bench_table = t.dataframe(headers=[('Model', '模型'), ('Actual device', '实际设备'), ('Runs', '次数'), ('Mean prediction ms', '预测均值 ms'),
                                              ('Prediction P95 ms', '预测 P95 ms'), ('Mean full call ms', '完整调用均值 ms'), ('Derived sequential FPS', '换算逐图 FPS')], interactive=False)
            bench_download = gr.File(label=tr('Download machine benchmark record', '下载本机测速记录'), visible=False)
            with gr.Accordion(tr('Related research: ONNX conversion · recorded CPU parity', '相关研究：ONNX 转换 · 已有 CPU 一致性记录'), open=False):
                gr.Markdown(tr('Conversion changes the model file/runtime. These archived tests compare converted outputs with PyTorch. They are not this computer’s live speed results.',
                               '转换会改变模型文件或运行框架。这组已有测试比较转换后与 PyTorch 的输出，不是上面的本机实时速度结果。'))
                variant = gr.Radio([('FP32', 'fp32'), ('FP16', 'fp16'), ('INT8 dynamic', 'int8_dynamic')], value='fp32', label=tr('Inspect a recorded export format', '查看一种已有导出格式'))
                export_initial = export_record('fp32')
                export_summary = gr.HTML(export_initial[0])
                export_table = t.dataframe(export_initial[1], headers=[('Execution provider', '执行后端'), ('Test images', '测试图片'), ('PyTorch boxes', 'PyTorch 框数'), ('ONNX boxes', 'ONNX 框数'),
                                                                     ('Unmatched PyTorch', '未配对 PyTorch'), ('Unmatched ONNX', '未配对 ONNX'), ('Max box drift px', '最大框偏差 px'), ('Max score drift', '最大分数偏差')], interactive=False)
                benchmark_path = RESULTS / 'runtime/export_benchmark.csv'
                if benchmark_path.is_file():
                    with benchmark_path.open(encoding='utf-8-sig', newline='') as handle:
                        archived = [[r['runtime'], r['precision'], r['ep'], r['eval_split'], r['model_size_mb'], r['mean_ms']] for r in csv.DictReader(handle)]
                    t.dataframe(archived, headers=[('Framework', '运行框架'), ('Precision', '计算精度'), ('Device / provider', '设备或后端'), ('Sampling protocol', '采样条件'), ('Size MB', '体积 MB'), ('Mean ms', '均值 ms')], interactive=False,
                                label=tr('Archive · GPU and CPU rows have different hardware and protocols', '历史记录 · GPU 与 CPU 行的硬件和采样条件不同'))
                gr.Markdown(tr('FP32 passed only the recorded 32-image numerical check. FP16 and INT8 failed. This page exports diagnostic JSON, not new ONNX model files; Jetson/TensorRT deployment remains unmeasured.',
                               'FP32 仅通过了记录中 32 图的数值检验，FP16 与 INT8 未通过。本页导出诊断 JSON，不生成新的 ONNX 模型；Jetson/TensorRT 部署尚未实测。'), elem_classes='notice')

    def load_reference(case_id, lang):
        if case_id is None:
            return gr.skip(), gr.skip(), gr.skip()
        value, item = guarded(lab.load_case, lang, case_id)
        return image_output(value), case_note(item, lang), item

    def import_workspace(value, case_id, lang):
        if value is None:
            raise gr.Error(words('Run or load an image in the workspace first.', '请先在工作台载入或检测一张图片。', lang))
        identity = case_id or words('uploaded image', '上传图片', lang)
        return image_output(value), note(f'Workspace image: {identity}. Analysis uses the settings shown above.', f'工作台图片：{identity}。分析使用上方明确列出的设置。', lang), None, gr.update(value=case_id if any(i['id'] == case_id for i in catalog) else None)

    load.click(load_reference, [selection, language], [image, source, source_state], api_name='load_diagnostic_case')
    selection.change(load_reference, [selection, language], [image, source, source_state], api_name=False)
    for control in (image, brightness, blur):
        control.change(lambda value, light, kernel: image_output(lab.transform(value, light, kernel)), [image, brightness, blur], preview, show_progress="hidden", api_name=False)

    def run_perspective(value, name, threshold, hardware, road_width, light, kernel, lang, context):
        token = context['token'] = fingerprint(value, name, threshold, hardware, road_width, light, kernel)
        first, second, record = guarded(lab.perspective, lang, value, name, threshold, hardware, road_width, light, kernel)
        if context['token'] != token:
            return (gr.skip(),) * 6
        return image_output(first), image_output(second), live_summary(record, lang), record['rows'], record, gr.File(value=save_record(record, lang), visible=True)

    ipm_outputs = [ipm_original, bird, ipm_summary, ipm_table, ipm_state, ipm_download]
    ipm_inputs = [image, model, confidence, device, width, brightness, blur, language, ipm_context]
    ipm_event = run_ipm.click(run_perspective, ipm_inputs, ipm_outputs,
                            api_name='analyze_perspective', concurrency_id='road-inference', concurrency_limit=1)

    def run_features(value, name, threshold, hardware, light, kernel, lang, context):
        token = context['token'] = fingerprint(value, name, threshold, hardware, light, kernel)
        first, second, third, fourth, record = guarded(lab.feature_maps, lang, value, name, threshold, hardware, light, kernel)
        if context['token'] != token:
            return (gr.skip(),) * 8
        return *[image_output(item) for item in (first, second, third, fourth)], live_summary(record, lang), record['rows'], record, gr.File(value=save_record(record, lang), visible=True)

    cam_outputs = [cam_original, p3, p4, p5, cam_summary, cam_table, cam_state, cam_download]
    cam_inputs = [image, model, confidence, device, brightness, blur, language, cam_context]
    cam_event = run_cam.click(run_features, cam_inputs, cam_outputs,
                            api_name='analyze_features', concurrency_id='road-inference', concurrency_limit=1)

    def run_countries(name, threshold, hardware, light, kernel, lang, context, progress=gr.Progress()):
        token = context['token'] = fingerprint(name, threshold, hardware, light, kernel)
        record = guarded(lab.country_check, lang, name, threshold, hardware, light, kernel, progress)
        if context['token'] != token:
            return (gr.skip(),) * 6
        pictures = []
        for case in record['cases']:
            raw, _ = lab.load_case(case['id'])
            display = render_boxes(lab.transform(raw, light, kernel), case['report'])
            pictures.append((image_output(display), f"{case['country']} · {case['id']} · TP {case['tp']} / FP {case['fp']} / FN {case['fn']}"))
        return live_summary(record, lang), live_chart(record, lang), record['rows'], pictures, record, gr.File(value=save_record(record, lang), visible=True)

    country_outputs = [country_summary, country_chart, country_table, gallery, country_state, country_download]
    country_inputs = [model, confidence, device, brightness, blur, language, country_context]
    country_event = run_country.click(run_countries, country_inputs, country_outputs,
                                     api_name='check_countries', concurrency_id='road-inference', concurrency_limit=1)

    def select_country(record, lang, event: gr.SelectData):
        if not record or not isinstance(event.index, int) or not 0 <= event.index < len(record['cases']):
            raise gr.Error(words('Run the country diagnostic first.', '请先运行跨国家验证。', lang))
        raw, item = lab.load_case(record['cases'][event.index]['id'])
        return image_output(raw), case_note(item, lang), item, gr.Tabs(selected='features'), gr.update(value=item['id'])

    gallery.select(select_country, [country_state, language], [image, source, source_state, tabs, selection], api_name=False)

    def run_benchmark(value, names, hardware, threshold, count, light, kernel, lang, context):
        token = context['token'] = fingerprint(value, names, hardware, threshold, count, light, kernel)
        record = guarded(lab.benchmark, lang, value, names, hardware, threshold, count, light, kernel)
        if context['token'] != token:
            return (gr.skip(),) * 4
        return live_summary(record, lang), record['rows'], record, gr.File(value=save_record(record, lang), visible=True)

    bench_outputs = [bench_summary, bench_table, bench_state, bench_download]
    bench_inputs = [image, bench_models, bench_devices, confidence, repeats, brightness, blur, language, bench_context]
    bench_event = run_bench.click(run_benchmark, bench_inputs, bench_outputs,
                                api_name='benchmark_local', concurrency_id='road-inference', concurrency_limit=1)
    def selected_adaptation(budget, metric, lang):
        _, rows, summary = adaptation(budget, metric, lang)
        return adaptation_chart(budget, metric, lang), rows, summary

    for control in (budget, metric):
        control.change(selected_adaptation, [budget, metric, language], [adapt_plot, adapt_table, adapt_summary], api_name=False)
    variant.change(export_record, [variant, language], [export_summary, export_table], api_name=False)

    def changed(context, *values):
        token = fingerprint(*values)
        if context['token'] == token:
            return False
        context['token'] = token
        return True
    def clear_ipm(value, name, threshold, hardware, road_width, light, kernel, lang, context):
        if not changed(context, value, name, threshold, hardware, road_width, light, kernel):
            return (gr.skip(),) * 6
        return None, None, ready(lang), [], None, gr.File(value=None, visible=False)
    def clear_cam(value, name, threshold, hardware, light, kernel, lang, context):
        if not changed(context, value, name, threshold, hardware, light, kernel):
            return (gr.skip(),) * 8
        return None, None, None, None, ready(lang), [], None, gr.File(value=None, visible=False)
    def clear_country(name, threshold, hardware, light, kernel, lang, context):
        if not changed(context, name, threshold, hardware, light, kernel):
            return (gr.skip(),) * 6
        return ready(lang), '', [], [], None, gr.File(value=None, visible=False)
    def clear_bench(value, names, hardware, threshold, count, light, kernel, lang, context):
        if not changed(context, value, names, hardware, threshold, count, light, kernel):
            return (gr.skip(),) * 4
        return ready(lang), [], None, gr.File(value=None, visible=False)
    for control in (image, model, confidence, device, brightness, blur):
        control.change(clear_ipm, ipm_inputs, ipm_outputs, queue=False, show_progress="hidden", api_name=False)
        control.change(clear_cam, cam_inputs, cam_outputs, queue=False, show_progress="hidden", api_name=False)
    width.change(clear_ipm, ipm_inputs, ipm_outputs, queue=False, show_progress="hidden", api_name=False)
    for control in (model, confidence, device, brightness, blur):
        control.change(clear_country, country_inputs, country_outputs, queue=False, show_progress="hidden", api_name=False)
    for control in (image, bench_models, bench_devices, confidence, repeats, brightness, blur):
        control.change(clear_bench, bench_inputs, bench_outputs, queue=False, show_progress="hidden", api_name=False)

    # Rendering a completed record is independent of the benchmark and other tabs.
    for saved, chart in ((ipm_state, ipm_chart), (cam_state, cam_chart),
                         (country_state, country_chart), (bench_state, bench_chart)):
        saved.change(live_chart, [saved, language], chart, queue=False, show_progress='hidden', api_name=False)
        language.change(live_chart, [saved, language], chart, queue=False, show_progress='hidden', api_name=False)

    # Language changes only re-render saved numbers; they never start inference.
    def refresh_live(record, lang):
        return live_summary(record, lang), gr.File(value=save_record(record, lang), visible=True) if record else gr.File(value=None, visible=False)
    for state, summary, file in ((ipm_state, ipm_summary, ipm_download), (cam_state, cam_summary, cam_download),
                                  (country_state, country_summary, country_download), (bench_state, bench_summary, bench_download)):
        language.change(refresh_live, [state, language], [summary, file], concurrency_id='road-inference', concurrency_limit=1, api_name=False)
    language.change(selected_adaptation, [budget, metric, language], [adapt_plot, adapt_table, adapt_summary], api_name=False)
    language.change(export_record, [variant, language], [export_summary, export_table], api_name=False)
    language.change(lambda item, lang: case_note(item, lang) if item else note('Image received from detection workspace.', '从检测工作台带入的图片。', lang),
                    [source_state, language], source, api_name=False)
    return {'image': image, 'source': source, 'state': source_state, 'selection': selection, 'import': import_workspace}
