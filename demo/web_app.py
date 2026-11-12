"""Road review workbench, recorded experiments and a traceable case gallery."""
import argparse
from html import escape
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.runtime_bootstrap import handoff_if_needed

if __name__ == '__main__':
    handoff_if_needed(ROOT, Path(__file__), sys.argv[1:])

_no_proxy = {value.strip() for value in (os.environ.get('NO_PROXY') or os.environ.get('no_proxy', '')).split(',') if value.strip()}
_no_proxy.update({'127.0.0.1', 'localhost'})
os.environ['NO_PROXY'] = os.environ['no_proxy'] = ','.join(sorted(_no_proxy))
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')

import gradio as gr
import torch
from demo.cases import CASE_DIR, CLASS_LABELS, case_choices, case_note, filter_cases, load_case, read_cases
from demo.experiments import build_experiments, experiment_files
from demo.i18n import Localizer, localized_record, locale_module_url, text
from demo.inference import EXPORT_DIR, MODELS, analyze_image, compare_images, export_json
from demo.media import image_output
from demo.presentation import CLASS_COLORS, compare_detections, empty_summary, render_boxes, summary_cards
from scripts.demo_runtime import available_devices, load_config, resolve_device, select_port

TABLE_HEADERS = ['模型', '实际设备', '框数', 'D00', 'D10', 'D20', 'D40', '复核优先级', '单次调用 ms']
MODES = ['双模型对比', '单模型检测']
ZOOM_SCRIPT = """
value => {
    const pair = document.getElementById('comparison-pair');
    if (pair) pair.style.setProperty('--zoom', value);
    if (Number(value) === 1) {
        document.querySelectorAll('#comparison-n img, #comparison-s img').forEach(img => {
            img.style.setProperty('--pan-x', '0px');
            img.style.setProperty('--pan-y', '0px');
        });
    }
    return [];
}
"""


def user_error(function, *arguments, language='zh'):
    try:
        return function(*arguments)
    except (ValueError, FileNotFoundError) as error:
        message = str(error)
        if language == 'en' and message.startswith('当前 Python 环境不能使用'):
            message = 'This Python environment cannot use the selected device. Choose Automatic or CPU, or install the matching PyTorch build.'
        elif language == 'en' and message.startswith('缺少模型权重'):
            message = 'Model weights are missing. Update the complete repository.'
        elif language == 'en' and any('\u4e00' <= char <= '\u9fff' for char in message) and text(message, language) == message:
            message = 'Analysis could not start. Check the model files and selected settings; details are shown in the run console.'
            print(error, file=sys.stderr, flush=True)
        raise gr.Error(text(message, language)) from error
    except RuntimeError as error:
        import traceback
        traceback.print_exc()
        raise gr.Error(text('推理未完成。可选择 CPU 重试，详细原因已记录在运行窗口。', language)) from error


def run_web_prediction(image_rgb, model_name, confidence, device, language='zh'):
    _, report = user_error(analyze_image, image_rgb, model_name, confidence, device, language=language)
    report = localized_record(report, language)
    status = prediction_status(report, language)
    return image_output(render_boxes(image_rgb, report)), status, report, export_json(report)


def run_comparison(image_rgb, confidence, device, language='zh'):
    _, _, rows, record, path = user_error(compare_images, image_rgb, confidence, device, language=language)
    if language == 'en':
        record = localized_record(record, language)
        path = export_json(record, 'road-comparison-')
    left, right = [render_boxes(image_rgb, report) for report in record['reports']]
    return image_output(left), image_output(right), rows, record, path


def prediction_status(report, language):
    review = report['review']
    return text('复核优先级 {priority} · 检测框 {count} · 设备 {device} · 预测调用 {ms} ms', language,
                priority=review['priority_level'], count=review['total_distress_count'], device=report['device'],
                ms=report['predict_call_ms_including_first_call'])


def run_analysis(image_rgb, mode, model, confidence, device, categories, language='zh'):
    if mode == MODES[0]:
        _, _, rows, record, path = run_comparison(image_rgb, confidence, device, language)
        left, right = [render_boxes(image_rgb, report, categories) for report in record['reports']]
        return (image_output(left), image_output(right), None, summary_cards(record, language), compare_detections(record, language), rows,
                record, gr.File(value=path, label=text('下载双模型 JSON', language), visible=True), record, text('对比完成 · 下方可查看模型分歧与完整记录。', language), 1)
    _, status, report, path = run_web_prediction(image_rgb, model, confidence, device, language)
    return (None, None, image_output(render_boxes(image_rgb, report, categories)), summary_cards(report, language), [], [],
            report, gr.File(value=path, label=text('下载单模型 JSON', language), visible=True), report, status, 1)


def clear_results(language='zh'):
    return None, None, None, empty_summary(language), [], [], None, gr.File(value=None, visible=False), None, text('等待运行', language), 1


def render_visible(image_rgb, record, categories):
    if image_rgb is None or record is None:
        return None, None, None
    if 'reports' in record:
        left, right = [render_boxes(image_rgb, report, categories) for report in record['reports']]
        return image_output(left), image_output(right), None
    return None, None, image_output(render_boxes(image_rgb, record, categories))


def update_cases(country, category, language='zh'):
    choices = case_choices(filter_cases(read_cases(), country, category), language)
    return gr.Dropdown(choices=choices, value=choices[0][1] if choices else None)


def load_example(case_id, language='zh'):
    image, reference, note = user_error(load_case, case_id, language, language=language)
    return image_output(image), image_output(reference), note


def load_example_view(case_id, language):
    return *load_example(case_id, language), case_id


def gallery_items(cases, language='zh'):
    return [(str(CASE_DIR / item['image']), f"{item['id']} · {', '.join(item['classes']) or text('无四类目标标注', language)}") for item in cases]


def update_gallery(country, category, language='zh'):
    cases = filter_cases(read_cases(), country, category)
    return gallery_items(cases, language), [item['id'] for item in cases], text('当前筛选共 **{count} 张**，点击缩略图可载入检测工作台。', language, count=len(cases))


def select_case(case_ids, language, event: gr.SelectData):
    if not isinstance(event.index, int) or not 0 <= event.index < len(case_ids):
        raise gr.Error(text('样例已变化，请重新选择图片。', language))
    case_id = case_ids[event.index]
    image, reference, note = load_example(case_id, language)
    return image, reference, note, gr.Tabs(selected='workspace'), case_id, case_id


def change_mode(mode):
    comparison = mode == MODES[0]
    return (gr.Row(visible=comparison), gr.Image(visible=not comparison), gr.Dropdown(visible=not comparison),
            gr.Accordion(visible=comparison), gr.Slider(visible=comparison))


def banner(language, device_label, count):
    t = lambda message, **values: text(message, language, **values)
    return (f'<div class="app-banner"><div><span class="eyebrow">{t("ROAD INSIGHT / 道路巡检")}</span>'
            f'<h1>{t("道路损伤检测与人工复核")}</h1><p>{t("观察道路细节，比较模型判断，追踪实验依据。")}</p></div>'
            f'<div class="header-badges"><span class="online">● {t("{device} 可用", device=escape(device_label))}</span>'
            f'<span>{t("{count} 张原始样例", count=count)}</span><span>{t("4 类目标")}</span></div></div>')


def legend(language):
    return '<div class="legend">'+''.join(f'<span><i style="background:{CLASS_COLORS[code]}"></i>{code} {text(name, language)}</span>' for code, name in CLASS_LABELS.items())+'</div>'


def library_intro(language):
    return ('<div class="section-intro"><span class="eyebrow">' + text('CASE LIBRARY / 原始样例', language)
            + '</span><h2>' + text('从不同国家与病害出发', language) + '</h2><p>'
            + text('图片来自组内提供的原始数据包，保留原标注、原划分和来源哈希。', language) + '</p></div>')


def refresh_record(record, language):
    """Localize an existing result without calling the detector or altering the image."""
    if not record:
        return empty_summary(language), [], None, gr.File(value=None, visible=False), None, text('等待运行', language)
    record = localized_record(record, language)
    comparison = 'reports' in record
    return (summary_cards(record, language), compare_detections(record, language) if comparison else [], record,
            gr.File(value=export_json(record), label=text('下载双模型 JSON' if comparison else '下载单模型 JSON', language), visible=True),
            record, text('对比完成 · 下方可查看模型分歧与完整记录。', language) if comparison else prediction_status(record, language))


def build_app(default_device='auto'):
    t = Localizer()
    cases = read_cases()
    devices = [(t('自动选择'), 'auto'), *available_devices(torch)]
    actual = resolve_device(default_device, torch)
    label = next(label for label, value in available_devices(torch) if value == actual)
    with gr.Blocks(title='Road Damage Detection & Review', fill_width=True) as app:
        with gr.Row(elem_id='language-bar'):
            gr.Markdown(t('CDS540 / ROAD INSIGHT'), elem_classes='project-mark')
            language = gr.Radio([('English', 'en'), ('中文', 'zh')], value='en', show_label=False,
                                container=False, elem_id='language-switch', min_width=200, scale=0)
        gr.HTML(t.html(lambda lang: banner(lang, label, len(cases))), elem_id='app-header')
        record_state = gr.State(None)
        source_case = gr.State(None)
        gallery_ids = gr.State([item['id'] for item in cases])
        with gr.Tabs(selected='workspace', elem_id='main-pages') as pages:
            with t.tab('检测工作台', id='workspace'):
                with gr.Row(elem_id='workspace-layout'):
                    with gr.Column(scale=1, min_width=260, elem_id='control-panel'):
                        gr.Markdown(t('### 选择道路图片'))
                        image = gr.Image(label=t('上传或拖入原图'), type='numpy', image_mode='RGB', format='png', height=170,
                                         elem_id='source-image', sources=['upload', 'clipboard'], placeholder=text('拖入道路图片\n或点击上传'))
                        quick_case = gr.Dropdown(case_choices(cases, 'en'), value=cases[0]['id'] if cases else None, label=t('快速选择样例'), filterable=True)
                        with gr.Row():
                            load = gr.Button(t('载入样例'), interactive=bool(cases), elem_id='quick-load', min_width=90)
                            browse = gr.Button(t('浏览案例库'), min_width=90)
                        gr.Markdown(t('### 设置检测方式'))
                        mode = gr.Radio([(t(value), value) for value in MODES], value=MODES[0], label=t('检测模式'), elem_id='analysis-mode')
                        model = gr.Dropdown(list(MODELS), value=list(MODELS)[0], label=t('选择模型'), visible=False)
                        confidence = gr.Slider(.05, .60, value=.25, step=.05, label=t('置信度阈值'))
                        with gr.Accordion(t('设备与图层'), open=False):
                            device = gr.Dropdown(devices, value=default_device, label=t('运行设备'))
                            categories = gr.CheckboxGroup([(t.pair(f'{code} {text(name)}', f'{code} {name}'), code) for code, name in CLASS_LABELS.items()],
                                                          value=list(CLASS_LABELS), label=t('显示的检测类别'))
                            gr.Markdown(t('图层筛选只影响显示，统计保留完整检测结果。'), elem_classes='small-note')
                        run = gr.Button(t('开始分析'), variant='primary', elem_id='run-analysis')
                        status = gr.Markdown(t('等待运行'), elem_classes='small-note')
                        inspect_result = gr.Button(t.pair('Investigate this image', '在分析区研究这张图'), elem_id='inspect-result')
                    with gr.Column(scale=3, min_width=560, elem_id='result-panel'):
                        summary = gr.HTML(empty_summary('en'))
                        gr.HTML(t.html(legend))
                        with gr.Row(elem_id='comparison-pair') as pair:
                            left = gr.Image(label=t('YOLO11n · 基线模型'), type='numpy', format='png', interactive=False, height=390,
                                            elem_id='comparison-n', elem_classes='result-image', buttons=['fullscreen', 'download'])
                            right = gr.Image(label=t('YOLO11s · 对照模型'), type='numpy', format='png', interactive=False, height=390,
                                             elem_id='comparison-s', elem_classes='result-image', buttons=['fullscreen', 'download'])
                        overlay = gr.Image(label=t('检测结果'), type='numpy', format='png', interactive=False, height=420,
                                           visible=False, elem_classes='result-image', buttons=['fullscreen', 'download'])
                        zoom = gr.Slider(1, 3, value=1, step=.25, label=t('双图同步缩放 · 放大后可拖动，双击图像归位'))
                        with gr.Accordion(t('原图与标注参考'), open=False):
                            sample_note = gr.Markdown(t('选择样例后可查看原始标注。'), elem_id='source-note')
                            reference = gr.Image(label=t('青色框为原始标注'), type='numpy', format='png', interactive=False, height=340)
                        with gr.Accordion(t('两模型的检出差异'), open=False) as difference_panel:
                            gr.Markdown(t('同类检测框按 IoU ≥ 0.5 配对。单侧检出可作为人工核对线索。'), elem_classes='small-note')
                            differences = t.dataframe(headers=['类别', '检出情况', 'YOLO11n 置信度', 'YOLO11s 置信度', '框 IoU'],
                                                       interactive=False, label=t('逐框比较'), max_height=260)
                        with gr.Accordion(t('统计、运行条件与完整记录'), open=False):
                            table = t.dataframe(headers=TABLE_HEADERS, interactive=False, label=t('本次调用记录'), max_height=180)
                            gr.Markdown(t('首次调用包含初始化；现有两份权重的训练预算不同。正式比较请结合统一协议的实验记录。'), elem_classes='small-note')
                            details = gr.JSON(label=t('完整记录'))
                        download = gr.File(label=t('下载本次分析 JSON'), visible=False)
                        gr.Markdown(t('复核优先级用于人工排序。请结合原图和标注检查检出与漏检情况。'), elem_classes='notice')
            with t.tab('实验分析', id='experiments'):
                experiments = build_experiments(t, language)
            with t.tab('案例库', id='library'):
                gr.HTML(t.html(library_intro))
                with gr.Row():
                    country = gr.Dropdown([(t(name), value) for name, value in [('全部国家', '全部'), ('捷克', 'Czech'), ('印度', 'India'), ('美国', 'United_States')]], value='全部', label=t('国家'))
                    category = gr.Dropdown([(t('全部类别'), '全部'), *[(t.pair(f'{code} · {text(name)}', f'{code} · {name}'), code) for code, name in CLASS_LABELS.items()],
                                            (t('无四类目标标注'), '无四类目标标注')], value='全部', label=t('原标注类别'))
                gallery_note = gr.Markdown(text('当前共 **{count} 张**，点击缩略图载入检测工作台。', count=len(cases)))
                if not cases:
                    gr.Markdown(t('暂未载入案例，可以先在工作台上传自己的原始道路图片。'))
                gallery = gr.Gallery(gallery_items(cases, 'en'), columns=4, rows=3, height=600, object_fit='cover',
                                     preview=False, allow_preview=False, interactive=False, label=t('原始道路样例'), elem_id='case-gallery')
                gr.Markdown(t('案例按原始标注筛选。该集合经过分层选样，总体指标请查看完整评估记录。'), elem_classes='notice')
        gr.HTML(t.html(lambda lang: '<div class="app-footnote">' + text('CDS540 · Road Damage Detection　/　当前页面用于检测、实验观察与人工复核。', lang) + '</div>'))

        inspect_result.click(experiments['import'], [image, source_case, language],
                             [experiments['image'], experiments['source'], experiments['state'], experiments['selection']], api_name=False).then(
            lambda: gr.Tabs(selected='experiments'), outputs=pages, api_name=False)

        result_outputs = [left, right, overlay, summary, differences, table, details, download, record_state, status, zoom]
        analysis_event = run.click(run_analysis, [image, mode, model, confidence, device, categories, language], result_outputs,
                                   api_name='analyze_view', concurrency_id='road-inference', concurrency_limit=1)
        load.click(load_example_view, [quick_case, language], [image, reference, sample_note, source_case], api_name='load_example_view')
        browse.click(lambda: gr.Tabs(selected='library'), outputs=pages, api_name=False)
        gallery.select(select_case, [gallery_ids, language], [image, reference, sample_note, pages, quick_case, source_case], api_name=False)
        country.change(update_cases, [country, category, language], quick_case, api_name='filter_cases_view').then(
            update_gallery, [country, category, language], [gallery, gallery_ids, gallery_note], api_name=False)
        category.change(update_cases, [country, category, language], quick_case, api_name=False).then(
            update_gallery, [country, category, language], [gallery, gallery_ids, gallery_note], api_name=False)
        mode.change(change_mode, mode, [pair, overlay, model, difference_panel, zoom], cancels=[analysis_event], api_name=False).then(clear_results, language, result_outputs, api_name=False)
        for component in (image, confidence, device, model):
            component.change(clear_results, language, result_outputs, cancels=[analysis_event], api_name=False)
        image.upload(lambda lang: (None, text('自选图片 · 当前没有关联原始标注。', lang), None), language, [reference, sample_note, source_case], api_name=False)
        image.clear(lambda lang: (None, text('选择样例后可查看原始标注。', lang), None), language, [reference, sample_note, source_case], api_name=False)
        categories.change(render_visible, [image, record_state, categories], [left, right, overlay], api_name=False)
        zoom.change(fn=None, inputs=zoom, js=ZOOM_SCRIPT, api_name=False)

        def switch_language(lang, record, loaded_case, source, selected_case, current_country, current_category):
            cards, matches, full_record, file, saved_record, message = refresh_record(record, lang)
            current_cases = filter_cases(read_cases(), current_country, current_category)
            choices = case_choices(current_cases, lang)
            note = case_note(loaded_case, lang) if source is not None else text('选择样例后可查看原始标注。', lang)
            headers = [gr.update(headers=[t.resolve(header, lang) for header in titles]) for _, titles in t.tables]
            # Updating only headers on the match table would preserve its old-language rows.
            headers[0]['value'] = matches
            return (cards, full_record, file, saved_record, message, note,
                    gr.update(choices=choices, value=selected_case), gallery_items(current_cases, lang),
                    text('当前筛选共 **{count} 张**，点击缩略图可载入检测工作台。', lang, count=len(current_cases)), *headers,
                    *[gr.Tab(label=t.resolve(title, lang)) for _, title in t.tabs], gr.update(placeholder=text('拖入道路图片\n或点击上传', lang)))

        language.change(switch_language, [language, record_state, source_case, image, quick_case, country, category],
                        [summary, details, download, record_state, status, sample_note, quick_case, gallery, gallery_note,
                         *[component for component, _ in t.tables], *[component for component, _ in t.tabs], image],
                        js='(...args) => { window.roadReviewLocale(args[0]); return args; }',
                        api_name=False, concurrency_id='road-inference', concurrency_limit=1)
        # Lazy Gradio components can mount after the startup JS. Reapply the
        # English default once the whole UI is ready, including native upload text.
        app.load(fn=None, js="async () => { const {changeLocale} = await import(new URL('"
                 + locale_module_url() + "', document.baseURI).href); changeLocale('en'); return []; }", api_name=False)

        # Preserve the existing public API shapes for scripts and group members.
        old_analysis = gr.Button(visible=False)
        old_analysis.click(run_analysis, [image, mode, model, confidence, device, categories], result_outputs,
                           api_name='analyze', concurrency_id='road-inference', concurrency_limit=1)
        old_load = gr.Button(visible=False)
        old_load.click(load_example, quick_case, [image, reference, sample_note], api_name='load_example')
        old_filter = gr.Button(visible=False)
        old_filter.click(update_cases, [country, category], quick_case, api_name='filter_cases')
        old_single = gr.Button(visible=False)
        old_single_outputs = [gr.Image(visible=False, format='png'), gr.Markdown(visible=False), gr.JSON(visible=False), gr.File(visible=False)]
        old_single.click(run_web_prediction, [image, model, confidence, device], old_single_outputs,
                         api_name='run_web_prediction', concurrency_id='road-inference', concurrency_limit=1)
        old_comparison = gr.Button(visible=False)
        old_compare_outputs = [gr.Image(visible=False, format='png'), gr.Image(visible=False, format='png'),
                               gr.Dataframe(headers=TABLE_HEADERS, visible=False), gr.JSON(visible=False), gr.File(visible=False)]
        old_comparison.click(run_comparison, [image, confidence, device], old_compare_outputs,
                             api_name='run_comparison', concurrency_id='road-inference', concurrency_limit=1)
    app.review_i18n = t.i18n
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int)
    parser.add_argument('--device', choices=['auto', 'cpu', '0', 'mps'])
    parser.add_argument('--strict-port', action='store_true')
    parser.add_argument('--open-browser', action='store_true')
    args = parser.parse_args()
    try:
        config = load_config()
        preferred = config['port'] if args.port is None else args.port
        if not 1 <= preferred <= 65535:
            parser.error('--port 必须在 1 到 65535 之间。')
        port = select_port(preferred, config['auto_port'] and not args.strict_port)
        if port != preferred:
            print(f'端口 {preferred} 已被占用，本次使用 {port}。', flush=True)
        app = build_app(args.device or config['device'])
        print(f'实际解释器：{sys.executable}\n演示地址：http://127.0.0.1:{port}', flush=True)
        theme = gr.themes.Soft(primary_hue='blue', secondary_hue='teal', neutral_hue='slate')
        app.launch(server_name='127.0.0.1', server_port=port, share=False, inbrowser=args.open_browser,
                   allowed_paths=[str(EXPORT_DIR), str(CASE_DIR), *experiment_files()],
                   theme=theme, css_paths=ROOT / 'demo/dashboard.css',
                   js=(ROOT / 'demo/dashboard.js').read_text(encoding='utf-8').replace('__GRADIO_I18N_MODULE__', locale_module_url()),
                   i18n=app.review_i18n, footer_links=[])
    except (ValueError, OSError) as error:
        parser.exit(1, f'启动未完成：{error}\n')


if __name__ == '__main__':
    main()
