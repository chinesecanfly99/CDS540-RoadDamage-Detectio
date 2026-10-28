"""Interactive road review: single image, model comparison and annotated cases."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.runtime_bootstrap import handoff_if_needed

if __name__ == '__main__':
    handoff_if_needed(ROOT, Path(__file__), sys.argv[1:])

# Keep Gradio's local health probe away from system HTTP proxies.
_no_proxy = {value.strip() for value in (os.environ.get('NO_PROXY') or os.environ.get('no_proxy', '')).split(',') if value.strip()}
_no_proxy.update({'127.0.0.1', 'localhost'})
os.environ['NO_PROXY'] = os.environ['no_proxy'] = ','.join(sorted(_no_proxy))
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')

import gradio as gr
import torch
from demo.cases import CASE_DIR, CLASS_LABELS, case_choices, filter_cases, load_case, read_cases
from demo.inference import EXPORT_DIR, MODELS, analyze_image, compare_images, export_json, load_model
from scripts.demo_runtime import available_devices, load_config, resolve_device, select_port


def user_error(function, *arguments):
    try:
        return function(*arguments)
    except (ValueError, FileNotFoundError) as error:
        raise gr.Error(str(error)) from error
    except RuntimeError as error:
        # Keep full traceback in the console while giving the audience a next step.
        import traceback
        traceback.print_exc()
        raise gr.Error('推理未完成。请先选 CPU 重试，并查看控制台错误；GPU 环境可运行 scripts/check_environment.py 检查。') from error


def run_web_prediction(image_rgb, model_name, confidence, device):
    overlay, report = user_error(analyze_image, image_rgb, model_name, confidence, device)
    review = report['review']
    status = (f"**复核优先级：{review['priority_level']}** · 检测框：{review['total_distress_count']} · "
              f"设备：{report['device']} · 预测调用：{report['predict_call_ms_including_first_call']} ms\n\n"
              '无检出不表示道路完好；首次调用包含初始化耗时。')
    return overlay, status, report, export_json(report)


def run_comparison(image_rgb, confidence, device):
    return user_error(compare_images, image_rgb, confidence, device)


def update_cases(country, category):
    choices = case_choices(filter_cases(read_cases(), country, category))
    return gr.Dropdown(choices=choices, value=choices[0][1] if choices else None)


def load_example(case_id):
    return user_error(load_case, case_id)


def build_app(default_device='auto'):
    cases = read_cases()
    devices = [('自动选择', 'auto'), *available_devices(torch)]
    actual = resolve_device(default_device, torch)
    label = next(label for label, value in available_devices(torch) if value == actual)
    with gr.Blocks(title='道路损伤检测与人工复核') as app:
        gr.Markdown('# 道路损伤检测与人工复核\n道路巡检实验台 · 单图检查 / 双模型对比 / 标注参考')
        gr.Markdown(f'**当前默认设备：{label}**　|　**可用样例：{len(cases)} 张**　|　**目标：4 类道路损坏**')
        with gr.Accordion('从样例库选择道路图片', open=bool(cases)):
            with gr.Row():
                country = gr.Dropdown(['全部', 'Czech', 'India', 'United_States'], value='全部', label='国家')
                category = gr.Dropdown(['全部', *CLASS_LABELS, '无四类目标标注'], value='全部', label='标注类别')
                case_id = gr.Dropdown(case_choices(cases), value=cases[0]['id'] if cases else None, label='样例')
            load = gr.Button('载入所选样例', interactive=bool(cases))
            if not cases:
                gr.Markdown('可以直接上传自己的图片。需要样例库时，按 docs/DEMO_GUIDE.md 从原数据生成。')
        with gr.Row():
            image = gr.Image(label='原始道路图片 · 可上传或拖入', type='numpy', image_mode='RGB', format='png', height=340)
            with gr.Column():
                sample_note = gr.Markdown('上传原始图片或载入样例。样例标注供人工核对，不能作为总体准确率证明。')
                with gr.Accordion('所选样例的原始标注参考', open=False):
                    reference = gr.Image(label='青色框：原始数据标注', type='numpy', format='png', interactive=False, height=260)
                gr.Markdown('D00 纵向裂缝 · D10 横向裂缝 · D20 网状裂缝 · D40 坑洞\n\n'
                            '阈值降低可能增加召回，同时增加误报。复核优先级只用于排序，不是实际损坏严重程度。')
        with gr.Row():
            confidence = gr.Slider(.05, .60, value=.25, step=.05, label='置信度阈值（两模型共用）')
            device = gr.Dropdown(devices, value=default_device, label='运行设备')
        with gr.Tab('单模型检测'):
            model = gr.Dropdown(list(MODELS), value=list(MODELS)[0], label='选择模型')
            run = gr.Button('开始检测', variant='primary')
            overlay = gr.Image(label='检测与复核结果', type='numpy', format='png', interactive=False, height=460)
            status = gr.Markdown()
            with gr.Accordion('检测与复核详情', open=False):
                details = gr.JSON(label='完整 JSON')
            download = gr.File(label='下载本次检测 JSON')
        with gr.Tab('双模型对比'):
            gr.Markdown('同图、同阈值、同推理尺寸，依次运行两个模型。现有权重训练预算不同；框更多不代表更准确。')
            compare = gr.Button('运行两个模型', variant='primary')
            with gr.Row():
                left = gr.Image(label='YOLO11n · baseline', type='numpy', format='png', interactive=False, height=420)
                right = gr.Image(label='YOLO11s · comparison', type='numpy', format='png', interactive=False, height=420)
            table = gr.Dataframe(headers=['模型', '实际设备', '框数', 'D00', 'D10', 'D20', 'D40', '复核优先级', '单次调用 ms'],
                                 interactive=False, label='本图结果（耗时含可能的首次初始化，不能用于正式速度排名）')
            with gr.Accordion('两模型完整记录', open=False):
                comparison_details = gr.JSON()
            comparison_download = gr.File(label='下载双模型对比 JSON')
        country.change(update_cases, [country, category], case_id, api_name='filter_cases')
        category.change(update_cases, [country, category], case_id, api_name=False)
        load.click(load_example, case_id, [image, reference, sample_note], api_name='load_example')
        # Changing an upload clears sample-specific references to avoid mismatching GT.
        image.upload(lambda: (None, '当前为自选图片，没有关联的原始标注。'), outputs=[reference, sample_note], api_name=False)
        image.clear(lambda: (None, '请上传图片或载入样例。'), outputs=[reference, sample_note], api_name=False)
        run.click(run_web_prediction, [image, model, confidence, device], [overlay, status, details, download],
                  api_name='run_web_prediction', concurrency_id='road-inference', concurrency_limit=1)
        compare.click(run_comparison, [image, confidence, device], [left, right, table, comparison_details, comparison_download],
                      api_name='run_comparison', concurrency_id='road-inference', concurrency_limit=1)
        result_outputs = [overlay, status, details, download, left, right, table, comparison_details, comparison_download]
        def clear_results():
            return None, '', None, None, None, None, [], None, None
        # Never leave another image's boxes or another threshold's results visible.
        image.change(clear_results, outputs=result_outputs, api_name=False)
        confidence.change(clear_results, outputs=result_outputs, api_name=False)
        device.change(clear_results, outputs=result_outputs, api_name=False)
        model.change(lambda: (None, '', None, None), outputs=[overlay, status, details, download], api_name=False)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int)
    parser.add_argument('--device', choices=['auto', 'cpu', '0', 'mps'])
    parser.add_argument('--strict-port', action='store_true', help='Fail instead of selecting the next free port')
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
        device = args.device or config['device']
        app = build_app(device)
        print(f'实际解释器：{sys.executable}\n演示地址：http://127.0.0.1:{port}', flush=True)
        app.launch(server_name='127.0.0.1', server_port=port, share=False,
                   inbrowser=args.open_browser, allowed_paths=[str(EXPORT_DIR), str(CASE_DIR)])
    except (ValueError, OSError) as error:
        parser.exit(1, f'启动未完成：{error}\n')


if __name__ == '__main__':
    main()
