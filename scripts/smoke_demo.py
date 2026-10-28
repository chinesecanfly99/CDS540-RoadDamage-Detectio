"""Start a temporary server and verify both model APIs and downloadable JSON."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.runtime_bootstrap import handoff_if_needed

if __name__ == '__main__':
    handoff_if_needed(ROOT, Path(__file__), sys.argv[1:])

import psutil
from gradio_client import Client, handle_file
from PIL import Image
from demo.cases import CASE_DIR, read_cases


def stop_owned_tree(process):
    try:
        parent = psutil.Process(process.pid)
        targets = parent.children(recursive=True)[::-1] + [parent]
        for target in targets:
            try:
                target.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(targets, timeout=5)
        for target in alive:
            target.kill()
    except psutil.NoSuchProcess:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', '0', 'mps'])
    parser.add_argument('--output', type=Path, default=ROOT / '.local/demo-smoke')
    args = parser.parse_args()
    cases = read_cases()
    source = args.source or (CASE_DIR / cases[0]['image'] if cases else None)
    if source is None or not source.is_file():
        parser.error('请用 --source 指定一张原图，或先生成演示样例库。')
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.update(NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    env = dict(os.environ, GRADIO_ANALYTICS_ENABLED='False', PYTHONUTF8='1', RDD_SKIP_HANDOFF='1')
    log_path = args.output / 'server.log'
    with log_path.open('wb') as log:
        process = subprocess.Popen([sys.executable, str(ROOT / 'demo/web_app.py'), '--port', str(port),
                                    '--strict-port', '--device', args.device], cwd=ROOT / 'demo', env=env,
                                    stdout=log, stderr=log)
        try:
            for _ in range(200):
                try:
                    with opener.open(url + '/config', timeout=1) as response:
                        json.load(response)
                    break
                except (OSError, ValueError):
                    if process.poll() is not None:
                        raise RuntimeError(f'服务启动失败，请查看 {log_path}')
                    time.sleep(.2)
            else:
                raise RuntimeError(f'服务启动超时，请查看 {log_path}')
            client = Client(url, verbose=False)
            if cases:
                sample = client.predict(cases[0]['id'], api_name='/load_example')
                with Image.open(CASE_DIR / cases[0]['image']) as original, Image.open(sample[0]) as returned:
                    if original.convert('RGB').tobytes() != returned.convert('RGB').tobytes():
                        raise RuntimeError('样例传回浏览器时像素发生变化，请检查是否使用有损输出格式。')
            output = client.predict(handle_file(str(source.absolute())), .25, args.device, api_name='/run_comparison')
            report = output[3]
            if len(report['reports']) != 2 or len(output[2]['data']) != 2:
                raise RuntimeError('未得到完整双模型对比结果。')
            if not all(Path(output[i]).is_file() for i in (0, 1, 4)):
                raise RuntimeError('叠框图或 JSON 未能下载。')
            if json.loads(Path(output[4]).read_text(encoding='utf-8')) != report:
                raise RuntimeError('下载的 JSON 与界面返回内容不一致。')
            forbidden = url + '/gradio_api/file=' + urllib.parse.quote(str(ROOT / 'models/baseline_best.pt'), safe='')
            try:
                opener.open(forbidden, timeout=5)
                raise RuntimeError('模型文件意外允许网页下载，请检查目录授权。')
            except urllib.error.HTTPError as error:
                if error.code != 403:
                    raise
            summary = {'status': 'passed', 'python': sys.executable, 'source': str(source),
                       'working_directory': 'demo', 'json_download': 'matches HTTP response',
                       'model_file_access': '403 blocked',
                       'sample_pixel_roundtrip': 'lossless' if cases else 'no library available',
                       'models': [{'name': r['model'], 'device': r['device'],
                                   'detections': r['review']['total_distress_count']} for r in report['reports']]}
            (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        finally:
            stop_owned_tree(process)


if __name__ == '__main__':
    main()
