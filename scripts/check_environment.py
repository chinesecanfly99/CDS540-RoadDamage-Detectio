"""Check the selected interpreter, native operators and bundled road weights."""
import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def inspect_environment(require_gpu=False):
    from scripts.demo_runtime import available_devices
    import torch
    from torchvision.ops import nms
    os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.local/ultralytics'))
    from ultralytics import YOLO

    report = {'python': sys.executable, 'python_version': sys.version.split()[0],
              'isolated': sys.prefix != sys.base_prefix,
              'versions': {name: metadata.version(name) for name in
                           ('torch', 'torchvision', 'ultralytics', 'gradio', 'numpy', 'opencv-python')},
              'cuda_runtime': torch.version.cuda, 'devices': available_devices(torch)}
    devices = [value for _, value in report['devices']]
    if require_gpu and '0' not in devices:
        raise RuntimeError('该解释器没有可用的 CUDA。请核对安装类型、显卡驱动和解释器路径。')
    report['operator_checks'] = {}
    for device in ('cpu', 'cuda:0') if '0' in devices else ('cpu',):
        a = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device=device)
        scores = torch.tensor([.9, .8], device=device)
        assert nms(a, scores, .5).cpu().tolist() == [0]
        assert (a @ a.T).isfinite().all().item()
        report['operator_checks'][device] = 'tensor computation and torchvision NMS passed'
    report['weights'] = {}
    for name in ('baseline_best.pt', 'compare_yolo11s_best.pt'):
        path = ROOT / 'models' / name
        if not path.is_file():
            raise RuntimeError(f'缺少权重 {name}，请更新或重新克隆完整仓库。')
        model = YOLO(str(path))
        if model.names != {0: 'D00', 1: 'D10', 2: 'D20', 3: 'D40'}:
            raise RuntimeError(f'{name} 类别映射不匹配。')
        report['weights'][name] = 'load and class mapping passed'
    check = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
    report['pip_check'] = check.stdout.strip() or check.stderr.strip()
    if check.returncode:
        raise RuntimeError('依赖存在冲突：' + report['pip_check'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-gpu', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        report = inspect_environment(args.require_gpu)
    except Exception as error:
        print(f'环境检查未通过：{error}\n实际解释器：{sys.executable}', file=sys.stderr)
        return 1
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
