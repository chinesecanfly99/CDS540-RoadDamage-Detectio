"""Create an isolated demo environment and select it for every launch entry."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.parse
import venv

ROOT = Path(__file__).resolve().parents[1]


def run(command):
    subprocess.run([str(item) for item in command], check=True, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['cpu', 'cuda118'], default='cpu')
    parser.add_argument('--env', type=Path, default=Path('.venv'))
    parser.add_argument('--index-url', help='Optional HTTPS package index for non-PyTorch dependencies')
    args = parser.parse_args()
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        parser.error('请使用 Python 3.10–3.12，推荐 3.11。')
    if args.backend == 'cuda118' and sys.platform not in ('win32', 'linux'):
        parser.error('CUDA 安装方案仅用于有 NVIDIA 显卡的 Windows / Linux 电脑。')
    if args.index_url and urllib.parse.urlsplit(args.index_url).scheme != 'https':
        parser.error('--index-url 必须是 HTTPS 地址。')
    target = (ROOT / args.env).resolve()
    # Keep local environments under the ignored workspace directory.
    if target != ROOT / '.venv' and not target.is_relative_to(ROOT / '.local'):
        parser.error('--env 仅支持 .venv 或项目 .local 下的路径。')
    config = target / 'pyvenv.cfg'
    if target.exists():
        if not config.is_file():
            parser.error(f'{target} 已存在且不是虚拟环境，请换一个 --env 路径。')
        text = config.read_text(encoding='utf-8').lower().replace(' ', '')
        if 'include-system-site-packages=true' in text:
            parser.error('旧环境混入了全局依赖。请用 --env .local/runtime-cpu 或 .local/runtime-gpu 创建独立环境。')
    else:
        venv.EnvBuilder(with_pip=True, system_site_packages=False).create(target)
    python = target / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    index = ['--index-url', args.index_url] if args.index_url else []
    run([python, '-m', 'pip', 'install', '--upgrade', 'pip', *index])
    if sys.platform == 'darwin':
        run([python, '-m', 'pip', 'install', 'torch==2.7.1', 'torchvision==0.22.1', *index])
    else:
        profile = ROOT / 'requirements' / f'torch-{args.backend}.txt'
        run([python, '-m', 'pip', 'install', '-r', profile])
    run([python, '-m', 'pip', 'install', '-r', ROOT / 'requirements/runtime.txt', *index])
    check = [python, ROOT / 'scripts/check_environment.py']
    if args.backend == 'cuda118':
        check.append('--require-gpu')
    run(check)
    local = ROOT / '.local'
    local.mkdir(exist_ok=True)
    pointer = local / 'runtime-env.json'
    temporary = pointer.with_suffix('.tmp')
    temporary.write_text(json.dumps({'python': python.relative_to(ROOT).as_posix(),
                                     'backend': args.backend}, indent=2), encoding='utf-8')
    temporary.replace(pointer)
    print('安装与检查通过。以后运行项目根目录 run_demo.py，或双击 start_demo.cmd。')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, subprocess.CalledProcessError) as error:
        print(f'安装未完成：{error}。修复网络或安装问题后，可重跑相同命令继续。', file=sys.stderr)
        raise SystemExit(1)
