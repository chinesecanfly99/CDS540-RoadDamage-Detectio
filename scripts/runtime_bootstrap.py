"""Select the project interpreter before importing third-party packages."""
import json
import os
from pathlib import Path
import subprocess
import sys


def configured_python(root):
    root = Path(root)
    pointer = root / '.local/runtime-env.json'
    if pointer.is_file():
        saved = json.loads(pointer.read_text(encoding='utf-8-sig'))
        value = saved.get('python') if isinstance(saved, dict) else None
        if not isinstance(value, str) or not value:
            raise RuntimeError(f'无效的环境配置：{pointer}')
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = root / candidate
        if not candidate.is_file():
            raise RuntimeError(f'项目环境不存在：{candidate}。请重新执行 scripts/setup_runtime.py。')
        return candidate.absolute()
    candidate = root / ('.venv/Scripts/python.exe' if os.name == 'nt' else '.venv/bin/python')
    if candidate.is_file():
        return candidate.absolute()
    if sys.prefix != sys.base_prefix:
        return Path(sys.executable).absolute()
    raise RuntimeError('请先执行：python scripts/setup_runtime.py --backend cpu')


def run_process(python, script, arguments):
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    print(f'项目运行环境：{python}', flush=True)
    process = subprocess.Popen([str(python), str(script), *arguments], env=env)
    try:
        return process.wait()
    except KeyboardInterrupt:
        if process.poll() is None:
            process.terminate()
        try:
            return process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            return process.wait()


def handoff_if_needed(root, script, arguments):
    if os.environ.get('RDD_SKIP_HANDOFF') == '1':
        return
    pointer = Path(root) / '.local/runtime-env.json'
    if not pointer.is_file():
        return
    python = configured_python(root)
    if os.path.normcase(str(python)) != os.path.normcase(str(Path(sys.executable).absolute())):
        raise SystemExit(run_process(python, script, arguments))
