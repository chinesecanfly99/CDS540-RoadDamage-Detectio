"""Shared, working-directory-independent configuration and device selection."""
import json
from pathlib import Path
import socket

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {'port': 7860, 'auto_port': True, 'device': 'auto'}


def load_config(root=ROOT, create=True):
    path = Path(root) / '.local/demo.json'
    values = dict(DEFAULTS)
    if path.exists():
        try:
            saved = json.loads(path.read_text(encoding='utf-8-sig'))
        except (ValueError, OSError) as error:
            raise ValueError(f'无法读取配置 {path}: {error}') from error
        if not isinstance(saved, dict) or set(saved) - set(DEFAULTS):
            raise ValueError(f'{path} 只接受 port、auto_port、device 三项配置。')
        values.update(saved)
    elif create:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open('x', encoding='utf-8') as output:
                json.dump(values, output, ensure_ascii=False, indent=2)
        except FileExistsError:
            return load_config(root, create=False)
    if isinstance(values['port'], bool) or not isinstance(values['port'], int) or not 1 <= values['port'] <= 65535:
        raise ValueError('port 必须是 1 到 65535 之间的整数。')
    if not isinstance(values['auto_port'], bool):
        raise ValueError('auto_port 必须是 true 或 false。')
    if not isinstance(values['device'], str) or values['device'] not in {'auto', 'cpu', '0', 'mps'}:
        raise ValueError('device 必须是 auto、cpu、0 或 mps。')
    return values


def available_devices(torch):
    devices = [('CPU', 'cpu')]
    if torch.cuda.is_available():
        devices.append(('NVIDIA GPU (CUDA:0)', '0'))
    mps = getattr(getattr(torch, 'backends', None), 'mps', None)
    if mps is not None and mps.is_available():
        devices.append(('Apple GPU (MPS)', 'mps'))
    return devices


def resolve_device(requested, torch):
    devices = dict((value, label) for label, value in available_devices(torch))
    if requested == 'auto':
        return '0' if '0' in devices else ('mps' if 'mps' in devices else 'cpu')
    if requested not in {'cpu', '0', 'mps'}:
        raise ValueError('请选择有效的运行设备。')
    if requested not in devices:
        raise ValueError(f'当前 Python 环境不能使用 {requested}；请选择自动或 CPU，GPU 需先安装对应版 PyTorch。')
    return requested


def select_port(preferred, auto_port=True):
    candidates = range(preferred, min(65536, preferred + 20)) if auto_port else [preferred]
    for port in candidates:
        try:
            with socket.socket() as listener:
                if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
                    listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                listener.bind(('127.0.0.1', port))
            return port
        except OSError:
            pass
    raise OSError(f'端口 {preferred} 被占用；请修改 .local/demo.json 的 port，或启用 auto_port。')
