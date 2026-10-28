import json
from pathlib import Path
import socket
import tempfile
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.demo_runtime import available_devices, load_config, resolve_device, select_port
from scripts.runtime_bootstrap import configured_python
from scripts.prepare_demo_samples import select_cases


def fake_torch(cuda=False, mps=False):
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda),
                           backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)))


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def write_config(self, value):
        path = self.root / '.local/demo.json'
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')

    def test_missing_config_creates_local_defaults(self):
        self.assertEqual(load_config(self.root)['port'], 7860)
        self.assertTrue((self.root / '.local/demo.json').is_file())

    def test_existing_port_is_retained(self):
        self.write_config({'port': 7866})
        self.assertEqual(load_config(self.root)['port'], 7866)

    def test_invalid_configuration_has_readable_error(self):
        for value in ({'port': True}, {'port': 0}, {'port': 65536}, {'port': '7860'},
                      {'auto_port': 1}, {'device': []}, {'device': 'cuda99'}, {'unknown': 1}, []):
            with self.subTest(value=value):
                self.write_config(value)
                with self.assertRaises(ValueError):
                    load_config(self.root)

    def test_occupied_port_uses_next_available_and_strict_mode_fails(self):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            listener.listen()
            port = listener.getsockname()[1]
            if port == 65535:
                self.skipTest('OS selected the final port')
            self.assertGreater(select_port(port), port)
            with self.assertRaises(OSError):
                select_port(port, False)

    def test_auto_device_prefers_available_acceleration(self):
        self.assertEqual(resolve_device('auto', fake_torch()), 'cpu')
        self.assertEqual(resolve_device('auto', fake_torch(mps=True)), 'mps')
        self.assertEqual(resolve_device('auto', fake_torch(cuda=True, mps=True)), '0')
        self.assertEqual(available_devices(fake_torch()), [('CPU', 'cpu')])

    def test_explicit_unavailable_gpu_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_device('0', fake_torch())
        self.assertEqual(resolve_device('cpu', fake_torch(cuda=True)), 'cpu')

    def test_relative_interpreter_pointer_independent_of_cwd(self):
        path = self.root / '.local/env with spaces/python'
        path.parent.mkdir(parents=True)
        path.touch()
        (self.root / '.local/runtime-env.json').write_text(json.dumps({'python': '.local/env with spaces/python'}))
        self.assertEqual(configured_python(self.root), path.absolute())

    def test_interpreter_symlink_is_not_resolved_to_base_environment(self):
        target = self.root / 'base-python'
        target.touch()
        link = self.root / '.local/python'
        link.parent.mkdir()
        try:
            link.symlink_to(target)
        except OSError:
            self.skipTest('Symlink creation requires OS privilege')
        (self.root / '.local/runtime-env.json').write_text(json.dumps({'python': '.local/python'}))
        self.assertEqual(configured_python(self.root), link.absolute())

    def test_missing_or_invalid_interpreter_pointer_fails(self):
        folder = self.root / '.local'
        folder.mkdir()
        for value in ([], {}, {'python': 'missing/python'}):
            with self.subTest(value=value):
                (folder / 'runtime-env.json').write_text(json.dumps(value))
                with self.assertRaises(RuntimeError):
                    configured_python(self.root)

    def test_case_selection_preserves_class_coverage_and_is_deterministic(self):
        values = [{'id': str(i), 'classes': [name] if name else []}
                  for i, name in enumerate(['D00', 'D10', 'D20', 'D40', '', 'D00', 'D10', 'D20', 'D40', ''])]
        chosen = select_cases(values, 5)
        self.assertEqual(chosen, select_cases(list(reversed(values)), 5))
        self.assertEqual(len({item['id'] for item in chosen}), 5)
        self.assertEqual({name for item in chosen for name in item['classes']}, {'D00', 'D10', 'D20', 'D40'})


if __name__ == '__main__':
    unittest.main()
