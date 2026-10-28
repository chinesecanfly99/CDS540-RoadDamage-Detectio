"""Run the demo with the configured project environment from any working directory."""
import argparse
from pathlib import Path
import sys
from scripts.runtime_bootstrap import configured_python, run_process

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-browser', action='store_true')
    args, remaining = parser.parse_known_args()
    if not args.no_browser:
        remaining.append('--open-browser')
    try:
        return run_process(configured_python(ROOT), ROOT / 'demo/web_app.py', remaining)
    except (ValueError, OSError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
