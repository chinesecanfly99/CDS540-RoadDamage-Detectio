"""Immutable per-output PNGs avoid concurrent Gradio writes to a shared hash path."""
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / '.local/web_exports/images'


def image_output(value):
    if value is None or isinstance(value, (str, Path)):
        return value
    image = Image.fromarray(np.ascontiguousarray(value)) if isinstance(value, np.ndarray) else value
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    # Return only after the unique file is complete and closed. Gradio preserves
    # this filename when copying to its cache, even for pixel-identical outputs.
    with tempfile.NamedTemporaryFile(dir=IMAGE_DIR, prefix='view-', suffix='.png', delete=False) as handle:
        image.save(handle, format='PNG')
        path = handle.name
    return path
