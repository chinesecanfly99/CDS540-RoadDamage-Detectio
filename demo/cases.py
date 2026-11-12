"""Read a local, provenance-preserving demo case library."""
import json
from pathlib import Path

from PIL import Image, ImageDraw
import numpy as np
from demo.i18n import text

ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / '.local/demo_samples'
CLASS_LABELS = {'D00': '纵向裂缝', 'D10': '横向裂缝', 'D20': '网状裂缝', 'D40': '坑洞'}


def read_cases():
    path = CASE_DIR / 'manifest.json'
    if not path.is_file():
        return []
    saved = json.loads(path.read_text(encoding='utf-8'))
    cases = saved['cases']
    for item in cases:
        image = (CASE_DIR / item['image']).resolve()
        if not image.is_relative_to(CASE_DIR.resolve()) or not image.is_file():
            raise ValueError('样例清单包含不存在或越界的图片，请重新生成样例库。')
    return cases


def filter_cases(cases, country='全部', category='全部'):
    return [item for item in cases if (country == '全部' or item['country'] == country)
            and (category == '全部' or (not item['targets'] if category == '无四类目标标注'
                                      else category in item['classes']))]


def case_choices(cases, language='zh'):
    return [(f"{item['id']} · {','.join(item['classes']) or text('无四类目标标注', language)}", item['id']) for item in cases]


def case_note(case_id, language='zh'):
    if case_id is None:
        return text('自选图片 · 当前没有关联原始标注。', language)
    item = next((case for case in read_cases() if case['id'] == case_id), None)
    if item is None:
        return text('选择样例后可查看原始标注。', language)
    return text('**{id}** · {country} · 原划分 `{split}` · 四类目标标注 {count} 个。\n\n'
                '标注仅供个例核对；已知原数据标注仍有待复核项。此样例库经过分层挑选，不能据此推断总体精度。',
                language, id=item['id'], country=item['country'], split=item['split'], count=len(item['targets']))


def load_case(case_id, language='zh'):
    item = next((case for case in read_cases() if case['id'] == case_id), None)
    if item is None:
        raise ValueError('请先选择一个有效样例。')
    with Image.open(CASE_DIR / item['image']) as image:
        image = image.convert('RGB')
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        for target in item['targets']:
            box = target['xyxy']
            draw.rectangle(box, outline='#00d9ff', width=3)
            draw.text((box[0], max(0, box[1] - 13)), target['class_name'], fill='#00d9ff', stroke_width=1)
        original = np.array(image)
    return original, np.array(overlay), case_note(case_id, language)
