"""Deterministic toy renderer; its factors are offline diagnostic truth only."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
SIZE = 512
COLORS = [(0.62, 0.23, 0.11), (0.13, 0.43, 0.23), (0.18, 0.30, 0.65), (0.65, 0.47, 0.12)]


def render(objects, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:SIZE, :SIZE].astype(np.float64) / SIZE
    background = np.stack([0.55 + 0.035 * yy + 0.008 * np.sin(xx * 25),
                           0.58 + 0.025 * yy, 0.60 + 0.02 * yy], -1)
    background += rng.normal(0, 0.0006, (SIZE, SIZE, 1))
    image = background.copy()
    # The renderer contains visibility tests; none are passed to model observers.
    for obj in objects:
        cx, cy, scale = obj['x'], obj['y'], obj.get('scale', 1.0)
        dx = (xx - cx) / (0.115 * scale)
        dy = (yy - cy) / (0.19 * scale)
        body = dx * dx + dy * dy < 1
        normal_z = np.sqrt(np.maximum(1 - dx * dx - dy * dy, 0))
        shade = 0.42 + 0.46 * normal_z + 0.08 * np.maximum(-dx, 0)
        texture = 1 + 0.055 * np.sin(72 * xx + seed * 0.2) * np.cos(60 * yy)
        radiance = np.array(obj['color'])[None, None] * (shade * texture)[..., None]
        image[body] = radiance[body]
        # Distinctive beak and eye provide an explicit source-heading carrier.
        direction = obj['heading']
        hx, hy = cx + direction * 0.020 * scale, cy - 0.105 * scale
        head = ((xx - hx) / (0.074 * scale)) ** 2 + ((yy - hy) / (0.075 * scale)) ** 2 < 1
        hshade = 0.58 + 0.30 * np.sqrt(np.maximum(1 - ((xx-hx)/(0.074*scale))**2
                                                       - ((yy-hy)/(0.075*scale))**2, 0))
        head_rgb = np.array(obj['color'])[None, None] * (hshade * texture)[..., None]
        image[head] = head_rgb[head]
        pil = Image.fromarray(np.uint8(np.rint(np.clip(image, 0, 1) * 255)))
        draw = ImageDraw.Draw(pil)
        beak = [(hx + direction*.060*scale, hy-.012*scale),
                (hx + direction*.112*scale, hy+.010*scale),
                (hx + direction*.060*scale, hy+.026*scale)]
        draw.polygon([(int(x*SIZE), int(y*SIZE)) for x,y in beak], fill=(164, 112, 54))
        ex, ey = hx + direction*.028*scale, hy-.018*scale
        er = .009*scale
        draw.ellipse(((ex-er)*SIZE, (ey-er)*SIZE, (ex+er)*SIZE, (ey+er)*SIZE), fill=(17, 18, 20))
        image = np.asarray(pil).astype(np.float64) / 255
    return Image.fromarray(np.uint8(np.rint(np.clip(image, 0, 1) * 255)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', default='v1')
    ap.add_argument('--count',type=int,default=16);ap.add_argument('--fresh-heldout-start',type=int,default=8)
    args = ap.parse_args()
    out = HERE / 'outputs' / args.tag / 'dataset'
    if (out/'manifest.json').exists():
        raise RuntimeError('Dataset tag already exists; use immutable artifacts')
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for scene in range(args.count):
        rng = np.random.default_rng(6311 + scene)
        left = dict(x=float(.29+rng.uniform(-.015,.015)), y=float(.59+rng.uniform(-.015,.015)),
                    heading=-1 if scene % 2 else 1, color=list(COLORS[scene % 4]))
        right = dict(x=float(.71+rng.uniform(-.015,.015)), y=float(.59+rng.uniform(-.015,.015)),
                     heading=1 if scene % 3 else -1,
                     color=list(COLORS[scene % 4 if scene % 2 else (scene+1) % 4]))
        target = [.10,.10,.105] if scene % 4 < 2 else [.91,.91,.88]
        target_word = 'black' if scene % 4 < 2 else 'white'
        source = [left, right]
        correct = copy.deepcopy(source); correct[0]['color'] = target
        partial = copy.deepcopy(source)
        partial[0]['color'] = (0.45*np.array(target)+0.55*np.array(left['color'])).tolist()
        wrong = copy.deepcopy(source); wrong[1]['color'] = target
        both = copy.deepcopy(correct); both[1]['color'] = target
        turned = copy.deepcopy(correct); turned[0]['heading'] *= -1
        shifted = copy.deepcopy(correct); shifted[0]['x'] += .055
        duplicate = copy.deepcopy(correct)
        extra = copy.deepcopy(left); extra.update(x=left['x']+.105, y=left['y']-.075, scale=.65)
        duplicate.append(extra)
        variants = dict(source=source, correct=correct, partial=partial, wrong_object=wrong,
                        both_objects=both, turned=turned, shifted=shifted, duplicate=duplicate)
        source_word = ['red','green','blue','yellow'][scene % 4]
        right_word = ['red','green','blue','yellow'][scene % 4 if scene % 2 else (scene+1) % 4]
        base_prompt = f'Two clay toy birds on a gray table. A {source_word} bird on the left and a {right_word} bird on the right.'
        target_prompt = f'Two clay toy birds on a gray table. A {target_word} bird on the left and a {right_word} bird on the right.'
        record = dict(scene=scene, split='development' if scene < 4 else 'calibration' if scene < 8 else 'adaptation' if scene < args.fresh_heldout_start else 'heldout',
                      seed=6311+scene, identical_source_material=scene % 2 == 1,
                      source_prompt=base_prompt, target_prompt=target_prompt,
                      instruction=f'Change only the left bird to {target_word}; keep its shape and pose, the right bird and the background.',
                      variants={})
        for name, objects in variants.items():
            path = out / f'scene{scene:02d}' / f'{name}.png'; path.parent.mkdir(parents=True, exist_ok=True)
            render(objects, 6311+scene).save(path)
            record['variants'][name] = dict(file=str(path.relative_to(HERE)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                          factors=objects, task_complete=name in ['correct','both_objects','turned','shifted','duplicate'],
                                          joint_success=name == 'correct')
        rows.append(record)
    payload = dict(scope='Known restricted rendering factors; no claim of natural-image semantics or physical global illumination.',
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), records=rows)
    (out/'manifest.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')
    print('DATASET_COMPLETE', len(rows), sum(len(row['variants']) for row in rows), flush=True)


if __name__ == '__main__': main()
