"""Correct the pose-scope test with actually camera-facing articulation.

Positive yaw rotates the +x muzzle toward -z, away from our +z camera. The
original pose_only therefore does NOT validate the toward-camera instruction.
Original shape-task error labels remain valid. Never overwrite old results.
"""
import hashlib
import json
from pathlib import Path

from dataset import render,HERE


def main():
    original=json.loads((HERE/'outputs/dataset_v2/manifest.json').read_text())
    out=HERE/'outputs/dataset_challenge';out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():raise RuntimeError('Immutable challenge already exists')
    records=[]
    for row in original['records']:
        record=dict(scene=row['scene'],seed=row['seed'],split=row['split'],variants={})
        record['variants']['source']=row['variants']['source']
        for name,factors in dict(pose_front_only=dict(head=-.85),
                                 wrong_head_front=dict(color=1.,morph=1.,head=-.85),
                                 wrong_heading_front=dict(color=1.,morph=1.,heading=-.85)).items():
            p=out/f"scene{row['scene']:02d}"/f'{name}.png';p.parent.mkdir(exist_ok=True)
            render(row['seed'],**factors).save(p)
            record['variants'][name]=dict(file=str(p.relative_to(HERE)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),factors=factors)
        records.append(record)
    manifest=dict(records=records,prompts=original['prompts'],
        correction='Old pose_only +0.85 points away and is invalid for toward-camera task selectivity; use pose_front_only -0.85.',
        scope='Posthoc direction challenge on the same v2 sources; not a new independent holdout.',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('CHALLENGE_COMPLETE',len(records),3*len(records),'new images',flush=True)


if __name__=='__main__':main()
