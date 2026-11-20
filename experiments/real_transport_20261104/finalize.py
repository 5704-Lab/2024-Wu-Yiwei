"""Final paired-noise ablation, accepted-step checks, and artifact manifest."""
import ast
import csv
import numpy as np
from PIL import Image,ImageDraw
from common import *


def main():
    out=HERE/'outputs/final_audit_v1'
    if (out/'result.json').exists():raise RuntimeError('Immutable final audit exists')
    data=json.loads((HERE/'outputs/repair_v3_shared/summary.json').read_text())['results']
    rgb=json.loads((HERE/'outputs/repair_shared_rgb_audit_v1/summary.json').read_text())['rows']
    rgb={r['pair']:r for r in rgb}
    checks=[]
    def check(name,condition):
        checks.append(dict(name=name,passed=bool(condition)))
        if not condition:raise AssertionError(name)
    rows=[];calls=0;accepted=0
    for r in data:
        calls+=24+4*len(r['steps'])+4*sum(len(s.get('trials',[])) for s in r['steps'])
        for step in r['steps']:
            if step['status']!='accepted':continue
            trial=next(t for t in step['trials'] if abs(t['correction_degrees']-step['accepted_correction_degrees'])<1e-10)
            check(r['pair']+'_accepted_'+str(step['step']),trial['passed'] and abs(trial['observer']['frame']['rotation_deg'])<=step['cap_degrees']+1e-9 and abs(trial['luminance_ratio']-.6)<=.05+1e-8)
            accepted+=1
        before=r['evaluations']['unchanged']['reference_center_rmse'];after=r['evaluations']['controlled']['reference_center_rmse']
        check(r['pair']+'_rgb_improved',after<before and abs(rgb[r['pair']]['scores']['controlled']['offline_residual_angle_deg'])<12)
        rows.append(dict(pair=r['pair'],accepted_steps=r['accepted_steps'],native_angle=r['final_observer']['frame']['rotation_deg'],
            independent_probe_bank_angle=r['final_independent_audit']['frame']['rotation_deg'],
            rgb_angle=rgb[r['pair']]['scores']['controlled']['offline_residual_angle_deg'],rmse_reduction=1-after/before,
            luminance_ratio=r['evaluations']['controlled']['luminance_ratio']))
    out.mkdir(parents=True,exist_ok=True)
    with (out/'paired_noise_recovery.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    size=224;canvas=Image.new('RGB',(size*4,(size+30)*len(rows)+32),'white');draw=ImageDraw.Draw(canvas)
    labels=['Dark + added rotation','Independent-noise feedback','Shared-noise feedback','Dark reference: EVAL ONLY']
    for j,label in enumerate(labels):draw.text((j*size+2,5),label,fill='black')
    for i,row in enumerate(rows):
        pair=row['pair'];y=32+i*(size+30);draw.text((3,y),pair,fill='black')
        paths=[HERE/'outputs/repair_v2'/pair/'initial.png',HERE/'outputs/repair_v2'/pair/'controlled.png',
               HERE/'outputs/repair_v3_shared'/pair/'controlled.png',HERE/'outputs/repair_v2'/pair/'offline_reference_dark.png']
        for j,p in enumerate(paths):canvas.paste(Image.open(p).convert('RGB').resize((size,size)),(j*size,y+24))
    canvas.save(out/'shared_and_independent_recovery.png')
    for p in HERE.glob('*.py'):ast.parse(p.read_text())
    snapshot=out/'code_snapshot';snapshot.mkdir()
    for p in HERE.glob('*.py'):(snapshot/p.name).write_bytes(p.read_bytes())
    write(out/'result.json',dict(checks=checks,passed=len(checks),total=len(checks),accepted_steps=accepted,
        shared_recovery_forward_calls=calls,shared_noise_results=rows,
        median_rmse_reduction=float(np.median([r['rmse_reduction'] for r in rows])),
        independent_and_shared_are_same_sources_not_independent_new_images=True,
        true_semantic_safety_certificate=False,code_sha256={p.name:sha(p) for p in HERE.glob('*.py')}))
    print('FINAL_SHARED_AUDIT',len(checks),calls,accepted,flush=True)


if __name__=='__main__':main()
