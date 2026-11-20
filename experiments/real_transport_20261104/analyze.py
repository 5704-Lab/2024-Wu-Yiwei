"""Develop -> lock -> calibrate -> held-out sources; no target reference at runtime."""
import argparse
import math
import numpy as np
from common import *

NONROT=['roundtrip','dark','warm','desaturate','stretch','translate']
ROT=['rotate_plus12','rotate_minus12','rotate_plus25','stretch_rotate']


def read(folder,pairs):
    rows=[]
    for pair in pairs:
        for p in sorted((folder/pair).glob('*.json')):
            d=json.loads(p.read_text());rows.append(d)
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['lock','calibrate','test'],required=True)
    ap.add_argument('--analysis',default='analysis_v3')
    args=ap.parse_args();out=HERE/'outputs'/args.analysis
    if args.stage=='lock':
        if (out/'selection.json').exists():raise RuntimeError('Immutable selection exists')
        rows=read(HERE/'outputs/development_weighted_v2',[p+'__00' for p in DEV])
        expected=len(DEV)*14
        assert len(rows)==expected,(len(rows),expected)
        keys=[k for k in rows[0]['results'] if k.startswith('independent/')]
        scores={}
        for k in keys:
            vals=[r['results'][k] for r in rows if r['variant'] in NONROT+ROT]
            ev=[r['evaluation'] for r in vals]
            angle=[r.get('global_frame_angle_error_deg',180.) for r in ev]
            pck=[r['pck32'] for r in ev]
            support=[r['mutual_fraction'] for r in vals]
            score=float(np.mean(angle)+10*(1-np.mean(pck))+10*max(0.,.3-np.mean(support)))
            scores[k]=dict(score=score,mean_angle_error_deg=float(np.mean(angle)),pck32=float(np.mean(pck)),mutual_fraction=float(np.mean(support)))
        primary=min(scores,key=lambda k:scores[k]['score'])
        candidates=list(dict.fromkeys([primary,primary.replace('independent/','shared/',1),
            'independent/O/l11/raw/sem0.0/nn','independent/post_adaln/l11/centered/sem0.0/nn']))
        write(out/'selection.json',dict(primary=primary,candidates=candidates,development_scores=scores,
            rule='independent noise primary; minimize mean angle error + 10*(1-PCK32) + 10*max(0,.3-mutual_fraction), only four development sources',
            protocol_scope='known whole-image affine/photometric measurement; not semantic editing completion',script_sha256=sha(__file__)))
        print('LOCKED',primary,scores[primary],flush=True);return
    selection=json.loads((out/'selection.json').read_text());keys=selection['candidates']
    if args.stage=='calibrate':
        if (out/'thresholds.json').exists():raise RuntimeError('Immutable thresholds exist')
        rows=read(HERE/'outputs/calibration_weighted_v1',[p+'__00' for p in CAL])
        assert len(rows)==len(CAL)*14
        limits={}
        for k in keys:
            v=[r['results'][k] for r in rows if r['variant'] in NONROT]
            allv=[r['results'][k] for r in rows if r['variant'] in NONROT+ROT]
            angles=[abs(r['global_frame']['rotation_deg']) for r in v if r['global_frame']['rotation_deg'] is not None]
            limits[k]=dict(rotation_tolerance_degrees=max(angles)*1.000001+1e-6,
                mutual_floor=.9*min(r['mutual_fraction'] for r in allv),
                fit_rmse_cap=1.1*max(r['global_frame']['fit_rmse'] for r in allv))
        write(out/'thresholds.json',dict(limits=limits,source_unit_count=4,independent_states=False,
            caveat='mechanism-class calibration, not a semantic error bound'))
        print('CALIBRATED',limits,flush=True);return
    if (out/'heldout.json').exists():raise RuntimeError('Immutable test result exists')
    rows=read(HERE/'outputs/heldout_weighted_v1',[p+'__00' for p in TEST])
    assert len(rows)==len(TEST)*14
    limits=json.loads((out/'thresholds.json').read_text())['limits']
    results={}
    for k in keys:
        limit=limits[k];groups={}
        for group,names in [('nonrotation',NONROT),('rotation',ROT),('mismatch',['mismatch']),('reflection',['mirror'])]:
            items=[]
            for row in rows:
                if row['variant'] not in names:continue
                v=row['results'][k];f=v['global_frame']
                quality=f.get('valid',False) and v['mutual_fraction']>=limit['mutual_floor'] and f['fit_rmse']<=limit['fit_rmse_cap']
                changed=f.get('reflection',False) or (f.get('rotation_deg') is not None and abs(f['rotation_deg'])>limit['rotation_tolerance_degrees'])
                items.append(dict(pair=row['pair'],variant=row['variant'],quality_supported=quality,rotation_detected=changed,
                    rotation_deg=f['rotation_deg'],reflection=f['reflection'],pck32=v['evaluation']['pck32'] if v['evaluation'] else None,
                    angle_error_deg=v['evaluation'].get('global_frame_angle_error_deg') if v['evaluation'] else None))
            groups[group]=dict(total=len(items),supported=sum(x['quality_supported'] for x in items),
                changed=sum(x['rotation_detected'] for x in items),supported_changed=sum(x['quality_supported'] and x['rotation_detected'] for x in items),
                median_pck32=float(np.median([x['pck32'] for x in items if x['pck32'] is not None])) if any(x['pck32'] is not None for x in items) else None,
                rows=items)
        results[k]=groups
    write(out/'heldout.json',dict(results=results,source_count=12,primary=selection['primary'],
        permission_scope='global affine/nonrotation mechanism only; no natural semantic edit quality guarantee'))
    for k,r in results.items():print('HELDOUT',k,{g:{x:v for x,v in s.items() if x!='rows'} for g,s in r.items()},flush=True)


if __name__=='__main__':main()
