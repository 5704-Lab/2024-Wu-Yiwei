"""Source-only texture weighting of measurements, not a latent update mask.

Keeps uniform-frame results alongside the adjusted estimate. No current geometry
or transform labels select source points. Uniform areas have little motion info.
"""
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from common import *
from match import coordinates,estimate_frame


def texture(pair):
    im=np.asarray(Image.open(HERE/'outputs/data_v1'/pair/'source.png').convert('RGB')).copy()/255.
    v=torch.tensor(im,dtype=torch.float32).permute(2,0,1)[None]
    dx=F.pad(v[:,:,:,1:]-v[:,:,:,:-1],(0,1,0,0))
    dy=F.pad(v[:,:,1:,:]-v[:,:,:-1,:],(0,0,0,1))
    energy=F.avg_pool2d((dx.square()+dy.square()).mean(1,keepdim=True),16).flatten().sqrt()
    return (.05+energy/energy.mean().clamp_min(1e-8)).clamp_max(10)


def augment(row,folder):
    weight=texture(row['pair']);p=coordinates(len(weight));arrays=np.load(folder/row['pair']/(row['variant']+'.npz'))
    for k,v in row['results'].items():
        j=torch.from_numpy(arrays[k].astype(np.int64));m=torch.from_numpy(arrays[k+'/mutual']).float()
        frame=estimate_frame(p,p[j],m*weight)
        v['uniform_frame']=v['global_frame'];v['global_frame']=frame
        # Existing per-point truth scores and object/local fits are unchanged.
        if v['evaluation'] is not None:
            ev=v['evaluation']
            for metric in ['angle_error_deg','matrix_error','translation_error']:
                ev['uniform_'+metric]=ev.pop('global_frame_'+metric,None)
    row['frame_rule']='source_RGB_gradient_RMS_patch_weights, floor .05, cap 10; observation only'
    return row


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();setup()
    folder=HERE/'outputs'/args.input;out=HERE/'outputs'/args.output
    if (out/'complete.json').exists():raise RuntimeError('Immutable weighted refit exists')
    manifest=json.loads((HERE/'outputs/data_v1/manifest.json').read_text())
    truth={(r['pair'],n):item for r in manifest['records'] for n,item in r['variants'].items()}
    count=0
    for p in sorted(folder.glob('*__00/*.json')):
        row=augment(json.loads(p.read_text()),folder)
        gt=truth.get((row['pair'],row['variant']))
        if gt and 'matrix' in gt:
            u,_,vh=np.linalg.svd(gt['matrix']);r=u@vh
            angle=np.degrees(np.arctan2(r[1,0],r[0,0])) if np.linalg.det(gt['matrix'])>0 else None
            for k,v in row['results'].items():
                f=v['global_frame']
                if f.get('valid'):
                    v['evaluation']['global_frame_matrix_error']=float(np.linalg.norm(np.array(f['matrix'])-gt['matrix']))
                    v['evaluation']['global_frame_translation_error']=float(np.linalg.norm(np.array(f['translation'])-gt['translation']))
                    if f.get('rotation_deg') is not None and angle is not None:
                        v['evaluation']['global_frame_angle_error_deg']=abs((f['rotation_deg']-angle+180)%360-180)
        write(out/row['pair']/p.name,row);count+=1
    write(out/'complete.json',dict(input=args.input,output=args.output,states=count,script_sha256=sha(__file__),
        changes_matching=False,source_only_observation_weights=True))
    print('WEIGHTED_REFIT',count,flush=True)


if __name__=='__main__':main()
