"""Independent RGB registration and resampling-matched negative control.

Correct dark images and known added angles are evaluation-only. No result here
is fed back to selection, inference, calibration or candidate acceptance.
"""
import numpy as np
import argparse
from PIL import Image
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from common import *
from repair import warp,encode


def load(path):
    a=np.asarray(Image.open(path).convert('RGB')).copy()/255.
    return torch.tensor(a,dtype=torch.float32,device='cuda:0').permute(2,0,1)[None]


def registration(im,ref):
    angles=torch.arange(-18.,18.001,.5,device=im.device)
    a=-angles*torch.pi/180
    theta=torch.zeros((len(a),2,3),device=im.device)
    theta[:,0,0]=a.cos();theta[:,0,1]=a.sin();theta[:,1,0]=-a.sin();theta[:,1,1]=a.cos()
    grid=F.affine_grid(theta,(len(a),3,512,512),align_corners=False)
    images=F.grid_sample(im.expand(len(a),-1,-1,-1),grid,align_corners=False,padding_mode='reflection')
    rms=(images[:,:,80:-80,80:-80]-ref[:,:,80:-80,80:-80]).square().mean((1,2,3)).sqrt()
    index=int(rms.argmin())
    return dict(offline_residual_angle_deg=float(angles[index]),registered_center_rmse=float(rms[index]),
        angle_grid_degrees=angles.cpu().tolist(),rmse_grid=rms.cpu().tolist())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',default='repair_v2');ap.add_argument('--output',default='repair_rgb_audit_v1')
    args=ap.parse_args();setup();out=HERE/'outputs'/args.output
    if (out/'summary.json').exists():raise RuntimeError('Immutable RGB audit exists')
    root=HERE/'outputs'/args.tag;summary=json.loads((root/'summary.json').read_text())
    vae=AutoencoderKL.from_pretrained(ROOT/'pretrained_models/sd3-medium/vae',local_files_only=True,
        torch_dtype=torch.float32).eval().requires_grad_(False).to('cuda:0')
    rows=[]
    with torch.no_grad():
        for result in summary['results']:
            pair=result['pair'];folder=root/pair
            data=torch.load(folder/'states.pt',weights_only=True);wrong=data['initial'].to('cuda:0')
            corrections=[r['accepted_correction_degrees'] for r in result['steps'] if r['status']=='accepted']
            for angle in corrections:
                im=image(vae,wrong);wrong=wrong+encode(vae,warp(im,-angle))-encode(vae,im)
            wrong_im=image(vae,wrong);save_image(wrong_im,out/pair/'resampling_matched_wrong_direction.png')
            ref=load(folder/'offline_reference_dark.png')
            scores={mode:registration(load(folder/(mode+'.png')),ref) for mode in ['unchanged','controlled','oracle_inverse']}
            scores['resampling_matched_wrong_direction']=dict(reference_center_rmse=float((wrong_im[:,:,80:-80,80:-80]-ref[:,:,80:-80,80:-80]).square().mean().sqrt()))
            rows.append(dict(pair=pair,split=split(pair),accepted_steps=len(corrections),scores=scores,
                acted_residual_angle=result['added_angle_truth_offline_only']+sum(corrections)))
            print('RGB_AUDIT',pair,{k:v.get('offline_residual_angle_deg') for k,v in scores.items()},flush=True)
    write(out/'summary.json',dict(rows=rows,reference_used_only_in_offline_evaluation=True,
        metric='center352x352 fixed evaluation crop; not runtime update mask; bilinear independent image registration',
        scope='2D global rotation alignment, not animal 3D pose or category correctness'))


if __name__=='__main__':main()
