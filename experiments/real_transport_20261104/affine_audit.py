"""Evaluation-only RGB affine registration, independent of native descriptors.

Six image-registration variables are optimized, no model weights are trained.
This checks global scale/translation side effects of latent lifting; it does not
measure 3D animal articulation or solve semantic validity.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
from common import *
from evaluate_repair import load


def fit(im,reference,angle):
    im=F.interpolate(im,(256,256),mode='bilinear',align_corners=False,antialias=True)
    ref=F.interpolate(reference,(256,256),mode='bilinear',align_corners=False,antialias=True)
    a=math.radians(angle)
    theta=torch.tensor([[math.cos(a),-math.sin(a),0.],[math.sin(a),math.cos(a),0.]],device=im.device).requires_grad_(True)
    optimizer=torch.optim.LBFGS([theta],lr=.5,max_iter=60,tolerance_grad=1e-8,tolerance_change=1e-10,line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        grid=F.affine_grid(theta[None],im.shape,align_corners=False)
        warped=F.grid_sample(im,grid,align_corners=False,padding_mode='reflection')
        loss=(warped[:,:,40:-40,40:-40]-ref[:,:,40:-40,40:-40]).square().mean()
        loss.backward();return loss
    optimizer.step(closure)
    value=float(closure().detach());matrix=theta[:,:2].detach().cpu().numpy()
    u,s,vh=np.linalg.svd(matrix);r=u@vh
    return dict(rotation_degrees=float(np.degrees(np.arctan2(r[1,0],r[0,0]))),
        stretch_singular_values=s.tolist(),translation_pixels=(theta[:,2].detach().cpu().numpy()/2*512).tolist(),
        registered_RMSE=math.sqrt(value),matrix=theta.detach().cpu().tolist())


def main():
    setup();out=HERE/'outputs/affine_rgb_audit_v1'
    if (out/'summary.json').exists():raise RuntimeError('Immutable affine audit exists')
    registered=json.loads((HERE/'outputs/repair_rgb_audit_v1/summary.json').read_text())['rows']
    rows=[]
    for row in registered:
        pair=row['pair'];folder=HERE/'outputs/repair_v2'/pair;ref=load(folder/'offline_reference_dark.png')
        results={}
        for mode in ['unchanged','controlled','oracle_inverse']:
            a=row['scores'][mode]['offline_residual_angle_deg']
            results[mode]=fit(load(folder/(mode+'.png')),ref,a)
        rows.append(dict(pair=pair,results=results))
        print('AFFINE_RGB_AUDIT',pair,results['controlled'],flush=True)
    write(out/'summary.json',dict(rows=rows,correct_reference_at_runtime=False,
        scope='evaluation-only global RGB affine fit; local minima possible; no anatomical pose guarantee'))


if __name__=='__main__':main()
