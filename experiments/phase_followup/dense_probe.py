"""Calibrate a dense, image-derived geometry observer before using it as control.

No editing masks, new network weights or external references. Optical flow remains
a proxy and must pass appearance-only counterexamples; no semantic pose claim.
"""
import sys,math,json
from pathlib import Path
import cv2,numpy as np
import torch
HERE=Path(__file__).parent
sys.path.insert(0,str(HERE.parent/'phase_permission'))
from common import source_rgb,seed_setup,Phase,warp,PAIRS,write


def prepare(rgb,normalized):
    gray=cv2.cvtColor(rgb.astype(np.float32),cv2.COLOR_RGB2GRAY)
    if normalized:
        mean=cv2.GaussianBlur(gray,(0,0),2)
        var=np.maximum(cv2.GaussianBlur(gray*gray,(0,0),2)-mean*mean,0)
        gray=np.clip((gray-mean)/np.sqrt(var+.005**2),-3,3)/6+.5
    return np.uint8(np.rint(np.clip(gray,0,1)*255))


def measure(a,b,normalized):
    flow=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(prepare(a,normalized),prepare(b,normalized),None)
    back=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(prepare(b,normalized),prepare(a,normalized),None)
    yy,xx=np.mgrid[:a.shape[0],:a.shape[1]].astype(np.float32)
    bsample=cv2.remap(back,xx+flow[:,:,0],yy+flow[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    cycle=np.linalg.norm(flow+bsample,axis=-1)
    return flow,cycle


def main():
    seed_setup();cv2.setNumThreads(4);rows=[]
    for pair in PAIRS+['flowers__00','corgi__00','penguins__00','dalmatian__00']:
        rgb,_=source_rgb(pair,dtype=torch.float64);phase=Phase(rgb);src=phase.source_small
        xx=(torch.arange(128,dtype=src.dtype)+.5)/128
        cases=dict(identity=src,channel_gain=src*torch.tensor([.55,.85,1.05])[None,:,None,None],
                   gamma=src.clamp_min(1e-5)**torch.tensor([.7,1.2,1.])[None,:,None,None],
                   spatial_gain=src*torch.exp(.25*torch.cos(4*math.pi*xx))[None,None,None,:],
                   shift_050=warp(src,tx=.5),shift_100=warp(src,tx=1.))
        a=src[0].permute(1,2,0).numpy().astype(np.float32)
        for name,im in cases.items():
            b=im[0].permute(1,2,0).numpy().astype(np.float32)
            truth=.5 if name=='shift_050' else 1. if name=='shift_100' else 0.
            for norm in [False,True]:
                flow,cycle=measure(a,b,norm)
                error=np.linalg.norm(flow-np.array([truth,0]),axis=-1)
                # Fixed interior evaluation avoids the synthetic translation boundary.
                vals=error[8:-8,8:-8];cy=cycle[8:-8,8:-8]
                rows.append(dict(pair=pair,case=name,normalized=norm,median_error=float(np.median(vals)),
                    p90_error=float(np.quantile(vals,.9)),cycle_median=float(np.median(cy)),
                    confident_fraction=float(np.mean(cy<.5)),unit='pixels at 128; multiply by four at 512'))
        write(HERE/'outputs/dense_probe.json',dict(rows=rows,scope='calibration only; no optical-flow corrections applied'))
        print(pair,flush=True)
    for norm in [False,True]:
        for case in cases:
            rs=[r for r in rows if r['normalized']==norm and r['case']==case]
            print(norm,case,np.median([r['median_error'] for r in rs]),np.median([r['p90_error'] for r in rs]),flush=True)


if __name__=='__main__':main()
