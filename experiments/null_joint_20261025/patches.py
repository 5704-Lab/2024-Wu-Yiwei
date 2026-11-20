"""Source-fixed patch features with bounded photometric nuisance removal.

Same existing frozen SigLIP vision encoder; no global text score, attention
editing mask, object region, or additional pretrained model is used.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend,sdpa_kernel

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/phase_permission'))
from common import CleanScorer,panel,save_image,seed_setup,write


def photometries(source):
    photos=[]
    for c in range(3):
        for sign in [-1.,1.]:
            gains=torch.ones((1,3,1,1),device=source.device);gains[:,c]=math.exp(sign*.4)
            photos.append((source*gains).clamp(0,1))
    photos.extend([source*.3,source*.6,source*.55+.45,source*.3+.7])
    return photos


class PatchRelation:
    def __init__(self,sc,source,rank=4,deadzone=True):
        self.sc=sc;self.rank=rank;self.deadzone=deadzone
        with torch.no_grad():
            self.source=self.features(source).detach()
            responses=torch.cat([self.features(im)-self.source for im in photometries(source)],0)
            # One nuisance subspace per fixed patch position, not an edit gate.
            matrix=responses.permute(1,2,0).double()
            u,s,_=torch.linalg.svd(matrix,full_matrices=False)
            self.u=u[:,:,:rank].float().detach()
            coeff=torch.einsum('bpd,pdr->bpr',responses,self.u)
            self.bounds=(coeff.abs().amax(0)*1.5+1e-5).detach()
            r=self.profile(responses)
            self.tolerance=(r.norm(dim=-1).amax(0)*1.25+.005).detach()
            self.singular_values=s.cpu()

    def features(self,image):
        raw=self.sc.model.vision_model(pixel_values=self.sc.preprocess(image)).last_hidden_state
        return raw/raw.norm(dim=-1,keepdim=True).clamp_min(1e-8)

    def profile(self,r):
        coeff=torch.einsum('bpd,pdr->bpr',r,self.u)
        coeff=torch.maximum(torch.minimum(coeff,self.bounds),-self.bounds) if hasattr(self,'bounds') else coeff
        return r-torch.einsum('bpr,pdr->bpd',coeff,self.u)

    def __call__(self,image):
        r=self.profile(self.features(image)-self.source)
        if self.deadzone:
            norm=r.norm(dim=-1,keepdim=True)
            r=r*(1-self.tolerance[None,:,None]/norm.clamp_min(1e-8)).clamp_min(0)
        return r/math.sqrt(r.shape[1])

    def diagnostics(self,image):
        with torch.no_grad():
            raw=self.features(image)-self.source;profile=self.profile(raw)
            o=self(image)
            return dict(raw=float(raw.square().sum(-1).mean()),profile=float(profile.square().sum(-1).mean()),
                 excess=float(o.square().sum()),active_patches=int((profile.norm(dim=-1)>self.tolerance).sum()),
                 median_tolerance=float(self.tolerance.median()))


class PatchGeometry:
    """Adapter for the same matrix-free restoration solver, all patches used."""
    def __init__(self,relation):
        self.relation=relation
        self.source=torch.zeros_like(relation.source)

    def __call__(self,image):return self.relation(image)

    def loss(self,obs,kappa=0.):return .5*obs.square().sum()

    def gn_weights(self,obs,kappa=0.):
        # The generic geometry solver divides by feature count. Here the
        # observer already normalizes by sqrt(number of patches).
        return torch.full(obs.shape[:2],obs.shape[-1],device=obs.device,dtype=obs.dtype)


def main():
    seed_setup();sc=CleanScorer(torch.device('cuda:0'),torch.float32,'native')
    out=HERE/'outputs/patch_calibration';rows=[]
    for pair in ['bear__00','brown_owl__00']:
        data=torch.load(ROOT/'experiments/stage1_full/inputs'/f'{pair}.pt',weights_only=True)
        x=data['x'].to(sc.device)
        with torch.no_grad():source=sc.image(x)
        lum=(source*torch.tensor([.299,.587,.114],device=sc.device)[None,:,None,None]).sum(1,keepdim=True)
        xx=torch.linspace(0,2*math.pi,512,device=sc.device)
        validation={'dark':source*.45,'white':source*.65+.35,'desaturate':source*.4+lum*.6,
            'warm':(source*torch.tensor([1.1,.9,1.05],device=sc.device)[None,:,None,None]).clamp(0,1),
            'spatial_light':(source*torch.exp(.15*xx.cos()[None,None,None,:])).clamp(0,1)}
        negative={'mirror':source.flip(-1),'shift16':torch.roll(source,16,-1)}
        actual={}
        for mode in ['BASE','CF','NO_NULL','JOINT']:
            state=torch.load(HERE/'outputs/v1_full/full'/pair/'seed0'/mode/'trajectory.pt',weights_only=True)['final'].to(sc.device)
            with torch.no_grad():actual[mode]=sc.image(state)
        for rank in [0,2,4,6,8]:
            relation=PatchRelation(sc,source,rank=rank)
            for group,items in [('positive',validation),('negative',negative),('actual',actual)]:
                for name,im in items.items():
                    row=dict(pair=pair,rank=rank,group=group,case=name,**relation.diagnostics(im));rows.append(row)
            write(out/'results.json',dict(rows=rows,scope='source-photometry nuisance calibration, fixed image geometry controls, and selected development outputs; no semantic ground truth'))
            print(pair,rank,[r for r in rows if r['pair']==pair and r['rank']==rank],flush=True)
        panel([('Source',source)]+[(k,v) for k,v in actual.items()],out/f'{pair}.png',title=pair)
    print('PATCH_CALIBRATION_COMPLETE',flush=True)


if __name__=='__main__':
    with sdpa_kernel(SDPBackend.MATH):main()
