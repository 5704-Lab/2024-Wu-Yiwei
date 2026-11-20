"""Locked selected native reader, finite observations of actual clean latents."""
import numpy as np
import torch
from common import *
from tap import NativeTap
from run import masks
from match import descriptor,role_distribution,transport,coordinates,estimate_frame
from weighted_frames import texture


class SelectedReader:
    def __init__(self,model,tokenizer,pair,key):
        self.model=model;self.tap=NativeTap(model);self.pair=pair;self.key=key
        bank,facet,layer,norm,sem,method=key.split('/')
        self.facet=facet;self.layer=int(layer[1:]);self.norm=norm;self.sem=float(sem[3:]);self.method=method
        self.independent=bank=='independent'
        data=torch.load(ROOT/'experiments/stage1_full/inputs'/f'{pair}.pt',weights_only=True)
        et=data['embeddings'][1];self.condition=dict(seq=et[0],pool=et[2])
        self.masks=masks(tokenizer,data['record']['target_prompt'],ROLES[pair.split('__')[0]],et[0].shape[1])
        self.x=data['x'].to('cuda:0');self.weight=texture(pair)
        self.source={}
        with torch.no_grad():
            for audit in [False,True]:
                rows=[]
                for tau,seed in PROBES:
                    eps=torch.randn(self.x.shape,generator=torch.Generator('cpu').manual_seed(seed+(20000000 if audit else 0))).to(self.x)
                    f,_=self.tap.observe(self.x,eps,tau,self.condition);rows.append(f[self.layer])
                self.source[audit]=rows

    @torch.no_grad()
    def __call__(self,y,audit=False):
        similarity=0
        for (tau,seed),source in zip(PROBES,self.source[audit]):
            offset=(20000000 if audit else 0)+(10000000 if self.independent else 0)
            eps=torch.randn(y.shape,generator=torch.Generator('cpu').manual_seed(seed+offset)).to(y)
            f,_=self.tap.observe(y,eps,tau,self.condition);current=f[self.layer]
            a=descriptor(source[self.facet][0],source[self.facet][0],self.norm)
            b=descriptor(current[self.facet][0],source[self.facet][0],self.norm)
            similarity=similarity+(a@b.T+self.sem*(role_distribution(source,self.masks)@role_distribution(current,self.masks).T))/len(PROBES)
        j,mutual,gap,alternative=transport(similarity,self.method)
        p=coordinates(len(j));jj=j.cpu();m=mutual.cpu().float()
        frame=estimate_frame(p,p[jj],m*self.weight)
        return dict(frame=frame,mutual_fraction=float(m.mean()),mean_gap=float(gap.mean()))
