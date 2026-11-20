"""Follow-up interventions on correction capacity, spectral emphasis and task floor.

Predefined variants, same seed/noises/CFG. No endpoint-dependent parameter selection.
Old experiment files are preserved. Canonical endpoint metrics use old observations.
"""
import sys,json,gc,hashlib
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend

HERE=Path(__file__).parent
sys.path.insert(0,str(HERE.parent/'phase_permission'))
from common import Phase,write,seed_setup
sys.path.insert(0,str(HERE))
import solver


class LowPhase(Phase):
    def __init__(self,source,*args,**kwargs):
        super().__init__(source,*args,**kwargs)
        band=torch.arange(36,device=source.device)%12
        multiplier=torch.where(band<4,1.,.25)[None,:,None,None]
        self.sqrt_weight=self.sqrt_weight*multiplier
        self.sqrt_weight=self.sqrt_weight/self.sqrt_weight.square().sum().sqrt()
        self.source_obs=self.embed(self.coefficients(source)).detach()


VARIANTS=dict(RADIUS=dict(radius=.6,low=False,disable_task=False),
              LOW_FREQ=dict(radius=.6,low=True,disable_task=False),
              LOW_NO_TASK=dict(radius=.6,low=True,disable_task=True))


def main():
    seed_setup()
    write(HERE/'protocol.json',dict(variants=VARIANTS,pairs=['bear__00','brown_owl__00'],
        comparisons='previous PERMIT .2 -> radius .6 -> low frequency -> no task floor',
        caution='exploratory follow-up on two development images; no independent pose truth',
        code={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}))
    for name,v in VARIANTS.items():
        solver.OUT=HERE/'outputs'/name
        solver.Phase=LowPhase if v['low'] else Phase
        solver.CONFIG.update(radius_ratio=v['radius'],disable_task=v['disable_task'])
        with sdpa_kernel(SDPBackend.MATH):
            solver.full(SimpleNamespace(pairs=['bear__00','brown_owl__00'],methods=['PERMIT']))
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
