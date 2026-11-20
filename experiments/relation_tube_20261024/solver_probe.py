"""Isolate truncation error on an identical actual SD3 intermediate state."""
import sys
from pathlib import Path
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run import Run,Relation,CFG,ROOT,seed_setup,write,save_image,metrics

def main():
 seed_setup();run=Run();ctx=run.setup('bear__00')
 saved=torch.load(ROOT/'experiments/flowedit_frequency_trajectory/outputs/bear__00/trajectory.pt',weights_only=True)
 y=saved['states'][33].to(run.device);delta=saved['states'][34].to(run.device)-y
 rel=Relation(run,ctx,'SHARED_FACT')
 with torch.no_grad():ob=rel.observe(run.sc.image(y))[0]
 theta,fit=rel.fit(ob);rows=[];out=HERE/'outputs/solver_probe'
 for count in [4,16,32]:
  CFG['cg_steps']=count
  state,th,info=rel.correct(y,delta,theta,audit=True)
  m,im=metrics(run,ctx,state);rows.append(dict(cg_steps=count,metrics=m,**info));save_image(im,out/f'cg{count}.png')
  write(out/'results.json',dict(rows=rows,fit=fit,scope='same real state and proposal; controls numerical solve accuracy, not a full trajectory comparison'))
  print(count,info['solver'],info['loss_after'],m,flush=True)
 print('SOLVER_PROBE_COMPLETE',flush=True)

if __name__=='__main__':
 with sdpa_kernel(SDPBackend.MATH):main()
