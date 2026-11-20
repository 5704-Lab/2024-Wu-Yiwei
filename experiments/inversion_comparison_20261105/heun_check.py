"""One-source solver diagnostic: partial ordinary ODE inversion using Heun both ways.

Same endpoints, prompts and CFG as the matched Euler experiment; twice the NFE.
"""
import json
import time
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
from diffusers import AutoencoderKL,SD3Transformer2DModel
from run import ROOT,HERE,sha_file,write,calc_v_sd3


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda:0');base=HERE/'outputs/bear_v1';out=HERE/'outputs/heun_partial_v1'
    if out.exists():raise RuntimeError('Use fresh output; no overwrite')
    out.mkdir(parents=True)
    p=json.loads((base/'protocol.json').read_text())
    inp=ROOT/'experiments/stage1_full/inputs/bear__00.pt'
    assert sha_file(inp)==p['input_sha256']
    data=torch.load(inp,map_location='cpu',weights_only=True)
    x=data['x'].to(device)
    es,et=[tuple(v.to(device) for v in e) for e in data['embeddings']]
    seq=torch.cat([es[1],es[0],et[1],et[0]]);pool=torch.cat([es[3],es[2],et[3],et[2]])
    sigmas=torch.tensor(p['sigmas'],dtype=torch.float32,device=device)
    schedule=json.loads((ROOT/'experiments/stage1_full/inputs/manifest.json').read_text())
    times=torch.tensor(schedule['timesteps']+[0.],dtype=torch.float32,device=device)
    write(out/'protocol.json',dict(source_protocol='../bear_v1/protocol.json',source_protocol_sha256=sha_file(base/'protocol.json'),
        input_sha256=sha_file(inp),solver='Heun explicit trapezoidal, two velocity evaluations per interval',
        precision='float32',source_cfg=3.5,target_cfg=13.5,partial_sigma=p['partial_sigma'],
        forward_intervals=33,reverse_intervals=33,inversion_branch_evaluations=132,
        target_branch_evaluations=132,reconstruction_branch_evaluations=132,
        source_image=data['record']['source_image'],prompts=data['record'],
        purpose='solver diagnostic, not same-compute method comparison',script_sha256=sha_file(__file__)))
    start=time.monotonic()
    model=SD3Transformer2DModel.from_pretrained(ROOT/'pretrained_models/sd3-medium/transformer',local_files_only=True,
        torch_dtype=torch.float32).eval().requires_grad_(False).to(device)
    pipe=SimpleNamespace(transformer=model,do_classifier_free_guidance=True)
    ss=torch.cat([es[1],es[0]]);sp=torch.cat([es[3],es[2]])
    def source_v(z,t):
        pred=model(hidden_states=torch.cat([z,z]),timestep=t.expand(2),encoder_hidden_states=ss,
                   pooled_projections=sp,return_dict=False)[0]
        u,c=pred.chunk(2);return u+3.5*(c-u)
    z=x.clone();asc=[z.cpu()]
    for j in range(49,16,-1):
        dt=sigmas[j]-sigmas[j+1]
        v1=source_v(z,times[j+1]);v2=source_v(z+dt*v1,times[j])
        z=z+dt*.5*(v1+v2)
        assert bool(torch.isfinite(z).all())
        asc.append(z.cpu());print('HEUN_INVERT',j,round(time.monotonic()-start,2),flush=True)
    torch.save(dict(ascending=asc,endpoint=z.cpu()),out/'inversion_checkpoint.pt')
    reconstruction=z.clone();target=z.clone();rh=[reconstruction.cpu()];th=[target.cpu()]
    def paired(a,b,t):
        return calc_v_sd3(pipe,torch.cat([a,a,b,b]),seq,pool,3.5,13.5,t)
    for i in range(17,50):
        dt=sigmas[i+1]-sigmas[i]
        sr,tr=paired(reconstruction,target,times[i])
        spred,tpred=paired(reconstruction+dt*sr,target+dt*tr,times[i+1])
        reconstruction=reconstruction+dt*.5*(sr+spred)
        target=target+dt*.5*(tr+tpred)
        assert bool(torch.isfinite(reconstruction).all() and torch.isfinite(target).all())
        rh.append(reconstruction.cpu());th.append(target.cpu())
        print('HEUN_REVERSE',i,round(time.monotonic()-start,2),flush=True)
    states=dict(ode_heun_partial=target.cpu(),ode_heun_partial_reconstruction=reconstruction.cpu(),source=x.cpu())
    torch.save(dict(states=states,ascending=asc,reconstruction=rh,target=th),out/'states.pt')
    model.cpu();del model,pipe
    torch.cuda.empty_cache()
    vae=AutoencoderKL.from_pretrained(ROOT/'pretrained_models/sd3-medium/vae',local_files_only=True,
        torch_dtype=torch.float32).eval().requires_grad_(False).to(device)
    decoded={}
    for name,latent in states.items():
        im=((vae.decode(latent.to(device)/vae.config.scaling_factor+vae.config.shift_factor,return_dict=False)[0]+1)/2).clamp(0,1)
        decoded[name]=im.cpu()
        array=(im[0].permute(1,2,0).cpu().numpy()*255).round().clip(0,255).astype('uint8')
        Image.fromarray(array).save(out/(name+'.png'))
    metrics={name:dict(pixel_rmse_from_vae_source=float((im.double()-decoded['source'].double()).square().mean().sqrt()),
         latent_rms_displacement=float((states[name].double()-x.cpu().double()).square().mean().sqrt()))
         for name,im in decoded.items()}
    write(out/'complete.json',dict(status='complete',metrics=metrics,seconds=time.monotonic()-start,
        single_branch_evaluations=396,transformer_calls=132,scope='same-source solver diagnostic',script_sha256=sha_file(__file__)))
    print('HEUN_COMPLETE',json.dumps(metrics),flush=True)


if __name__=='__main__':main()
