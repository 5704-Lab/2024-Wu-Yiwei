"""Run official SelFix and FlowEdit's velocity-difference algorithm on the SAME FLUX.

SelFix solver and native attention injection are imported verbatim from pinned vendor.
Only local weight/text/VAE I/O and experiment logging are adapted.
"""
import copy
import gc
import json
import time
import torch
from diffusers import AutoencoderKL
from common import ROOT,HERE,setup,write,sha,load_model,base_info,model_inp,decode_state,save_image,get_schedule,denoise_selfix


@torch.inference_mode()
def main():
    setup();device=torch.device('cuda:0');out=HERE/'outputs/comparison_v1'
    if out.exists():raise RuntimeError('Output exists; use a fresh version')
    out.mkdir(parents=True);start=time.monotonic()
    payload=torch.load(HERE/'outputs/inputs_v1/inputs.pt',map_location='cpu',weights_only=True)
    inputs=payload['inputs'];size=payload['size']
    protocol=dict(date='2026-11-05',source=payload['source'],model='FLUX.1-dev',resolution=size,
        model_dtype='bfloat16',source_latent_dtype='bfloat16',vae_dtype='float32',
        official_selfix_commit='c549b2f934c8cde7d3ff46a020dc05422059d435',
        official_solver_sha256=sha(ROOT/'vendor/selfix/src/flux/sampling.py'),
        official_layers_sha256=sha(ROOT/'vendor/selfix/src/flux/modules/layers.py'),
        input_sha256=sha(HERE/'outputs/inputs_v1/inputs.pt'),
        script_sha256=sha(__file__),common_sha256=sha(HERE/'common.py'),
        source_count=1,seed=0,negative_cfg_branches=False,
        spatial_edit_mask=False,attention_injection='SelFix official value replacement, first reverse step, single layers 0..37',
        proposed_reconstruction_gate=dict(reference='decoded unchanged packed BF16 source',psnr_db_min=28.,ssim_min=.9),
        recipes=[dict(name='selfix_default',steps=15,iterations=4,momentum=.1,alpha1=.5,delta=2.,target_guidance=[2.]),
                 dict(name='selfix_high_fidelity',steps=15,iterations=10,momentum=.5,alpha1=.5,delta=.125,target_guidance=[2.,3.5])],
        recipe_scope='default editing recipe plus official reconstruction inversion budget transferred to editing, explicitly separate',
        flowedit_recipes=[dict(name='flowedit_author',T=28,n_max=24,n_min=0,n_avg=1,source_guidance=1.5,target_guidance=5.5),
                          dict(name='flowedit_guidance_matched',T=28,n_max=24,n_min=0,n_avg=1,source_guidance=1.,target_guidance=2.)])
    write(out/'protocol.json',protocol)
    model=load_model(device,out/'weight_layout_audit.json')
    inp_s=model_inp(inputs,'source',device);inp_t=model_inp(inputs,'target',device)
    states={'source':inp_s['img'].cpu()};traces={};counts={};calls=0;phase='load'
    def count_hook(module,args,output):
        nonlocal calls
        calls+=1
        if calls%10==0:print(json.dumps(dict(stage=phase,nfe=calls,seconds=round(time.monotonic()-start,2))),flush=True)
    handle=model.register_forward_hook(count_hook)
    def save_checkpoint():torch.save(dict(states=states,traces=traces,counts=counts),out/'checkpoint.pt')

    for recipe in protocol['recipes']:
        name=recipe['name'];phase=name+'_invert';before=calls
        info=base_info(1)
        info.update(num_iterations=recipe['iterations'],momentum=recipe['momentum'],alpha1=recipe['alpha1'],delta=recipe['delta'])
        schedule=get_schedule(15,inp_s['img'].shape[1],shift=True)
        z,info=denoise_selfix(model,**inp_s,timesteps=schedule,inverse=True,info=info,guidance=1.)
        assert bool(torch.isfinite(z).all())
        assert len(info['feature'])==38,('Missing official injection cache',len(info['feature']))
        counts[name+'_inversion']=calls-before
        states[name+'_inverted']=z.cpu()
        traces[name+'_inversion']=copy.deepcopy(info['trace'])
        # The no-edit control uses EXACTLY the same inverted state as these edits.
        phase=name+'_source_reconstruction';before=calls
        ri={**info,'inject_step':0,'trace':{},'latent_trajectory':{}}
        source_query={**inp_s,'img':z.clone()}
        rec,ri=denoise_selfix(model,**source_query,timesteps=schedule,inverse=False,info=ri,guidance=1.)
        states[name+'_reconstruction']=rec.cpu();counts[name+'_reconstruction']=calls-before
        traces[name+'_reconstruction']=ri['trace']
        for guidance in recipe['target_guidance']:
            label=name+'_g'+str(guidance).replace('.','p')
            phase=label+'_edit';before=calls
            ti={**info,'trace':{},'latent_trajectory':{}}
            target_query={**inp_t,'img':z.clone()}
            edited,ti=denoise_selfix(model,**target_query,timesteps=schedule,inverse=False,info=ti,guidance=guidance)
            states[label]=edited.cpu();counts[label]=calls-before
            traces[label]=ti['trace']
            assert bool(torch.isfinite(edited).all())
        save_checkpoint()
        del info,ri,ti,z,rec,edited;gc.collect();torch.cuda.empty_cache()

    # Faithful FlowEdit n_min=0 equations, applied to original FLUX packed latents.
    # Local legacy wrapper's image-size API predates installed diffusers; do not use
    # that wrapper's obsolete size conversion on today's pipeline.
    for recipe in protocol['flowedit_recipes']:
        phase=recipe['name'];before=calls
        x=inp_s['img'];y=x.clone();history=[y.cpu()]
        schedule=get_schedule(recipe['T'],x.shape[1],shift=True)
        generator=torch.Generator('cpu').manual_seed(0)
        info=base_info(0);info.update(inject=False,inverse=False,second_order=False,t=0.,feature_injection_mode='off')
        for i in range(recipe['T']-recipe['n_max'],recipe['T']):
            t,tn=schedule[i:i+2]
            noise=torch.randn(x.shape,generator=generator,dtype=torch.bfloat16).to(device)
            qs=(1-t)*x+t*noise;qt=y+qs-x
            velocities=[]
            for query,condition,guidance in [(qs,inp_s,recipe['source_guidance']),(qt,inp_t,recipe['target_guidance'])]:
                prediction,_=model(img=query,img_ids=condition['img_ids'],txt=condition['txt'],txt_ids=condition['txt_ids'],
                    y=condition['vec'],timesteps=torch.full((1,),t,device=device,dtype=x.dtype),
                    guidance=torch.full((1,),guidance,device=device,dtype=x.dtype),info=info)
                velocities.append(prediction)
            delta=velocities[1]-velocities[0]
            y=(y.float()+(tn-t)*delta).to(delta.dtype)
            assert bool(torch.isfinite(y).all())
            history.append(y.cpu())
        states[recipe['name']]=y.cpu();counts[recipe['name']]=calls-before
        traces[recipe['name']]=dict(states=history,timesteps=schedule[recipe['T']-recipe['n_max']:])
        save_checkpoint()
    handle.remove()
    peak=torch.cuda.max_memory_allocated(device)
    del model,inp_s,inp_t;gc.collect();torch.cuda.empty_cache()
    print('DECODE_ALL',flush=True)
    vae=AutoencoderKL.from_pretrained(ROOT/'pretrained_models/flux1-dev/vae',local_files_only=True,
        torch_dtype=torch.float32).eval().requires_grad_(False).to(device)
    images={}
    for name,latent in states.items():
        if name.endswith('_inverted'):continue
        im=decode_state(vae,latent,size,device)
        save_image(im,out/(name+'.png'));images[name]=im.cpu()
    torch.save(dict(states=states,traces=traces,images=images,source=payload['source']),out/'results.pt')
    write(out/'complete.json',dict(status='complete',nfe=counts,total_nfe=calls,seconds=time.monotonic()-start,
        peak_gpu_bytes=peak,recipes_completed=[r['name'] for r in protocol['recipes']],
        all_same_source_and_model=True,script_sha256=sha(__file__)))
    print('COMPARISON_COMPLETE',counts,flush=True)


if __name__=='__main__':main()
