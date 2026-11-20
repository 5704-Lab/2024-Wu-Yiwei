"""Local weight I/O for unmodified official SelFix FLUX model and solver.

Diffusers tensor layout is reversed using the installed official converter's map.
Every tensor is round-trip checked, and original Flux strict loading is required.
"""
from pathlib import Path
import gc
import hashlib
import json
import sys
import time
import numpy as np
from PIL import Image
import torch
from safetensors.torch import load_file

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'vendor/selfix/src'))
from flux.model import Flux
from flux.util import configs
from flux.sampling import get_schedule,denoise_selfix,unpack


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


def reverse_flux(sd):
    src=dict(sd);dst={}
    def move(new,old):dst[new]=src.pop(old)
    for name,df in [('time_in','timestep_embedder'),('vector_in','text_embedder'),('guidance_in','guidance_embedder')]:
        for suffix in ['weight','bias']:
            move(f'{name}.in_layer.{suffix}',f'time_text_embed.{df}.linear_1.{suffix}')
            move(f'{name}.out_layer.{suffix}',f'time_text_embed.{df}.linear_2.{suffix}')
    for suffix in ['weight','bias']:
        move(f'img_in.{suffix}',f'x_embedder.{suffix}');move(f'txt_in.{suffix}',f'context_embedder.{suffix}')
    for i in range(19):
        o=f'double_blocks.{i}.';d=f'transformer_blocks.{i}.'
        for suffix in ['weight','bias']:
            move(o+'img_mod.lin.'+suffix,d+'norm1.linear.'+suffix)
            move(o+'txt_mod.lin.'+suffix,d+'norm1_context.linear.'+suffix)
            for modality,projections in [('img',['to_q','to_k','to_v']),('txt',['add_q_proj','add_k_proj','add_v_proj'])]:
                dst[o+modality+'_attn.qkv.'+suffix]=torch.cat([src.pop(d+'attn.'+p+'.'+suffix) for p in projections])
            for org,dif in [('img_mlp.0','ff.net.0.proj'),('img_mlp.2','ff.net.2'),
                            ('txt_mlp.0','ff_context.net.0.proj'),('txt_mlp.2','ff_context.net.2'),
                            ('img_attn.proj','attn.to_out.0'),('txt_attn.proj','attn.to_add_out')]:
                move(o+org+'.'+suffix,d+dif+'.'+suffix)
        for org,dif in [('img_attn.norm.query_norm.scale','attn.norm_q.weight'),
                        ('img_attn.norm.key_norm.scale','attn.norm_k.weight'),
                        ('txt_attn.norm.query_norm.scale','attn.norm_added_q.weight'),
                        ('txt_attn.norm.key_norm.scale','attn.norm_added_k.weight')]:move(o+org,d+dif)
    for i in range(38):
        o=f'single_blocks.{i}.';d=f'single_transformer_blocks.{i}.'
        for suffix in ['weight','bias']:
            move(o+'modulation.lin.'+suffix,d+'norm.linear.'+suffix)
            dst[o+'linear1.'+suffix]=torch.cat([src.pop(d+k+'.'+suffix) for k in
                ['attn.to_q','attn.to_k','attn.to_v','proj_mlp']])
            move(o+'linear2.'+suffix,d+'proj_out.'+suffix)
        move(o+'norm.query_norm.scale',d+'attn.norm_q.weight')
        move(o+'norm.key_norm.scale',d+'attn.norm_k.weight')
    for suffix in ['weight','bias']:
        move('final_layer.linear.'+suffix,'proj_out.'+suffix)
        a,b=src.pop('norm_out.linear.'+suffix).chunk(2,dim=0)
        dst['final_layer.adaLN_modulation.1.'+suffix]=torch.cat([b,a])
    assert not src,list(src)
    return dst


def load_model(device,audit_path):
    from diffusers.loaders.single_file_utils import convert_flux_transformer_checkpoint_to_diffusers
    start=time.monotonic();state={};folder=ROOT/'pretrained_models/flux1-dev/transformer'
    shards=sorted(folder.glob('*.safetensors'))
    for path in shards:
        print('LOAD_SHARD',path.name,flush=True);state.update(load_file(str(path),device='cpu'))
    print('REVERSE_WEIGHT_LAYOUT',len(state),flush=True)
    original=reverse_flux(state)
    restored=convert_flux_transformer_checkpoint_to_diffusers(dict(original))
    assert set(restored)==set(state)
    for key,value in state.items():assert torch.equal(value,restored[key]),key
    verified=len(state);del restored,state;gc.collect()
    with torch.device('meta'):model=Flux(configs['flux-dev'].params).to(torch.bfloat16)
    model.load_state_dict(original,strict=True,assign=True)
    del original;gc.collect()
    print('MOVE_OFFICIAL_FLUX_TO_GPU',flush=True)
    model=model.eval().requires_grad_(False).to(device)
    write(audit_path,dict(layout_roundtrip_bitwise=True,tensors_verified=verified,
        original_module_strict_loading=True,official_model_file_sha256=sha(ROOT/'vendor/selfix/src/flux/model.py'),
        weight_shards=[dict(file=p.name,bytes=p.stat().st_size) for p in shards],
        device=str(device),dtype='bfloat16',seconds=time.monotonic()-start))
    return model


def base_info(inject=0):
    return dict(feature_path='',feature={},inject_step=inject,start_layer_index=0,end_layer_index=37,
        reuse_v=1,editing_strategy='replace_v',qkv_ratio=[1.,1.,1.],inv_stop_ratio=1.,method='selfix',
        initialization='clone',window_size=1,save_trace=True,save_latent_trajectory=True)


def model_inp(inputs,which,device):
    return {k:v.to(device) for k,v in inputs[which].items()}


def save_image(tensor,path):
    a=(tensor[0].permute(1,2,0).float().cpu().numpy()*255).round().clip(0,255).astype('uint8')
    Image.fromarray(a).save(path)


def decode_state(vae,packed,size,device):
    latent=unpack(packed.to(device).float(),size,size)
    return ((vae.decode(latent/vae.config.scaling_factor+vae.config.shift_factor,return_dict=False)[0]+1)/2).clamp(0,1)
