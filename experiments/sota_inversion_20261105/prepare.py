"""Prepare source and both complete text conditions using existing local FLUX weights."""
import gc
import json
import time
from einops import rearrange
from PIL import Image
import numpy as np
import torch
from diffusers import AutoencoderKL
from transformers import CLIPTokenizer,CLIPTextModel,T5Tokenizer,T5EncoderModel
from common import ROOT,HERE,setup,write,sha,save_image


@torch.inference_mode()
def main():
    setup();out=HERE/'outputs/inputs_v1'
    if out.exists():raise RuntimeError('Inputs exist; use a versioned directory')
    out.mkdir(parents=True);start=time.monotonic();device=torch.device('cuda:0')
    source=torch.load(ROOT/'experiments/stage1_full/inputs/bear__00.pt',weights_only=True,map_location='cpu')['record']
    prompts=[source['source_prompt'],source['target_prompt']]
    modelroot=ROOT/'pretrained_models/flux1-dev'
    source_image=Image.open(ROOT/source['source_image']).convert('RGB')
    assert source_image.size==(1024,1024)
    source_image.save(out/'source.png');size=1024
    vae=AutoencoderKL.from_pretrained(modelroot/'vae',local_files_only=True,torch_dtype=torch.float32).eval().requires_grad_(False).to(device)
    pixels=torch.from_numpy(np.array(source_image).copy()).permute(2,0,1)[None].to(device).float()/255
    latent=(vae.encode(2*pixels-1).latent_dist.mode()-vae.config.shift_factor)*vae.config.scaling_factor
    reconstruction=((vae.decode(latent/vae.config.scaling_factor+vae.config.shift_factor,return_dict=False)[0]+1)/2).clamp(0,1)
    save_image(reconstruction,out/'vae_reconstruction.png')
    packed=rearrange(latent.to(torch.bfloat16),'b c (h ph) (w pw)->b (h w) (c ph pw)',ph=2,pw=2)
    vae_source=reconstruction.cpu();del vae,pixels,latent,reconstruction;gc.collect();torch.cuda.empty_cache()
    print('ENCODE_CLIP',flush=True)
    tokenizer=CLIPTokenizer.from_pretrained(modelroot/'tokenizer',local_files_only=True)
    text=CLIPTextModel.from_pretrained(modelroot/'text_encoder',local_files_only=True,torch_dtype=torch.bfloat16).eval().requires_grad_(False).to(device)
    clip_inputs=tokenizer(prompts,max_length=77,padding='max_length',truncation=True,return_tensors='pt')
    clip=text(input_ids=clip_inputs['input_ids'].to(device),attention_mask=None,output_hidden_states=False).pooler_output.cpu()
    clip_ids=clip_inputs['input_ids'];del text;gc.collect();torch.cuda.empty_cache()
    print('ENCODE_T5',flush=True)
    tokenizer=T5Tokenizer.from_pretrained(modelroot/'tokenizer_2',local_files_only=True)
    text=T5EncoderModel.from_pretrained(modelroot/'text_encoder_2',local_files_only=True,torch_dtype=torch.bfloat16).eval().requires_grad_(False).to(device)
    tokens=tokenizer(prompts,max_length=512,padding='max_length',truncation=True,return_tensors='pt')
    txt=text(input_ids=tokens['input_ids'].to(device),attention_mask=None,output_hidden_states=False).last_hidden_state.cpu()
    t5_ids=tokens['input_ids'];del text;gc.collect();torch.cuda.empty_cache()
    h=size//16
    ids=torch.zeros(1,h,h,3);ids[...,1]=torch.arange(h)[:,None];ids[...,2]=torch.arange(h)[None,:]
    ids=ids.reshape(1,h*h,3)
    inputs={which:dict(img=packed.cpu(),img_ids=ids,txt=txt[i:i+1],txt_ids=torch.zeros(1,512,3),vec=clip[i:i+1])
            for i,which in enumerate(['source','target'])}
    torch.save(dict(inputs=inputs,vae_source=vae_source,source=source,clip_ids=clip_ids,t5_ids=t5_ids,size=size),out/'inputs.pt')
    write(out/'protocol.json',dict(date='2026-11-05',model='FLUX.1-dev',resolution=size,
        source=source,source_sha256=sha(ROOT/source['source_image']),source_encoding='original VAE posterior mode, FP32 then packed BF16',
        text_conditions='full CLIP pooler + T5 sequence, BF16, no attention mask; official SelFix convention',
        text_lengths=dict(clip=(clip_ids!=0).sum(1).tolist(),t5=(t5_ids!=0).sum(1).tolist()),
        extra_models=False,code_sha256=sha(__file__),seconds=time.monotonic()-start))
    print('PREPARE_COMPLETE',round(time.monotonic()-start,2),flush=True)


if __name__=='__main__':main()
