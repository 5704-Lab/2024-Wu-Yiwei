"""Cache the existing SD3 text encoders' exact FP32 outputs; no training."""
import gc
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from diffusers import StableDiffusion3Pipeline

from dataset import SOURCE_PROMPT,TARGET_PROMPT,COLOR_PROMPT,POSE_PROMPT

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PROMPTS=dict(source=SOURCE_PROMPT,shape=TARGET_PROMPT,color=COLOR_PROMPT,pose=POSE_PROMPT,null='')


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    path=HERE/'outputs'/'conditions_v1.pt'
    if path.exists():raise RuntimeError('Condition cache already exists')
    start=time.monotonic()
    pipe=StableDiffusion3Pipeline.from_pretrained(ROOT/'pretrained_models/sd3-medium',
        torch_dtype=torch.float32,local_files_only=True,transformer=None,vae=None)
    items={name:{} for name in PROMPTS}
    for idx,encoder in enumerate([pipe.text_encoder,pipe.text_encoder_2]):
        encoder.eval().requires_grad_(False).to('cuda:0')
        for name,prompt in PROMPTS.items():
            with torch.no_grad():s,p=pipe._get_clip_prompt_embeds(prompt,device='cuda:0',clip_model_index=idx)
            items[name][f'clip{idx}']=(s.cpu(),p.cpu())
        encoder.cpu();gc.collect();torch.cuda.empty_cache();print('CLIP_COMPLETE',idx,flush=True)
    pipe.text_encoder_3.eval().requires_grad_(False).to('cuda:0')
    for name,prompt in PROMPTS.items():
        with torch.no_grad():s=pipe._get_t5_prompt_embeds(prompt,device='cuda:0',max_sequence_length=256)
        items[name]['t5']=s.cpu()
    result={}
    for name,item in items.items():
        c0,p0=item['clip0'];c1,p1=item['clip1'];t=item['t5']
        seq=torch.cat([F.pad(torch.cat([c0,c1],-1),(0,t.shape[-1]-c0.shape[-1]-c1.shape[-1])),t],1)
        result[name]=dict(seq=seq,pool=torch.cat([p0,p1],-1))
    torch.save(dict(prompts=PROMPTS,conditions=result,seconds=time.monotonic()-start,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),path)
    print('ENCODING_COMPLETE',time.monotonic()-start,flush=True)


if __name__=='__main__':main()
