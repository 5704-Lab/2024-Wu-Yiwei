"""Use the existing SD3 text encoders, sequential FP32 GPU residency."""
import gc
import time
import torch
import torch.nn.functional as F
from diffusers import StableDiffusion3Pipeline

from common import ROOT, OUT, PAIRS, SHORT, write, protocol


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    protocol("encode_conditions",dict(sequential_encoder_residency=True,added_model=False))
    path=OUT/"short_condition_encodings.pt"
    if path.exists():
        print("EXISTING_CONDITION_ENCODINGS",flush=True);return
    started=time.monotonic()
    pipe=StableDiffusion3Pipeline.from_pretrained(ROOT/"pretrained_models/sd3-medium",
        torch_dtype=torch.float32,local_files_only=True,transformer=None,vae=None)
    payloads={p:torch.load(ROOT/"experiments/stage1_full/inputs"/(p+".pt"),weights_only=True) for p in PAIRS}
    prompts=[""]+[payloads[p]["record"]["source_prompt"] for p in PAIRS]+[
        t for p in PAIRS for t in SHORT[p]]
    items={t:{} for t in prompts}
    for idx,encoder in enumerate([pipe.text_encoder,pipe.text_encoder_2]):
        encoder.eval().requires_grad_(False).to("cuda:0")
        for text in prompts:
            with torch.no_grad():s,p=pipe._get_clip_prompt_embeds(text,device="cuda:0",clip_model_index=idx)
            items[text]["clip"+str(idx)]=(s.cpu(),p.cpu())
        encoder.cpu();torch.cuda.empty_cache()
        print("CLIP_ENCODER_COMPLETE",idx,flush=True)
    pipe.text_encoder_3.eval().requires_grad_(False).to("cuda:0")
    for text in prompts:
        with torch.no_grad():t=pipe._get_t5_prompt_embeds(text,device="cuda:0",max_sequence_length=256)
        items[text]["t5"]=t.cpu()
    pipe.text_encoder_3.cpu();torch.cuda.empty_cache()
    result={}
    for text,item in items.items():
        c0,p0=item["clip0"];c1,p1=item["clip1"];t=item["t5"]
        seq=torch.cat([F.pad(torch.cat([c0,c1],dim=-1),(0,t.shape[-1]-c0.shape[-1]-c1.shape[-1])),t],dim=1)
        result[text]=(seq,torch.cat([p0,p1],dim=-1))
    audits=[];shorts={}
    negative=result[""]
    for p in PAIRS:
        old=payloads[p]["embeddings"][0]
        new=result[payloads[p]["record"]["source_prompt"]]
        errors=[float((new[0]-old[0]).norm()/old[0].norm()),
                float((new[1]-old[2]).norm()/old[2].norm()),
                float((negative[0]-old[1]).norm()/old[1].norm()),
                float((negative[1]-old[3]).norm()/old[3].norm())]
        audits.append(dict(pair=p,relative_errors=errors,max_error=max(errors)))
        if max(errors)>1e-6:raise RuntimeError("FP32 text encoding compatibility failed: "+p)
        shorts[p]=[(result[t][0],negative[0],result[t][1],negative[1]) for t in SHORT[p]]
    torch.save(dict(embeddings=shorts,short=SHORT,audits=audits),path)
    write(OUT/"encoding_audit.json",dict(audits=audits,precision="fp32",seconds=time.monotonic()-started))
    print("CONDITION_ENCODINGS_COMPLETE",audits,flush=True)


if __name__=="__main__":main()
