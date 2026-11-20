"""Post hoc bounded-fit sensitivity on saved 8-bit endpoints; no image changes."""
import json
import numpy as np
import torch
from PIL import Image
from common import OUT,PAIRS,Phase,Appearance,seed_setup,write,code_hashes


def tensor(path):
    a=np.asarray(Image.open(path).convert('RGB')).copy()/255.
    return torch.tensor(a,dtype=torch.float64).permute(2,0,1)[None]


def main():
    seed_setup();rows=[]
    if (OUT/'fit_sensitivity.json').exists():rows=json.loads((OUT/'fit_sensitivity.json').read_text())['rows']
    for pair in PAIRS:
        source=OUT/'real'/pair/'source_vae.png'
        if not source.exists():continue
        phase=Phase(tensor(source));app=Appearance(phase.source_small,2)
        for method in ['BASE','PERMIT','RAW_PHASE','NORM_CONTROL']:
            dest=OUT/'full'/pair/method
            if not (dest/'metrics.json').exists():continue
            if any(r['pair']==pair and r['method']==method for r in rows):continue
            target=phase(tensor(dest/'final.png')).detach()
            old=json.loads((dest/'metrics.json').read_text())['metrics']['permitted_phase']
            theta,ref,fit=app.fit(phase,target,maxiter=400)
            ref_image=app(theta).detach()
            rows.append(dict(pair=pair,method=method,inference_float32_40iter=old,
                             png_float64_400iter=float((target-ref).norm()),fit=fit,
                             reference_min=float(ref_image.min()),reference_max=float(ref_image.max()),
                             reference_outside_display_fraction=float(((ref_image<0)|(ref_image>1)).float().mean())))
            write(OUT/'fit_sensitivity.json',dict(rows=rows,scope='8-bit PNG quantization and longer fit both differ; diagnostic only',code=code_hashes()))
            print(pair,method,old,float((target-ref).norm()),fit['success'],flush=True)
    print('POLISH_COMPLETE',flush=True)


if __name__=='__main__':main()
