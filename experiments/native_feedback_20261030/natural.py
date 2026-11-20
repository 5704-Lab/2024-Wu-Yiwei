"""One real bear source: actual saved clean trajectory and geometry controls.

No invented correctness label for the edited endpoint. This diagnoses whether
two-instance binding alone explains the failure on controlled clay images.
"""
import json
import torch
from transformers import AutoTokenizer
from diffusers import SD3Transformer2DModel
from probe import ROOT, HERE, NativeProbe, cpu_observation, digest, write
from run import PROBES
from analyze import spatial, wrapped


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = HERE/'outputs/natural_bear_v1'
    if out.exists():
        raise RuntimeError('Immutable natural diagnostic exists')
    inp = torch.load(ROOT/'experiments/stage1_full/inputs/bear__00.pt',weights_only=True)
    target = inp['embeddings'][1]
    cond = dict(seq=target[0],pool=target[2])
    masks = []
    for folder, length, offset in [('tokenizer',77,0),('tokenizer_3',256,77)]:
        tok = AutoTokenizer.from_pretrained(ROOT/'pretrained_models/sd3-medium'/folder,local_files_only=True)
        t = tok(inp['record']['target_prompt'], padding='max_length',max_length=length,truncation=True,return_tensors='pt')
        m = t.attention_mask[0].bool()
        for special in tok.all_special_ids:
            m &= t.input_ids[0] != special
        full = torch.zeros(cond['seq'].shape[1],dtype=torch.bool)
        full[offset:offset+length] = m
        masks.append(full)
    cond['content_masks'] = masks
    bank = torch.load(HERE/'outputs/bank_v1/bank.pt',weights_only=True)
    model = SD3Transformer2DModel.from_pretrained(ROOT/'pretrained_models/sd3-medium/transformer',
        torch_dtype=torch.float32, local_files_only=True).eval().requires_grad_(False).to('cuda:0')
    probe = NativeProbe(model,bank)
    folder = ROOT/'experiments/semantic_feedback_20261028/outputs/natural_v1/bear__00'
    x = torch.load(folder/'source.pt',weights_only=True)['y'].to('cuda:0')
    src = []
    with torch.no_grad():
        for tau, seed in PROBES:
            eps = torch.randn(x.shape,generator=torch.Generator('cpu').manual_seed(seed)).to(x)
            src.append(probe.observe(x,eps,tau,cond))
    write(out/'protocol.json',dict(pair='bear__00', condition=inp['record']['target_prompt'],
        source_hash=digest(x),prompt_hash=digest(cond['seq'])+digest(cond['pool']),
        probes=PROBES,concepts=bank['concepts'], note='generic head/nose/body only for this single-instance diagnostic; no natural category correctness labels'))
    records = {}
    for name in ['source','source_dark','source_mirror','source_shift','step25','step33','step41','baseline','JOINT']:
        y = torch.load(folder/f'{name}.pt',weights_only=True)['y'].to(x)
        readings = []
        with torch.no_grad():
            for (tau,seed),s in zip(PROBES,src):
                eps = torch.randn(y.shape,generator=torch.Generator('cpu').manual_seed(seed)).to(y)
                readings.append(cpu_observation(probe.observe(y,eps,tau,cond,s['queries'])))
        torch.save(dict(observations=readings,variant=name),out/f'{name}.pt')
        records[name] = readings
        print('NATURAL',name,flush=True)
    result = {}
    a = (torch.arange(32).double()+.5)/32-.5
    yy,xx = torch.meshgrid(a,a,indexing='ij')
    xy = torch.stack([xx,yy],-1).reshape(-1,2)
    for branch in ['dynamic','anchored']:
        for mode in ['qk','cos_competition']:
            for layer in [3,11,20]:
                stem = f'{branch}/{mode}/l{layer}'
                per_state = {}
                for name,readings in records.items():
                    lengths,angles = [],[]
                    for obs in readings:
                        p = spatial(obs['layers'][layer],branch,mode)
                        centers = p@xy
                        v = centers[2]-centers[1]  # generic nose - head
                        lengths.append(v.norm())
                        angles.append(torch.atan2(v[1],v[0]))
                    per_state[name] = dict(part_separation_pixels=float(torch.stack(lengths).mean()*512),
                        angles=torch.stack(angles).tolist())
                source_angle = torch.tensor(per_state['source']['angles'])
                for name,row in per_state.items():
                    row['mean_absolute_source_angle_delta_degrees'] = float(wrapped(torch.tensor(row['angles'])-source_angle).abs().mean()*180/torch.pi)
                result[stem] = per_state
    write(out/'result.json',dict(results=result, independent_sources=1,
        endpoint_success_evaluated=False, extra_trained_models=False,
        limitation='single real example; geometric center spacing is not a physical landmark accuracy certificate'))


if __name__ == '__main__':
    main()
