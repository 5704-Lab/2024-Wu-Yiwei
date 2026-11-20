"""Additional saved clean states; separate manifest, preserves first data stage."""
import torch
from diffusers import AutoencoderKL
from common import *


def main():
    setup();out=HERE/'outputs/trajectory_data_v1'
    if (out/'manifest.json').exists():raise RuntimeError('Immutable data exists')
    vae=AutoencoderKL.from_pretrained(ROOT/'pretrained_models/sd3-medium/vae',local_files_only=True,
        torch_dtype=torch.float32).eval().requires_grad_(False).to('cuda:0')
    records=[]
    for pair in PAIRS:
        inp=torch.load(ROOT/'experiments/stage1_full/inputs'/f'{pair}.pt',weights_only=True)
        saved=torch.load(ROOT/'experiments/stage1_full/outputs'/f'{pair}__seed0/states.pt',weights_only=True)
        dest=out/pair;dest.mkdir(parents=True,exist_ok=True)
        torch.save(dict(y=inp['x']),dest/'source.pt')
        states={f'flow_step{s["step"]}':s['zE'] for s in saved['snapshots'] if s['step']!=33}
        states['flow_final']=saved['baseline_final']
        variants={}
        with torch.no_grad():
            for name,y in states.items():
                torch.save(dict(y=y),dest/f'{name}.pt');save_image(image(vae,y.to('cuda:0')),dest/f'{name}.png')
                variants[name]=dict(kind='actual_flowedit_clean_state',known_geometric_truth=False)
        records.append(dict(pair=pair,split=split(pair),record=inp['record'],variants=variants))
        print('TRAJECTORY_DATA',pair,list(states),flush=True)
    write(out/'manifest.json',dict(records=records,script_sha256=sha(__file__),source_count=len(records),
        input_coordinate_verified='original zE=z+delta, n_min=0 clean FlowEdit; final is baseline_final'))


if __name__=='__main__':main()
