"""Interface tests and paired native observations of actual clean latents."""
import argparse
import hashlib
import json
import time
import platform
import torch
import diffusers
from diffusers import SD3Transformer2DModel
from probe import ROOT, HERE, NativeProbe, cpu_observation, write, digest, LAYERS, CONCEPTS

OLD = ROOT/'experiments/semantic_feedback_20261028'
PROBES = [(.20, 19571030), (.20, 19571031), (.35, 19571030), (.35, 19571031)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='dataset_v2')
    ap.add_argument('--latents', default='features_v2')
    ap.add_argument('--tag', default='native_v1')
    ap.add_argument('--scenes', nargs='+', type=int)
    ap.add_argument('--variants', nargs='+')
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    bank = torch.load(HERE/'outputs/bank_v1/bank.pt', weights_only=True)
    conditions = torch.load(HERE/'outputs/bank_v1/conditions.pt', weights_only=True)
    cond = conditions['conditions']['shape']
    manifest = json.loads((OLD/'outputs'/args.dataset/'manifest.json').read_text())
    out = HERE/'outputs'/args.tag
    weight_hash = hashlib.sha256()
    with (ROOT/'pretrained_models/sd3-medium/transformer/diffusion_pytorch_model.safetensors').open('rb') as f:
        for chunk in iter(lambda: f.read(16*1024*1024), b''):
            weight_hash.update(chunk)
    protocol = dict(dataset=args.dataset, dataset_hash=hashlib.sha256((OLD/'outputs'/args.dataset/'manifest.json').read_bytes()).hexdigest(),
        layers=LAYERS, probes=PROBES, prompt=conditions['prompts']['shape'],
        condition_hash=digest(cond['seq'])+digest(cond['pool']), concept_hash=digest(bank['initial']), concepts=CONCEPTS,
        source_queries='source dynamic query/key/value frozen independently per layer/probe; current image K/V live',
        dynamic='all prefix blocks 0 through 20; original context norm/gates/MLP; concept reads image + concept bank only',
        precision='float32, native SDPA; original processor untouched', factors_used_in_observation=False,
        model_weights_sha256=weight_hash.hexdigest(), model_config_sha256=hashlib.sha256((ROOT/'pretrained_models/sd3-medium/transformer/config.json').read_bytes()).hexdigest(),
        latent_convention='original cached scaled SD3 VAE posterior mode; source latent hash in every record',
        libraries=dict(torch=torch.__version__, diffusers=diffusers.__version__, cuda=torch.version.cuda, python=platform.python_version()),
        coordinates='32x32 patch-center normalized image coordinates; no resampling alignment',
        map_definitions='QK conditional spatial probability; unnormalized O dot B; O/B cosine; concept competition evaluated offline without spatial update mask',
        code_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'probe.py', HERE/'run.py']})
    if (out/'protocol.json').exists():
        assert json.loads((out/'protocol.json').read_text()) == json.loads(json.dumps(protocol)), 'Use fresh tag'
    else:
        write(out/'protocol.json', protocol)
    device = 'cuda:0'
    model = SD3Transformer2DModel.from_pretrained(ROOT/'pretrained_models/sd3-medium/transformer',
        torch_dtype=torch.float32, local_files_only=True).eval().requires_grad_(False).to(device)
    probe = NativeProbe(model, bank)
    start = time.monotonic()
    count = 0
    for row in manifest['records']:
        scene = row['scene']
        if args.scenes is not None and scene not in args.scenes:
            continue
        folder = OLD/'outputs'/args.latents/f'scene{scene:02d}'
        source = torch.load(folder/'source.pt', weights_only=True)['y'].to(device)
        source_bank = []
        with torch.no_grad():
            for tau, seed in PROBES:
                eps = torch.randn(source.shape, generator=torch.Generator('cpu').manual_seed(seed)).to(source)
                source_bank.append(probe.observe(source, eps, tau, cond))
        if not (out/'interface.json').exists():
            with torch.no_grad():
                tau, seed = PROBES[0]
                eps = torch.randn(source.shape, generator=torch.Generator('cpu').manual_seed(seed)).to(source)
                native = model(hidden_states=(1-tau)*source+tau*eps,
                    timestep=torch.tensor([tau*1000], device=device), encoder_hidden_states=cond['seq'].to(source),
                    pooled_projections=cond['pool'].to(source), return_dict=False)[0]
                repeated = probe.observe(source, eps, tau, cond, source_bank[0]['queries'])
                errors = {f'{l}:{k}': float((v-repeated['layers'][l][k]).abs().max())
                    for l, layer in source_bank[0]['layers'].items() for k, v in layer.items()}
                anchor_errors = {f'{l}:{facet}': float((repeated['layers'][l]['dynamic_'+facet]-repeated['layers'][l]['anchored_'+facet]).abs().max())
                    for l in LAYERS for facet in ['B', 'probability', 'mass', 'cosine_map', 'dot_map']}
            write(out/'interface.json', dict(original_vs_observed_max_abs=float((native-source_bank[0]['prediction']).abs().max()),
                identical_pair_max_abs=errors, source_anchor_dynamic_identity_max_abs=anchor_errors,
                hook_handles_after_call=sum(len(m._forward_pre_hooks) for m in model.modules()),
                all_prefix_blocks_propagated=True, exact_zero_expected=True))
            print('INTERFACE', json.loads((out/'interface.json').read_text()), flush=True)
        for name in row['variants']:
            if args.variants is not None and name not in args.variants:
                continue
            path = out/f'scene{scene:02d}'/f'{name}.pt'
            if path.exists():
                continue
            y = torch.load(folder/f'{name}.pt', weights_only=True)['y'].to(device)
            readings = []
            with torch.no_grad():
                for (tau, seed), src in zip(PROBES, source_bank):
                    eps = torch.randn(y.shape, generator=torch.Generator('cpu').manual_seed(seed)).to(y)
                    obs = probe.observe(y, eps, tau, cond, src['queries'])
                    readings.append(cpu_observation(obs))
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(scene=scene, variant=name, source_hash=digest(source), latent_hash=digest(y), observations=readings), path)
            count += 1
            print('NATIVE', args.tag, scene, name, count, round(time.monotonic()-start, 2), flush=True)
        del source_bank
    write(out/f'timing_{int(time.time())}.json', dict(states_this_run=count, seconds=time.monotonic()-start,
        peak_memory_bytes=torch.cuda.max_memory_allocated()))


if __name__ == '__main__':
    main()
