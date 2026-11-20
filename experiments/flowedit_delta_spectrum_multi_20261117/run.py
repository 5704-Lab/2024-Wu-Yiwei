"""Paired real FLUX trajectories, split over two independent GPU workers."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs/study_v1"
sys.path.insert(0, str(ROOT / "experiments/sota_inversion_20261105"))
from common import setup, load_model, base_info, get_schedule, write, sha
sys.path.insert(0, str(ROOT / "experiments/flowedit_delta_spectrum_20261117"))
from analyze import unpack, matrix, tensor
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("multi_prepare", HERE / "prepare.py")
prepare = module_from_spec(spec)
spec.loader.exec_module(prepare)


def transform(delta, method):
    if method == "full":
        return delta
    z = unpack(delta)
    if method in ("svd4", "scalar4"):
        a = matrix(z)
        _, v = torch.linalg.eigh(a.T @ a)
        v4 = v[:, -4:]
        selected = tensor((a @ v4) @ v4.T, z.shape)
        if method == "scalar4":
            selected = z * (selected.square().sum()/z.square().sum()).sqrt()
    elif method == "shift":
        selected = torch.roll(z, 8, -1)
    else:
        f = torch.fft.fft2(z, norm="ortho")
        fy = torch.fft.fftfreq(z.shape[-2], device=z.device)
        fx = torch.fft.fftfreq(z.shape[-1], device=z.device)
        radius = (fy[:, None].square()+fx[None, :].square()).sqrt()
        mask = radius < .0625 if method in ("fft_low", "scalar_low") else radius >= .25
        selected = torch.fft.ifft2(f*mask, norm="ortho").real
        if method.startswith("scalar"):
            selected = z * (selected.square().sum()/z.square().sum()).sqrt()
    return prepare.pack(selected)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, choices=[0, 1], required=True)
    args = ap.parse_args()
    setup()
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert (OUT / "prepared.json").exists()
    device = torch.device(f"cuda:{args.worker}")
    torch.cuda.set_device(device)
    start = time.monotonic()
    model = load_model(device, OUT / f"weight_audit_worker{args.worker}.json")
    write(OUT / f"ready_worker{args.worker}.json", dict(seconds=time.monotonic()-start, script_sha256=sha(Path(__file__))))
    calls, done, resumed = 0, 0, 0
    all_stats = []
    for record in protocol["records"][args.worker::2]:
        pair = record["pair_id"]
        payload = torch.load(OUT / "inputs" / pair / "input.pt", weights_only=True)
        x = payload["x"].to(device)
        img_ids = payload["img_ids"].to(device)
        conditions = {n:{k:v.to(device) for k,v in d.items()} for n,d in payload["conditions"].items()}
        for seed in protocol["seeds"]:
            methods = protocol["arms"] if seed == 0 and pair in protocol["online_sources"] else ["full"]
            # Each independent arm consumes precisely the same fixed noises.
            gen = torch.Generator("cpu").manual_seed(seed)
            noises = [torch.randn(x.shape, generator=gen, dtype=torch.bfloat16).to(device) for _ in range(24)]
            for method in methods:
                dest = OUT / "runs" / pair / f"seed{seed}" / method
                dest.mkdir(parents=True, exist_ok=True)
                if (dest / "complete.json").exists():
                    resumed += 1
                    continue
                job_start = time.monotonic()
                y = x.clone()
                history, updates, applied, rounding = [y.cpu()], [], [], []
                schedule = get_schedule(28, x.shape[1], shift=True)[4:]
                info = base_info(0)
                info.update(inject=False, inverse=False, second_order=False, t=0., feature_injection_mode="off")
                for k, (t, tn) in enumerate(zip(schedule[:-1], schedule[1:])):
                    qs = (1-t)*x + t*noises[k]
                    qt = y + qs - x
                    velocities = []
                    for query, condition, g in [(qs, conditions["source"], 1.5), (qt, conditions["target"], 5.5)]:
                        prediction, _ = model(img=query, img_ids=img_ids, txt=condition["txt"], txt_ids=condition["txt_ids"],
                            y=condition["vec"], timesteps=torch.full((1,), t, device=device, dtype=x.dtype),
                            guidance=torch.full((1,), g, device=device, dtype=x.dtype), info=info)
                        velocities.append(prediction)
                        calls += 1
                    raw = (tn-t)*(velocities[1].float()-velocities[0].float())
                    proposal = transform(raw, method)
                    next_y = (y.float()+proposal).to(torch.bfloat16)
                    assert bool(torch.isfinite(next_y).all())
                    actual = next_y.float()-y.float()
                    rounding.append(dict(step=k+1, proposed_rms=float(proposal.square().mean().sqrt()),
                        actual_rms=float(actual.square().mean().sqrt()),
                        rounding_relative=float((actual-proposal).norm()/proposal.norm().clamp_min(1e-20)),
                        unchanged_fraction=float((actual==0).float().mean())))
                    updates.append(raw.cpu())
                    applied.append(proposal.cpu())
                    history.append(next_y.cpu())
                    y = next_y
                torch.save(dict(record=record, seed=seed, method=method, states=torch.stack(history),
                                raw_updates=torch.stack(updates), transformed_updates=torch.stack(applied),
                                timesteps=schedule, rounding=rounding), dest / "trajectory.pt")
                row = dict(pair=pair, seed=seed, method=method, transformer_calls=48,
                           seconds=time.monotonic()-job_start, complete=True)
                write(dest / "complete.json", row)
                all_stats.append(row)
                done += 1
                print(json.dumps(dict(worker=args.worker, done=done, **row), ensure_ascii=False), flush=True)
        del payload, x, img_ids, conditions, noises, history, updates, applied, next_y, y, prediction, raw, proposal, actual
        gc.collect()
    write(OUT / f"worker{args.worker}_complete.json", dict(worker=args.worker, new_runs=done, resumed=resumed,
        transformer_calls=calls, seconds=time.monotonic()-start, runs=all_stats, script_sha256=sha(Path(__file__))))
    print("WORKER_COMPLETE", args.worker, done, calls, flush=True)


if __name__ == "__main__":
    main()
