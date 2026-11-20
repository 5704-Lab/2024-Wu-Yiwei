"""Measure real trajectories and controlled decoder responses, one GPU shard.

Can run alongside inference: each source is processed after all its runs exist.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from diffusers import AutoencoderKL

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs/study_v1"
sys.path.insert(0, str(ROOT / "experiments/flowedit_delta_spectrum_20261117"))
from analyze import unpack, matrix, tensor, decompose, png
sys.path.insert(0, str(ROOT / "experiments/sota_inversion_20261105"))
from common import setup, write, sha

LOCAL = ["full", "svd4", "scalar4", "svd4_equal_norm", "tail4", "fft_low",
         "scalar_low", "fft_high", "scalar_high", "shift", "random4_matched", "smooth_low", "channel_flip"]
REPLAY = ["full", "svd4", "scalar4", "fft_low", "fft_high", "shift"]


def image_metrics(image, before, full):
    d, reference = image.double()-before.double(), full.double()-before.double()
    energy = float(d.square().mean())
    full_energy = float(reference.square().mean())
    h = image.shape[-2]
    return dict(rgb_rms=energy**.5, rgb_energy_relative=energy/max(full_energy,1e-30),
                rgb_cosine_to_full=float((d*reference).sum()/(d.norm()*reference.norm()).clamp_min(1e-30)),
                rgb_error_to_full=float((image.double()-full.double()).square().mean().sqrt()),
                top_window_rms=float(d[:,:round(h*220/1024)].square().mean().sqrt()),
                bottom_window_rms=float(d[:,round(h*850/1024):].square().mean().sqrt()),
                out_of_display_range_fraction=float(((image<0)|(image>1)).float().mean()))


def variants_for(z, v):
    """Add the predeclared energy and coordinate controls to the first pilot."""
    # Match the new main study's translation distance (64 nominal RGB pixels).
    v["shift"] = torch.roll(z, 8, -1)
    for band in ("low", "high"):
        v["scalar_"+band] = z*(v["fft_"+band].square().sum()/z.square().sum()).sqrt()
    generator = torch.Generator().manual_seed(20261117)
    q = torch.linalg.qr(torch.randn(16, 4, generator=generator, dtype=torch.float64)).Q
    a = matrix(z)
    random = (a@q)@q.T
    random *= (matrix(v["svd4"]).square().sum()/random.square().sum()).sqrt()
    v["random4_matched"] = tensor(random, z.shape)
    fy = torch.fft.fftfreq(z.shape[-2], dtype=torch.float64)
    fx = torch.fft.fftfreq(z.shape[-1], dtype=torch.float64)
    r = (fy[:, None].square()+fx[None, :].square()).sqrt()
    # Raised-cosine taper, transition at 0.046875..0.078125, compared at
    # the same latent energy as the hard low-pass filter.
    gain = .5*(1+torch.cos(np.pi*((r-.046875)/.03125).clamp(0, 1)))
    smooth = torch.fft.ifft2(torch.fft.fft2(z, norm="ortho")*gain, norm="ortho").real
    smooth *= (v["fft_low"].square().sum()/smooth.square().sum()).sqrt()
    v["smooth_low"] = smooth
    flip = z.clone()
    flip[:, ::2] *= -1
    v["channel_flip"] = flip
    return v


def extra_stats(z, modes):
    a = matrix(z)
    spatial = a.square().sum(1)
    total = spatial.sum()
    power = modes["power"]
    fy = torch.fft.fftfreq(z.shape[-2], dtype=torch.float64)
    fx = torch.fft.fftfreq(z.shape[-1], dtype=torch.float64)
    radius = (fy[:, None].square()+fx[None, :].square()).sqrt()
    eigen = 4*torch.sin(np.pi*fy[:, None]).square()+4*torch.sin(np.pi*fx[None, :]).square()
    return dict(spatial_participation_fraction=float(total.square()/spatial.square().sum()/len(spatial)),
                spectral_centroid=float((power*radius).sum()), laplacian_roughness=float((power*eigen).sum()),
                rank4_relative_gap=float((modes["s"][3]-modes["s"][4])/modes["s"][0]))


def board(path, images, title, cols):
    size = 192
    canvas = Image.new("RGB", (len(cols)*size, size+65), "white")
    d = ImageDraw.Draw(canvas)
    d.text((5, 4), title, fill="black")
    for i, name in enumerate(cols):
        d.text((i*size+3, 35), name, fill="black")
        canvas.paste(png(images[name]).resize((size, size)), (i*size, 62))
    canvas.save(path)


def additivity(images, before, step, alpha):
    delta = {m:x.double()-before.double() for m,x in images.items()}
    a, b, full = delta["svd4"], delta["tail4"], delta["full"]
    return dict(step=step, alpha=alpha,
        rgb_remainder=float((full-a-b).norm()/full.norm().clamp_min(1e-30)),
        decoded_component_cosine=float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-30)))


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, choices=[0, 1], required=True)
    ap.add_argument("--wait", action="store_true")
    args = ap.parse_args()
    setup()
    device = f"cuda:{args.worker}"
    torch.cuda.set_device(device)
    protocol = json.loads((OUT / "protocol.json").read_text())
    start = time.monotonic()
    vae = AutoencoderKL.from_pretrained(ROOT/"pretrained_models/flux1-dev/vae", local_files_only=True,
                                        torch_dtype=torch.float32).eval().requires_grad_(False).to(device)
    calls = 0
    def decode(z):
        nonlocal calls
        result = (vae.decode(z.to(device, dtype=torch.float32)/vae.config.scaling_factor+vae.config.shift_factor,
                             return_dict=False)[0]+1)/2
        calls += 1
        return result[0].cpu()

    for record in protocol["records"][args.worker::2]:
        pair = record["pair_id"]
        dest = OUT/"measurements"/pair
        dest.mkdir(parents=True, exist_ok=True)
        if (dest/"complete.json").exists():
            continue
        expected = [(seed, method) for seed in protocol["seeds"]
                    for method in (protocol["arms"] if seed==0 and pair in protocol["online_sources"] else ["full"])]
        while not all((OUT/"runs"/pair/f"seed{s}"/m/"complete.json").exists() for s,m in expected):
            if not args.wait:
                raise RuntimeError(f"Missing runs for {pair}")
            # Worker process wait; the assistant remains responsive separately.
            time.sleep(5)
        pair_start, before_calls = time.monotonic(), calls
        rows, algebra, trajectory_summaries, endpoints = [], [], [], {}
        representations = []
        replay_states = None
        seed0 = None
        for seed in protocol["seeds"]:
            run = torch.load(OUT/"runs"/pair/f"seed{seed}"/"full"/"trajectory.pt", weights_only=True)
            states = [unpack(s.double()) for s in run["states"]]
            if seed == 0:
                seed0 = run
                replay_states = {m:states[0].clone() for m in REPLAY}
            all_modes = []
            for k in range(24):
                actual = states[k+1]-states[k]
                intended = unpack(run["raw_updates"][k].double())
                for kind, z in [("actual", actual), ("intended", intended)]:
                    stats, audit, variants, modes = decompose(z)
                    stats.update(extra_stats(z, modes))
                    stats.update(pair=pair, category=record["category"], seed=seed, step=k+1, kind=kind,
                                 h=run["timesteps"][k]-run["timesteps"][k+1], **{
                                     "rounding_relative":run["rounding"][k]["rounding_relative"],
                                     "unchanged_fraction":run["rounding"][k]["unchanged_fraction"]})
                    stats["rate_rms"] = stats["rms"]/stats["h"]
                    rows.append(stats)
                    algebra.append(max(audit.values()))
                    if kind == "actual":
                        all_modes.append(modes)
                        if seed == 0:
                            variants["shift"] = torch.roll(z, 8, -1)
                            for m in REPLAY:
                                replay_states[m] += variants[m]
            representations.append(np.stack([m["vh"].numpy() for m in all_modes]))
            total = states[-1]-states[0]
            sum_energy = sum((states[k+1]-states[k]).square().sum() for k in range(24))
            trajectory_summaries.append(dict(seed=seed, temporal_accumulation_ratio=float(total.square().sum()/sum_energy),
                                            final_displacement_rms=float(total.square().mean().sqrt())))
            endpoints[f"full_seed{seed}"] = decode(states[-1])
            png(endpoints[f"full_seed{seed}"]).save(dest/f"full_seed{seed}.png")
        source_z = unpack(seed0["states"][0].double())
        source_image = decode(source_z)
        endpoints["source_vae"] = source_image
        online_metrics = []
        for seed, method in expected:
            if method == "full":
                image = endpoints[f"full_seed{seed}"]
            else:
                run = torch.load(OUT/"runs"/pair/f"seed{seed}"/method/"trajectory.pt", weights_only=True)
                image = decode(unpack(run["states"][-1].double()))
                endpoints[method] = image
                png(image).save(dest/f"online_{method}.png")
            row = image_metrics(image, source_image, endpoints[f"full_seed{seed}"])
            row.update(pair=pair, seed=seed, method=method, category=record["category"])
            online_metrics.append(row)

        replay_images = {m:decode(z) for m,z in replay_states.items()}
        replay_error = float((replay_images["full"]-endpoints["full_seed0"]).abs().max())
        assert replay_error == 0
        board(dest/"frozen_replay.png", {"source":source_image, **replay_images},
              f"{pair}: FROZEN increments, seed0; no velocity re-query", ["source"]+REPLAY)
        for m, im in replay_images.items():
            png(im).save(dest/f"replay_{m}.png")
        replay_metrics = [dict(method=m, **image_metrics(im, source_image, endpoints["full_seed0"])) for m,im in replay_images.items()]

        local_metrics, remainder = [], []
        for step in [1, 12, 24]:
            before_z = unpack(seed0["states"][step-1].double())
            delta = unpack(seed0["states"][step].double())-before_z
            _, _, variants, modes = decompose(delta)
            variants = variants_for(delta, variants)
            before_image = decode(before_z)
            images = {m:decode(before_z+variants[m]) for m in LOCAL}
            for m,image in images.items():
                stats = image_metrics(image, before_image, images["full"])
                stats.update(step=step, method=m, pair=pair, category=record["category"],
                             latent_energy_relative=float(variants[m].square().sum()/delta.square().sum()))
                local_metrics.append(stats)
            remainder.append(additivity(images, before_image, step, 1.))
            board(dest/f"local_step{step:02d}.png", {"before":before_image, **images},
                  f"{pair}: LOCAL step {step}; all arms share pre-state", ["before", "full", "svd4", "scalar4", "random4_matched", "fft_low", "fft_high", "shift"])
            if step == 24:
                for alpha in [.25, .5]:
                    small = {m:decode(before_z+alpha*variants[m]) for m in ["full", "svd4", "tail4"]}
                    remainder.append(additivity(small, before_image, step, alpha))
                channel = variants["channel_flip"]
                # Orthogonal channel sign flip keeps BOTH singular values and
                # channel-summed spatial Fourier power, but not decoder response.
                channel_s = torch.linalg.svdvals(matrix(channel))
                channel_p = torch.fft.fft2(channel, norm="ortho").abs().square().sum((0,1))
                orig_p = torch.fft.fft2(delta, norm="ortho").abs().square().sum((0,1))
                assert float((channel_s-modes["s"]).norm()/modes["s"].norm()) < 1e-10
                assert float((channel_p-orig_p).norm()/orig_p.norm()) < 1e-10
                board(dest/"two_blind_spots.png", {"before":before_image, **images},
                      f"{pair}: SAME singular values + summed Fourier power; changed spatial/channel alignment", ["before", "full", "shift", "channel_flip"])

        write(dest/"step_statistics.json", rows)
        write(dest/"online_metrics.json", online_metrics)
        write(dest/"replay_metrics.json", replay_metrics)
        write(dest/"local_metrics.json", local_metrics)
        write(dest/"nonlinearity.json", remainder)
        write(dest/"trajectory_metrics.json", trajectory_summaries)
        np.savez_compressed(dest/"channel_bases.npz", vh=np.stack(representations))
        if pair in protocol["online_sources"]:
            board(dest/"online.png", endpoints, f"{pair}: ONLINE state-dependent FlowEdit, seed0", ["source_vae", "full_seed0"]+protocol["arms"][1:])
        board(dest/"seeds.png", endpoints, f"{pair}: three seeds, same source and text", ["source_vae", "full_seed0", "full_seed1", "full_seed2"])
        write(dest/"complete.json", dict(pair=pair, max_algebra_error=max(algebra),
            vae_calls=calls-before_calls, seconds=time.monotonic()-pair_start, source_count=1,
            full_replay_rgb_max_error=replay_error, measurement_script_sha256=sha(Path(__file__))))
        print(json.dumps(dict(pair=pair, measured=True, vae_calls=calls, seconds=time.monotonic()-start)), flush=True)
        gc.collect()
    write(OUT/f"measurement_worker{args.worker}_complete.json", dict(worker=args.worker, vae_calls=calls, seconds=time.monotonic()-start, script_sha256=sha(Path(__file__))))


if __name__ == "__main__":
    main()
