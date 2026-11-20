"""Describe cached FlowEdit increments and decode controlled offline interventions.

No Transformer calls. Frozen-increment replay is not a new FlowEdit trajectory.
Run from the repository root with .venv/bin/python.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
METHODS = ["full", "svd1", "svd4", "svd8", "tail4", "svd4_equal_norm",
           "scalar4", "dc_centered4", "fft_dc", "fft_low", "fft_mid",
           "fft_high", "shift"]
DISPLAY = ["full", "svd4", "scalar4", "svd4_equal_norm", "tail4",
           "fft_low", "fft_high", "shift"]


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unpack(packed):
    """Native FLUX: (b,h*w,c*2*2) -> (b,c,h*2,w*2)."""
    b, n, d = packed.shape
    h = int(n ** .5)
    assert h * h == n and d % 4 == 0
    return packed.reshape(b, h, h, d // 4, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(b, d // 4, h * 2, h * 2)


def matrix(z):
    return z[0].flatten(1).T.contiguous()


def tensor(a, shape):
    return a.T.reshape(shape).contiguous()


def masks_for(z):
    h, w = z.shape[-2:]
    fy, fx = torch.fft.fftfreq(h, dtype=z.dtype), torch.fft.fftfreq(w, dtype=z.dtype)
    radius = (fy[:, None].square() + fx[None, :].square()).sqrt()
    return [radius < .0625, (radius >= .0625) & (radius < .25), radius >= .25]


def decompose(z):
    a = matrix(z)
    u, s, vh = torch.linalg.svd(a, full_matrices=False)
    energy = a.square().sum()
    p = s.square() / energy
    mean = a.mean(0, keepdim=True)
    ac = a - mean
    uc, sc, vc = torch.linalg.svd(ac, full_matrices=False)
    cov = ac.T @ ac / (len(a) - 1)
    eigenvalues = torch.linalg.eigvalsh(cov).flip(0)
    f = torch.fft.fft2(z, norm="ortho")
    power = f.abs().square().sum((0, 1))
    bands = masks_for(z)
    band_energy = [power[m].sum() for m in bands]
    # Fourier acts only on U's spatial axis. Orthogonality of V makes this
    # joint rank/frequency table additive even after frequency restriction.
    spatial_modes = (u * s).T.reshape(len(s), *z.shape[-2:])
    mode_power = torch.fft.fft2(spatial_modes, norm="ortho").abs().square()
    joint = torch.stack([mode_power[:, m].sum(1) for m in bands], 1)
    truncated = {r: (u[:, :r] * s[:r]) @ vh[:r] for r in (1, 4, 8)}
    rho4 = float(p[:4].sum())
    variants = {"full": z.clone()}
    variants.update({f"svd{r}": tensor(v, z.shape) for r, v in truncated.items()})
    variants["tail4"] = z - variants["svd4"]
    variants["svd4_equal_norm"] = variants["svd4"] / rho4 ** .5
    variants["scalar4"] = z * rho4 ** .5
    variants["dc_centered4"] = tensor(mean + (uc[:, :4] * sc[:4]) @ vc[:4], z.shape)
    dc = torch.zeros_like(power, dtype=torch.bool)
    dc[0, 0] = True
    variants["fft_dc"] = torch.fft.ifft2(f * dc, norm="ortho").real
    for name, mask in zip(("fft_low", "fft_mid", "fft_high"), bands):
        variants[name] = torch.fft.ifft2(f * mask, norm="ortho").real
    variants["shift"] = torch.roll(z, (0, 16), (-2, -1))
    shifted = matrix(variants["shift"])
    shifted_power = torch.fft.fft2(variants["shift"], norm="ortho").abs().square().sum((0, 1))
    reconstruction = tensor((u * s) @ vh, z.shape)
    checks = dict(
        svd_reconstruction_relative=float((reconstruction - z).norm() / z.norm()),
        parseval_relative=float(abs(power.sum() - energy) / energy),
        covariance_svd_relative=float((eigenvalues - sc.square() / (len(a) - 1)).norm() / eigenvalues.norm()),
        band_reconstruction_relative=float((sum(variants[n] for n in ("fft_low", "fft_mid", "fft_high")) - z).norm() / z.norm()),
        joint_marginal_rank_relative=float((joint.sum(1) - s.square()).norm() / s.square().norm()),
        joint_marginal_frequency_relative=float((joint.sum(0) - torch.stack(band_energy)).norm() / energy),
        shift_singular_values_relative=float((torch.linalg.svdvals(shifted) - s).norm() / s.norm()),
        shift_power_relative=float((shifted_power - power).norm() / power.norm()),
        shift_covariance_relative=float((shifted.T @ shifted - a.T @ a).norm() / (a.T @ a).norm()),
        rank4_residual_identity_relative=float(abs(variants["tail4"].square().sum() - s[4:].square().sum()) / energy),
        equal_energy_control_relative=float(abs(variants["svd4"].square().sum() - variants["scalar4"].square().sum()) / energy),
    )
    assert max(checks.values()) < 1e-9, checks
    stats = dict(energy=float(energy), rms=float(z.square().mean().sqrt()),
                 rho1=float(p[0]), rho4=rho4, rho8=float(p[:8].sum()),
                 stable_rank=float(energy / s[0].square()),
                 effective_rank=float(torch.exp(-(p * p.clamp_min(1e-300).log()).sum())),
                 dc_fraction=float(power[0, 0] / energy),
                 low_fraction=float(band_energy[0] / energy),
                 mid_fraction=float(band_energy[1] / energy),
                 high_fraction=float(band_energy[2] / energy),
                 top4_high_fraction=float(joint[:4, 2].sum() / joint[:4].sum()))
    return stats, checks, variants, dict(s=s, p=p, vh=vh, joint=joint / energy,
                                       power=power / energy, spatial_modes=spatial_modes)


def png(rgb):
    return Image.fromarray((rgb.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8))


def image_metrics(image, before, full):
    # Metrics in unclipped float RGB, not PNG. Clipped display is separate.
    d, ref = image.double() - before.double(), full.double() - before.double()
    e, ref_e = float(d.square().mean()), float(ref.square().mean())
    cosine = float((d * ref).sum() / (d.norm() * ref.norm()).clamp_min(1e-30))
    return dict(rgb_rms=e ** .5, rgb_energy_relative=e / max(ref_e, 1e-30),
                rgb_cosine_to_full=cosine,
                rgb_error_to_full=float((image.double() - full.double()).square().mean().sqrt()),
                top_window_rms=float(d[:, :220].square().mean().sqrt()),
                bottom_window_rms=float(d[:, 850:].square().mean().sqrt()),
                out_of_display_range_fraction=float(((image < 0) | (image > 1)).float().mean()))


def make_board(path, images, base, title, columns, labels=None):
    size = 224
    max_delta = max(float((images[n] - base).square().mean(0).sqrt().max()) for n in columns)
    # All arms share a linear scale within this board. Saved numerical RMS is
    # the comparison across boards; each board explicitly names its maximum.
    max_delta = max(max_delta, 1e-8)
    canvas = Image.new("RGB", (size * (len(columns) + 1), size * 2 + 100), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), title, fill="black")
    draw.text((8, 23), f"Below: RGB change magnitude from reference, shared 0..{max_delta:.4f}. Display RGB clipped only for PNG.", fill="black")
    canvas.paste(png(base).resize((size, size)), (0, 78))
    draw.text((5, 55), "Reference", fill="black")
    cmap = plt.get_cmap("magma")
    for i, name in enumerate(columns, 1):
        im = images[name]
        text = name if labels is None else labels.get(name, name)
        draw.text((i * size + 4, 55), text, fill="black")
        canvas.paste(png(im).resize((size, size)), (i * size, 78))
        mag = (im - base).square().mean(0).sqrt().numpy()
        heat = Image.fromarray((cmap(np.clip(mag / max_delta, 0, 1))[..., :3] * 255).astype(np.uint8))
        canvas.paste(heat.resize((size, size)), (i * size, 78 + size))
    canvas.save(path)


def plot_stats(out, results):
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    colors = ["tab:blue", "tab:orange"]
    for (name, data), color in zip(results.items(), colors):
        rows = data["rows"]
        k = np.arange(1, len(rows) + 1)
        for col, key in enumerate(("rho4", "effective_rank", "high_fraction", "dc_fraction")):
            axes[0, col].plot(k, [r[key] for r in rows], label=name, color=color)
            axes[0, col].set_title(key)
        axes[1, 0].plot(k, [r["rms"] for r in rows], label=name, color=color)
        axes[1, 1].plot(k, [r["rate_rms"] for r in rows], color=color)
        axes[1, 2].plot(k[1:], [r["subspace_overlap4"] for r in rows[1:]], color=color)
        axes[1, 3].plot(k, [r["top4_high_fraction"] for r in rows], color=color)
    for ax, title in zip(axes[1], ("Increment RMS", "Increment / step width RMS",
                                  "Consecutive channel subspace overlap (r=4)", "High-frequency share inside top-4")):
        ax.set_title(title)
    for ax in axes.flat:
        ax.set_xlabel("active step (1-based)")
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "trajectory_statistics.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "experiments/sota_inversion_20261105/outputs/comparison_v1/results.pt")
    ap.add_argument("--output", type=Path, default=HERE / "outputs/pilot_v1")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--statistics-only", action="store_true")
    args = ap.parse_args()
    if args.output.exists():
        raise RuntimeError("Use a fresh output directory; existing results are preserved")
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.monotonic()
    payload = torch.load(args.input, map_location="cpu", weights_only=True)
    names = ["flowedit_author", "flowedit_guidance_matched"]
    protocol = dict(date="2026-11-17", input=str(args.input.relative_to(ROOT)), input_sha256=sha(args.input),
                    script_sha256=sha(Path(__file__)), torch_version=torch.__version__,
                    source=payload["source"], methods=METHODS, local_steps=[1, 12, 24],
                    latent_layout="unpacked BCHW=(1,16,128,128); SVD matrix=(16384,16)",
                    increment="difference of cached BF16 states, cast to float64 before subtraction",
                    scope="offline local intervention and frozen-increment replay; no online FlowEdit rollout",
                    transformer_calls=0, vae_calls=0, spatial_edit_mask=False,
                    fft="orthonormal periodic 2D; radii <.0625, [.0625,.25), >=.25 cycles/latent pixel; DC in low",
                    shift="circular +16 latent pixels horizontally; all channels; approx +128 RGB pixels",
                    metrics="unclipped float RGB in [nominal 0,1] units; PNG clipped to [0,1]",
                    replay="z_source + sum_k T_k(delta_k); later increments never re-evaluated",
                    predeclared_visual_arms=DISPLAY,
                    statistics_only=args.statistics_only)
    write_json(args.output / "protocol.json", protocol)
    results = {}
    for name in names:
        trace = payload["traces"][name]
        states = [unpack(z.double()) for z in trace["states"]]
        times = trace["timesteps"]
        rows, checks, decompositions = [], [], []
        replay = {m: states[0].clone() for m in METHODS}
        previous_v = None
        for k, (before, after) in enumerate(zip(states[:-1], states[1:]), 1):
            delta = after - before
            stats, audit, variants, modes = decompose(delta)
            h = times[k - 1] - times[k]
            stats.update(step=k, time_before=times[k - 1], time_after=times[k], h=h,
                         rate_rms=stats["rms"] / h,
                         subspace_overlap4=None if previous_v is None else float((previous_v @ modes["vh"][:4].T).square().sum() / 4))
            previous_v = modes["vh"][:4]
            rows.append(stats)
            checks.append(audit)
            decompositions.append(modes)
            for m in METHODS:
                replay[m] += variants[m]
        replay_error = float((replay["full"] - states[-1]).norm() / states[-1].norm())
        assert replay_error < 1e-12
        results[name] = dict(rows=rows, checks=checks, decompositions=decompositions,
                             replay=replay, states=states, replay_error=replay_error)
        directory = args.output / name
        directory.mkdir()
        write_json(directory / "steps.json", rows)
        write_json(directory / "audits.json", dict(steps=checks, full_replay_relative_error=replay_error))
        with (directory / "steps.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        np.savez_compressed(directory / "decompositions.npz",
                            singular_values=np.stack([m["s"].numpy() for m in decompositions]),
                            energy_fraction=np.stack([m["p"].numpy() for m in decompositions]),
                            joint_rank_frequency=np.stack([m["joint"].numpy() for m in decompositions]),
                            normalized_latent_power=np.stack([m["power"].numpy() for m in decompositions]))
        print(json.dumps(dict(stage="statistics_complete", name=name, seconds=time.monotonic() - start)), flush=True)
    plot_stats(args.output, results)

    # Joint tables show that SVD ordering is not a frequency ordering.
    fig, axes = plt.subplots(2, 3, figsize=(12, 9))
    for row, (name, data) in enumerate(results.items()):
        for col, k in enumerate((1, 12, 24)):
            ax = axes[row, col]
            table = data["decompositions"][k - 1]["joint"].numpy()
            im = ax.imshow(table, aspect="auto", vmin=0, vmax=.25, cmap="magma")
            ax.set_title(f"{name} | active step {k}", fontsize=9)
            ax.set_xticks([0, 1, 2], ["low (incl DC)", "mid", "high"])
            ax.set_ylabel("SVD mode index")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="fraction of whole increment energy", fraction=.02)
    fig.savefig(args.output / "joint_rank_frequency.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    all_visual_metrics = {}
    if not args.statistics_only:
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(ROOT / "pretrained_models/flux1-dev/vae",
                                            local_files_only=True, torch_dtype=torch.float32).eval().requires_grad_(False).to(args.device)
        def decode(z):
            with torch.inference_mode():
                result = (vae.decode(z.to(args.device, dtype=torch.float32) / vae.config.scaling_factor
                                     + vae.config.shift_factor, return_dict=False)[0] + 1) / 2
            protocol["vae_calls"] += 1
            return result[0].float().cpu()

        # Both guidance settings have stats. Predeclared visual pilot uses author setting only.
        name = "flowedit_author"
        data = results[name]
        directory = args.output / name
        local_metrics, decoded_additivity = [], []
        for k in (1, 12, 24):
            before, after = data["states"][k - 1:k + 1]
            delta = after - before
            _, _, variants, modes = decompose(delta)
            before_image = decode(before)
            images = {m: decode(before + variants[m]) for m in METHODS}
            for m, image in images.items():
                dest = directory / f"step_{k:02d}"
                dest.mkdir(exist_ok=True)
                png(image).save(dest / f"{m}.png")
                v = variants[m]
                stats = image_metrics(image, before_image, images["full"])
                stats.update(step=k, method=m, latent_energy_relative=float(v.square().sum() / delta.square().sum()),
                             latent_cosine_to_full=float((v * delta).sum() / (v.norm() * delta.norm())),
                             rgb_to_latent_rms_gain=stats["rgb_rms"] / float(v.square().mean().sqrt()))
                local_metrics.append(stats)
            rgb_deltas = {m: image.double() - before_image.double() for m, image in images.items()}
            da, db = rgb_deltas["svd4"], rgb_deltas["tail4"]
            decoded_additivity.append(dict(step=k,
                svd4_tail4_rgb_cosine=float((da * db).sum() / (da.norm() * db.norm())),
                svd4_tail4_nonlinear_remainder_relative=float((rgb_deltas["full"] - da - db).norm() / rgb_deltas["full"].norm()),
                fourier_bands_nonlinear_remainder_relative=float((rgb_deltas["full"] - sum(rgb_deltas[m] for m in ("fft_low", "fft_mid", "fft_high"))).norm() / rgb_deltas["full"].norm())))
            make_board(directory / f"local_step_{k:02d}.png", images, before_image,
                       f"LOCAL intervention | active step {k} | same pre-step state; no later rollout", DISPLAY)
            # Show spatial patterns without decoding a delta as an absolute latent.
            fig, axes = plt.subplots(1, 4, figsize=(12, 3))
            maps = modes["spatial_modes"][:4].numpy()
            limit = max(abs(maps).max(), 1e-10)
            for i, ax in enumerate(axes):
                ax.imshow(maps[i], cmap="RdBu_r", vmin=-limit, vmax=limit)
                ax.set_title(f"mode {i+1}; energy={float(modes['p'][i]):.1%}")
                ax.axis("off")
            fig.suptitle(f"Unpacked latent spatial factors U*S; arbitrary sign; step {k}")
            fig.tight_layout()
            fig.savefig(directory / f"spatial_modes_step_{k:02d}.png", dpi=150)
            plt.close(fig)
            print(json.dumps(dict(stage="local_decode_complete", step=k, vae_calls=protocol["vae_calls"], seconds=time.monotonic() - start)), flush=True)

        source_image = decode(data["states"][0])
        endpoint_images = {m: decode(data["replay"][m]) for m in METHODS}
        replay_metrics = []
        for m, image in endpoint_images.items():
            png(image).save(directory / f"replay_{m}.png")
            stats = image_metrics(image, source_image, endpoint_images["full"])
            stats.update(method=m, top_source_psnr=-10 * np.log10(max(float((image[:, :220].double() - source_image[:, :220].double()).square().mean()), 1e-30)),
                         bottom_source_psnr=-10 * np.log10(max(float((image[:, 850:].double() - source_image[:, 850:].double()).square().mean()), 1e-30)))
            replay_metrics.append(stats)
        make_board(directory / "frozen_replay.png", endpoint_images, source_image,
                   "FROZEN-INCREMENT REPLAY | all 24 cached increments transformed; no velocity re-query", DISPLAY)
        make_board(directory / "spectral_blind_spot.png", endpoint_images, source_image,
                   "Same singular values + same latent Fourier power at EVERY step; different spatial update", ["full", "shift"])
        baseline_error = float((endpoint_images["full"].clamp(0, 1) - payload["images"][name][0]).square().mean().sqrt())
        all_visual_metrics = dict(local=local_metrics, frozen_replay=replay_metrics,
                                  decoded_additivity=decoded_additivity,
                                  cached_baseline_clipped_rgb_rms_error=baseline_error,
                                  decoder="same local FLUX VAE; FP32; no tiling; no Transformer")
        write_json(directory / "visual_metrics.json", all_visual_metrics)

    summary = dict(status="complete", seconds=time.monotonic() - start, transformer_calls=0,
                   vae_calls=protocol["vae_calls"], source_count=1, trajectories=2, active_increments=48,
                   max_algebra_relative_error=max(v for d in results.values() for c in d["checks"] for v in c.values()),
                   visual_scope="author configuration; local steps 1/12/24 and frozen replay" if not args.statistics_only else "none",
                   configurations={n: {key: dict(min=min(r[key] for r in d["rows"]),
                                                max=max(r[key] for r in d["rows"]),
                                                mean=float(np.mean([r[key] for r in d["rows"]])))
                                       for key in ("rho1", "rho4", "rho8", "effective_rank", "dc_fraction", "high_fraction", "top4_high_fraction")}
                                   for n, d in results.items()})
    write_json(args.output / "protocol.json", protocol)
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
