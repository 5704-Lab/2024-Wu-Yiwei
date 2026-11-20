"""First real-model stage-1 pilot: one SD3 source image, three actual states.

This audits numerical conditional derivatives, not semantic editing success.
Default configuration matches FlowEdit's SD3 T=50, n_max=33, no target tail.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
import numpy as np
from PIL import Image
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel


def dump_json(value, **kwargs):
    def clean(item):
        if isinstance(item, float) and not np.isfinite(item):
            return None
        if isinstance(item, list):
            return [clean(x) for x in item]
        if isinstance(item, dict):
            return {k: clean(x) for k, x in item.items()}
        return item
    return json.dumps(clean(value), allow_nan=False, **kwargs)


def finite_cos(a, b):
    aa, bb = a.flatten().double(), b.flatten().double()
    if not bool(torch.isfinite(aa).all() and torch.isfinite(bb).all()):
        return None
    denom = aa.norm() * bb.norm()
    return None if denom < 1e-20 else float(aa @ bb / denom)


def relative_error(a, b):
    return float((a.float() - b.float()).norm() / (1 + b.float().norm()))


def wait_for_model(path, wait):
    manifest = path / "download_manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    rows = json.loads(manifest.read_text())["files"]
    deadline = time.monotonic() + wait
    last_print = 0.0
    while True:
        ready = sum(
            (path / r["path"]).is_file() and (path / r["path"]).stat().st_size == r["bytes"]
            for r in rows
        )
        if ready == len(rows):
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Model incomplete: {ready}/{len(rows)} files")
        if time.monotonic() - last_print > 45:
            print(f"waiting for verified model files: {ready}/{len(rows)}", flush=True)
            last_print = time.monotonic()
        time.sleep(10)


def derivative_audit(model, state, embeddings, beta, precision, seed):
    device = state["q"].device
    dtype = next(model.parameters()).dtype
    low_precision = dtype in (torch.float16, torch.bfloat16)
    q = state["q"].to(dtype)
    tnn = state["network_time"].to(device=device, dtype=torch.float32).expand(1)
    seq, neg_seq, pooled, neg_pooled = [v.to(device=device, dtype=dtype).detach() for v in embeddings]
    def attention_context():
        return nullcontext()

    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        reference_uncond = model(
            hidden_states=q, timestep=tnn, encoder_hidden_states=neg_seq,
            pooled_projections=neg_pooled, return_dict=False,
        )[0].float()
        reference_cond = model(
            hidden_states=q, timestep=tnn, encoder_hidden_states=seq,
            pooled_projections=pooled, return_dict=False,
        )[0].float()
        reference_velocity = reference_uncond + beta * (reference_cond - reference_uncond)
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), attention_context():
        uncond = model(
            hidden_states=q, timestep=tnn, encoder_hidden_states=neg_seq,
            pooled_projections=neg_pooled, return_dict=False,
        )[0].float()

    def fn(sequence, pool):
        pred = model(
            hidden_states=q, timestep=tnn, encoder_hidden_states=sequence,
            pooled_projections=pool, return_dict=False,
        )[0].float()
        return uncond + beta * (pred - uncond)

    with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH), attention_context():
        velocity, vjp_fn = torch.func.vjp(fn, seq, pooled)
        raw_probe = (-state["h"] * velocity).detach()
        probe_scale = max(float(raw_probe.norm()), 1.0) if low_precision else 1.0
        probe = raw_probe / probe_scale
        gseq, gpool = vjp_fn(probe)
        response_scale = max(float(torch.sqrt(gseq.float().square().sum() + gpool.float().square().sum())), 1.0) if low_precision else 1.0
        gen = torch.Generator("cpu").manual_seed(seed)
        tangents = tuple(
            torch.randn(value.shape, generator=gen, dtype=torch.float32).to(device=device, dtype=dtype)
            for value in (seq, pooled)
        )
        random_probe = torch.randn(velocity.shape, generator=gen).to(device)
        random_probe_scale = max(float(random_probe.norm()), 1.0) if low_precision else 1.0
        random_g = vjp_fn(random_probe / random_probe_scale)
        random_response_scale = max(float(torch.sqrt(sum(g.float().square().sum() for g in random_g))), 1.0) if low_precision else 1.0
    # no_grad disables reverse graph retention, while forward AD still works.
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), attention_context():
        # Torch 2.5 forward-mode LayerNorm promotes BF16 tangents and fails
        # at the following linear layer. Reverse-over-reverse computes the
        # same JVP by automatic differentiation, without finite differences.
        jvp = torch.autograd.functional.jvp if low_precision else torch.func.jvp
        _, tangent_velocity = jvp(fn, (seq, pooled), tangents)
        _, response = jvp(fn, (seq, pooled), (gseq.detach() / response_scale, gpool.detach() / response_scale))
        _, random_response = jvp(fn, (seq, pooled), tuple(v.detach() / random_response_scale for v in random_g))
    response = response.float() * (probe_scale * response_scale)
    random_response = random_response.float() * (random_probe_scale * random_response_scale)
    gseq_raw, gpool_raw = gseq.float() * probe_scale, gpool.float() * probe_scale
    lhs = (gseq_raw.double() * tangents[0].double()).sum() + (gpool_raw.double() * tangents[1].double()).sum()
    rhs = (raw_probe.double() * tangent_velocity.double()).sum()
    psd = (random_probe.double() * random_response.double()).sum()
    row = {
        "precision": precision, "beta": beta, "h": state["h"],
        "attention_backend": "torch_math_sdpa",
        "jvp_backend": "reverse_over_reverse" if low_precision else "forward_mode",
        "probe_scale": probe_scale, "response_scale": response_scale,
        "reference_sdpa_primal_error": relative_error(velocity, reference_velocity),
        "dual_residual": float(abs(lhs - rhs) / (1 + abs(lhs) + abs(rhs))),
        "dual_lhs": float(lhs), "dual_rhs": float(rhs),
        "dual_absolute_error": float(abs(lhs - rhs)),
        "random_response_psd_inner": float(psd),
        "vjp_sequence_norm": float(gseq_raw.norm()),
        "vjp_pooled_norm": float(gpool_raw.norm()),
        "response_norm": float(response.float().norm()),
        "finite": all(bool(torch.isfinite(x).all()) for x in (velocity, gseq, gpool, response, random_response)),
        "max_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
    }
    tensors = {
        "velocity": velocity.detach().float().cpu(),
        "gseq": gseq_raw.detach().cpu(),
        "gpool": gpool_raw.detach().cpu(),
        "response": response.detach().float().cpu(),
    }
    del vjp_fn, velocity, gseq, gpool, random_g, response, random_response
    gc.collect()
    torch.cuda.empty_cache()
    return row, tensors


def resume_bf16(args, records, output):
    """Reuse already saved FP32 states, avoiding another generation trajectory."""
    device = torch.device(args.device)
    label = args.compare_precision
    model = SD3Transformer2DModel.from_pretrained(
        args.model / "transformer", torch_dtype=getattr(torch, "float16" if label == "fp16" else "bfloat16"), local_files_only=True,
    ).eval().requires_grad_(False).to(device)
    rows, comparisons = [], []
    start = time.monotonic()
    for case in records:
        case_out = output / f"{case['pair_id']}__seed{args.seed}"
        saved = torch.load(case_out / "fp32_states.pt", map_location="cpu", weights_only=True)
        fp32_rows = json.loads((case_out / "fp32_rows.json").read_text())
        rows.extend(fp32_rows)
        for snapshot in saved["snapshots"]:
            state = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in snapshot["state"].items()}
            torch.cuda.reset_peak_memory_stats(device)
            row, tensors = derivative_audit(model, state, snapshot["embedding"], snapshot["beta"], label, 20261005 + snapshot["step"])
            row.update(pair_id=case["pair_id"], seed=args.seed, step=snapshot["step"], t=snapshot["t"], branch=snapshot["branch"])
            rows.append(row)
            comparison = {k: row[k] for k in ("pair_id", "seed", "step", "t", "branch")}
            for name in ("velocity", "gseq", "gpool", "response"):
                comparison[name + "_cosine"] = finite_cos(snapshot["fp32"][name], tensors[name])
                comparison[name + "_relative_error"] = relative_error(tensors[name], snapshot["fp32"][name])
            comparisons.append(comparison)
            snapshot[label] = tensors
            print(dump_json({"metrics": row, "comparison": comparison}), flush=True)
            (case_out / f"rows_{label}_scaled.json").write_text(dump_json(rows, indent=2) + "\n")
            (case_out / f"comparisons_{label}_scaled.json").write_text(dump_json(comparisons, indent=2) + "\n")
        torch.save(saved, case_out / f"diagnostic_states_{label}_scaled.pt")
    reference_rows = [r for r in rows if r["precision"] == "fp32"]
    cosines = [r["response_cosine"] for r in comparisons if r["response_cosine"] is not None]
    summary = {
        "stage": 1, "scope": "Initial real-model compute pilot; not the complete 24-image/two-seed gate or semantic validation.",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(args.model), "resolution": args.resolution,
        "T": 50, "n_max": 33, "n_min": 0, "n_avg": 1,
        "source_guidance": 3.5, "target_guidance": 13.5,
        "source_images": len(records), "seeds": [args.seed], "rows": len(rows),
        "comparison_precision": label, "probe_scaling_enabled": True,
        "finite_all": all(r["finite"] for r in rows),
        "fp32_dual_residual_max": max(r["dual_residual"] for r in reference_rows),
        "fp32_baseline_primal_error_max": max(r["baseline_primal_error"] for r in reference_rows),
        "low_precision_response_cosine_min": min(cosines) if cosines else None,
        "low_precision_sdpa_primal_error_max": max(r["reference_sdpa_primal_error"] for r in rows if r["precision"] == label),
        "low_precision_dual_residual_max": max(r["dual_residual"] for r in rows if r["precision"] == label),
        "elapsed_bf16_resume_seconds": time.monotonic() - start,
        "training_steps": 0,
        "precision_note": f"{label} uses automatic reverse-over-reverse JVP because Torch 2.5 low-precision LayerNorm forward AD promotes tangent dtype. Same math SDPA and unchanged model weights; no finite differences. Network timesteps remain FP32. Probe scaling is inverted in saved responses.",
    }
    summary["pilot_numerics_pass"] = (
        summary["finite_all"] and summary["fp32_dual_residual_max"] <= 1e-4
        and summary["fp32_baseline_primal_error_max"] <= 1e-4
        and (summary["low_precision_response_cosine_min"] or 0) >= 0.99
    )
    (output / f"pilot_summary_{label}_scaled.json").write_text(dump_json(summary, indent=2) + "\n")
    print(dump_json(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "pretrained_models/sd3-medium")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--wait-model", type=int, default=0)
    parser.add_argument("--resume-bf16", action="store_true")
    parser.add_argument("--compare-precision", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--reference-only", action="store_true", help="Audit only the viable FP32 route; low-precision failures remain separate.")
    args = parser.parse_args()
    if not json.loads((ROOT / "experiments/stage0/outputs/summary.json").read_text())["passed"]:
        raise RuntimeError("Stage 0 has not passed")
    wait_for_model(args.model, args.wait_model)
    output = ROOT / "experiments/stage1/outputs"
    output.mkdir(parents=True, exist_ok=True)
    records = [json.loads(s) for s in (ROOT / "data/flowedit/manifests/stage1_pilot.jsonl").read_text().splitlines()][:args.limit]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.resume_bf16:
        resume_bf16(args, records, output)
        return
    device = torch.device(args.device)
    print("Loading original SD3 Medium pipeline locally, FP32 reference.", flush=True)
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model, torch_dtype=torch.float32, local_files_only=True,
    )
    modules = [pipe.transformer, pipe.vae, pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3]
    for module in modules:
        module.eval().requires_grad_(False)
    pipe.to(device)
    pipe._guidance_scale = 13.5
    all_rows = []
    comparisons = []
    start = time.monotonic()
    for case in records:
        case_out = output / f"{case['pair_id']}__seed{args.seed}"
        case_out.mkdir(exist_ok=True)
        with torch.no_grad():
            embeddings = [
                pipe.encode_prompt(
                    prompt=prompt, prompt_2=None, prompt_3=None, device=device,
                    do_classifier_free_guidance=True,
                    negative_prompt="", max_sequence_length=256,
                )
                for prompt in (case["source_prompt"], case["target_prompt"])
            ]
            image = Image.open(ROOT / case["source_image"]).convert("RGB")
            image = image.resize((args.resolution, args.resolution), Image.Resampling.BICUBIC)
            source_pixels = pipe.image_processor.preprocess(image).to(device=device, dtype=torch.float32)
            latent = pipe.vae.encode(source_pixels).latent_dist.mode()
            x = (latent - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        for encoder in (pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3):
            encoder.cpu()
        pipe.vae.cpu()
        torch.cuda.empty_cache()
        pipe.scheduler.set_timesteps(50, device=device)
        sigmas = pipe.scheduler.sigmas.float()
        active = list(range(17, 50))
        noise_times = np.array([float(sigmas[i]) for i in active])
        targets = np.quantile(noise_times, [0.8, 0.5, 0.2])
        selected = sorted(set(active[int(np.argmin(abs(noise_times - value)))] for value in targets))
        z = x.clone()
        noise_generator = torch.Generator("cpu").manual_seed(args.seed)
        snapshots = []
        fingerprints = []
        for i in active:
            t, next_t = sigmas[i], sigmas[i + 1]
            h = float(t - next_t)
            noise_cpu = torch.randn(x.shape, generator=noise_generator, dtype=torch.float32)
            fingerprints.append(hashlib.sha256(noise_cpu.numpy().tobytes()).hexdigest())
            qs = (1 - t) * x + t * noise_cpu.to(device)
            qa = qs + z - x
            src, tar = embeddings
            with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
                prediction = pipe.transformer(
                    hidden_states=torch.cat((qs, qs, qa, qa)),
                    timestep=pipe.scheduler.timesteps[i].expand(4),
                    encoder_hidden_states=torch.cat((src[1], src[0], tar[1], tar[0])),
                    pooled_projections=torch.cat((src[3], src[2], tar[3], tar[2])),
                    return_dict=False,
                )[0]
                su, sc, au, ac = prediction.chunk(4)
                vs = su + 3.5 * (sc - su)
                va = au + 13.5 * (ac - au)
            if i in selected:
                for branch, q, beta, embedding, velocity in (
                    ("source", qs, 3.5, src, vs), ("target", qa, 13.5, tar, va),
                ):
                    state = {"q": q.detach(), "network_time": pipe.scheduler.timesteps[i], "h": h}
                    torch.cuda.reset_peak_memory_stats(device)
                    row, tensors = derivative_audit(pipe.transformer, state, embedding, beta, "fp32", 20261005 + i)
                    row.update(pair_id=case["pair_id"], seed=args.seed, step=i, t=float(t), branch=branch)
                    row["baseline_primal_error"] = relative_error(tensors["velocity"].to(device), velocity)
                    all_rows.append(row)
                    snapshots.append({
                        "branch": branch, "step": i, "t": float(t), "beta": beta,
                        "state": {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in state.items()},
                        "embedding": [v.cpu() for v in embedding], "fp32": tensors,
                    })
                    print(dump_json(row), flush=True)
                    (case_out / "fp32_rows.json").write_text(dump_json(all_rows, indent=2) + "\n")
            z = z + h * (vs - va)
        with torch.no_grad():
            pipe.vae.to(device)
            decoded = pipe.vae.decode(z / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor, return_dict=False)[0]
            pipe.image_processor.postprocess(decoded, output_type="pil")[0].save(case_out / "baseline.png")
            pipe.vae.cpu()
        torch.save({"snapshots": snapshots, "noise_sha256": fingerprints}, case_out / "fp32_states.pt")
        if args.reference_only:
            (case_out / "rows_fp32.json").write_text(dump_json(all_rows, indent=2) + "\n")
            if case is not records[-1]:
                for encoder in (pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3):
                    encoder.to(device)
                pipe.vae.to(device)
            continue
        pipe.transformer.to(dtype=torch.bfloat16)
        for snapshot in snapshots:
            state = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in snapshot["state"].items()}
            embedding = snapshot["embedding"]
            torch.cuda.reset_peak_memory_stats(device)
            row, tensors = derivative_audit(
                pipe.transformer, state, embedding, snapshot["beta"], "bf16", 20261005 + snapshot["step"],
            )
            row.update(pair_id=case["pair_id"], seed=args.seed, step=snapshot["step"], t=snapshot["t"], branch=snapshot["branch"])
            all_rows.append(row)
            comparison = {k: row[k] for k in ("pair_id", "seed", "step", "t", "branch")}
            for name in ("velocity", "gseq", "gpool", "response"):
                comparison[name + "_cosine"] = finite_cos(snapshot["fp32"][name], tensors[name])
                comparison[name + "_relative_error"] = relative_error(tensors[name], snapshot["fp32"][name])
            comparisons.append(comparison)
            snapshot["bf16"] = tensors
            print(dump_json(comparison), flush=True)
        torch.save({"snapshots": snapshots, "noise_sha256": fingerprints}, case_out / "diagnostic_states.pt")
        (case_out / "rows.json").write_text(dump_json(all_rows, indent=2) + "\n")
        (case_out / "comparisons.json").write_text(dump_json(comparisons, indent=2) + "\n")
        # A multi-case FP32 audit needs fresh original weights after BF16 casting.
        if case is not records[-1]:
            pipe.transformer = type(pipe.transformer).from_pretrained(
                args.model / "transformer", torch_dtype=torch.float32, local_files_only=True,
            ).eval().requires_grad_(False).to(device)
            for encoder in (pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3):
                encoder.to(device)
            pipe.vae.to(device)
    fp32_rows = [r for r in all_rows if r["precision"] == "fp32"]
    response_cos = [r["response_cosine"] for r in comparisons if r["response_cosine"] is not None]
    summary = {
        "stage": 1, "scope": "Initial real-model compute pilot; not the complete 24-image/two-seed gate or semantic validation.",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(args.model), "resolution": args.resolution,
        "source_guidance": 3.5, "target_guidance": 13.5,
        "T": 50, "n_max": 33, "n_min": 0, "n_avg": 1,
        "source_images": len(records), "seeds": [args.seed], "rows": len(all_rows),
        "finite_all": all(r["finite"] for r in all_rows),
        "fp32_dual_residual_max": max(r["dual_residual"] for r in fp32_rows),
        "fp32_baseline_primal_error_max": max(r["baseline_primal_error"] for r in fp32_rows),
        "bf16_response_cosine_min": min(response_cos) if response_cos else None,
        "elapsed_seconds": time.monotonic() - start,
        "training_steps": 0,
    }
    summary["pilot_fp32_numerics_pass"] = (
        summary["finite_all"] and summary["fp32_dual_residual_max"] <= 1e-4
        and summary["fp32_baseline_primal_error_max"] <= 1e-4
    )
    summary["pilot_numerics_pass"] = (
        summary["finite_all"] and summary["fp32_dual_residual_max"] <= 1e-4
        and summary["fp32_baseline_primal_error_max"] <= 1e-4
        and (summary["bf16_response_cosine_min"] or 0) >= 0.99
    )
    summary_name = "pilot_summary_fp32.json" if args.reference_only else "pilot_summary.json"
    if args.reference_only:
        summary["low_precision_comparison_performed"] = False
        summary["pilot_numerics_pass"] = None
    (output / summary_name).write_text(dump_json(summary, indent=2) + "\n")
    print(dump_json(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
