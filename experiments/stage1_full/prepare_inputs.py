"""Cache fixed FP32 encodings for the complete 24-source development audit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
import torch
from PIL import Image
from diffusers import StableDiffusion3Pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records = [json.loads(s) for s in (ROOT / "data/flowedit/manifests/stage1_pilot.jsonl").read_text().splitlines()]
    dest = ROOT / "experiments/stage1_full/inputs"
    dest.mkdir(parents=True, exist_ok=True)
    model = ROOT / "pretrained_models/sd3-medium"
    start = time.monotonic()
    pipe = StableDiffusion3Pipeline.from_pretrained(model, torch_dtype=torch.float32, local_files_only=True)
    for module in (pipe.transformer, pipe.vae, pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3):
        module.eval().requires_grad_(False)
    # Only text encoders reside on the GPU during text encoding.
    encoders = (pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3)
    for encoder in encoders:
        encoder.to(args.device)
    for index, record in enumerate(records):
        target = dest / f"{record['pair_id']}.pt"
        if target.exists():
            print(f"existing input {index+1}/24 {record['pair_id']}", flush=True)
            continue
        with torch.no_grad():
            embeddings = [
                tuple(t.cpu() for t in pipe.encode_prompt(
                    prompt=prompt, prompt_2=None, prompt_3=None,
                    device=args.device, do_classifier_free_guidance=True,
                    negative_prompt="", max_sequence_length=256,
                ))
                for prompt in (record["source_prompt"], record["target_prompt"])
            ]
        torch.save({"record": record, "embeddings": embeddings, "precision": "fp32"}, target)
        print(f"encoded prompts {index+1}/24 {record['pair_id']}", flush=True)
    for encoder in encoders:
        encoder.cpu()
    torch.cuda.empty_cache()
    pipe.vae.to(args.device)
    fingerprints = []
    for index, record in enumerate(records):
        target = dest / f"{record['pair_id']}.pt"
        payload = torch.load(target, weights_only=True)
        if "x" not in payload:
            image = Image.open(ROOT / record["source_image"]).convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)
            with torch.no_grad():
                pixels = pipe.image_processor.preprocess(image).to(args.device)
                latent = pipe.vae.encode(pixels).latent_dist.mode()
                x = (latent - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
            payload.update(x=x.cpu(), resolution=512, vae_encoding="mode", dtype="float32")
            torch.save(payload, target)
        fingerprints.append({"pair_id": record["pair_id"], "file": target.name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
        print(f"encoded image {index+1}/24 {record['pair_id']}", flush=True)
    pipe.scheduler.set_timesteps(50)
    manifest = {
        "source_images": len(records), "inputs": fingerprints,
        "model": "SD3 Medium", "resolution": 512, "dtype": "fp32",
        "timesteps": pipe.scheduler.timesteps.tolist(), "scheduler_sigmas": pipe.scheduler.sigmas.tolist(),
        "T": 50, "n_max": 33, "n_min": 0, "n_avg": 1,
        "source_guidance": 3.5, "target_guidance": 13.5,
        "time_coordinate": "official SD3 implementation: network timestep / 1000",
        "elapsed_seconds": time.monotonic() - start,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
