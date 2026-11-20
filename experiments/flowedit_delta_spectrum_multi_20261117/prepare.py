"""Register the multi-source experiment and encode frozen FLUX inputs."""
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
from diffusers import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel, T5Tokenizer, T5EncoderModel

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs/study_v1"
sys.path.insert(0, str(ROOT / "experiments/sota_inversion_20261105"))
from common import setup, write, sha, save_image

CATEGORIES = {
    "castle": "material", "flowers_draw": "replacement", "bus": "replacement",
    "penguins": "material", "flowers": "color", "dalmatian": "pose",
    "corgi": "material", "brown_owl": "color", "mountain_black": "replacement",
    "piece_of_cake": "addition", "bear": "color", "cat_crown": "accessory",
    "dogs": "replacement", "sign": "text", "puppies": "replacement",
    "luna": "text", "dogs2": "replacement", "yellow_bulldog": "replacement",
    "kid_running": "replacement", "beer_glass": "replacement",
    "rocks": "replacement", "japanese_castle": "material", "tiger": "replacement", "dog": "pose",
}
ONLINE = ["castle__00", "flowers_draw__00", "bus__00", "penguins__00",
          "flowers__00", "dalmatian__00", "brown_owl__00", "piece_of_cake__00",
          "cat_crown__00", "sign__00", "kid_running__00", "dog__00"]
ARMS = ["full", "svd4", "scalar4", "fft_low", "scalar_low", "fft_high", "scalar_high", "shift"]


def pack(z):
    b, c, h, w = z.shape
    return z.reshape(b, c, h//2, 2, w//2, 2).permute(0, 2, 4, 1, 3, 5).reshape(b, h*w//4, c*4)


@torch.inference_mode()
def main():
    setup()
    if OUT.exists():
        raise RuntimeError("Existing registered study must be preserved")
    OUT.mkdir(parents=True)
    inpdir = OUT / "inputs"
    inpdir.mkdir()
    start = time.monotonic()
    manifest = json.loads((ROOT / "experiments/stage1_full/inputs/manifest.json").read_text())
    records = [torch.load(ROOT / "experiments/stage1_full/inputs" / r["file"], weights_only=True)["record"] for r in manifest["inputs"]]
    assert len(records) == len({r["source_sha256"] for r in records}) == 24
    for r in records:
        r["category"] = CATEGORIES[r["pair_id"].split("__")[0]]
    protocol = dict(date="2026-11-17", source_count=24, records=records,
                    source_split="existing development partition, one pre-existing task per source; no held-out claims",
                    model="local FLUX.1-dev, frozen native model", resolution=512,
                    resize="RGB, LANCZOS square 512 for every source and arm",
                    seeds=[0, 1, 2], baseline_runs=72, online_sources=ONLINE, arms=ARMS,
                    online_extra_runs=len(ONLINE)*(len(ARMS)-1), total_runs=156,
                    T=28, n_max=24, n_min=0, n_avg=1, source_guidance=1.5, target_guidance=5.5,
                    planned_transformer_calls=7488, local_steps=[1, 12, 24],
                    dtype="BF16 network/state; FP32 velocity subtraction, timestep product and accumulation; BF16 state rounding",
                    precision_note="new uniform 512 protocol; not bitwise replay of previous 1024 cached baseline",
                    spectral_units="cycles/latent pixel; orthonormal FFT; radial cuts .0625 and .25, DC in low",
                    online_svd="unpacked spatial-by-16 channel matrix; top-4 from FP32 Gram eigendecomposition",
                    online_shift="horizontal circular +8 latent pixels = nominal 64 RGB pixels",
                    online_noise="identical CPU BF16 random draws per pair+seed across arms, independent of transform",
                    registered_before_inference=True, spatial_edit_masks=False, training=False,
                    selection="24 fixed cached task IDs; online subset fixed by task variety before viewing new outputs",
                    additional_local_controls=["random4_matched", "svd4_equal_norm", "smooth_low", "channel_flip"],
                    inference_script_sha256=None, prepare_script_sha256=sha(Path(__file__)))
    write(OUT / "protocol.json", protocol)
    device = "cuda:0"
    modelroot = ROOT / "pretrained_models/flux1-dev"
    vae = AutoencoderKL.from_pretrained(modelroot / "vae", local_files_only=True, torch_dtype=torch.float32).eval().requires_grad_(False).to(device)
    latents = {}
    for r in records:
        pair = r["pair_id"]
        dest = inpdir / pair
        dest.mkdir()
        im = Image.open(ROOT / r["source_image"]).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
        im.save(dest / "source.png")
        pixels = torch.from_numpy(np.array(im).copy()).permute(2, 0, 1)[None].to(device).float()/255
        z = (vae.encode(pixels*2-1).latent_dist.mode()-vae.config.shift_factor)*vae.config.scaling_factor
        z = z.to(torch.bfloat16)
        latents[pair] = pack(z).cpu()
        rgb = ((vae.decode(z.float()/vae.config.scaling_factor+vae.config.shift_factor, return_dict=False)[0]+1)/2).clamp(0, 1)
        save_image(rgb, dest / "source_vae.png")
    del vae, pixels, z, rgb
    gc.collect()
    torch.cuda.empty_cache()
    prompts = [r[k] for r in records for k in ["source_prompt", "target_prompt"]]
    tokenizer = CLIPTokenizer.from_pretrained(modelroot / "tokenizer", local_files_only=True)
    clip_tokens = tokenizer(prompts, max_length=77, padding="max_length", truncation=True, return_tensors="pt")
    clip_model = CLIPTextModel.from_pretrained(modelroot / "text_encoder", local_files_only=True, torch_dtype=torch.bfloat16).eval().requires_grad_(False).to(device)
    clip = torch.cat([clip_model(input_ids=clip_tokens["input_ids"][i:i+8].to(device)).pooler_output.cpu() for i in range(0, len(prompts), 8)])
    clip_lengths = [len(tokenizer(p)["input_ids"]) for p in prompts]
    del clip_model
    gc.collect()
    torch.cuda.empty_cache()
    tokenizer = T5Tokenizer.from_pretrained(modelroot / "tokenizer_2", local_files_only=True)
    t5_tokens = tokenizer(prompts, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
    t5_model = T5EncoderModel.from_pretrained(modelroot / "text_encoder_2", local_files_only=True, torch_dtype=torch.bfloat16).eval().requires_grad_(False).to(device)
    txt = torch.cat([t5_model(input_ids=t5_tokens["input_ids"][i:i+4].to(device)).last_hidden_state.cpu() for i in range(0, len(prompts), 4)])
    t5_lengths = [len(tokenizer(p)["input_ids"]) for p in prompts]
    del t5_model
    gc.collect()
    torch.cuda.empty_cache()
    ids = torch.zeros(1, 32, 32, 3)
    ids[..., 1] = torch.arange(32)[:, None]
    ids[..., 2] = torch.arange(32)[None, :]
    for i, r in enumerate(records):
        conditions = {which: dict(txt=txt[2*i+j:2*i+j+1], txt_ids=torch.zeros(1, 512, 3), vec=clip[2*i+j:2*i+j+1]) for j, which in enumerate(["source", "target"])}
        torch.save(dict(record=r, x=latents[r["pair_id"]], img_ids=ids.reshape(1, 1024, 3), conditions=conditions), inpdir / r["pair_id"] / "input.pt")
    write(OUT / "prepared.json", dict(seconds=time.monotonic()-start, vae_encode_calls=24, vae_decode_calls=24,
        clip_lengths=clip_lengths, t5_lengths=t5_lengths, clip_truncated_count=sum(x>77 for x in clip_lengths),
        t5_truncated_count=sum(x>512 for x in t5_lengths),
        input_hashes={r["pair_id"]:sha(inpdir/r["pair_id"]/"input.pt") for r in records}))
    print("PREPARED", len(records), round(time.monotonic()-start, 2), flush=True)


if __name__ == "__main__":
    main()
