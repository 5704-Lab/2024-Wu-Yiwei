"""Frozen, own-condition paired descriptors. No edit-region masks."""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
OUT = HERE / "outputs"
sys.path.insert(0, str(ROOT / "experiments/siglip_state_response"))
from run_precheck import CleanScorer

PAIRS = ["bear__00", "brown_owl__00", "bus__00", "cat_crown__00"]
SHORT = {
    "bear__00": ["A brown bear.", "A black bear."],
    "brown_owl__00": ["A brown owl.", "A white owl."],
    "bus__00": ["A classic yellow van.", "A modern SUV."],
    "cat_crown__00": ["A gray cat wearing a crown.", "A gray cat wearing a top hat."],
}
VARIANTS = ["BASE", "SHORT_SOURCE", "SHORT_TARGET", "SHORT_BOTH", "NO_EDIT_ASYMMETRIC_CFG",
            "MATCHED_CFG_FULL", "MATCHED_CFG_SHORT_BOTH",
            "COUNTERFACTUAL_FULL", "COUNTERFACTUAL_SHORT_BOTH"]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def save_image(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    a = image[0].detach().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(np.rint(a.clip(0, 1) * 255).astype("uint8")).save(path)


def original(rec, device):
    a = np.asarray(Image.open(ROOT / rec["source_image"]).convert("RGB").resize(
        (512, 512), Image.Resampling.BILINEAR), dtype=np.float32) / 255
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device)


def rotate(vector, es, et):
    c = es @ et
    def skew(v):
        return et * (es @ v) - es * (et @ v)
    return vector + skew(vector) + skew(skew(vector)) / (1 + c)


def residual(f, e):
    a = f @ e
    q = f - a * e
    rho = q.norm()
    if float(rho.detach()) < 1e-5:
        raise RuntimeError("degenerate own-condition residual")
    return a, q / rho, rho


class Paired:
    def __init__(self, device="cuda:0"):
        self.device = torch.device(device)
        self.sc = CleanScorer(self.device, torch.float32, "native")
        self.ctx = {}

    def feature_image(self, image, hidden=False):
        with sdpa_kernel(SDPBackend.MATH):
            out = self.sc.model.vision_model(pixel_values=self.sc.preprocess(image),
                                            output_hidden_states=hidden)
        f = F.normalize(out.pooler_output, dim=-1)[0]
        return f, out

    def feature(self, y):
        with sdpa_kernel(SDPBackend.MATH):
            image = self.sc.image(y)
            f, _ = self.feature_image(image)
        return f

    def setup(self, pair):
        data = torch.load(ROOT / "experiments/stage1_full/inputs" / (pair + ".pt"),
                          weights_only=True)
        rec = data["record"]
        texts = [rec["source_prompt"], rec["target_prompt"], *SHORT[pair]]
        es, lengths = self.sc.encode(texts)
        with torch.no_grad():
            rgb = original(rec, self.device)
            fs, _ = self.feature_image(rgb)
            vae = self.sc.image(data["x"].to(self.device))
            fv, _ = self.feature_image(vae)
        source_a, ns, rho = residual(fs, es[0])
        m = rotate(ns, es[0], es[1])
        _, ns_short, _ = residual(fs, es[2])
        ms = rotate(ns_short, es[2], es[1])
        ctx = dict(data=data, rec=rec, texts=texts, embeddings=es, lengths=lengths,
                   fs=fs, fv=fv, ns=ns, m=m, m_short_source=ms,
                   x=data["x"].to(self.device), source_rgb=rgb, source_vae=vae,
                   source_pair_cosine=float(source_a), source_rho=float(rho))
        self.ctx[pair] = ctx
        save_image(rgb, OUT / pair / "source_rgb.png")
        save_image(vae, OUT / pair / "source_vae.png")
        return ctx

    def values(self, pair, f):
        c = self.ctx[pair]
        e = c["embeddings"]
        a, n, rho = residual(f, e[1])
        _, nshort, _ = residual(f, e[3])
        logs = self.sc.alpha * (e @ f) + self.sc.bias
        k = 0.5 * (n - c["m"]).square().sum()
        raw = 0.5 * (n - c["ns"]).square().sum()
        globe = 1 - f @ c["fs"]
        return dict(target_cosine=float(a), target_logit=float(logs[1]),
                    source_logit=float(logs[0]), short_target_logit=float(logs[3]),
                    short_source_logit=float(logs[2]), target_rho=float(rho),
                    K=float(k), raw_unit_residual=float(raw), global_distance=float(globe),
                    K_short_source=float(.5 * (n - c["m_short_source"]).square().sum()),
                    K_short_both=float(.5 * (nshort - rotate(
                        residual(c["fs"],e[2])[1],e[2],e[3])).square().sum()),
                    full_residual_length_change=float(rho - c["source_rho"]))

    def evaluate(self, pair, y, hidden=False):
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            image = self.sc.image(y)
            f, output = self.feature_image(image, hidden)
            row = self.values(pair, f)
            row["latent_source_distance"] = float((y-self.ctx[pair]["x"]).norm())
            row["rgb_rmse_source_vae"] = float((image-self.ctx[pair]["source_vae"]).square().mean().sqrt())
            if hidden:
                row["middle_distance"] = float((F.normalize(output.hidden_states[6][0],dim=-1) -
                    self.ctx[pair]["source_middle"]).square().mean().sqrt())
        return row, image, f


def protocol(stage, extra=None):
    write(OUT / ("protocol_" + stage + ".json"), dict(
        stage=stage, pairs=PAIRS, date="2026-10-13", weights_frozen=True,
        source_anchor="actual input RGB, not VAE reconstruction", device="cuda:0",
        precision="fp32", seed=0, no_extra_reference=True, no_spatial_mask=True,
        model="SD3 Medium + SigLIP2 Base", short=SHORT,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        extra=extra or {}))
