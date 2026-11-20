"""Frozen clean-state response calibration and task-excess feedback."""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

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
METHODS = ["BASE", "DRIVE", "RAW", "CAL", "FILTER", "ISO"]
CFG = dict(mu=.25, probes=4, ridge_ratio=.05, ridge_floor=1e-6,
           beta_cap=4., rank_rtol=1e-4, drive_ratio=.05, kappa=.1,
           correction_cap=.2, backtracks=6, task_abs_tolerance=.05,
           task_relative_tolerance=.1, seed=0, probe_seed=1701)
TASKS = {
    "bear__00": ("a brown bear", "a black bear", "a stream of water",
                 "the bear has brown fur", "the bear has black fur"),
    "brown_owl__00": ("a brown owl", "a white owl", "a patch of grass",
                      "the owl has brown feathers", "the owl has white feathers"),
    "bus__00": ("a classic yellow van", "a modern SUV", "a house and garage",
                "a parked classic van", "a parked modern SUV"),
    "cat_crown__00": ("a cat wearing a crown", "a cat wearing a top hat", "a black cloth",
                     "a crown on the cat's head", "a top hat on the cat's head"),
}


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def tensor_norm(x, diagonal):
    return torch.sqrt((x.double().square() * diagonal.double()).sum().clamp_min(0))


class Experiment:
    def __init__(self, device):
        self.device = torch.device(device)
        self.sc = CleanScorer(self.device, torch.float32, "native")
        self.context = {}
        self.counts = {}

    def forward(self, y, hidden=False):
        with sdpa_kernel(SDPBackend.MATH):
            image = self.sc.image(y)
            outputs = self.sc.model.vision_model(
                pixel_values=self.sc.preprocess(image), output_hidden_states=hidden)
        pooled = F.normalize(outputs.pooler_output, dim=-1)
        patches = F.normalize(outputs.last_hidden_state, dim=-1)
        n = int(math.sqrt(patches.shape[1]))
        coarse = F.adaptive_avg_pool2d(
            patches.transpose(1, 2).reshape(1, -1, n, n), (4, 4)).flatten()
        return image, pooled, coarse, outputs

    def setup(self, pair):
        payload = torch.load(ROOT / "experiments/stage1_full/inputs" / f"{pair}.pt",
                             weights_only=True)
        rec = payload["record"]
        source, target, context, source2, target2 = TASKS[pair]
        texts = [
            rec["target_prompt"], rec["source_prompt"],
            "a photo of " + target, "a photo of " + source, target2, source2,
            "a brightly lit photograph", "a dimly lit photograph",
            "a high contrast photograph", "a low contrast photograph",
            "a brightly lit " + context, "a dimly lit " + context,
            "a clear image showing " + target, "a clear image showing " + source,
        ]
        embeddings, lengths = self.sc.encode(texts)
        x = payload["x"].to(self.device)
        with torch.no_grad():
            image, pooled, feature, outputs = self.forward(x, hidden=True)
            middle = F.normalize(outputs.hidden_states[6][0], dim=-1)
        gen = torch.Generator("cpu").manual_seed(CFG["probe_seed"])
        probes = torch.randint(0, 2, (CFG["probes"], feature.numel()),
                               generator=gen).float().mul(2).sub(1).to(self.device)
        ctx = dict(payload=payload, rec=rec, x=x, source_image=image.detach(),
                   source_feature=feature.detach(), embeddings=embeddings,
                   probes=probes, texts=texts, lengths=lengths)
        self.context[pair] = ctx
        current = self.derivatives(pair, x, scale=1.)
        scale = float(current["gn_diag"].mean().clamp_min(1e-10))
        ctx["scale"] = scale
        torch.save(dict(middle=middle.cpu(),feature=feature.cpu()),
                   OUT/pair/"source_features.pt")
        write_json(OUT / pair / "input.json", dict(
            record=rec, texts=texts, text_lengths=lengths,
            truncated=[n > 64 for n in lengths], metric_scale=scale))
        return ctx

    def values(self, pooled, embeddings):
        logits = (self.sc.alpha * (pooled @ embeddings.T) + self.sc.bias)[0]
        task = torch.stack([logits[0] - logits[1],
                            .5 * (logits[2] - logits[3] + logits[4] - logits[5])])
        control = torch.stack([logits[6] - logits[7], logits[8] - logits[9],
                               logits[10] - logits[11]])
        return logits, task, control

    def derivatives(self, pair, y, scale=None):
        ctx = self.context[pair]
        scale = ctx.get("scale", 1.) if scale is None else scale
        q = y.detach().requires_grad_(True)
        image, pooled, feature, _ = self.forward(q)
        logits, task, control = self.values(pooled, ctx["embeddings"])
        nuisance = torch.stack([image * (1 - image),
                                image * (1 - image) * (2 * image - 1)])
        all_outputs = list(task.unbind()) + list(control.unbind())
        image_gradients = [
            torch.autograd.grad(v, image, retain_graph=True)[0].detach()
            for v in all_outputs
        ]
        response = torch.stack([
            (v.unsqueeze(0) * nuisance.detach()).flatten(1).sum(1)
            for v in image_gradients
        ])
        alpha = response[:2].double()
        B = response[2:].double()
        gram = B.T @ B
        ridge = CFG["ridge_ratio"] * float(gram.trace() / 2) + CFG["ridge_floor"]
        beta = (B @ torch.linalg.solve(gram + ridge * torch.eye(2, device=self.device,
                    dtype=torch.float64), alpha.T)).T
        beta_norm = beta.norm(dim=1, keepdim=True)
        beta = beta * torch.clamp(CFG["beta_cap"] / beta_norm.clamp_min(1e-12), max=1)
        raw_grad = []
        calibrated_grad = []
        for j in range(2):
            raw_grad.append(torch.autograd.grad(
                image, q, grad_outputs=image_gradients[j], retain_graph=True)[0].detach())
            eff = image_gradients[j] - sum(
                beta[j, k].float() * image_gradients[2+k] for k in range(3))
            calibrated_grad.append(torch.autograd.grad(
                image, q, grad_outputs=eff, retain_graph=True)[0].detach())
        residual = feature - ctx["source_feature"]
        source_cost = .5 * residual.square().sum() / scale + .5 * CFG["mu"] * (
            q - ctx["x"]).square().sum()
        b = torch.autograd.grad(source_cost, q, retain_graph=True)[0].detach()
        diagonal = torch.zeros_like(q)
        for i, probe in enumerate(ctx["probes"]):
            vjp = torch.autograd.grad(feature, q, grad_outputs=probe,
                                      retain_graph=i < CFG["probes"]-1)[0].detach()
            diagonal += vjp.square() / CFG["probes"]
        H = CFG["mu"] + diagonal / scale
        result = dict(raw=torch.stack(raw_grad).flatten(1),
                      cal=torch.stack(calibrated_grad).flatten(1),
                      b=b, H=H, gn_diag=diagonal, beta=beta.float(),
                      alpha=alpha.float(), B=B.float(),
                      residual=(alpha-beta @ B).float(), ridge=ridge,
                      task=task.detach(), control=control.detach(),
                      logits=logits.detach(), source_cost=float(source_cost.detach()),
                      feature_mse=float(residual.square().mean().detach()),
                      clip_fraction=float(((image <= 0) | (image >= 1)).float().mean()))
        for key in ["raw", "cal", "b", "H", "beta"]:
            if not torch.isfinite(result[key]).all():
                raise RuntimeError(f"nonfinite derivative: {pair} {key}")
        return result

    def evaluate(self, pair, y, beta=None, center_control=None, hidden=False):
        ctx = self.context[pair]
        with torch.no_grad():
            image, pooled, feature, outputs = self.forward(y, hidden=hidden)
            logits, task, control = self.values(pooled, ctx["embeddings"])
            cost = .5 * (feature-ctx["source_feature"]).square().sum()/ctx["scale"]
            cost += .5*CFG["mu"]*(y-ctx["x"]).square().sum()
            calibrated = task if beta is None else task-beta@(control-center_control)
        result = dict(logits=logits, task=task, control=control,
                      calibrated=calibrated, source_cost=float(cost),
                      image=image, pooled=pooled, feature=feature)
        if hidden:
            middle = F.normalize(outputs.hidden_states[6][0], dim=-1)
            result["middle"] = middle
        return result

    def geometry(self, dr, calibrated):
        C = dr["cal"] if calibrated else dr["raw"]
        inverse = 1 / dr["H"].flatten()
        row_norm = torch.sqrt((C.double().square()*inverse.double()).sum(1)).float()
        Cn = C / row_norm.clamp_min(1e-10)[:, None]
        Gram = (Cn.double()*inverse.double()) @ Cn.double().T
        vals, vec = torch.linalg.eigh(Gram)
        keep = vals > max(float(vals.max())*CFG["rank_rtol"], 1e-8)
        U = vec[:, keep].float()
        Ca = U.T @ Cn
        G = (Ca.double()*inverse.double()) @ Ca.double().T
        Gi = torch.linalg.pinv(G, rtol=1e-10).float()
        lift = inverse[:, None]*Ca.T
        def project(z):
            return lift @ (Gi @ (Ca @ z.flatten()))
        w = U.T @ torch.full((2,), .5, device=self.device)
        g = Ca.T @ w
        return dict(project=project, g=g.reshape_as(dr["b"]), C=Ca,
                    row_norm=row_norm, rank=int(keep.sum()),
                    eigenvalues=vals.cpu().tolist(),
                    Cp_residual=float((Ca @ (dr["b"].flatten()-project(dr["b"]))).norm()
                                      / dr["b"].norm().clamp_min(1e-8)))

    def proposal(self, dr, d, h, method, kappa):
        calibrated = method != "RAW"
        geo = self.geometry(dr, calibrated)
        H = dr["H"]
        dnorm = float(tensor_norm(d, H))
        invg = geo["g"] / H
        gnorm = float(tensor_norm(invg, H))
        valid = geo["rank"] > 0 and gnorm > 1e-8
        drive = CFG["drive_ratio"] * dnorm * invg / max(gnorm, 1e-8) if valid else torch.zeros_like(d)
        ref = d + drive
        bvec = dr["b"]/H
        Pd = geo["project"](d).reshape_as(d)
        Pb = geo["project"](bvec).reshape_as(d)
        if method == "DRIVE" or not valid:
            correction = torch.zeros_like(d)
        elif method == "ISO":
            correction = -(kappa/(1+kappa))*(ref+bvec)
        else:
            anchor = torch.zeros_like(d) if method == "FILTER" else bvec-Pb
            correction = -(kappa/(1+kappa))*(d-Pd+anchor)
        raw_cnorm = float(tensor_norm(correction, H))
        cap = CFG["correction_cap"] * dnorm
        factor = min(1., cap/max(raw_cnorm, 1e-30))
        correction = factor*correction
        pred_task = (dr["cal"] if calibrated else dr["raw"]) @ correction.flatten()
        return ref, correction, geo, dict(
            metric_increment_norm=dnorm, metric_drive_norm=float(tensor_norm(drive, H)),
            raw_repair_norm=raw_cnorm, repair_cap=cap, repair_cap_factor=factor,
            task_rank=geo["rank"], task_eigenvalues=geo["eigenvalues"],
            task_projection_residual=geo["Cp_residual"],
            linear_repair_task=pred_task.cpu().tolist(), valid_task=valid,
            kappa=0. if method=="DRIVE" else kappa,
            lambda_value=0. if method=="DRIVE" else kappa/h)

    def corrected_step(self, pair, y, d, h, method, kappa):
        dr = self.derivatives(pair, y)
        ref, correction, geo, info = self.proposal(dr, d, h, method, kappa)
        calibrated = method != "RAW"
        beta = dr["beta"] if calibrated else torch.zeros_like(dr["beta"])
        ref_state = (y + ref).detach()
        reference = self.evaluate(pair, ref_state, beta, dr["control"])
        origin = dr["task"]
        tolerance = CFG["task_abs_tolerance"] + CFG["task_relative_tolerance"] * (
            reference["calibrated"]-origin).abs()
        alpha = 1.
        accepted = False
        trials = []
        best = reference
        new = ref_state
        if method == "DRIVE" or float(correction.norm()) == 0:
            accepted = True
        else:
            for attempt in range(CFG["backtracks"]):
                candidate = (y+ref+alpha*correction).detach()
                observed = self.evaluate(pair, candidate, beta, dr["control"])
                task_ok = bool(torch.all(observed["calibrated"] >= (
                    reference["calibrated"]-tolerance)))
                source_ok = observed["source_cost"] <= reference["source_cost"] + 1e-6
                trials.append(dict(alpha=alpha, task_ok=task_ok, source_ok=source_ok,
                    source_delta=observed["source_cost"]-reference["source_cost"],
                    task_delta=(observed["calibrated"]-reference["calibrated"]).cpu().tolist()))
                if task_ok and source_ok:
                    accepted=True;new=candidate;best=observed;break
                alpha *= .5
        if not accepted:
            alpha=0.
        actual_u = new-y
        actual_repair = new-ref_state
        norm_before = (alpha*correction).norm()
        info.update(accepted=accepted, repair_alpha=alpha, trials=trials,
                    optimization_calls=1, metric_repair_norm=float(tensor_norm(actual_repair, dr["H"])),
                    repair_to_increment=float(actual_repair.norm()/d.norm().clamp_min(1e-12)),
                    actual_total_correction=float((actual_u-d).norm()),
                    increment_ratio=float(actual_u.norm()/d.norm().clamp_min(1e-12)),
                    source_cost_before=dr["source_cost"],
                    source_cost_reference=reference["source_cost"],
                    source_cost_after=best["source_cost"],
                    task_before=dr["task"].cpu().tolist(),
                    task_after=best["task"].cpu().tolist(),
                    calibrated_after=best["calibrated"].cpu().tolist(),
                    calibrated_reference=reference["calibrated"].cpu().tolist(),
                    beta=dr["beta"].cpu().tolist(),
                    nuisance_derivative_raw=dr["alpha"].cpu().tolist(),
                    nuisance_derivative_residual=dr["residual"].cpu().tolist(),
                    H_min=float(dr["H"].min()), H_max=float(dr["H"].max()),
                    source_gradient_norm=float(dr["b"].norm()),
                    clip_fraction=dr["clip_fraction"])
        info["predicted_actual_task"] = (
            (dr["cal"] if calibrated else dr["raw"]) @ actual_u.flatten()).cpu().tolist()
        info["actual_step_local_task"] = (best["calibrated"]-dr["task"]).cpu().tolist()
        return new, info


def save_image(tensor, path):
    arr = tensor[0].permute(1, 2, 0).detach().cpu().numpy()
    Image.fromarray(np.clip(np.rint(arr*255), 0, 255).astype("uint8")).save(path)


def calibration(exp, pairs):
    all_rows = []
    for pair in pairs:
        ctx = exp.setup(pair)
        states = torch.load(ROOT / "experiments/flowedit_frequency_trajectory/outputs" /
                            pair / "trajectory.pt", weights_only=True)
        for name, y in [("source",ctx["x"]), ("FlowEdit_mid",states["states"][34].to(exp.device)),
                        ("FlowEdit_final",states["final"].to(exp.device))]:
            started=time.monotonic()
            dr=exp.derivatives(pair,y)
            source=exp.evaluate(pair,y)
            row=dict(pair=pair,state=name,task=dr["task"].cpu().tolist(),
                     beta=dr["beta"].cpu().tolist(),raw=dr["alpha"].cpu().tolist(),
                     residual=dr["residual"].cpu().tolist(),
                     coefficient_norm=dr["beta"].norm(dim=1).cpu().tolist(),
                     raw_gradient_norm=dr["raw"].norm(dim=1).cpu().tolist(),
                     calibrated_gradient_norm=dr["cal"].norm(dim=1).cpu().tolist(),
                     transforms=[])
            image=source["image"]
            for axis in [0,1]:
                for strength in [-.15,-.03,.03,.15]:
                    direction=image*(1-image)
                    if axis:direction=direction*(2*image-1)
                    altered=image+strength*direction
                    with torch.no_grad():
                        pooled=F.normalize(exp.sc.model.get_image_features(
                            pixel_values=exp.sc.preprocess(altered)),dim=-1)
                        _,task,control=exp.values(pooled,ctx["embeddings"])
                        adj=task-dr["beta"]@(control-dr["control"])
                    row["transforms"].append(dict(axis=axis,strength=strength,
                        task_delta=(task-dr["task"]).cpu().tolist(),
                        calibrated_delta=(adj-dr["task"]).cpu().tolist()))
            row["seconds"]=time.monotonic()-started
            all_rows.append(row)
            write_json(OUT/"calibration.json",dict(config=CFG,rows=all_rows))
            print(json.dumps(dict(pair=pair,state=name,raw=row["raw"],residual=row["residual"])),flush=True)
    print("CALIBRATION_COMPLETE",flush=True)


def local(exp,pairs):
    all_rows=[]
    meta=json.loads((ROOT/"experiments/stage1_full/inputs/manifest.json").read_text())
    for pair in pairs:
        exp.setup(pair)
        saved=torch.load(ROOT/"experiments/flowedit_frequency_trajectory/outputs"/pair/"trajectory.pt",weights_only=True)
        for step in [33]:
            y=saved["states"][step].to(exp.device)
            d=saved["states"][step+1].to(exp.device)-y
            times=meta["timesteps"];h=(times[step]-times[step+1])/1000
            dest=OUT/pair/"local";dest.mkdir(parents=True,exist_ok=True)
            save_image(exp.evaluate(pair,y+d)["image"],dest/f"BASE_step{step}.png")
            for kappa in [.03,.1,.3]:
                for method in METHODS[1:]:
                    new,info=exp.corrected_step(pair,y,d,h,method,kappa)
                    vals=exp.evaluate(pair,new)
                    row=dict(pair=pair,step=step,method=method,kappa_grid=kappa,
                             target_logit=float(vals["logits"][0]),
                             source_logit=float(vals["logits"][1]),**info)
                    all_rows.append(row)
                    if kappa==CFG["kappa"]:
                        save_image(vals["image"],dest/f"{method}_step{step}.png")
                    write_json(OUT/"local.json",dict(config=CFG,rows=all_rows))
            print(json.dumps(dict(pair=pair,local_complete=True)),flush=True)
    print("LOCAL_COMPLETE",flush=True)


def full(exp,pairs,methods):
    from diffusers import SD3Transformer2DModel
    sys.path.insert(0,str(ROOT/"vendor/flowedit"))
    from FlowEdit_utils import calc_v_sd3
    transformer=SD3Transformer2DModel.from_pretrained(
        ROOT/"pretrained_models/sd3-medium/transformer",torch_dtype=torch.float32,
        local_files_only=True).eval().requires_grad_(False).to(exp.device)
    pipe=SimpleNamespace(transformer=transformer,do_classifier_free_guidance=True)
    meta=json.loads((ROOT/"experiments/stage1_full/inputs/manifest.json").read_text())
    times=torch.tensor(meta["timesteps"],device=exp.device)
    results=[]
    for pair in pairs:
        ctx=exp.setup(pair);payload=ctx["payload"];x=ctx["x"]
        es,et=[tuple(t.to(exp.device) for t in e) for e in payload["embeddings"]]
        seq=torch.cat([es[1],es[0],et[1],et[0]])
        pool=torch.cat([es[3],es[2],et[3],et[2]])
        gen=torch.Generator("cpu").manual_seed(CFG["seed"])
        noise=[torch.randn(x.shape,generator=gen).to(exp.device) for _ in range(33)]
        pairdir=OUT/pair;pairdir.mkdir(parents=True,exist_ok=True)
        save_image(ctx["source_image"],pairdir/"source_vae.png")
        for method in methods:
            dest=pairdir/method;dest.mkdir(parents=True,exist_ok=True)
            if (dest/"summary.json").exists():
                results.append(json.loads((dest/"summary.json").read_text()))
                continue
            started=time.monotonic();y=x.clone();rows=[];latents=[y.cpu()]
            for k in range(17,50):
                t=times[k]/1000
                tn=times[k+1]/1000 if k<49 else torch.zeros_like(t)
                h=float(t-tn);qs=(1-t)*x+t*noise[k-17];qa=qs+y-x
                with torch.no_grad():
                    vs,va=calc_v_sd3(pipe,torch.cat([qs,qs,qa,qa]),seq,pool,3.5,13.5,times[k])
                    d=-h*(va-vs)
                if method=="BASE":
                    new=(y+d).detach();obs=exp.evaluate(pair,new)
                    info=dict(optimization_calls=0,source_cost_after=obs["source_cost"],
                              task_after=obs["task"].cpu().tolist(),repair_alpha=0,
                              metric_repair_norm=0,actual_total_correction=0)
                else:
                    new,info=exp.corrected_step(pair,y,d,h,method,CFG["kappa"])
                    obs=exp.evaluate(pair,new)
                if not torch.isfinite(new).all():
                    raise RuntimeError("nonfinite state")
                row=dict(step=k,t=float(t),h=h,
                         target_logit=float(obs["logits"][0]),
                         source_logit=float(obs["logits"][1]),
                         heldout_contrast=float(obs["logits"][12]-obs["logits"][13]),
                         rgb_rmse_to_source_vae=float(torch.sqrt(
                             (obs["image"]-ctx["source_image"]).square().mean())),
                         mean_srgb_luma=float((obs["image"]*torch.tensor(
                             [0.2126,0.7152,0.0722],device=exp.device).reshape(1,3,1,1)).sum(1).mean()),
                         control_response=obs["control"].cpu().tolist(),
                         latent_distance_to_source=float((new-x).norm()),**info)
                rows.append(row);latents.append(new.cpu());y=new
                if k in [17,22,27,33,38,43,47,49]:
                    save_image(obs["image"],dest/f"step_{k:02d}.png")
                write_json(dest/"progress.json",dict(pair=pair,method=method,rows=rows))
                if k%8==1:
                    print(json.dumps(dict(pair=pair,method=method,step=k,
                                         seconds=time.monotonic()-started)),flush=True)
            torch.cuda.synchronize(exp.device)
            final=exp.evaluate(pair,y,hidden=True)
            save_image(final["image"],dest/"final.png")
            cached=torch.load(ROOT/"experiments/flowedit_frequency_trajectory/outputs"/pair/"trajectory.pt",weights_only=True)["final"]
            baseline_error=float((y.cpu().double()-cached.double()).norm()/cached.double().norm())
            if method=="BASE" and baseline_error>1e-7:
                raise RuntimeError(f"baseline mismatch {baseline_error}")
            summary=dict(pair=pair,method=method,seconds=time.monotonic()-started,
                         active_steps=33,optimization_calls=sum(r["optimization_calls"] for r in rows),
                         nonzero_repairs=sum(r["metric_repair_norm"]>1e-8 for r in rows),
                         target_logit=float(final["logits"][0]),
                         source_logit=float(final["logits"][1]),
                         task=final["task"].cpu().tolist(),
                         heldout_contrast=float(final["logits"][12]-final["logits"][13]),
                         source_cost=final["source_cost"],latent_distance=float((y-x).norm()),
                         relative_to_cached_flowedit=baseline_error)
            torch.save(dict(record=ctx["rec"],states=torch.stack(latents),final=y.cpu(),
                            rows=rows,summary=summary,middle=final["middle"].cpu()),
                       dest/"trajectory.pt")
            write_json(dest/"summary.json",summary)
            results.append(summary);write_json(OUT/"full_summary.json",dict(config=CFG,results=results))
            print(json.dumps(summary),flush=True)
    print("FULL_COMPLETE",flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage",choices=["calibration","local","full"],required=True)
    ap.add_argument("--pairs",nargs="+",default=PAIRS)
    ap.add_argument("--methods",nargs="+",default=METHODS)
    ap.add_argument("--device",default="cuda:0")
    args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    protocol=dict(stage=args.stage,pairs=args.pairs,methods=args.methods,config=CFG,
                  torch=torch.__version__,device=args.device,
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  metric="four fixed VJP probes, positive diagonal Gauss-Newton approximation",
                  T=50,n_max=33,n_min=0,n_avg=1,source_cfg=3.5,target_cfg=13.5)
    write_json(OUT/f"protocol_{args.stage}.json",protocol)
    exp=Experiment(args.device)
    if args.stage=="calibration":calibration(exp,args.pairs)
    elif args.stage=="local":local(exp,args.pairs)
    else:full(exp,args.pairs,args.methods)


if __name__=="__main__":
    main()
