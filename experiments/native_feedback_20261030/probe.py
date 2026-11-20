"""Frozen SD3 native, read-only measurements. No model processors are replaced.

Each observe call owns its hooks and concept state. Concepts read image K/V and
their own K/V, never write to the main image/text stream. Units: pre-out-proj.
This is an experimental measurement, not an established semantic fact detector.
"""
from pathlib import Path
import hashlib
import json
import math
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LAYERS = (3, 11, 20)
CONCEPTS = ['bear', 'head', 'nose', 'ears', 'body', 'legs', 'gray table',
            'blue wall', 'yellow ball', 'bear on the left', 'bear on the right',
            'head of the bear on the left', 'nose of the bear on the left',
            'body of the bear on the left', 'head of the bear on the right',
            'nose of the bear on the right', 'body of the bear on the right']


def digest(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def pool_tokens(x, side=8):
    n = int(math.isqrt(x.shape[1]))
    assert n*n == x.shape[1]
    return F.adaptive_avg_pool2d(x.transpose(1, 2).reshape(x.shape[0], -1, n, n), side).flatten(2).transpose(1, 2)


def heads(x, attn):
    return x.reshape(x.shape[0], -1, attn.heads, x.shape[-1]//attn.heads).transpose(1, 2)


def merge(x):
    return x.transpose(1, 2).flatten(2)


def concept_read(q, image_k, image_v, concept_k, concept_v):
    k = torch.cat([image_k, concept_k], 2)
    v = torch.cat([image_v, concept_v], 2)
    p = (q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])).softmax(-1)
    output = merge(p @ v)
    spatial = p[..., :image_k.shape[2]].mean(1)
    mass = spatial.sum(-1)
    conditional = spatial / mass[..., None].clamp_min(1e-12)
    return output, conditional, mass


class NativeProbe:
    def __init__(self, model, bank):
        self.model, self.bank = model, bank

    def observe(self, y, eps, tau, condition, source_queries=None):
        """No detach in the differentiable observations. Caller controls no_grad.

        source_queries selects anchored queries; the dynamic branch is always
        propagated through *every* block in the prefix, not just saved layers.
        """
        model = self.model
        seq = condition['seq'].to(y)
        pooled = condition['pool'].to(y)
        time = torch.full((y.shape[0],), tau*1000, device=y.device, dtype=y.dtype)
        temb = model.time_text_embed(time, pooled)
        state = {'concept': model.context_embedder(self.bank['initial'].to(y)[None])}
        records, queries, handles = {}, {}, []
        def pre_attention(index, block):
            def hook(attn, args, kwargs):
                if index > max(LAYERS):
                    return
                image = kwargs['hidden_states']
                cn, gate, shift, scale, gate_ff = block.norm1_context(state['concept'], emb=temb)
                ik, iv = heads(attn.to_k(image), attn), heads(attn.to_v(image), attn)
                if attn.norm_k is not None:
                    ik = attn.norm_k(ik)
                cq = heads(attn.add_q_proj(cn), attn)
                ck = heads(attn.add_k_proj(cn), attn)
                cv = heads(attn.add_v_proj(cn), attn)
                if attn.norm_added_q is not None:
                    cq = attn.norm_added_q(cq)
                if attn.norm_added_k is not None:
                    ck = attn.norm_added_k(ck)
                b, p, mass = concept_read(cq, ik, iv, ck, cv)
                # The side branch reuses this projection; its pre-hook must not
                # mistake a concept tensor for the native prompt C tensor.
                state['in_side_projection'] = True
                try:
                    b_projected = attn.to_add_out(b)
                finally:
                    state['in_side_projection'] = False
                c = state['concept'] + gate[:, None]*b_projected
                c_norm = block.norm2_context(c)*(1+scale[:, None])+shift[:, None]
                state['concept'] = c + gate_ff[:, None]*block.ff_context(c_norm)
                if index in LAYERS:
                    queries[index] = {'q': cq, 'k': ck, 'v': cv}
                    row = {'dynamic_B': b, 'dynamic_probability': p, 'dynamic_mass': mass,
                           'image_K': pool_tokens(merge(ik)), 'image_V': pool_tokens(merge(iv))}
                    if source_queries is not None:
                        s = {k: v.to(y) for k, v in source_queries[index].items()}
                        ab, ap, am = concept_read(s['q'], ik, iv, s['k'], s['v'])
                        row.update(anchored_B=ab, anchored_probability=ap, anchored_mass=am)
                    records[index] = row
            return hook
        def capture_image(index):
            def hook(module, args):
                o = args[0]
                row = records[index]
                row['image_O'] = pool_tokens(o)
                for branch in ['dynamic', 'anchored']:
                    if branch+'_B' in row:
                        row[branch+'_cosine_map'] = F.normalize(row[branch+'_B'], dim=-1) @ F.normalize(o, dim=-1).transpose(-1, -2)
                        row[branch+'_dot_map'] = row[branch+'_B'] @ o.transpose(-1, -2)
            return hook
        def capture_context(index):
            def hook(module, args):
                if state.get('in_side_projection', False):
                    return
                c = args[0]
                masks = condition['content_masks']
                row = records[index]
                row['context_C'] = torch.stack([c[:, mask.to(c.device)].mean(1) for mask in masks], 1)
            return hook
        try:
            for i, block in enumerate(model.transformer_blocks):
                if i <= max(LAYERS):
                    handles.append(block.attn.register_forward_pre_hook(pre_attention(i, block), with_kwargs=True))
                if i in LAYERS:
                    handles.append(block.attn.to_out[0].register_forward_pre_hook(capture_image(i)))
                    handles.append(block.attn.to_add_out.register_forward_pre_hook(capture_context(i)))
            prediction = model(hidden_states=(1-tau)*y+tau*eps, timestep=time,
                encoder_hidden_states=seq, pooled_projections=pooled, return_dict=False)[0]
        finally:
            for handle in handles:
                handle.remove()
        return {'layers': records, 'queries': queries, 'prediction': prediction}


def cpu_observation(obs, keep_queries=False):
    out = {'layers': {l: {k: v.detach().cpu() for k, v in row.items()} for l, row in obs['layers'].items()}}
    if keep_queries:
        out['queries'] = {l: {k: v.detach().cpu() for k, v in row.items()} for l, row in obs['queries'].items()}
    return out
