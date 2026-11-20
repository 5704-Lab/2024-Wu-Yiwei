"""Concept initial rows from original frozen T5 content-token means.

Content tokens exclude pad/eos. The main prompt is the exact existing cache;
only its content masks are obtained from the original tokenizers.
"""
import time
import torch
from transformers import T5EncoderModel, AutoTokenizer
from probe import ROOT, HERE, CONCEPTS, write, digest


def main():
    torch.set_num_threads(4)
    out = HERE/'outputs/bank_v1'
    if (out/'bank.pt').exists():
        raise RuntimeError('Immutable output exists')
    start = time.monotonic()
    tok = AutoTokenizer.from_pretrained(ROOT/'pretrained_models/sd3-medium/tokenizer_3', local_files_only=True)
    clip_tok = AutoTokenizer.from_pretrained(ROOT/'pretrained_models/sd3-medium/tokenizer', local_files_only=True)
    model = T5EncoderModel.from_pretrained(ROOT/'pretrained_models/sd3-medium/text_encoder_3',
        local_files_only=True, torch_dtype=torch.float32).eval().requires_grad_(False).to('cuda:0')
    tokens = tok(CONCEPTS, padding='max_length', max_length=256, truncation=True, return_tensors='pt')
    # SD3's native T5 call does not pass attention_mask; reproduce that convention.
    with torch.no_grad():
        hidden = model(input_ids=tokens.input_ids.to('cuda:0'))[0].cpu()
    valid = tokens.attention_mask.bool() & (tokens.input_ids != tok.eos_token_id)
    initial = torch.stack([h[m].mean(0) for h, m in zip(hidden, valid)])
    cache = torch.load(ROOT/'experiments/semantic_feedback_20261028/outputs/conditions_v1.pt', weights_only=True)
    for name, c in cache['conditions'].items():
        masks = []
        for tokenizer, length, offset in [(clip_tok, 77, 0), (tok, 256, 77)]:
            t = tokenizer(cache['prompts'][name], padding='max_length', max_length=length,
                truncation=True, return_tensors='pt')
            m = t.attention_mask[0].bool()
            for special in tokenizer.all_special_ids:
                m &= t.input_ids[0] != special
            if not m.any():
                m[0] = True  # empty-prompt baseline is explicitly not semantic content
            full = torch.zeros(c['seq'].shape[1], dtype=torch.bool)
            full[offset:offset+length] = m
            masks.append(full)
        c['content_masks'] = masks
    out.mkdir(parents=True)
    torch.save(dict(initial=initial, concepts=CONCEPTS, content_token_ids=[t[m].tolist() for t, m in zip(tokens.input_ids, valid)]), out/'bank.pt')
    torch.save(cache, out/'conditions.pt')
    write(out/'protocol.json', dict(concepts=CONCEPTS, initial_hash=digest(initial),
        initialization='original frozen T5 content token mean, EOS/PAD excluded; native no attention_mask',
        token_padding='native max_length 256', main_prompt_cache='semantic_feedback_20261028/outputs/conditions_v1.pt',
        seconds=time.monotonic()-start))
    print('BANK_COMPLETE', time.monotonic()-start, flush=True)


if __name__ == '__main__':
    main()
