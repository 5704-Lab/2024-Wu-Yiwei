"""Auxiliary frozen SigLIP2 endpoint probe, not a task-success oracle."""
import argparse
import json
from pathlib import Path
import sys
import time

from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE/"outputs/study_v1"
sys.path.insert(0,str(ROOT/"experiments/sota_inversion_20261105"))
from common import write, setup, sha

SHORT = {
    "castle": ["a castle", "a castle made of Lego bricks"],
    "flowers_draw": ["a watercolor painting of three red flowers", "a watercolor painting of three sunflowers"],
    "bus": ["a yellow vintage van", "a modern SUV"],
    "penguins": ["two penguins standing in a lake", "two origami penguins standing in a lake"],
    "flowers": ["a bouquet of pink, red and white flowers", "a bouquet of orange, yellow and white flowers"],
    "dalmatian": ["a Dalmatian dog looking up", "a Dalmatian dog looking at the camera"],
    "corgi": ["a brown and white dog", "a brown and white dog made of Lego bricks"],
    "brown_owl": ["a brown owl", "a white owl"],
    "mountain_black": ["a green hill in a black landscape", "an Egyptian pyramid in a black landscape"],
    "piece_of_cake": ["a chocolate cake", "a chocolate cake with a cherry on top"],
    "bear": ["a brown bear walking in a stream", "a black bear walking in a stream"],
    "cat_crown": ["a gray cat wearing a crown", "a gray cat wearing a black top hat"],
    "dogs": ["two dingo dogs", "two red foxes"],
    "sign": ["a billboard saying LOVE IS ALL YOU NEED", "a billboard saying CVPR IS ALL YOU NEED"],
    "puppies": ["two golden retriever puppies", "two husky puppies"],
    "luna": ["a neon sign saying Luna", "a neon sign saying Sol"],
    "dogs2": ["two dogs standing on a grassy hillside", "two wolves standing on a grassy hillside"],
    "yellow_bulldog": ["a yellow dog figurine", "a yellow lion figurine"],
    "kid_running": ["a young boy running through a grassy field", "a shiny silver robot running through a grassy field"],
    "beer_glass": ["a wine glass filled with beer", "a wine glass filled with a red cocktail and an orange wedge"],
    "rocks": ["a stack of gray rocks on a beach", "a stack of colorful macarons on a beach"],
    "japanese_castle": ["a Japanese castle", "a Japanese castle made of Lego bricks"],
    "tiger": ["a tiger standing in a swamp", "a lion standing in a swamp"],
    "dog": ["a dog sitting on grass", "a dog standing on grass"],
}


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--wait",action="store_true")
    args=ap.parse_args()
    setup()
    device="cuda:1"
    root=ROOT/"pretrained_models/siglip2-base-patch16-256"
    model=AutoModel.from_pretrained(root,local_files_only=True,torch_dtype=torch.float32,attn_implementation="eager").eval().requires_grad_(False).to(device)
    processor=AutoProcessor.from_pretrained(root,local_files_only=True)
    protocol=json.loads((OUT/"protocol.json").read_text())
    dest=OUT/"scores"
    dest.mkdir(exist_ok=True)
    write(dest/"protocol.json",dict(role="auxiliary endpoint text-image proxy, never an optimization objective or success certificate",
        short_prompts=SHORT, input="saved clipped 8bit PNG endpoints", metric="normalized cosine target-minus-source",
        model="frozen local SigLIP2 base patch16 256", script_sha256=sha(Path(__file__)), independent_human_rating=False))
    all_rows=[]
    for record in protocol["records"]:
        pair=record["pair_id"]
        directory=OUT/"measurements"/pair
        while not (directory/"complete.json").exists():
            if not args.wait:
                raise RuntimeError(f"Missing measured source {pair}")
            time.sleep(5)
        texts=SHORT[pair.split("__")[0]]
        tokens=processor(text=texts,padding="max_length",max_length=64,truncation=True,return_tensors="pt").to(device)
        text=model.get_text_features(**tokens)
        text=text/text.norm(dim=-1,keepdim=True)
        entries=[("source",-1,OUT/"inputs"/pair/"source_vae.png")]
        entries += [("full",seed,directory/f"full_seed{seed}.png") for seed in protocol["seeds"]]
        if pair in protocol["online_sources"]:
            entries += [(method,0,directory/f"online_{method}.png") for method in protocol["arms"][1:]]
        images=[Image.open(path).convert("RGB") for _,_,path in entries]
        pixels=processor(images=images,return_tensors="pt")["pixel_values"].to(device)
        features=model.get_image_features(pixel_values=pixels)
        features=features/features.norm(dim=-1,keepdim=True)
        cosine=features@text.T
        source_similarity=features@features[0]
        rows=[]
        for i,(method,seed,path) in enumerate(entries):
            rows.append(dict(pair=pair,category=record["category"],method=method,seed=seed,
                source_cosine=float(cosine[i,0]),target_cosine=float(cosine[i,1]),
                margin=float(cosine[i,1]-cosine[i,0]),
                margin_gain_from_source=float((cosine[i,1]-cosine[i,0])-(cosine[0,1]-cosine[0,0])),
                image_source_feature_cosine=float(source_similarity[i])))
        write(dest/f"{pair}.json",rows)
        all_rows.extend(rows)
        print("SCORED",pair,flush=True)
    write(dest/"all_scores.json",all_rows)
    write(dest/"complete.json",dict(sources=24,images=len(all_rows),script_sha256=sha(Path(__file__))))


if __name__=="__main__":
    main()
