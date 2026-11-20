"""Natural relation controls; raw and reflection-antisymmetric scores."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.nn.attention import sdpa_kernel,SDPBackend
HERE=Path(__file__).parent
sys.path.insert(0,str(HERE.parent/'phase_permission'))
from common import ROOT,CleanScorer,write,seed_setup,panel

CASES=[
('cat_and_dog','horizontal',[
('A dog to the left of a cat.','A cat to the left of a dog.'),
('A dog on the left and a cat on the right.','A cat on the left and a dog on the right.'),
('A cat sitting to the right of a dog.','A dog sitting to the right of a cat.')]),
('cat_and_dog2','horizontal',[
('A cat to the left of a dog.','A dog to the left of a cat.'),
('A cat on the left and a dog on the right.','A dog on the left and a cat on the right.'),
('A dog lying to the right of a cat.','A dog lying to the left of a cat.')]),
('butterfly','vertical',[
('A butterfly above a purple flower.','A purple flower above a butterfly.'),
('A purple flower below a butterfly.','A butterfly below a purple flower.'),
('A butterfly at the top and a purple flower at the bottom.','A purple flower at the top and a butterfly at the bottom.')]),
('gray_bird','horizontal',[
('A bird facing left.','A bird facing right.'),
('A bird looking to the left.','A bird looking to the right.'),
('A bird with its beak pointing left.','A bird with its beak pointing right.')]),
('parrots2','horizontal',[
('The parrot on the left is higher than the parrot on the right.','The parrot on the right is higher than the parrot on the left.'),
('A higher parrot on the left and a lower parrot on the right.','A lower parrot on the left and a higher parrot on the right.'),
('Two parrots, with the left parrot above the right one.','Two parrots, with the right parrot above the left one.')])]

def main():
 seed_setup();out=HERE/'outputs';sc=CleanScorer(torch.device('cuda:0'),torch.float32,'native');rows=[]
 for name,axis,pairs in CASES:
  im=Image.open(ROOT/'data/flowedit/Images'/f'{name}.png').convert('RGB').resize((512,512))
  original=torch.tensor(np.asarray(im).copy(),device=sc.device,dtype=torch.float32).permute(2,0,1)[None]/255
  flip=original.flip(-1 if axis=='horizontal' else -2)
  e,lengths=sc.encode([t for pair in pairs for t in pair])
  with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
   scores=sc.logits_image(torch.cat([original,flip]),e).cpu().numpy()
  for j,(pos,neg) in enumerate(pairs):
   raw0=float(scores[0,2*j]-scores[0,2*j+1]);raw1=float(scores[1,2*j]-scores[1,2*j+1])
   anti=.5*(raw0-raw1)
   rows.append(dict(image=name,axis=axis,text_positive=pos,text_negative=neg,raw_original=raw0,
                    raw_transformed=raw1,raw_pair_correct=raw0>0 and raw1<0,
                    antisymmetric_original=anti,antisymmetric_pair_correct=anti>0,
                    note='antisymmetry is guaranteed; correct sign is not'))
  panel([('Original: first relation',original),('Transformed: reversed relation',flip)],out/f'{name}_binding.png',title=name)
  write(out/'binding.json',dict(rows=rows,raw_pair_correct=sum(r['raw_pair_correct'] for r in rows),
       antisymmetric_pair_correct=sum(r['antisymmetric_pair_correct'] for r in rows),total=len(rows),
       scope='5 visually verified natural source images x 3 paraphrases; not independent test of general binding',
       sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
  print(name,[(r['raw_pair_correct'],r['antisymmetric_pair_correct'],round(r['antisymmetric_original'],3)) for r in rows if r['image']==name],flush=True)
 print('BINDING_COMPLETE',flush=True)
if __name__=='__main__':main()
