"""Real-source correspondence experiment; frozen original SD3 only."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DEV=['bear','brown_owl','bus','cat_crown']
CAL=['corgi','dalmatian','dog','castle']
TEST=['tiger','puppies','dogs','dogs2','penguins','kid_running','japanese_castle','rocks','beer_glass','piece_of_cake','luna','yellow_bulldog']
PAIRS=[x+'__00' for x in DEV+CAL+TEST]
LAYERS=[3,11,20]
PROBES=[(.2,20261104),(.2,20261105),(.35,20261104),(.35,20261105)]
ROLES={
 'bear':['bear','water'],'brown_owl':['owl','grass'],'bus':['SUV','house'],'cat_crown':['cat','cloth'],
 'corgi':['dog','ground'],'dalmatian':['dog','background'],'dog':['dog','field'],'castle':['castle','sky'],
 'tiger':['lion','water'],'puppies':['puppies','field'],'dogs':['foxes','wooded'],'dogs2':['wolves','hillside'],
 'penguins':['penguins','lake'],'kid_running':['robot','field'],'japanese_castle':['castle','trees'],
 'rocks':['macarons','beach'],'beer_glass':['glass','cocktail'],'piece_of_cake':['cake','table'],
 'luna':['sign','street'],'yellow_bulldog':['figurine','scene']}


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def setup():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False


def image(vae,y):
    return ((vae.decode(y/vae.config.scaling_factor+vae.config.shift_factor,return_dict=False)[0]+1)/2).clamp(0,1)


def save_image(im,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    a=im.detach().cpu()[0].permute(1,2,0).numpy()
    Image.fromarray(np.uint8(np.rint(a.clip(0,1)*255))).save(path)


def split(pair):
    stem=pair.split('__')[0]
    return 'development' if stem in DEV else 'calibration' if stem in CAL else 'heldout'
