"""Offline factor-controlled 3D clay animals, never an inference-time oracle.

Analytic ellipsoid ray rendering separates intrinsic morphology, global heading,
head articulation, position, object identity and background. This is a controlled
shape/appearance task, not evidence of realistic species recognition.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
SOURCE_PROMPT = ('Two brown clay bear figurines standing on a gray table, with a blue wall '
                 'and a small yellow ball behind them.')
TARGET_PROMPT = ('Two clay bear figurines standing on a gray table, with a blue wall and a '
                 'small yellow ball behind them. The bear on the left is black with larger '
                 'rounded ears and a more slender muzzle. The bear on the right is brown.')
COLOR_PROMPT = ('Two clay bear figurines standing on a gray table, with a blue wall and a '
                'small yellow ball behind them. The bear on the left is black. '
                'The bear on the right is brown.')
POSE_PROMPT = ('Two brown clay bear figurines standing on a gray table, with a blue wall '
               'and a small yellow ball behind them. The bear on the left turns its head '
               'towards the camera.')


def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def render(seed, color=0., morph=0., heading=0., head=0., background=False,
           right_color=0., right_morph=0., shift=0., size=256,camera_side=1.):
    rng = np.random.default_rng(seed)
    camera = np.array([3.4*camera_side, 2.1, 7.5])
    camera += np.array([rng.uniform(-.35,.35),rng.uniform(-.1,.15),0.])
    target = np.array([0.,.75,0.])
    forward = target-camera; forward /= np.linalg.norm(forward)
    side = np.cross(forward,[0.,1.,0.]); side /= np.linalg.norm(side)
    up = np.cross(side,forward)
    yy,xx=np.mgrid[:size,:size]
    sx=((xx+.5)/size-.5)*4.85
    sy=(.5-(yy+.5)/size)*4.85
    ray = forward[None,None]+sx[...,None]/7.8*side+sy[...,None]/7.8*up
    ray /= np.linalg.norm(ray,axis=-1,keepdims=True)
    tground = -camera[1]/ray[...,1]
    ground = camera+tground[...,None]*ray
    wall = np.array([.43,.57,.69]) if not background else np.array([.65,.40,.49])
    table = np.array([.62,.63,.62]) if not background else np.array([.48,.56,.50])
    im = np.broadcast_to(wall,(size,size,3)).copy()
    is_ground=(tground>0)&(ground[...,2]>-1.5)
    modulation=.97+.025*np.sin(ground[...,0]*3.8)+.008*np.sin(ground[...,2]*5)
    im[is_ground]=(table*modulation[...,None])[is_ground]
    depth=np.where(is_ground,tground,np.inf)
    light=np.array([-2.,5.,5.]);light/=np.linalg.norm(light)
    brown=np.array([.40,.22,.12]);black=np.array([.075,.085,.095])
    primitives=[]

    def ell(center,radii,col,R=None):
        primitives.append((np.array(center),np.array(radii),np.array(col),np.eye(3) if R is None else R))

    # Source conditions vary independently of edit and remain fixed per source.
    base_yaws=[rng.uniform(-.25,.15),rng.uniform(-.4,.15)]
    scales=[rng.uniform(.94,1.06),rng.uniform(.92,1.05)]
    for index,(xpos,c,m) in enumerate([(-1.05,color,morph),(1.05,right_color,right_morph)]):
        yaw=base_yaws[index]+(heading if index==0 else 0.)
        R=rotation(yaw)
        origin=np.array([xpos+(shift if index==0 else 0.),0.,rng.uniform(-.04,.04)])
        scale=scales[index]
        fur=(1-c)*brown+c*black
        def bodyell(center,radii,col=fur,localR=None):
            center=origin+scale*(R@np.array(center))
            ell(center,scale*np.array(radii),col,R if localR is None else R@localR)
        bodyell([0.,.78,0.],[.63-.035*m,.37,.28-.015*m])
        bodyell([.27,1.,0.],[.30-.07*m,.25-.07*m,.25])
        for lx in [-.39,.37]:
            for lz in [-.19,.19]:
                bodyell([lx,.37,lz],[.115,.34,.11])
                bodyell([lx+.045,.075,lz],[.17,.075,.12])
        bodyell([-.60,.91,0.],[.10,.10,.10])
        Rh=rotation(head if index==0 else 0.)
        neck=np.array([.49,1.12,0.])
        def headell(offset,radii,col=fur):
            bodyell(neck+Rh@np.array(offset),radii,col,Rh)
        headell([.05,.04,0.],[.26-.025*m,.23,.23-.03*m])
        headell([.29,.00,0.],[.23+.035*m,.13-.035*m,.145-.04*m],fur*.91+.025)
        headell([.47,.025,0.],[.07,.065,.09-.018*m],[.025,.028,.031])
        for ez in [-.16,.16]:
            headell([-.035,.255,ez],[.10+.04*m,.105+.045*m,.075+.025*m])
            headell([.17,.13,ez*1.17],[.027,.03,.022],[.012,.014,.016])
    ell([-.2,.14,-.8],[.14,.14,.14],[.78,.62,.16])
    for center,radii,col,R in primitives:
        o=((camera-center)@R)/radii
        dr=(ray@R)/radii
        aa=np.sum(dr*dr,axis=-1);bb=2*np.sum(dr*o,axis=-1);cc=np.sum(o*o)-1
        discr=bb*bb-4*aa*cc
        distance=(-bb-np.sqrt(np.maximum(discr,0)))/(2*aa)
        visible=(discr>0)&(distance>0)&(distance<depth)
        hit=camera+distance[...,None]*ray
        normal=(((hit-center)@R)/(radii*radii))@R.T
        normal/=np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-12)
        diffuse=np.maximum(np.sum(normal*light,axis=-1),0)
        spec=np.maximum(np.sum(normal*(light-ray)/np.maximum(np.linalg.norm(light-ray,axis=-1,keepdims=True),1e-8),axis=-1),0)**35
        rgb=col*(.44+.55*diffuse[...,None])+spec[...,None]*.075
        im[visible]=rgb[visible];depth[visible]=distance[visible]
    im=np.clip(im,0,1)
    return Image.fromarray(np.uint8(np.rint(im*255))).resize((512,512),Image.Resampling.LANCZOS)


VARIANTS = {
    'source':{},
    'color_only':dict(color=1.),
    'partial':dict(color=.55,morph=.5),
    'valid':dict(color=1.,morph=1.),
    'valid_alt':dict(color=1.,morph=.8),
    'wrong_heading':dict(color=1.,morph=1.,heading=.85),
    'wrong_head':dict(color=1.,morph=1.,head=.85),
    'wrong_background':dict(color=1.,morph=1.,background=True),
    'wrong_object':dict(right_color=1.,right_morph=1.),
    'both_objects':dict(color=1.,morph=1.,right_color=1.,right_morph=1.),
    'shifted':dict(color=1.,morph=1.,shift=.25),
    'pose_only':dict(head=.85),
}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--count',type=int,default=16)
    ap.add_argument('--tag',default='dataset_v1');ap.add_argument('--seed-base',type=int,default=281031)
    ap.add_argument('--balanced-camera',action='store_true')
    args=ap.parse_args();out=HERE/'outputs'/args.tag
    if (out/'manifest.json').exists():raise RuntimeError('Immutable dataset already exists')
    out.mkdir(parents=True,exist_ok=True);records=[]
    for scene in range(args.count):
        camera_side=-1. if args.balanced_camera and scene%2 else 1.
        row=dict(scene=scene,seed=args.seed_base+scene,split='development' if scene<4 else 'calibration' if scene<8 else 'heldout',camera_side=camera_side,variants={})
        for name,factors in VARIANTS.items():
            p=out/f'scene{scene:02d}'/f'{name}.png';p.parent.mkdir(exist_ok=True)
            render(row['seed'],**factors,camera_side=camera_side).save(p)
            row['variants'][name]=dict(file=str(p.relative_to(HERE)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                factors=factors,legal_shape_task=name in ['source','color_only','partial','valid','valid_alt'],
                complete_shape_task=name in ['valid','valid_alt','wrong_heading','wrong_head','wrong_background','both_objects','shifted'],
                joint_shape_success=name in ['valid','valid_alt'])
        records.append(row);print('SCENE_COMPLETE',scene,flush=True)
    manifest=dict(scope='Factor-controlled explicit morphology/appearance editing; diagnostic truth only, not natural species validation.',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        prompts=dict(source=SOURCE_PROMPT,shape=TARGET_PROMPT,color=COLOR_PROMPT,pose=POSE_PROMPT),records=records)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    canvas=Image.new('RGB',(6*256,2*290),'white');draw=ImageDraw.Draw(canvas)
    for i,name in enumerate(VARIANTS):
        x=(i%6)*256;y=(i//6)*290
        canvas.paste(Image.open(out/'scene00'/f'{name}.png').resize((256,256)),(x,y+25));draw.text((x+4,y+5),name,fill='black')
    canvas.save(out/'contact.png')
    print('DATASET_COMPLETE',len(records),len(records)*len(VARIANTS),flush=True)


if __name__=='__main__':main()
