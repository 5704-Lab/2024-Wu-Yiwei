"""Independent decoded-image fidelity measurements; no scores control model inference."""
import argparse
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from diffusers import AutoencoderKL
from skimage.metrics import structural_similarity
from common import ROOT,HERE,setup,write,sha,decode_state,save_image

FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def metrics(image,reference):
    a=image[0].float().permute(1,2,0).numpy()
    b=reference[0].float().permute(1,2,0).numpy()
    mse=float(np.mean((a.astype('float64')-b.astype('float64'))**2))
    return dict(rmse=mse**.5,psnr_db=None if mse==0 else float(-10*np.log10(mse)),
        ssim=float(structural_similarity(a,b,channel_axis=2,data_range=1.,gaussian_weights=True,
                                         sigma=1.5,use_sample_covariance=False)))


def panel(columns,path,title,subtitle):
    side=384;canvas=Image.new('RGB',(side*len(columns),side+110),'white');draw=ImageDraw.Draw(canvas)
    draw.text((10,7),title,font=ImageFont.truetype(FONT,19),fill='black')
    draw.text((10,35),subtitle,font=ImageFont.truetype(FONT,12),fill='#333333')
    for i,(label,source) in enumerate(columns):
        draw.text((side*i+8,60),label,font=ImageFont.truetype(FONT,15),fill='black')
        im=Image.open(source).convert('RGB').resize((side,side),Image.Resampling.LANCZOS)
        canvas.paste(im,(side*i,84))
    draw.text((10,side+91),'One real source; native method configurations and all reconstruction controls are reported.',
              font=ImageFont.truetype(FONT,12),fill='#444444')
    canvas.save(path)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preview',action='store_true')
    ap.add_argument('--preview-tag',default='preview');args=ap.parse_args()
    setup();out=HERE/'outputs/comparison_v1';device=torch.device('cuda:0')
    path=out/('checkpoint.pt' if args.preview else 'results.pt')
    data=torch.load(path,map_location='cpu',weights_only=True)
    states=data['states'];size=1024
    inp=torch.load(HERE/'outputs/inputs_v1/inputs.pt',map_location='cpu',weights_only=True)
    if args.preview:
        vae=AutoencoderKL.from_pretrained(ROOT/'pretrained_models/flux1-dev/vae',local_files_only=True,
            torch_dtype=torch.float32).eval().requires_grad_(False).to(device)
        images={}
        for name,latent in states.items():
            if name.endswith('_inverted'):continue
            im=decode_state(vae,latent,size,device)
            images[name]=im.cpu();save_image(im,out/(args.preview_tag+'_'+name+'.png'))
    else:images=data['images']
    original=torch.from_numpy(np.array(Image.open(HERE/'outputs/inputs_v1/source.png')).copy()).permute(2,0,1)[None].float()/255
    rows={}
    for name,image in images.items():
        rows[name]=dict(vs_vae_source=metrics(image,images['source']),vs_original=metrics(image,original))
        # Offline fixed source-coordinate evaluation windows; not fed into editing.
        for region,(y0,y1) in dict(top_background=(0,220),bottom_water=(850,1024)).items():
            rows[name][region]=metrics(image[:,:,y0:y1,:],images['source'][:,:,y0:y1,:])
    gates={}
    for bank in ['selfix_default','selfix_high_fidelity']:
        name=bank+'_reconstruction'
        if name not in rows:continue
        r=rows[name]['vs_vae_source']
        gates[bank]=dict(psnr_db=r['psnr_db'],ssim=r['ssim'],
                        passed=r['psnr_db'] is not None and r['psnr_db']>=28 and r['ssim']>=.9,
                        same_inverse_as_edits=True,source_reconstruction_without_feature_injection=True)
    prefix=args.preview_tag+'_' if args.preview else ''
    write(out/(prefix+'metrics.json'),dict(scope='source fidelity, not full semantic success certification',
        reconstruction_gates=gates,metrics=rows,script_sha256=sha(__file__),
        background_windows='offline manually fixed source coordinates, never editing masks'))
    if not args.preview:
        src=HERE/'outputs/inputs_v1/source.png'
        panel([('Real source',src),('FlowEdit (author config)',out/'flowedit_author.png'),
               ('SelFix (fidelity budget)',out/'selfix_high_fidelity_g2p0.png')],out/'comparison.png',
              'Same FLUX.1-dev: FlowEdit vs SelFix high-fidelity inversion editing',
              '1024 x 1024 | Same real source and full text | Native method settings and all NFE reported')
        panel([('Real source',src),('FlowEdit (author config)',out/'flowedit_author.png'),
               ('SelFix (official editing)',out/'selfix_default_g2p0.png'),
               ('SelFix (fidelity budget)',out/'selfix_high_fidelity_g2p0.png')],out/'all_recipes.png',
              'FlowEdit vs recent inversion editing: SAME FLUX.1-dev, source and text',
              '1024 x 1024 | SelFix official solver + value injection; all settings and NFE reported separately')
        panel([('Real source',src),('Unchanged VAE source',out/'source.png'),
               ('SelFix default inversion',out/'selfix_default_reconstruction.png'),
               ('SelFix fidelity inversion',out/'selfix_high_fidelity_reconstruction.png')],out/'reconstruction.png',
              'Source reconstruction from the EXACT inversions used for editing',
              'Source prompt, guidance=1, no feature injection in the reconstruction control')
        panel([('Real source',src),('FlowEdit guidance 1 / 2',out/'flowedit_guidance_matched.png'),
               ('SelFix fidelity g=2',out/'selfix_high_fidelity_g2p0.png'),
               ('SelFix fidelity g=3.5',out/'selfix_high_fidelity_g3p5.png')],out/'guidance_controls.png',
              'Additional guidance controls: all trials retained',
              'Source conditioning, inverse state and injection settings documented in protocol.json')
    print(json.dumps(dict(preview=args.preview,reconstruction_gates=gates),indent=2),flush=True)


if __name__=='__main__':main()
