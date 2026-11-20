"""Make fixed-layout scientific comparison figures from actual model outputs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def figure(folder,name,title,columns,subtitle,crop=None):
    width=384
    height=330 if crop else 384
    canvas=Image.new('RGB',(width*len(columns),height+130),'white')
    draw=ImageDraw.Draw(canvas)
    draw.text((12,8),title,font=ImageFont.truetype(FONT,20),fill='black')
    draw.text((12,37),subtitle,font=ImageFont.truetype(FONT,13),fill='#333333')
    for i,(label,filename) in enumerate(columns):
        im=Image.open(folder/filename).convert('RGB')
        if crop: im=im.crop(crop)
        im=im.resize((width,height),Image.Resampling.LANCZOS)
        canvas.paste(im,(i*width,84))
        draw.text((i*width+10,61),label,font=ImageFont.truetype(FONT,16),fill='black')
    draw.text((12,height+94),'No feature injection, optimization or spatial edit mask; solver and steps are labeled above.',
              font=ImageFont.truetype(FONT,14),fill='#333333')
    draw.text((12,height+113),'One source only. Source pixel distance is not a measure of complete editing success.',
              font=ImageFont.truetype(FONT,12),fill='#555555')
    canvas.save(folder/name)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',default='bear_v1');args=ap.parse_args()
    folder=HERE/'outputs'/args.tag
    result=json.loads((folder/'complete.json').read_text())
    protocol=json.loads((folder/'protocol.json').read_text())
    columns=[('Source real image','source.png'),('FlowEdit: 33 steps','flowedit.png'),
             ('ODE partial: 33 + 33','ode_partial.png'),('ODE full: 50 + 50','ode_full.png')]
    subtitle='Brown bear -> black bear | SD3 Medium, FP32, 512 x 512 | Source CFG 3.5; target CFG 13.5'
    figure(folder,'comparison.png','Real-image editing: FlowEdit versus direct ODE inversion',columns,subtitle)
    figure(folder,'head_detail.png','Same source-coordinate detail; no alignment or region-dependent update',columns,
           'Fixed crop x=256:512, y=90:310 in every 512 x 512 output.',crop=(256,90,512,310))
    figure(folder,'reconstruction.png','Source reconstruction controls: no target prompt',[
        ('Original source','source.png'),('VAE reconstruction','vae_reconstruction.png'),
        ('ODE partial reconstruction','ode_partial_reconstruction.png'),
        ('ODE full reconstruction','ode_full_reconstruction.png')],
        'Every ODE reconstruction uses the source prompt and CFG 3.5 in both directions.')
    with (folder/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['method']+list(next(iter(result['metrics'].values()))))
        writer.writeheader()
        for name,metrics in result['metrics'].items():writer.writerow(dict(method=name,**metrics))
    files=['comparison.png','head_detail.png','reconstruction.png','source.png','flowedit.png','ode_partial.png',
           'ode_full.png','vae_reconstruction.png','ode_partial_reconstruction.png','ode_full_reconstruction.png',
           'protocol.json','complete.json','states.pt','metrics.csv']
    audit=dict(source_count=1,model_runs='new actual model inference',
               no_runtime_target_reference=True,source_reconstruction_diagnostic=True,
               numerical_baseline_replay=result['stages']['flowedit'],
               input_reencoding_error=result['source_reencoding_relative_error'],
               scripts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'run.py',Path(__file__)]},
               files={name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in files},
               partial_start_sigma=protocol['partial_sigma'])
    (folder/'artifact_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(output=str(folder/'comparison.png'),metrics=result['metrics'],
                         total_seconds=result['elapsed_seconds']),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
