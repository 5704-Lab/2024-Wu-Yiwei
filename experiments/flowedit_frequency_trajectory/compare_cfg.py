"""Compare fixed-noise original CFG and equal CFG1 trajectories."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).parent;OLD=BASE/'outputs';NEW=BASE/'outputs_cfg1';pairs=json.loads((NEW/'summary.json').read_text())['protocol']['pairs'];rows=[]
font=ImageFont.truetype('DejaVuSans.ttf',16);canvas=Image.new('RGB',(1280,len(pairs)*310),'white');draw=ImageDraw.Draw(canvas)
fig,axes=plt.subplots(len(pairs),2,figsize=(12,3*len(pairs)),squeeze=False)
for j,pair in enumerate(pairs):
 old=json.loads((OLD/pair/'metrics.json').read_text());new=json.loads((NEW/pair/'metrics.json').read_text());top=j*310
 changes={'bear__00':'brown bear -> black bear','brown_owl__00':'brown owl -> white owl','bus__00':'classic yellow van -> modern SUV','cat_crown__00':'crown -> top hat'}
 draw.text((6,top+4),pair+' | '+changes[pair],font=font,fill='black')
 for i,(label,path) in enumerate([('Source RGB',NEW/pair/'frames/source_rgb.png'),('Source VAE',NEW/pair/'frames/source_vae.png'),('FlowEdit CFG=3.5 / 13.5',OLD/pair/'frames/step_49.png'),('FlowEdit CFG=1 / 1',NEW/pair/'frames/step_49.png')]):
  draw.text((320*i+3,top+26),label,font=font,fill='black');canvas.paste(Image.open(path).convert('RGB').resize((256,256)),(320*i,top+50))
 for label,metrics in [('CFG3.5/13.5',old),('CFG1/1',new)]:
  r=metrics['steps'];h=np.array([x['time_before']-x['time_after'] for x in r]);axes[j,0].plot(np.arange(50),np.array([x['delta_rgb_rms'] for x in r])/h,label=label);axes[j,1].plot(np.arange(50),[x['band_fraction'][2] for x in r],label=label)
 axes[j,0].set_title(changes[pair]+' | RGB change rate');axes[j,1].set_title('High frequency fraction (>=0.25 cycles/pixel)')
 for ax in axes[j]:ax.set_xlabel('step k');ax.legend(fontsize=8);ax.grid(alpha=.2)
 oldE=sum(x['delta_energy'] for x in old['steps']);newE=sum(x['delta_energy'] for x in new['steps'])
 rows.append(dict(pair=pair,source_cfg=1,target_cfg=1,sum_increment_energy_original_cfg=oldE,sum_increment_energy_cfg1=newE,energy_ratio_cfg1_to_original=newE/oldE,max_rms_cfg1=max(x['delta_rgb_rms'] for x in new['steps']),max_parseval_error_cfg1=new['max_parseval_error'],latent_relative_difference_to_original_cfg=new['baseline_final_relative_error'],note='different CFG: endpoint distance is not a reproduction error; increment energy sum is not final-edit energy'))
canvas.save(NEW/'cfg_final_comparison.png');fig.tight_layout();fig.savefig(NEW/'cfg_frequency_comparison.png',dpi=150);plt.close(fig)
(NEW/'cfg_comparison.json').write_text(json.dumps(dict(rows=rows,all_other_generation_settings_fixed=True),indent=2)+'\n');print(json.dumps(rows,indent=2))
