"""Add exact source/target captions and honest per-step condition ribbons."""
import argparse,json,textwrap
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,default=Path(__file__).parent/'outputs');args=ap.parse_args()
OUT=args.output_dir;summary=json.loads((OUT/'summary.json').read_text());ref=summary['shared_power_reference'];cm=plt.get_cmap('magma');source_cfg=summary['protocol']['source_cfg'];target_cfg=summary['protocol']['target_cfg']
changes={'bear__00':'brown bear -> black bear','brown_owl__00':'brown owl -> white owl','bus__00':'classic yellow van -> modern SUV (possibly electric)','cat_crown__00':'crown -> top hat; gray cat remains the subject'}
data=[]
for entry in summary['pairs']:
 root=OUT/entry['pair'];metrics=json.loads((root/'metrics.json').read_text())
 with np.load(root/'spectra.npz') as archive:powers={key:archive[key] for key in archive.files}
 data.append(dict(pair=entry['pair'],root=root,record=metrics['record'],rows=metrics['steps'],powers=powers))
def font(size):return ImageFont.truetype('DejaVuSans.ttf',size)
def spectrum(p,size):
 db=10*np.log10(np.maximum(np.fft.fftshift(p)/ref,1e-10));rgb=(cm(np.clip((db+100)/100,0,1))[...,:3]*255).astype('uint8')
 return Image.fromarray(rgb).resize((size,size),Image.Resampling.BILINEAR)
def panel(indices,size,path):
 width=160+(len(indices)+2)*size;head=135;stride=head+2*size+35;height=len(data)*stride+65
 canvas=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(canvas);f=font(16 if size>=128 else 14);small=font(12)
 for j,a in enumerate(data):
  top=j*stride;record=a['record'];draw.text((6,top+3),a['pair']+' | Requested: '+changes[a['pair']],font=f,fill='black')
  draw.text((160,top+28),'SOURCE TEXT: '+record['source_prompt'],font=f,fill='#965400')
  draw.text((160,top+51),'TARGET TEXT: '+record['target_prompt'],font=f,fill='#005b99')
  draw.text((160,top+75),f'Text conditions are fixed. Every active step evaluates source c_s (CFG {source_cfg:g}) and target c_tar (CFG {target_cfg:g}).',font=small,fill='#333333')
  names=['source_rgb','source_vae']+[f'step_{k:02d}' for k in indices]
  ps=[a['powers']['original_power'],a['powers']['source_vae_power']]+[a['powers']['step_delta_power'][k] for k in indices]
  labels=['Source RGB','Source VAE']+[f"k={k:02d} t={a['rows'][k]['time_after']:.3f}" for k in indices]
  ribbons=['source reference','reconstruction']+['c_s + c_tar' if k>=17 else 'idle: no update' for k in indices]
  for i,(name,p,label,ribbon) in enumerate(zip(names,ps,labels,ribbons)):
   left=160+i*size;draw.text((left+2,top+98),label,font=small,fill='black')
   color='#e3edf7' if ribbon=='c_s + c_tar' else '#eeeeee';draw.rectangle((left,top+head-17,left+size-1,top+head-1),fill=color);draw.text((left+2,top+head-16),ribbon,font=font(10),fill='black')
   image=Image.open(a['root']/'frames'/f'{name}.png').convert('RGB');canvas.paste(image.resize((size,size)),(left,top+head));canvas.paste(spectrum(p,size),(left,top+head+size))
  draw.text((6,top+head+20),'FlowEdit clean',font=small,fill='black');draw.text((6,top+head+size+20),'Source power /',font=small,fill='black');draw.text((6,top+head+size+35),'step delta power',font=small,fill='black')
 draw.text((6,height-50),'Shared spectra: -100..0 dB relative to source peak; center = DC. Step power = FFT(I_after - I_before).',font=f,fill='black')
 draw.text((6,height-27),'Requested source-to-target change is textual; no interpolation of text embeddings or switching captions across time.',font=f,fill='black');canvas.save(path)

panel(list(range(50)),96,OUT/'all_50_steps.png');panel(list(range(17,50)),128,OUT/'all_33_active_steps.png')
for start in [0,10,20,30,40]:panel(list(range(start,start+10)),192,OUT/f'steps_{start:02d}_{start+9:02d}.png')

for rate in [False,True]:
 ncols=2 if rate else 3;fig,axes=plt.subplots(len(data),ncols,figsize=(16 if not rate else 12,3.5*len(data)),squeeze=False)
 for j,a in enumerate(data):
  rows=a['rows'];k=np.arange(50);h=np.array([r['time_before']-r['time_after'] for r in rows]);rms=np.array([r['delta_rgb_rms'] for r in rows])
  axes[j,0].plot(k,rms/h if rate else rms);axes[j,0].set_title(changes[a['pair']]+'\n'+('RMS(delta RGB / h)' if rate else 'RMS(delta RGB)'),fontsize=10);axes[j,0].axvline(17,color='gray',ls='--')
  for b,label in enumerate(['low <0.0625','mid 0.0625-0.25','high >=0.25']):axes[j,1].plot(k,[r['band_energy'][b]/(h[i]**2 if rate else 1) for i,r in enumerate(rows)],label=label)
  axes[j,1].set_yscale('symlog',linthresh=1e-7 if rate else 1e-9);axes[j,1].set_title('Band power'+(' / h^2' if rate else ''));axes[j,1].legend(fontsize=7)
  if not rate:
   radial=a['powers']['radial_delta_energy'];im=axes[j,2].imshow((10*np.log10(np.maximum(radial/ref,1e-10))).T,origin='lower',aspect='auto',extent=[-.5,49.5,0,np.sqrt(.5)],cmap='magma',vmin=-100,vmax=0);axes[j,2].set_title('Radial delta energy');axes[j,2].set_ylabel('cycles/pixel')
  for ax in axes[j]:ax.set_xlabel('FlowEdit step k')
 fig.suptitle(f'Fixed source + target text at active steps | source CFG {source_cfg:g}, target CFG {target_cfg:g} | k<17 idle',fontsize=11)
 if not rate:fig.colorbar(im,ax=axes[:,2].tolist(),label='shared dB',fraction=.018,pad=.02)
 fig.subplots_adjust(top=.94,hspace=.62,wspace=.3,right=.9);fig.savefig(OUT/('frequency_rate_evolution.png' if rate else 'frequency_evolution.png'),dpi=150);plt.close(fig)
(OUT/'condition_annotations.json').write_text(json.dumps(dict(full_texts=[dict(pair=a['pair'],source=a['record']['source_prompt'],target=a['record']['target_prompt'],requested_change=changes[a['pair']]) for a in data],conditions_fixed=True,source_cfg=source_cfg,target_cfg=target_cfg,only_rendering_changed=True),ensure_ascii=False,indent=2)+'\n')
print('CONDITIONS_RENDERED_NO_GENERATION_RERUN')
