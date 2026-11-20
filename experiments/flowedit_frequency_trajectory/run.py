"""Original FlowEdit full clean-state trajectory and aligned RGB delta spectra."""
import argparse,hashlib,json,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diffusers import SD3Transformer2DModel,AutoencoderKL

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'vendor/flowedit'));from FlowEdit_utils import calc_v_sd3
ap=argparse.ArgumentParser();ap.add_argument('--pairs',nargs='+',default=['bear__00','brown_owl__00','bus__00','cat_crown__00']);ap.add_argument('--seed',type=int,default=0)
ap.add_argument('--source-cfg',type=float,default=3.5);ap.add_argument('--target-cfg',type=float,default=13.5);ap.add_argument('--output-dir',type=Path,default=Path(__file__).parent/'outputs');args=ap.parse_args()
OUT=args.output_dir;OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
device=torch.device('cuda:0');meta=json.loads((ROOT/'experiments/stage1_full/inputs/manifest.json').read_text());times=torch.tensor(meta['timesteps'],device=device)
protocol=dict(pairs=args.pairs,seed=args.seed,T=50,n_max=33,n_min=0,n_avg=1,source_cfg=args.source_cfg,target_cfg=args.target_cfg,
 first_active_step=17,inactive_steps='0 through 16: unchanged clean state, explicit zero deltas',resolution=512,
 spectrum='RGB-channel-mean 2D power, orthonormal FFT divided by H*W; sum equals mean squared RGB signal',
 delta='decoded clipped float RGB I_after minus I_before; not difference between two power spectra',
 original='dataset RGB resized to512 with bilinear; source VAE reconstruction separately shown',
 code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
model=SD3Transformer2DModel.from_pretrained(ROOT/'pretrained_models/sd3-medium/transformer',torch_dtype=torch.float32,local_files_only=True).eval().requires_grad_(False).to(device)
vae=AutoencoderKL.from_pretrained(ROOT/'pretrained_models/sd3-medium/vae',torch_dtype=torch.float32,local_files_only=True).eval().requires_grad_(False).to(device);pipe=SimpleNamespace(transformer=model,do_classifier_free_guidance=True)
def decode(y):
 with torch.no_grad():p=((vae.decode(y/vae.config.scaling_factor+vae.config.shift_factor,return_dict=False)[0]+1)/2).clamp(0,1)
 return p[0].permute(1,2,0).cpu().numpy()
def power(a):return np.mean(np.abs(np.fft.fft2(a,axes=(0,1),norm='ortho'))**2,axis=2)/(a.shape[0]*a.shape[1])
def png(a):return Image.fromarray(np.clip(np.rint(a*255),0,255).astype('uint8'))
rad=np.sqrt(np.fft.fftfreq(512)[:,None]**2+np.fft.fftfreq(512)[None,:]**2);bands=[rad<.0625,(rad>=.0625)&(rad<.25),rad>=.25];freq_edges=np.linspace(0,np.sqrt(.5),65);binid=np.clip(np.digitize(rad,freq_edges)-1,0,63)
all_data=[]
for pair in args.pairs:
 start=time.monotonic();dest=OUT/pair;dest.mkdir(exist_ok=True);frames=dest/'frames';frames.mkdir(exist_ok=True)
 inp=torch.load(ROOT/'experiments/stage1_full/inputs'/f'{pair}.pt',weights_only=True);x=inp['x'].to(device);y=x.clone();es,et=[tuple(t.to(device) for t in e) for e in inp['embeddings']]
 seq=torch.cat([es[1],es[0],et[1],et[0]]);pool=torch.cat([es[3],es[2],et[3],et[2]])
 gen=torch.Generator('cpu').manual_seed(args.seed);noise=[torch.randn(x.shape,generator=gen).to(device) for _ in range(33)]
 source=np.asarray(Image.open(ROOT/inp['record']['source_image']).convert('RGB').resize((512,512),Image.Resampling.BILINEAR),dtype=np.float64)/255;initial=decode(x);previous=initial
 png(source).save(frames/'source_rgb.png');png(initial).save(frames/'source_vae.png');spectra=[];radial=[];rows=[];latents=[y.cpu()];frame_imgs=[]
 for k in range(50):
  t=times[k]/1000;tn=times[k+1]/1000 if k<49 else torch.zeros_like(t);h=float(t-tn)
  if k>=17:
   qs=(1-t)*x+t*noise[k-17];qa=qs+y-x
   with torch.no_grad():vs,va=calc_v_sd3(pipe,torch.cat([qs,qs,qa,qa]),seq,pool,args.source_cfg,args.target_cfg,times[k]);y=y-h*(va-vs)
   current=decode(y)
  else:current=previous
  delta=current.astype(np.float64)-previous.astype(np.float64);p=power(delta);energy=float(p.sum());mse=float(np.mean(delta**2));spectra.append(p.astype(np.float32));radial.append(np.bincount(binid.flatten(),weights=p.flatten(),minlength=64))
  rows.append(dict(step=k,active=k>=17,time_before=float(t),time_after=float(tn),delta_rgb_rms=float(np.sqrt(mse)),delta_energy=energy,parseval_error=abs(energy-mse),dc_energy=float(p[0,0]),band_energy=[float(p[b].sum()) for b in bands],band_fraction=[float(p[b].sum()/energy) if energy>0 else 0 for b in bands]))
  png(current).save(frames/f'step_{k:02d}.png');frame_imgs.append(png(current));latents.append(y.cpu());previous=current
 torch.save(dict(record=inp['record'],states=torch.stack(latents),steps=rows,final=y.cpu()),dest/'trajectory.pt')
 old=ROOT/'experiments/stage1_full/outputs'/f'{pair}__seed{args.seed}'/'states.pt';error=None
 if old.exists():baseline=torch.load(old,weights_only=True)['baseline_final'];error=float((y.cpu().double()-baseline.double()).norm()/baseline.double().norm())
 np.savez_compressed(dest/'spectra.npz',original_power=power(source).astype(np.float32),source_vae_power=power(initial).astype(np.float32),step_delta_power=np.stack(spectra),radial_delta_energy=np.stack(radial),radial_frequency_centers=(freq_edges[:-1]+freq_edges[1:])/2)
 summary=dict(pair=pair,record=inp['record'],steps=rows,baseline_final_relative_error=error,reference_baseline_cfg=dict(source=3.5,target=13.5),same_cfg_as_reference=args.source_cfg==3.5 and args.target_cfg==13.5,max_parseval_error=max(r['parseval_error'] for r in rows),seconds=time.monotonic()-start)
 (dest/'metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');all_data.append(dict(pair=pair,source=png(source),initial=png(initial),images=frame_imgs,spectra=spectra,source_power=power(source),initial_power=power(initial),rows=rows,radial=np.stack(radial)))
 print(json.dumps(dict(pair=pair,complete=True,baseline_error=error,seconds=summary['seconds'])),flush=True)

# One shared reference and color range for source and every signed-delta spectrum.
reference=max(float(a['source_power'].max()) for a in all_data);cmap=plt.get_cmap('magma')
def spectrum_image(p,size):
 db=10*np.log10(np.maximum(np.fft.fftshift(p)/reference,1e-10));rgb=(cmap(np.clip((db+100)/100,0,1))[...,:3]*255).astype('uint8')
 return Image.fromarray(rgb).resize((size,size),Image.Resampling.BILINEAR)
def panel(indices,size,path):
 cols=len(indices)+2;width=cols*size+160;height=len(all_data)*(2*size+65)+55;canvas=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(canvas)
 for j,a in enumerate(all_data):
  top=j*(2*size+65);draw.text((4,top+4),a['pair'],fill='black')
  ims=[a['source'],a['initial']]+[a['images'][k] for k in indices];powers=[a['source_power'],a['initial_power']]+[a['spectra'][k] for k in indices]
  labels=['Source RGB','Source VAE']+[f"k={k:02d} t={a['rows'][k]['time_after']:.3f}"+(' idle' if k<17 else '') for k in indices]
  for i,(im,p,label) in enumerate(zip(ims,powers,labels)):
   left=160+i*size;draw.text((left+2,top+3),label,fill='black');canvas.paste(im.resize((size,size)),(left,top+23));canvas.paste(spectrum_image(p,size),(left,top+23+size))
  draw.text((3,top+35),'FlowEdit clean',fill='black');draw.text((3,top+size+40),'Source power /',fill='black');draw.text((3,top+size+54),'step delta power',fill='black')
 draw.text((4,height-35),'Shared scale: -100 to 0 dB relative to maximum source RGB power. Brightness = absolute spectral energy.',fill='black');draw.text((4,height-20),'Step spectrum is FFT(I_after - I_before); original RGB-to-VAE reconstruction is NOT counted as an editing step.',fill='black');canvas.save(path)
panel(list(range(50)),96,OUT/'all_50_steps.png')
panel(list(range(17,50)),128,OUT/'all_33_active_steps.png')
for start in [0,10,20,30,40]:panel(list(range(start,start+10)),192,OUT/f'steps_{start:02d}_{start+9:02d}.png')

fig,axes=plt.subplots(len(all_data),3,figsize=(16,3.1*len(all_data)),squeeze=False)
for j,a in enumerate(all_data):
 rows=a['rows'];k=np.arange(50);axes[j,0].plot(k,[r['delta_rgb_rms'] for r in rows]);axes[j,0].set_title(a['pair']+' | signed RGB increment RMS');axes[j,0].set_ylabel('RGB [0,1]');axes[j,0].axvline(17,color='gray',ls='--')
 for bi,label in enumerate(['low <0.0625','mid 0.0625-0.25','high >=0.25']):axes[j,1].plot(k,[r['band_energy'][bi] for r in rows],label=label)
 axes[j,1].set_yscale('symlog',linthresh=1e-9);axes[j,1].set_title('Absolute band energy incl. DC');axes[j,1].legend(fontsize=7)
 lograd=10*np.log10(np.maximum(a['radial']/reference,1e-10));im=axes[j,2].imshow(lograd.T,origin='lower',aspect='auto',extent=[-.5,49.5,0,np.sqrt(.5)],cmap='magma',vmin=-100,vmax=0);axes[j,2].set_title('Radial summed delta energy, shared dB');axes[j,2].set_ylabel('cycles/pixel')
 for ax in axes[j]:ax.set_xlabel('original FlowEdit step k')
fig.colorbar(im,ax=axes[:,2].tolist(),label='dB relative to source peak',fraction=.018,pad=.02);fig.subplots_adjust(hspace=.5,wspace=.3,right=.9);fig.savefig(OUT/'frequency_evolution.png',dpi=150);plt.close(fig)
(OUT/'summary.json').write_text(json.dumps(dict(protocol=protocol,shared_power_reference=reference,spectrum_db_range=[-100,0],pairs=[dict(pair=a['pair'],max_parseval_error=max(r['parseval_error'] for r in a['rows'])) for a in all_data]),indent=2)+'\n')
print('TRAJECTORY_AND_SPECTRUM_COMPLETE',flush=True)
