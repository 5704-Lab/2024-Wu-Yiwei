"""Full-size per-step spectra and shared legend, from saved numerical powers."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,default=Path(__file__).parent/'outputs');args=ap.parse_args()
OUT=args.output_dir;summary=json.loads((OUT/'summary.json').read_text());ref=summary['shared_power_reference'];cm=plt.get_cmap('magma')
for entry in summary['pairs']:
 root=OUT/entry['pair'];dest=root/'spectra';dest.mkdir(exist_ok=True)
 with np.load(root/'spectra.npz') as archive:data={key:archive[key] for key in archive.files}
 for name,p in [('source_rgb',data['original_power']),('source_vae',data['source_vae_power'])]+[(f'step_{i:02d}',p) for i,p in enumerate(data['step_delta_power'])]:
  db=10*np.log10(np.maximum(np.fft.fftshift(p)/ref,1e-10));rgb=(cm(np.clip((db+100)/100,0,1))[...,:3]*255).astype('uint8')
  im=Image.new('RGB',(570,575),'white');d=ImageDraw.Draw(im);im.paste(Image.fromarray(rgb),(30,30));d.text((30,8),entry['pair']+' | '+name,fill='black');d.text((30,548),'-0.5',fill='black');d.text((282,548),'0',fill='black');d.text((518,548),'+0.5',fill='black');d.text((80,562),'cycles/pixel; center=DC; same -100..0 dB scale',fill='black');im.save(dest/f'{name}.png')
fig,ax=plt.subplots(figsize=(8,1.1));fig.colorbar(ScalarMappable(norm=Normalize(-100,0),cmap='magma'),cax=ax,orientation='horizontal',label='dB relative to maximum source RGB power; zero energy uses display floor -100 dB');fig.tight_layout();fig.savefig(OUT/'spectrum_colorbar.png',dpi=150);plt.close(fig)
print('FULL_SIZE_SPECTRA_EXPORTED')
