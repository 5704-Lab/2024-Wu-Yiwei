"""Step-size-aware interpretation of saved original FlowEdit image increments."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,default=Path(__file__).parent/'outputs');args=ap.parse_args()
OUT=args.output_dir;summary=json.loads((OUT/'summary.json').read_text());result=[]
fig,axes=plt.subplots(len(summary['pairs']),2,figsize=(12,3*len(summary['pairs'])),squeeze=False)
for j,entry in enumerate(summary['pairs']):
 root=OUT/entry['pair'];m=json.loads((root/'metrics.json').read_text());rows=m['steps'];h=np.array([r['time_before']-r['time_after'] for r in rows]);rms=np.array([r['delta_rgb_rms'] for r in rows]);rate=rms/h;k=np.arange(50)
 peak=max(rows,key=lambda x:x['delta_energy']);rate_peak=int(np.argmax(rate));phase=[]
 for name,lo,hi in [('early',17,28),('middle',28,39),('late',39,50)]:
  a=rows[lo:hi];energy=sum(r['delta_energy'] for r in a)
  phase.append(dict(phase=name,steps=[lo,hi-1],sum_increment_energy=energy,high_frequency_fraction=sum(r['band_energy'][2] for r in a)/energy,dc_fraction=sum(r['dc_energy'] for r in a)/energy,mean_increment_rms=float(rms[lo:hi].mean()),mean_time_normalized_rms=float(rate[lo:hi].mean())))
 record=dict(pair=entry['pair'],peak_increment_step=peak['step'],peak_increment_rms=peak['delta_rgb_rms'],peak_rate_step=rate_peak,peak_rate_rms=float(rate[rate_peak]),phases=phase);result.append(record)
 axes[j,0].plot(k,rate,label='RMS(delta RGB / h)');axes[j,0].axvline(17,color='gray',ls='--');axes[j,0].set_title(entry['pair']+' | average change rate');axes[j,0].set_ylabel('RGB / time')
 for bi,label in enumerate(['low <0.0625','middle 0.0625-0.25','high >=0.25']):axes[j,1].plot(k,[r['band_energy'][bi]/(h[i]**2) for i,r in enumerate(rows)],label=label)
 axes[j,1].set_yscale('symlog',linthresh=1e-7);axes[j,1].set_title('Band power of delta RGB / h');axes[j,1].legend(fontsize=8)
 for ax in axes[j]:ax.set_xlabel('original FlowEdit step k');ax.grid(alpha=.15)
fig.tight_layout();fig.savefig(OUT/'frequency_rate_evolution.png',dpi=150);plt.close(fig)
(OUT/'analysis.json').write_text(json.dumps(dict(pairs=result,phase_definition='three equal groups of 11 active steps; band fractions weighted by absolute increment energy',limitations=['Unwindowed FFT contains boundary effects and DC.','Sum of per-step energies is not final edit energy.','RGB spectral changes are not semantic labels or latent velocity spectra.','Dividing by h estimates an average finite-step image change rate.']),indent=2)+'\n')
print(json.dumps(result,indent=2))
