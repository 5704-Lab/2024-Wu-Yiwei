"""Summarize completed local pilots; no editing-quality inference."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];P=Path(__file__).parent/'outputs'
pairs=['bear__00','brown_owl__00','bus__00','cat_crown__00']
rows=[]
fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for pair,ax in zip(pairs,axes.flat):
 r=json.loads((P/f'{pair}_step33_float32_checkpoint.json').read_text())
 ax.axhline(r['cosine_before'],color='black',linestyle='--',label='Original FlowEdit (w=0)')
 xs=[t['norm']/(r['radius']/.05) for t in r['trials']]
 ax.semilogx(xs,[t['cosine_after'] for t in r['trials']],'o-',label='Clean-response correction, FP32')
 fp64=P/f'{pair}_step33_float64_checkpoint_shard.json'
 if fp64.exists():
  rr=json.loads(fp64.read_text())
  ax.semilogx([t['norm']/(rr['radius']/.05) for t in rr['trials']],[t['cosine_after'] for t in rr['trials']],'s:',label='FP64 diagnostic')
 ax.set(title=pair,xlabel='Correction norm / original update norm',ylabel='Target response direction cosine',ylim=(-1.05,1.05))
 ax.legend(fontsize=7);ax.grid(alpha=.2)
 rows.append(dict(pair=pair,angle_before=r['angle_before_degrees'],mixed_chain_error=r['mixed_chain_error'],registered_two_trials_pass=any(t['armijo_without_numerical_margin'] for t in r['trials'] if t['scale'] in [1.,.5]),diagnostic_passing_scales=[t['scale'] for t in r['trials'] if t['armijo_without_numerical_margin']],seconds=r['elapsed_seconds'],scope='local proxy; numerical margin and full quality gate not completed'))
fig.suptitle('Original FlowEdit clean states | step 33 | response time 0')
fig.savefig(P/'local_response_comparison.png',dpi=160);fig.savefig(P/'local_response_comparison.svg');plt.close(fig)
files=[p for p in P.glob('*.json')]+[p for p in Path(__file__).parent.glob('*.py')]
summary=dict(scope='4 development source images, one cached original FlowEdit state each; no complete edited-image experiment',rows=rows,sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
(P/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(rows,ensure_ascii=False,indent=2))
