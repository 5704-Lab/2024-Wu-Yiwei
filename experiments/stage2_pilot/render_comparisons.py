from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from correctflow.comparison import render_comparison
OUT=ROOT/'experiments/stage2_pilot/outputs'
for p in OUT.glob('*/complete.json'):
    case=json.loads(p.read_text());dest=p.parent
    images={'source':dest/'source.png','flowedit':dest/'official.png','matched interface':dest/'baseline.png','VAE reconstruction':dest/'vae_reconstruction.png'}
    for arm in case['arms']:
        if arm not in {'baseline','official','vae_reconstruction'}:images[arm]=dest/f'{arm}.png'
    render_comparison(images,dest/'overview.png',title=case['record']['pair_id']+' | seed 0 | checkpoint 33')
    print(dest/'overview.png')
