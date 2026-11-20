"""Fetch a pinned public Google SigLIP2 evaluator, with per-file SHA256."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib,json,time,requests
ROOT=Path(__file__).resolve().parents[1];dest=ROOT/'pretrained_models/siglip2-base-patch16-256';dest.mkdir(parents=True,exist_ok=True)
repo='google/siglip2-base-patch16-256'
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
for attempt in range(5):
 try:
  meta=requests.get(f'https://huggingface.co/api/models/{repo}',params={'blobs':'true'},timeout=60);meta.raise_for_status();meta=meta.json();break
 except requests.RequestException:
  if attempt==4:raise
  time.sleep(2)
rev=meta['sha']
files=[s for s in meta['siblings'] if not s['rfilename'].startswith('.')]
def fetch(row):
 name=row['rfilename'];p=dest/name;expected=row.get('lfs',{}).get('sha256')
 if not p.exists() or (expected and digest(p)!=expected):
  for attempt in range(4):
   try:
    with requests.get(f'https://huggingface.co/{repo}/resolve/{rev}/{name}',stream=True,timeout=(20,90)) as r:
     r.raise_for_status()
     with p.with_suffix(p.suffix+'.part').open('wb') as f:
      for b in r.iter_content(4*1024**2):f.write(b)
    partial=p.with_suffix(p.suffix+'.part')
    if expected and digest(partial)!=expected:raise RuntimeError('SHA mismatch')
    partial.replace(p);break
   except Exception:
    if attempt==3:raise
    time.sleep(2)
 result={'path':name,'bytes':p.stat().st_size,'sha256':digest(p),'upstream_lfs_sha256':expected};print(json.dumps(result),flush=True);return result
with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(fetch,files))
(dest/'download_manifest.json').write_text(json.dumps({'repository':repo,'revision':rev,'files':results,'complete':True,'purpose':'frozen independent evaluator, no image-editing guidance'},indent=2)+'\n')
print('DOWNLOAD_COMPLETE',flush=True)
