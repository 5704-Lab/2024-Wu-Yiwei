"""Saved-image diagnostics and figures. No optimization or parameter selection."""
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT,OUT,PAIRS,write,panel


def load(path):return np.asarray(Image.open(path).convert('RGB'),dtype=np.float32)/255


def gradient_tensor_distance(source,image):
    source=cv2.resize(source,(128,128),interpolation=cv2.INTER_AREA)
    image=cv2.resize(image,(128,128),interpolation=cv2.INTER_AREA)
    def field(im):
        gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
        gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)/8
        gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)/8
        energy=gx*gx+gy*gy
        tensor=np.stack([gx*gx-gy*gy,2*gx*gy],-1)/(energy[...,None]+1e-5)
        return tensor,energy
    a,energy=field(source);b,_=field(image)
    weights=energy/(energy+1e-4)
    return float(np.sqrt(np.sum(weights*np.sum((a-b)**2,axis=-1))/weights.sum()))


def sift_diagnostic(source,image):
    a=np.uint8(np.rint(source*255));b=np.uint8(np.rint(image*255))
    sift=cv2.SIFT_create(nfeatures=1500)
    ka,da=sift.detectAndCompute(cv2.cvtColor(a,cv2.COLOR_RGB2GRAY),None)
    kb,db=sift.detectAndCompute(cv2.cvtColor(b,cv2.COLOR_RGB2GRAY),None)
    if da is None or db is None:return dict(matches=0,median_displacement=None)
    matcher=cv2.BFMatcher()
    ab=matcher.knnMatch(da,db,k=2);ba=matcher.knnMatch(db,da,k=2)
    reverse={m.queryIdx:m.trainIdx for m,n in ba if m.distance<.75*n.distance}
    good=[m for m,n in ab if m.distance<.75*n.distance and reverse.get(m.trainIdx)==m.queryIdx]
    ds=[np.linalg.norm(np.array(ka[m.queryIdx].pt)-np.array(kb[m.trainIdx].pt)) for m in good]
    return dict(matches=len(good),median_displacement=float(np.median(ds)) if ds else None,
                fraction_within_2px=float(np.mean(np.array(ds)<2)) if ds else None,
                limitation='matches can be background-dominated; not an object pose/identity metric')


def main():
    rows=[];grids=[]
    for pair in PAIRS:
        source_path=OUT/'real'/pair/'source_vae.png'
        if not source_path.exists():continue
        source=load(source_path)
        import torch
        data=torch.load(ROOT/'experiments/stage1_full/inputs'/f'{pair}.pt',weights_only=True)
        columns=[('Source RGB',ROOT/data['record']['source_image']),('Source VAE',source_path)]
        for method in ['BASE','PERMIT','RAW_PHASE','NORM_CONTROL']:
            dest=OUT/'full'/pair/method
            if not (dest/'metrics.json').exists():continue
            saved=json.loads((dest/'metrics.json').read_text());image=load(dest/'final.png')
            metrics=saved['metrics']
            row=dict(pair=pair,method=method,**metrics,gradient_tensor_distance=gradient_tensor_distance(source,image),
                     sift=sift_diagnostic(source,image),seconds=saved['seconds'],
                     baseline_relative_error=saved['baseline_relative_error'],
                     accepted=sum(r['accepted'] for r in saved['steps']),
                     nonzero=sum(r['correction_norm']>1e-8 for r in saved['steps']),
                     mean_increment_ratio=float(np.mean([r['increment_ratio'] for r in saved['steps']])))
            rows.append(row)
            label='Original FlowEdit' if method=='BASE' else method
            columns.append((label,dest/'final.png'))
        if len(columns)>2:
            path=OUT/f'{pair}_comparison.png'
            panel(columns,path,title=f'{pair} | SD3 FP32, seed 0, CFG 3.5/13.5',size=320)
            grids.append(Image.open(path))
    if grids:
        canvas=Image.new('RGB',(max(im.width for im in grids),sum(im.height for im in grids)),'white')
        offset=0
        for im in grids:canvas.paste(im,(0,offset));offset+=im.height
        canvas.save(OUT/'all_comparison.png')
    write(OUT/'evaluation.json',dict(rows=rows,scope='descriptive four-image development/stress evaluation; no population significance'))
    if (OUT/'mechanism.json').exists():
        data=json.loads((OUT/'mechanism.json').read_text())
        names=['channel_gain','smooth_gain','shift_025','shift_050','shift_100','texture']
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for order in [1,2,3]:
            med=[np.median([r['residual_fraction'] for r in data['rows'] if r['order']==order and r['case']==name]) for name in names]
            axes[0].plot(range(len(names)),med,'o-',label=f'{order}x{order} appearance basis')
        axes[0].set_xticks(range(len(names)),['Channel gain','Smooth gain','Shift 1px','Shift 2px','Shift 4px','Texture'],rotation=25)
        axes[0].set_ylabel('Remaining / original phase distance');axes[0].set_ylim(-.03,1.05);axes[0].legend(fontsize=8)
        axes[0].set_title('Known interventions on 8 natural images (median)')
        for pair in PAIRS:
            inf=[next(r for r in data['information'] if r['pair']==pair and r['order']==order) for order in [1,2,3]]
            axes[1].plot([1,2,3],[r['geometry_information_eigenvalues'][0] for r in inf],'o-',label=pair)
        axes[1].set_xlabel('Appearance cosine order');axes[1].set_ylabel('Minimum conditional information eigenvalue')
        axes[1].set_title('More appearance freedom reduces geometry information');axes[1].legend(fontsize=8)
        fig.tight_layout();fig.savefig(OUT/'mechanism_validation.png',dpi=180);plt.close(fig)
    if len(rows)>=4:
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for ax,pair in zip(axes.flat,PAIRS):
            rs=[r for r in rows if r['pair']==pair]
            for r in rs:
                ax.scatter(r['permitted_phase'],r['short_margin'],s=65)
                ax.annotate(r['method'],(r['permitted_phase'],r['short_margin']),fontsize=8,xytext=(4,4),textcoords='offset points')
            ax.set_title(pair);ax.set_xlabel('Permitted phase residual (lower is better proxy)')
            ax.set_ylabel('Short target - source logit (task proxy only)')
        fig.tight_layout();fig.savefig(OUT/'task_structure_tradeoff.png',dpi=180);plt.close(fig)
    print('ANALYSIS_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
