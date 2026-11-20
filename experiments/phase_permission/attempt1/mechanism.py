"""Natural-image intervention identification; no generative-model inference."""
import time
import numpy as np
import torch
from common import OUT,PAIRS,Phase,Appearance,source_rgb,warp,orthogonal_basis,seed_setup,write,code_hashes,panel


def main():
    seed_setup()
    pairs=PAIRS+['flowers__00','corgi__00','penguins__00','dalmatian__00']
    rows=[];information=[]
    write(OUT/'protocol_mechanism.json',dict(pairs=pairs,source='dataset RGB at 128, float64 CPU',
         legal=['uniform_gain','channel_gain','smooth_gain'],geometric=['shift_025','shift_050','shift_100','mixed_shift'],
         unknown=['texture'],orders=[1,2,3],pose_unit='pixels at 128 (multiply by 4 at 512)',code=code_hashes()))
    for pair in pairs:
        started=time.monotonic()
        rgb,_=source_rgb(pair,dtype=torch.float64)
        phase=Phase(rgb);src=phase.source_small
        base=phase(src).detach()
        app2=Appearance(src,2)
        theta=torch.zeros(app2.n,dtype=src.dtype)
        theta[0]=-.35;theta[4]=-.15;theta[8]=.05
        theta[1]=.04;theta[6]=-.03;theta[11]=.02
        smooth=app2(theta).detach()
        x=torch.arange(128,dtype=src.dtype)[None,None,None,:]
        cases=dict(identity=src,uniform_gain=src*.65,
                   channel_gain=src*torch.tensor([.55,.85,1.05])[None,:,None,None],
                   smooth_gain=smooth,shift_025=warp(src,tx=.25),shift_050=warp(src,tx=.5),
                   shift_100=warp(src,tx=1.),mixed_shift=warp(smooth,tx=.5),
                   texture=src+.025*torch.sin(2*np.pi*8*x/128))
        # Derivative units are pixels at 128. Symmetric interpolation derivative.
        eps=.01
        with torch.no_grad():
            mx=((phase(warp(src,tx=eps))-phase(warp(src,tx=-eps)))/(2*eps)).flatten()
            my=((phase(warp(src,ty=eps))-phase(warp(src,ty=-eps)))/(2*eps)).flatten()
            m=torch.stack([mx,my],1)
        mats=[]
        for order in [1,2,3]:
            app=Appearance(src,order)
            zero=torch.zeros(app.n,dtype=src.dtype)
            v=app.jacobian(phase,zero,eps=1e-4)
            u,s,rank=orthogonal_basis(v)
            qm=m-u@(u.T@m)
            inf=qm.T@qm
            vals=torch.linalg.eigvalsh(inf)
            um,_,_=torch.linalg.svd(m,full_matrices=False)
            angles=torch.linalg.eigvalsh(um.T@(um-u@(u.T@um)))
            mats.append(inf)
            information.append(dict(pair=pair,order=order,parameter_count=app.n,effective_rank=rank,
                nuisance_singular_values=s,geometry_information_eigenvalues=vals.tolist(),
                normalized_information_eigenvalues=angles.tolist(),condition_number=float(vals.max()/vals.min().clamp_min(1e-30))))
            for name,image in cases.items():
                target=phase(image).detach()
                t,ref,fit=app.fit(phase,target,maxiter=40)
                w=(target-base).flatten()
                qw=w-u@(u.T@w)
                est=torch.linalg.lstsq(qm,qw).solution
                row=dict(pair=pair,order=order,case=name,raw_phase_distance=float(w.norm()),
                         profiled_distance=float((target-ref).norm()),
                         residual_fraction=float((target-ref).norm()/w.norm().clamp_min(1e-12)),
                         linear_displacement_estimate=est.tolist(),fit=fit)
                rows.append(row)
            write(OUT/'mechanism.json',dict(rows=rows,information=information))
        monotonic=[float(torch.linalg.eigvalsh(mats[i]-mats[i+1]).min()) for i in range(2)]
        panel([('Source',src),('Allowed channel gain',cases['channel_gain']),('Allowed smooth gain',smooth),
               ('Shift 4 px at 512',cases['shift_100']),('Unknown texture',cases['texture'])],
              OUT/'mechanism_images'/f'{pair}.png',title=pair)
        print(dict(pair=pair,seconds=round(time.monotonic()-started,1),information_monotonic_min_eigs=monotonic),flush=True)
    summary={}
    for order in [1,2,3]:
        for name in cases:
            rs=[r for r in rows if r['order']==order and r['case']==name]
            summary[f'{order}/{name}']=dict(raw_median=float(np.median([r['raw_phase_distance'] for r in rs])),
                profiled_median=float(np.median([r['profiled_distance'] for r in rs])),
                retained_median=float(np.median([r['residual_fraction'] for r in rs])))
    write(OUT/'mechanism_summary.json',summary)
    print('MECHANISM_COMPLETE',flush=True)


if __name__=='__main__':main()
