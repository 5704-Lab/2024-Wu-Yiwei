"""Separate, registered follow-up to distinguish finite-scale nonlinearity from AD error.

Does not replace the initial audit or alter any editing trajectory. Run after the
main process releases its model. The smaller scales are fixed before this run.
"""
import gc
from pathlib import Path
import time
import torch

from run import ROOT,OUT,Execution,CleanObjective,rms,cosine,projection
from common import setup,write,sha,load_model


def main():
    setup()
    out=OUT/'gradient_refinement_v1'
    if out.exists():
        raise RuntimeError('Do not overwrite a follow-up audit')
    out.mkdir()
    write(out/'protocol.json',dict(reason='initial FP32 gradient-direction differences at RMS >=5e-4',
         perturbation_rms=[1e-4,2e-5,5e-6,1e-6],relative_tolerance=.05,
         additional_probe='Jacobian bilinear antisymmetry along fixed full-gradient and field directions',
         stalled_state_candidate_rms=[.004,.001,.00025,.0000625],
         initial_audit_retained=True,edit_trajectories_changed=False,script_sha256=sha(__file__),
         registered_before_transformer_execution=True))
    data=torch.load(OUT/'inputs.pt',map_location='cpu',weights_only=True)
    model=load_model(torch.device('cuda:0'),out/'weight_audit.json')
    e=Execution(model,data['inputs']);e.stage='fp32_gradient_refinement';e.partition_fp32()
    x=data['inputs']['source']['img'].to('cuda:0')
    obj=CleanObjective(e,x)
    initial=obj.field(x)
    probe=x+.08*initial/rms(initial)
    value,g,detached=obj.value_gradient(probe)
    u=g/rms(g)
    predicted=float((g.double()*u.double()).sum())
    rows=[]
    with torch.no_grad():
        repeated=[obj.value(probe) for _ in range(2)]
        for h in [1e-4,2e-5,5e-6,1e-6]:
            plus=obj.value(probe+h*u);minus=obj.value(probe-h*u)
            fd=(plus-minus)/(2*h)
            error=abs(fd-predicted)/max(abs(fd),abs(predicted),1e-12)
            row=dict(perturbation_rms=h,autograd=predicted,finite_difference=fd,
                     relative_error=error,passed=error<=.05)
            rows.append(row);print(row,flush=True)
            write(out/'partial.json',dict(rows=rows))
    field=obj.field(probe)
    v=field/max(rms(field),1e-30)
    jtu=torch.zeros_like(probe);jtv=torch.zeros_like(probe)
    for condition,sign in [('source',1.),('target',-1.)]:
        variable=probe.detach().clone().requires_grad_(True)
        prediction=e.velocity(obj.alpha*variable,obj.t,condition)
        loss_u=sign*obj.factor*(prediction.double()*u.double()).mean()
        loss_v=sign*obj.factor*(prediction.double()*v.double()).mean()
        gu,=torch.autograd.grad(loss_u,variable,retain_graph=True)
        gv,=torch.autograd.grad(loss_v,variable)
        jtu+=gu.detach();jtv+=gv.detach();e.vjps+=2
        del prediction,loss_u,loss_v,variable,gu,gv
    uv=float((jtu.double()*v.double()).sum())
    vu=float((jtv.double()*u.double()).sum())
    antisym=dict(uT_J_v_per_dimension=uv,vT_J_u_per_dimension=vu,
                 absolute_gap=abs(uv-vu),relative_gap=abs(uv-vu)/max(abs(uv),abs(vu),1e-30),
                 scope='local numerical bilinear probe; not a global curl bound')
    checkpoint_data=torch.load(OUT/'checkpoint.pt',map_location='cpu',weights_only=True)
    stalled=checkpoint_data['states']['clean_full_gradient'].to('cuda:0')
    stalled_value,stalled_g,_=obj.value_gradient(stalled)
    stall_rows=[]
    with torch.no_grad():
        for step in [.004,.001,.00025,.0000625]:
            candidate=projection(stalled+step*stalled_g/rms(stalled_g),x,.35)
            delta=candidate-stalled
            linear=float((stalled_g.double()*delta.double()).sum())
            measured=obj.value(candidate)
            gain=measured-stalled_value
            passed=linear>0 and gain>=max(.25*linear,1e-7) and rms(candidate-x)<=.3500001
            stall_rows.append(dict(candidate_rms=step,actual_gain=gain,directional_gain=linear,
                                   source_rms=rms(candidate-x),accepted=passed))
    result=dict(value=value,repeat_value_range=max(repeated)-min(repeated),rows=rows,
         full_gradient_vs_detached_cosine=cosine(g,detached),forward_calls=e.calls,
         local_jacobian_antisymmetry=antisym,
         stalled_state_smaller_steps=stall_rows,
         vjps=e.vjps,seconds=time.monotonic()-e.start,
         any_refined_scale_passed=any(r['passed'] for r in rows),
         conclusion_scope='local numerical differentiation at a single fixed probe; no semantic certificate')
    write(out/'result.json',result)
    print('GRADIENT_REFINEMENT_COMPLETE',result,flush=True)


if __name__=='__main__':
    main()
