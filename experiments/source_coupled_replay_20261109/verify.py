"""Analytic/low-dimensional audit of the revised theory. No image models."""
import json
import math
from pathlib import Path
import numpy as np
from scipy.linalg import expm

from reference import (CountedField,integrate,local_lift,dense_heun,paired_replay,
    local_linear_error_bound,piecewise_defect_bound,cycle_to_edit_bound,
    invert_euler_replay,rms)

HERE=Path(__file__).resolve().parent


class AffineFlow:
    def __init__(self,A,b0,b1):
        self.A=np.asarray(A,float);self.b0=np.asarray(b0,float);self.b1=np.asarray(b1,float)
        self.n=len(b0)
        self.M=np.zeros((self.n+2,self.n+2))
        self.M[:self.n,:self.n]=self.A
        self.M[:self.n,self.n]=self.b0
        self.M[:self.n,self.n+1]=self.b1
        self.M[self.n+1,self.n]=1.

    def __call__(self,q,t):
        return self.A@q+self.b0+t*self.b1

    def exact(self,q,start,end):
        return (expm((end-start)*self.M)@np.r_[q,1.,start])[:self.n]


def main():
    checks=[]
    def record(name,passed,**metrics):
        if not passed:raise AssertionError(dict(name=name,**metrics))
        checks.append(dict(name=name,passed=bool(passed),**metrics))
    rng=np.random.default_rng(20261109)
    source=np.array([.8,-.3])
    fs=AffineFlow([[.45,-.2],[.1,.3]],[.12,-.08],[.1,.04])
    fa=AffineFlow([[.41,-.16],[.13,.27]],[-.25,.15],[.06,-.02])
    L=np.linalg.norm(fs.A,2);M=np.linalg.norm(fs.b1)
    v0=fs(source,0)
    for T in [.02,.04,.08,.16]:
        approx=local_lift(source,T,fs,1)
        error=np.linalg.norm(approx-fs.exact(source,0,T))
        bound=local_linear_error_bound(T,L,M,np.linalg.norm(v0))
        record('local_linear_conditional_bound_'+str(T),error<=bound+1e-14,error=error,bound=bound)
    e1=[];e2=[]
    for T in [.08,.04,.02]:
        exact=fs.exact(source,0,T)
        e1.append(np.linalg.norm(local_lift(source,T,fs,1)-exact))
        e2.append(np.linalg.norm(local_lift(source,T,fs,2)-exact))
    record('linear_lift_second_order',e1[0]/e1[1]>3.8 and e1[1]/e1[2]>3.8,errors=e1)
    record('heun_lift_third_order_local',e2[0]/e2[1]>7.7 and e2[1]/e2[2]>7.7,errors=e2)
    ft=AffineFlow(np.zeros((2,2)),[.2,-.1],[.3,.4])
    error=np.linalg.norm(local_lift(source,.2,ft,1)-ft.exact(source,0,.2))
    bound=local_linear_error_bound(.2,0,np.linalg.norm(ft.b1),np.linalg.norm(ft(source,0)))
    record('zero_state_L_limit',abs(error-bound)<1e-14,error=error,bound=bound)
    segment=dense_heun(source,0.,.2,fs)
    qend,_=segment.evaluate(.2)
    record('dense_heun_endpoint',np.array_equal(qend,local_lift(source,.2,fs,2)))
    offset=.073;eps=1e-6
    q,dq=segment.evaluate(offset)
    fd=(segment.evaluate(offset+eps)[0]-segment.evaluate(offset-eps)[0])/(2*eps)
    record('dense_heun_derivative',np.linalg.norm(fd-dq)<1e-9,error=float(np.linalg.norm(fd-dq)))
    # Valid Lipschitz bound on each quadratic segment's residual.
    residuals=[np.linalg.norm(segment.evaluate(s)[1]-fs(segment.evaluate(s)[0],s))
               for s in np.linspace(0,.2,101)]
    Mr=np.linalg.norm(segment.k2-segment.k1)/.2+L*max(np.linalg.norm(segment.k1),np.linalg.norm(segment.k2))+M
    sup_bound=max(residuals)+Mr*.002/2
    final_bound=piecewise_defect_bound([.2],[sup_bound],L)[-1]
    actual=np.linalg.norm(qend-fs.exact(source,0,.2))
    record('dense_defect_to_state_bound',actual<=final_bound,actual=actual,bound=float(final_bound))
    record('piecewise_bound_refinement_consistency',abs(piecewise_defect_bound([.1,.1],[.03,.03],.4)[-1]-
        piecewise_defect_bound([.2],[.03],.4)[-1])<1e-14)

    T=.6;times=np.linspace(T,0,25);qT=local_lift(source,T,fs,1)
    for method in ['euler','heun']:
        for index in range(5):
            arbitrary=rng.normal(size=2)
            result=paired_replay(source,arbitrary,fs,fs,times,method,cycle_rms_max=1e6)
            record(method+'_identity_arbitrary_lift_'+str(index),
                   np.array_equal(result.candidate,source) and np.all(result.clean_history==source),
                   source_cycle_rms=result.source_cycle_rms)
    cs,ca=CountedField(fs),CountedField(fa)
    result=paired_replay(source,qT,cs,ca,times,'euler',cycle_rms_max=1e6)
    record('euler_replay_call_accounting',cs.calls==24 and ca.calls==24,source_calls=cs.calls,target_calls=ca.calls)
    expected=source+(result.target_history-result.source_history)
    record('discrete_clean_state_identity',np.array_equal(expected,result.clean_history))
    raw=result.target_history[-1];reconstruction=result.source_history[-1]
    record('correction_equals_source_cycle',np.linalg.norm(result.candidate-raw+(reconstruction-source))<1e-14)
    record('candidate_status_does_not_claim_semantic_success',result.status=='candidate_requires_task_evaluation' and not result.semantic_success)
    cs,ca=CountedField(fs),CountedField(fa)
    rejected=paired_replay(source,qT+2,cs,ca,times,'heun',cycle_rms_max=1e-6)
    record('source_gate_precedes_target',rejected.status=='rejected_source_cycle' and ca.calls==0 and rejected.candidate is None,
           source_calls=cs.calls,target_calls=ca.calls)
    exhausted=paired_replay(source,qT,CountedField(fs,budget=2),CountedField(fa),times,'euler',1.)
    record('finite_budget_exit',exhausted.status=='budget_exhausted' and exhausted.candidate is None)

    # Exact continuous map certificate: reference lift is the exact source flow.
    exact_lift=fs.exact(source,0,T)
    Gs=expm(-T*fs.A);Ga=expm(-T*fa.A)
    inverse_L=1/np.linalg.svd(Gs,compute_uv=False).min()
    difference_L=np.linalg.norm(Ga-Gs,2)
    exact_target=fa.exact(exact_lift,T,0)
    worst_ratio=0.
    for i in range(20):
        proposal=exact_lift+.1*rng.normal(size=2)
        sr=fs.exact(proposal,T,0);tr=fa.exact(proposal,T,0)
        candidate=source+(tr-sr)
        actual=np.linalg.norm(candidate-exact_target)
        cert=cycle_to_edit_bound(np.linalg.norm(sr-source),difference_L,inverse_L)
        worst_ratio=max(worst_ratio,actual/max(cert,1e-30))
        assert actual<=cert+1e-14
    record('source_cycle_to_edit_difference_map_bound',worst_ratio<=1+1e-12,worst_ratio=worst_ratio)
    record('same_condition_difference_constant_zero',cycle_to_edit_bound(100.,0.,inverse_L)==0.)
    record('paired_map_bound_sharper_than_sum',difference_L<np.linalg.norm(Ga,2)+np.linalg.norm(Gs,2),
        difference_L=difference_L,sum_L=float(np.linalg.norm(Ga,2)+np.linalg.norm(Gs,2)))

    inverse,records=invert_euler_replay(source,fs,times,tolerance=1e-12,max_iterations=100)
    cycle=rms(integrate(fs,inverse,times,'euler')[-1]-source)
    record('discrete_inverse_reference',inverse is not None and cycle<1e-10,cycle_rms=cycle,
           inverse_field_calls=sum(r['iterations'] for r in records))
    near=paired_replay(source,inverse,fs,fa,times,'euler',cycle_rms_max=1e-9)
    record('zero_cycle_recovers_discrete_target_replay',np.linalg.norm(near.candidate-near.target_history[-1])<1e-10)
    # The numerical inverse is generally not identical to the continuous one.
    record('continuous_discrete_inverse_distinguished',np.linalg.norm(inverse-exact_lift)>1e-5,
           difference=float(np.linalg.norm(inverse-exact_lift)))
    divergent=AffineFlow(np.eye(2)*5,[0.,0.],[0.,0.])
    inv,log=invert_euler_replay(source,divergent,[.6,0.],max_iterations=5)
    record('failed_inverse_not_accepted',inv is None and not log[-1]['converged'])

    # Exact source reconstruction does not imply protected-factor preservation.
    x=np.array([.2,.4,-.3])
    source_zero=AffineFlow(np.zeros((3,3)),[0.,0.,0.],[0.,0.,0.])
    counter_T=.6
    A=np.zeros((3,3));A[1:,1:]=np.array([[0.,-math.pi],[math.pi,0.]])/counter_T
    target_rotate=AffineFlow(A,[-1./counter_T,0.,0.],[0.,0.,0.])
    sr=source_zero.exact(x,counter_T,0.);tr=target_rotate.exact(x,counter_T,0.)
    candidate=x+(tr-sr)
    record('zero_cycle_not_semantic_preservation',np.linalg.norm(sr-x)<1e-14 and np.linalg.norm(candidate[1:]-x[1:])>.5,
           source_cycle=float(np.linalg.norm(sr-x)),protected_factor_change=float(np.linalg.norm(candidate[1:]-x[1:])))

    # Approximate source replay emits its own state evolution; no independent
    # dense-source forcing is injected into the target recurrence.
    heun=paired_replay(source,qT,fs,fa,times,'heun',1e6)
    independent_target=integrate(fa,qT,times,'heun')
    record('target_replay_is_its_own_discrete_flow',np.array_equal(heun.target_history,independent_target))
    record('identity_survives_large_raw_source_error',any(row.get('source_cycle_rms',0.)>1. for row in checks))
    output=dict(date='2026-11-09',scope='analytic affine flows, discrete operators and counterexamples; no FLUX inference',
                passed=len(checks),total=len(checks),checks=checks)
    out=HERE/'outputs';out.mkdir(exist_ok=True)
    (out/'math_checks.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=len(checks),total=len(checks),output=str(out/'math_checks.json'))))


if __name__=='__main__':
    main()
