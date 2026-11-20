"""Independent mathematical review of the 2026-11-09 design.

No image-model inference. Positive and negative examples audit claim scope;
they do not claim that an actual FLUX solver has the constructed pathology.
"""
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
from scipy.integrate import solve_ivp,quad_vec
from scipy.linalg import expm

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
sys.path.insert(0,str(PARENT))
from reference import integrate,local_lift,paired_replay,cycle_to_edit_bound,rms
from verify import AffineFlow


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checks=[]
    def record(name,passed,**metrics):
        if not passed:raise AssertionError(dict(name=name,**metrics))
        checks.append(dict(name=name,passed=bool(passed),**metrics))

    # Exact Gaussian RF fields. For affine velocity, the original FlowEdit
    # expectation is obtained exactly by setting its source query to the mean.
    T=.8;x=1.;sigma_s=1.;sigma_a=2.
    def velocity(q,t,sigma):
        alpha=1-t;variance=alpha*alpha*sigma*sigma+t*t
        return (t-alpha*sigma*sigma)/variance*q
    def fe_rhs(t,y):
        qs=(1-t)*x
        return [velocity(qs+y[0]-x,t,sigma_a)-velocity(qs,t,sigma_s)]
    fe=solve_ivp(fe_rhs,[T,0],[x],rtol=1e-12,atol=1e-13).y[0,-1]
    alpha=1-T;Vs=alpha**2*sigma_s**2+T*T;Va=alpha**2*sigma_a**2+T*T
    inverse_lift=math.sqrt(Vs)/sigma_s*x
    inverse_target=sigma_a/math.sqrt(Va)*inverse_lift
    gaussian=dict(time=T,source_instance=x,source_variance=sigma_s**2,target_variance=sigma_a**2,
                  original_flowedit_expectation_output=float(fe),partial_inverse_output=inverse_target,
                  absolute_difference=abs(inverse_target-fe),exact_source_cycle=0.,
                  FE_expectation_computed_by_high_accuracy_ODE=True)
    record('zero_cycle_limit_is_not_original_FlowEdit',abs(inverse_target-fe)>.3,**gaussian)

    # A perfectly valid inverse error estimate can be huge even at tiny cycle.
    a=1e-4;q=1.;source=0.
    gs=a*q;ga=q;y=source+ga-gs;rho=abs(gs-source)
    bound=cycle_to_edit_bound(rho,1-a,1/a)
    ill_conditioned=dict(source_map_gain=a,inverse_source_L=1/a,cycle=rho,
        corrected_output=y,reference_target=0.,error=abs(y),conditional_bound=bound)
    record('small_source_cycle_not_small_edit_error_without_conditioning',rho<.02 and abs(y)>.9,
           **ill_conditioned)
    record('P5_bound_remains_correct_with_true_constants',abs(abs(y)-bound)<1e-12)

    # Smooth affine-in-state example: the Heun map collapses, but the exact
    # continuous source flow is invertible. Interior time forcing makes the
    # actual L1 initializer inaccurate while the coarse cycle still passes.
    h=.1;J=np.array([[0.,-1.],[1.,0.]])
    A=(np.eye(2)+J)/h;c=np.array([2.,-1.]);d=np.array([.4,.2]);src=-h*c/2
    field=lambda z,t:A@z+c*(1-t/h)+d*np.sin(np.pi*t/h)
    lift=local_lift(src,h,field,1)
    raw_map=integrate(field,np.array([.7,-.2]),[h,0.],'heun')[-1]
    second_map=integrate(field,np.array([-.8,.6]),[h,0.],'heun')[-1]
    zero=lambda z,t:np.zeros_like(z)
    candidate=paired_replay(src,lift,field,zero,[h,0.],'heun',.02)
    forcing,quad_error=quad_vec(lambda u:expm(A*(h-u))@(c*(1-u/h)+d*np.sin(np.pi*u/h)),
                               0,h,epsabs=1e-13,epsrel=1e-13)
    exact_lift=expm(A*h)@src+forcing
    R=np.eye(2)-h*A+.5*h*h*A@A
    singular=dict(step=h,heun_step_matrix_norm=float(np.linalg.norm(R,2)),
        different_inputs_output_distance=float(np.linalg.norm(raw_map-second_map)),
        exact_ODE_step_determinant=float(np.linalg.det(expm(-h*A))),
        L1_source_cycle_rms=candidate.source_cycle_rms,
        candidate_status=candidate.status,
        error_to_exact_inverse_target=float(np.linalg.norm(candidate.candidate-exact_lift)),
        quadrature_error_estimate=float(quad_error),
        role='analytic counterexample, not a measured FLUX failure')
    record('Heun_can_be_singular_while_ODE_is_invertible',np.linalg.norm(R,2)<1e-12 and
           np.linalg.det(expm(-h*A))>.1,**singular)
    record('source_gate_can_pass_bad_continuous_inverse_due_to_solver',
           candidate.source_cycle_rms<1e-12 and np.linalg.norm(candidate.candidate-exact_lift)>.04)
    record('identity_control_cannot_detect_Heun_collapse',
           np.array_equal(paired_replay(src,lift,field,field,[h,0.],'heun',.02).candidate,src))

    # Sufficient global Heun inverse condition: c_H=hL+.5(hL)^2<1.
    rng=np.random.default_rng(20261110)
    ratios=[]
    B=np.array([[.4,-.15],[.05,.3]])
    stable=AffineFlow(B,[.1,-.03],[.04,.02])
    h=.1;L=np.linalg.norm(B,2);coefficient=h*L+.5*(h*L)**2
    for _ in range(20):
        u=rng.normal(size=2);v=rng.normal(size=2)
        gu=integrate(stable,u,[h,0.],'heun')[-1]
        gv=integrate(stable,v,[h,0.],'heun')[-1]
        ratio=np.linalg.norm(gu-gv)/np.linalg.norm(u-v)
        ratios.append(float(ratio))
    record('Heun_colipschitz_sufficient_condition',coefficient<1 and min(ratios)>=1-coefficient,
           hL=float(h*L),lower_bound=float(1-coefficient),minimum_realized_ratio=min(ratios))

    # Source residual correction can introduce a protected-factor change.
    T=.5;eps=.01;src=np.zeros(2)
    source_f=lambda q,t:np.array([0.,-2*eps*t/T**2])
    target_f=lambda q,t:np.array([-1/T,0.])
    lift=local_lift(src,T,source_f,1)
    result=paired_replay(src,lift,source_f,target_f,np.linspace(T,0,25),'heun',.02)
    raw=result.target_history[-1]
    correction=dict(source_cycle_rms=result.source_cycle_rms,raw_target=raw.tolist(),
        corrected_target=result.candidate.tolist(),protected_coordinate=1,
        raw_protected_change=abs(float(raw[1]-src[1])),
        corrected_protected_change=abs(float(result.candidate[1]-src[1])),
        role='analytic task with protected second coordinate, not a measured image result')
    record('residual_correction_can_worsen_protected_factor',result.source_cycle_rms<.02 and
           abs(raw[1])<1e-14 and abs(result.candidate[1])>.009,**correction)
    record('bounded_correction_identity_is_still_true',abs(rms(result.candidate-raw)-result.source_cycle_rms)<1e-14)

    # Positive theorem checks; large T is not covered by small-time order alone.
    f=AffineFlow([[4.]],[0.],[0.]);src=np.array([1.]);T=.9
    error1=abs(float(local_lift(src,T,f,1)[0]-f.exact(src,0,T)[0]))
    error2=abs(float(local_lift(src,T,f,2)[0]-f.exact(src,0,T)[0]))
    record('local_orders_do_not_guarantee_high_horizon_accuracy',error1>30 and error2>20,
           T=T,L1_error=error1,L2_error=error2)

    oldplan=json.loads((PARENT/'plan.json').read_text())
    protocol=json.loads((PARENT/'protocol.json').read_text())
    dev=set(protocol['development_pair_ids'])
    missing=[pair for pair in sorted(dev) if not any(j['pair_id']==pair and
        j['method']=='paired_source_guarded' and j['horizon']>=.6 for j in oldplan['jobs'])]
    record('v1_development_gate_has_no_main_method_jobs',len(missing)==4,missing_pair_ids=missing)
    output=dict(date='2026-11-10',scope='mathematical audit and synthetic counterexamples only; no model inference',
        passed=len(checks),total=len(checks),checks=checks,
        gaussian_method_difference=gaussian,ill_conditioned_gate=ill_conditioned,
        Heun_singularity=singular,residual_correction_counterexample=correction,
        script_sha256=sha(Path(__file__)),
        original_plan_sha256=sha(PARENT/'plan.json'),original_protocol_sha256=sha(PARENT/'protocol.json'),
        original_reference_sha256=sha(PARENT/'reference.py'))
    (HERE/'review_checks.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=len(checks),total=len(checks),gaussian=gaussian,
                         Heun_singularity=singular,missing_development_jobs=missing),ensure_ascii=False))


if __name__=='__main__':
    main()
