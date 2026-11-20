"""Post-hoc CPU analysis of frozen 2026-11-06 outputs; no model inference.

Separates target/source score changes, numerical gain ratios, source query
geometry, paired error cancellation and exact analytic counterexamples.
Never edits the original experiment or its protocol/manifest.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
RUN=HERE/'outputs/run_v1'
OUT=HERE/'analysis_v1'


def rms(value):
    return float(value.double().square().mean().sqrt())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    inputs=torch.load(RUN/'inputs.pt',weights_only=True,map_location='cpu')
    states=torch.load(RUN/'results.pt',weights_only=True,map_location='cpu')
    paths=torch.load(RUN/'source_paths.pt',weights_only=True,map_location='cpu')
    image_metrics=json.loads((RUN/'image_metrics.json').read_text())['metrics']
    x=inputs['inputs']['source']['img'].double()
    ref=paths['reference']
    baseline=image_metrics['source']['siglip_logits']
    decomposition={}
    for name,metrics in image_metrics.items():
        ds=metrics['siglip_logits'][0]-baseline[0]
        dt=metrics['siglip_logits'][1]-baseline[1]
        dm=dt-ds
        decomposition[name]=dict(delta_source_logit=ds,delta_target_logit=dt,
            delta_margin=dm,source_decrease_component=-ds,
            fraction_from_source_decrease=None if abs(dm)<1e-15 else -ds/dm,
            scope='algebraic scorer decomposition, not likelihood or causal attribution')
    endpoint=dict(source_rms=rms(x),velocity_rms=rms(ref['v'][0]),
                  velocity_plus_source_rms=rms(ref['v'][0].double()+x),
                  relative_to_source=rms(ref['v'][0].double()+x)/rms(x))
    query_rows=[]
    for i in range(1,33):
        t=.3*i/32
        delta=(1-t)*x-ref['q'][i].double()
        e=rms(delta)
        query_rows.append(dict(time=t,deterministic_query_error_rms=e,error_over_t=e/t,
            error_over_t_squared=e/t**2,
            random_query_expected_mse_per_coordinate=e*e+t*t,
            sqrt_random_query_expected_mse_per_coordinate=math.sqrt(e*e+t*t),
            random_value_is_analytic_expectation_not_sampled=True))
    forcing={}
    for name,key in [('reference','path_reference'),('euler2','path_euler2'),('euler8','path_euler8')]:
        trace=paths['edit_traces'][name]
        history=states['histories'][key]
        rows=[]
        for k in range(8):
            qs=trace[k]['q'].double()
            qn=trace[k+1]['q'].double() if k<7 else x
            y=history[k].double();yn=history[k+1].double()
            dt=-.3/8
            rho=qn-qs-dt*trace[k]['vs'].double()
            target_residual=(qn+yn-x)-(qs+y-x)-dt*trace[k]['vt'].double()
            rows.append(dict(step=k,source_discrete_defect_rms=rms(rho),
                             target_discrete_residual_rms=rms(target_residual),
                             identity_error_rms=rms(target_residual-rho)))
        forcing[name]=rows
    cancellation={}
    for name in ['euler2','euler8']:
        rows=[]
        for k,(a,b) in enumerate(zip(paths['edit_traces'][name],paths['edit_traces']['reference'])):
            es=a['vs'].double()-b['vs'].double()
            ea=a['vt'].double()-b['vt'].double()
            denominator=max(rms(es)+rms(ea),1e-30)
            cos=float((es*ea).sum()/((es.square().sum()*ea.square().sum()).sqrt()).clamp_min(1e-30))
            rows.append(dict(step=k,source_velocity_error_rms=rms(es),target_velocity_error_rms=rms(ea),
                             difference_error_rms=rms(es-ea),
                             difference_over_sum=rms(es-ea)/denominator,error_cosine=cos))
        cancellation[name]=rows
    gain_ratios={}
    for name in ['clean_full_gradient','clean_field_proposal']:
        trace=json.loads((RUN/(name+'_trace.json')).read_text())
        gain_ratios[name]=[dict(step=row['step'],factor=row['factor'],
            actual_over_reported_linear_gain=row['gain']/row['directional_gain'],
            actual_gain=row['gain'],accepted=row['accepted'],
            denominator_is_true_gradient_only=name=='clean_full_gradient') for row in trace['trials']]
    refinement=json.loads((RUN/'gradient_refinement_v1/result.json').read_text())
    gain_ratios['stalled_full_gradient_refinement']=[dict(row,
        actual_over_reported_linear_gain=row['actual_gain']/row['directional_gain'])
        for row in refinement['stalled_state_smaller_steps']]

    # Exact normalized Gaussian densities: ratio ascent can lower target density.
    y=np.linspace(1.,1.35,101)
    log_ps=-.5*y*y-.5*np.log(2*np.pi)
    log_pa=-y*y/8-.5*np.log(8*np.pi)
    F=log_pa-log_ps
    direction=.75*y
    target_derivative=(-y/4)*direction
    ratio_derivative=direction**2
    counterexample=dict(source_variance=1.,target_variance=4.,source_instance=1.,
        interval=[1.,1.35],source_ball_radius=.35,
        final_ratio_gain=float(F[-1]-F[0]),
        final_target_log_density_gain=float(log_pa[-1]-log_pa[0]),
        all_ratio_derivatives_positive=bool(np.all(ratio_derivative>0)),
        all_target_derivatives_negative=bool(np.all(target_derivative<0)),
        role='exact mathematical counterexample; not a measured FLUX density')

    # Uniformly tiny field error need not yield a small radial-potential gradient
    # error: b=(1+eps*sin(z2/eps^2),0), x=0, y=(1,0).
    ripple=[]
    for eps in [.1,.03,.01]:
        variable=torch.tensor([1.,0.],dtype=torch.float64,requires_grad=True)
        nodes=(torch.arange(8,dtype=torch.float64)+.5)/8
        z=nodes[:,None]*variable[None,:]
        b=torch.stack((1+eps*torch.sin(z[:,1]/eps**2),torch.zeros(8,dtype=torch.float64)),dim=1)
        potential=(b*variable[None,:]).sum(1).mean()
        g,=torch.autograd.grad(potential,variable)
        predicted=torch.tensor([1.,1/(2*eps)],dtype=torch.float64)
        ripple.append(dict(epsilon=eps,uniform_field_error_bound=eps,
            value_at_probe=float(potential.detach()),actual_gradient=g.tolist(),
            exact_gradient=predicted.tolist(),gradient_error_norm=float((g-predicted).norm()),
            gradient_cosine_with_ideal=float(g[0]/g.norm())))

    # For b(y)=(I+2J)y, radial S=.5||y||^2. At y=(.1,0),
    # the true ascent derivative along b is positive, but using ||b||^2
    # as the Armijo predictor rejects every sufficiently small step.
    cheap_gate=[]
    state=np.array([.1,0.]);field=np.array([.1,.2])
    for h in [.08,.04,.02,.01]:
        candidate=state+h*field
        gain=float((candidate@candidate-state@state)/2)
        prediction=float(h*(field@field))
        cheap_gate.append(dict(step=h,actual_gain=gain,proxy_prediction=prediction,
            gain_ratio=gain/prediction,accepted=gain>=.25*prediction,
            source_ball_feasible=bool(np.linalg.norm(candidate)<=.35)))

    checks=dict(logit_decomposition_consistent=all(abs(row['delta_margin']-(
        image_metrics[name]['siglip_full_prompt_margin']-image_metrics['source']['siglip_full_prompt_margin']))<1e-10
        for name,row in decomposition.items()),
        discrete_forcing_identity=max(row['identity_error_rms'] for rows in forcing.values() for row in rows)<1e-7,
        triangle_cancellation_ratios_valid=all(0<=row['difference_over_sum']<=1+1e-12
            for rows in cancellation.values() for row in rows),
        gaussian_ratio_target_counterexample=counterexample['all_ratio_derivatives_positive'] and
            counterexample['all_target_derivatives_negative'],
        ripple_derivative_counterexample=max(row['gradient_error_norm'] for row in ripple)<1e-10,
        cheap_armijo_counterexample=all(row['actual_gain']>0 and not row['accepted'] and
            row['source_ball_feasible'] for row in cheap_gate))
    assert all(checks.values()),checks
    result=dict(scope='post-hoc analysis of existing outputs; no new image/model inference',
        scorer_decomposition=decomposition,source_endpoint=endpoint,query_geometry=query_rows,
        source_target_discrete_forcing=forcing,paired_velocity_error_cancellation=cancellation,
        finite_step_gain_ratios=gain_ratios,ideal_gaussian_counterexample=counterexample,
        small_field_error_large_gradient_counterexample=ripple,consistency_checks=checks,
        cheap_field_armijo_counterexample=cheap_gate,
        script_sha256=sha(Path(__file__)),input_sha256={name:sha(RUN/name) for name in
            ['inputs.pt','results.pt','source_paths.pt','image_metrics.json','clean_full_gradient_trace.json',
             'clean_field_proposal_trace.json','gradient_refinement_v1/result.json']})
    (OUT/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

    fig,axes=plt.subplots(1,3,figsize=(13.5,3.9))
    names=['clean_full_gradient','clean_field_proposal','flowedit_equal_g','path_reference']
    labels=['Full gradient','Field proposal','FlowEdit equal g','Source ODE ref.']
    target=np.array([decomposition[name]['delta_target_logit'] for name in names])
    source=np.array([decomposition[name]['source_decrease_component'] for name in names])
    idx=np.arange(len(names))
    axes[0].bar(idx,target,color='#4C78A8',label='Target logit change')
    axes[0].bar(idx,source,bottom=np.maximum(target,0),color='#F2A541',label='Minus source logit change')
    axes[0].scatter(idx,target+source,marker='D',s=25,color='black',label='Net margin change',zorder=3)
    axes[0].axhline(0,color='black',linewidth=.6)
    axes[0].set_xticks(idx,labels,rotation=18,ha='right',fontsize=8)
    axes[0].set_title('Offline score decomposition');axes[0].set_ylabel('SigLIP logit change')
    axes[0].legend(fontsize=7)
    times=np.array([r['time'] for r in query_rows[:8]])
    ratios=np.array([r['error_over_t'] for r in query_rows[:8]])
    axes[1].plot(times,ratios,'o-',label='Measured path error / t')
    axes[1].axhline(endpoint['velocity_plus_source_rms'],color='#D1495B',linestyle='--',
        label='RMS(v(source,0) + source)')
    axes[1].set_title('Actual clean-end mismatch');axes[1].set_xlabel('RF time t')
    axes[1].set_ylabel('Latent RMS / t');axes[1].legend(fontsize=7)
    axes[2].plot(y-1,F-F[0],label='Log density ratio gain')
    axes[2].plot(y-1,log_pa-log_pa[0],label='Target log density gain')
    axes[2].axhline(0,color='black',linewidth=.6)
    axes[2].set_title('Exact ideal-Gaussian counterexample')
    axes[2].set_xlabel('Displacement from source instance');axes[2].legend(fontsize=7)
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'research_diagnosis.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(checks=checks,source_endpoint=endpoint,
         counterexample=counterexample,output=str(OUT/'analysis.json')),ensure_ascii=False))


if __name__=='__main__':
    main()
