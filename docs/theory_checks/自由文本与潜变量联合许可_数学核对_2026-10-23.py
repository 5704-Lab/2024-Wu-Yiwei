"""Conditional algebra/counterexamples, not free-language or image-model validation."""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

rows=[]
def check(name,passed,**metrics):
    rows.append(dict(name=name,passed=bool(passed),metrics=metrics))
    assert passed,(name,metrics)

# State=(left-object attribute,right-object attribute,unrequested pose).
proposal=np.array([.6,.6,.8])
global_score=np.array([1.,1.,2.])
left=np.array([1.,0.,0.]);right=np.array([0.,1.,0.])
def solve(task,keep,beta=.4):
    cons=[dict(type='ineq',fun=lambda d:task@d-beta,jac=lambda d:task)]
    for k in keep:
        cons += [dict(type='ineq',fun=lambda d,k=k:.05-k@d,jac=lambda d,k=k:-k),
                 dict(type='ineq',fun=lambda d,k=k:.05+k@d,jac=lambda d,k=k:k)]
    opt=minimize(lambda d:.5*np.sum((d-proposal)**2),np.zeros(3),jac=lambda d:d-proposal,
                 constraints=cons,method='SLSQP',options=dict(ftol=1e-12,maxiter=200))
    assert opt.success,opt.message
    return opt.x

dl=solve(left,[right,np.array([0.,0.,1.])])
dr=solve(right,[left,np.array([0.,0.,1.])])
check('left_bound_task_progress',left@dl>=.4-1e-10)
check('preserve_other_entity',abs(right@dl)<=.05+1e-10)
check('preserve_unrequested_pose',abs(dl[2])<=.05+1e-10)
check('same_state_different_text_different_permission',np.linalg.norm(dl-dr)>.5,
      left_solution=dl.tolist(),right_solution=dr.tolist())
check('global_score_can_prefer_forbidden_pose',global_score@np.array([0.,0.,.5])>global_score@np.array([.4,0.,0.]))

# Same intended fact, distinct decoder tangent: text alone cannot decide reachability.
ja=np.array([[1.,0.],[0.,1.],[0.,1.]])
jb=np.array([[0.,0.],[1.,0.],[0.,1.]])
check('same_text_state_dependent_task_jacobian',np.linalg.norm(ja.T@left)>0 and np.linalg.norm(jb.T@left)==0)
check('unreachable_fact_must_not_license_pose',np.linalg.norm(jb.T@left)==0 and np.linalg.norm(jb.T@global_score)>0)

# Mechanism selection should be joint: individual first derivatives can vanish.
def synergy(a):return a[0]*a[1]
check('individual_zero_response_not_unnecessary',synergy([1,1])==1 and synergy([1,0])==synergy([0,1])==0)
check('joint_necessary_action_energy',abs(.5*np.dot([1.,1.],[1.,1.])-1.)<1e-12)

# A text-independent nuisance probe can remove known local score contamination.
g=np.array([1.,2.]);geom=np.array([[0.],[1.]])
q=np.eye(2)-geom@np.linalg.pinv(geom)
check('known_nuisance_removed',np.linalg.norm(q@g-np.array([1.,0.]))<1e-12)
check('pure_nuisance_proxy_exposes_no_task_evidence',np.linalg.norm(q@np.array([0.,2.]))==0)

# Local response margin: gradient evidence has uncertainty, rather than an oracle label.
c=np.array([.4,-.1]);err=.05;a=np.array([.7,.2])
worst=c-err*a/np.linalg.norm(a)
check('robust_response_lower_bound',abs(worst@a-(c@a-err*np.linalg.norm(a)))<1e-12)

# Jointly learning free permission from the same residual makes it open itself.
residual=3.;epsilon=.01
cost=lambda permission:(1-permission)*residual**2+epsilon*permission
check('unconstrained_permission_gate_collapses',cost(1)<cost(0))

# Per-location constraints prevent sparse unwanted change being averaged away.
errors=np.r_[np.zeros(999),1.]
check('global_average_hides_local_error',errors.mean()<.01 and errors.max()==1.)
qtail=.01;k=int(qtail*len(errors))
check('tail_aggregation_keeps_sparse_error',np.sort(errors)[-k:].mean()>errors.mean()*50)

# Reanchoring to accepted states permits unlimited cumulative movement.
positions=np.arange(11)*.04
check('reanchored_small_steps_do_not_preserve_source',np.diff(positions).max()<.05 and positions[-1]>.3)

# Model-relative necessity: permitting colour meets task with zero motion; a bad
# task proxy asks only for motion and must be diagnosed, not treated as semantics.
check('minimum_necessary_phase_is_not_zero_phase',np.linalg.norm(np.array([1.,1.])*.4)>0)
check('binding_is_not_identified_by_bag_of_words',sorted('red cup and blue bowl'.split())==sorted('blue cup and red bowl'.split()))

path=Path(__file__)
result=dict(passed=sum(r['passed'] for r in rows),total=len(rows),checks=rows,
            scope='known toy fact operators, no automatic parsing/binding or pretrained image verification',
            script_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
path.with_suffix('.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(dict(passed=result['passed'],total=result['total']))
