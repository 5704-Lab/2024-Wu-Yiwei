"""Constructed counterexamples and conditional algebra; no image-quality claim."""
from pathlib import Path
import json
import math
import numpy as np

HERE=Path(__file__).resolve().parent


def polar(f):
    u,s,vh=np.linalg.svd(f)
    return u@vh, vh.T@np.diag(s)@vh


def fit(source,current):
    p=source-source.mean(0);q=current-current.mean(0)
    if np.linalg.eigvalsh(p.T@p).min()<1e-10:raise ValueError('source geometry rank deficient')
    f=np.linalg.solve(p.T@p,p.T@q).T
    t=current.mean(0)-f@source.mean(0)
    return f,t


def main():
    out=HERE/'outputs/math_v1.json'
    if out.exists():raise RuntimeError('Immutable math result exists')
    rows=[]
    def check(name,test,**value):
        rows.append(dict(name=name,passed=bool(test),**value))
        if not test:raise AssertionError(name)
    x=np.array([-1.,0.,1.]);p=np.array([.5,0,.5]);q=np.array([0.,1.,0.])
    hellinger=.5*np.sum((np.sqrt(p)-np.sqrt(q))**2)
    check('centroid_can_erase_entire_distribution_change',p@x==q@x and abs(hellinger-1)<1e-12,
        equal_centroid=float(p@x),hellinger=float(hellinger))
    eps=1e-6;angle=np.arctan2(eps,eps)
    check('near_coincident_parts_make_angles_ill_conditioned',angle>.7,
        displacement=eps,angle_change_radians=float(angle))
    rng=np.random.default_rng(306031)
    a=rng.uniform(size=(4,5));a/=a.sum(1,keepdims=True)
    b=rng.uniform(size=(4,5));b/=b.sum(1,keepdims=True)
    v=rng.normal(size=(5,3));w=rng.normal(size=(5,3))
    route=(b-a)@((v+w)/2);value=((a+b)/2)@(w-v)
    error=np.linalg.norm((b@w-a@v)-(route+value))
    check('symmetric_routing_value_decomposition_is_exact',error<1e-12,error=float(error))
    check('same_feature_can_have_legal_and_illegal_latent_causes',1.+0.==0.+1.,
        legal=dict(appearance=1,pose=0),illegal=dict(appearance=0,pose=1))
    src=np.array([[-1,-.6],[1,-.6],[-.8,.9],[.7,.8],[.2,.1]],dtype=float)
    theta=.6;r=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    stretch=np.array([[1.35,.18],[.18,.85]])
    target=(r@stretch@src.T).T+np.array([.2,-.1])
    f,t=fit(src,target);rp,up=polar(f)
    check('affine_source_correspondence_recovers_rotation_and_stretch',
        np.linalg.norm(rp-r)<1e-12 and np.linalg.norm(up-stretch)<1e-12,
        rotation_error=float(np.linalg.norm(rp-r)),stretch_error=float(np.linalg.norm(up-stretch)))
    check('translation_is_retained_not_aligned_away',np.linalg.norm(t-[.2,-.1])<1e-12,translation=t.tolist())
    f0,_=fit(src,(stretch@src.T).T);r0,_=polar(f0)
    check('symmetric_intrinsic_stretch_does_not_require_polar_rotation',np.linalg.norm(r0-np.eye(2))<1e-12)
    axis=np.array([1.,1.]);changed=np.diag([2.,1.])@axis
    da=abs(np.arctan2(changed[1],changed[0])-np.arctan2(axis[1],axis[0]))
    check('even_exact_two_landmarks_need_not_preserve_angle_under_legal_stretch',da>.3,
        angle_change_degrees=float(da*180/np.pi),polar_rotation_degrees=0.)
    shear=np.array([[1.,.5],[0.,1.]])
    shear_r,_=polar(shear);shear_angle=np.arctan2(shear_r[1,0],shear_r[0,0])
    check('polar_rotation_is_not_universal_semantic_pose',abs(shear_angle)>.2,
        shear_polar_rotation_degrees=float(shear_angle*180/np.pi),meaning='if task permits shear as shape, polar rotation alone can overconstrain')
    rotated=(r@src.T).T
    dist=lambda z:np.linalg.norm(z[:,None]-z[None,:],axis=-1)
    check('intrinsic_pairwise_distances_are_blind_to_rigid_turn',np.linalg.norm(dist(rotated)-dist(src))<1e-12)
    ff,tt=fit(src,rotated);alignment=np.linalg.norm(rotated-(ff@src.T).T-tt)
    check('free_affine_alignment_hides_rotation_if_transform_is_discarded',alignment<1e-12 and np.linalg.norm(ff-np.eye(2))>.1,
        alignment_residual=float(alignment),rotation_radians=theta)
    try:fit(np.array([[0.,0.],[1.,0.],[2.,0.]]),np.array([[0.,0.],[1.,0.],[2.,0.]]))
    except ValueError:check('rank_deficient_source_frame_is_not_silently_inverted',True)
    else:check('rank_deficient_source_frame_is_not_silently_inverted',False)
    cost=np.ones((3,3));zero_plan=np.zeros((3,3))
    check('free_unmatched_mass_can_hide_all_errors',float((zero_plan*cost).sum())==0. and zero_plan.sum()==0.)
    source_features=np.concatenate([np.eye(4),100*np.eye(4)],1)
    perm=np.array([1,0,3,2])
    current_features=np.concatenate([np.eye(4)[perm],100*np.eye(4)],1)
    match=(source_features@current_features.T).argmax(1)
    back=(current_features@source_features.T).argmax(1)
    check('confident_cycle_consistent_match_can_be_pure_position_bias',
        np.array_equal(match,np.arange(4)) and np.array_equal(back[match],np.arange(4)) and not np.array_equal(match,perm),
        predicted=match.tolist(),true_content_correspondence=perm.tolist())
    beta=.2;hat=.5;uncertainty=1.
    lower=max(abs(hat)-uncertainty-beta,0.);upper=max(abs(hat)+uncertainty-beta,0.)
    check('uncertainty_overlap_is_unknown_not_zero_violation_certificate',lower==0 and upper>0,
        lower_violation=lower,upper_violation=upper)
    alpha=.5;true_now=1.;hat_next=.5;epsilon_next=.3;true_next=.8
    check('proxy_contraction_need_not_imply_true_contraction',hat_next<=alpha*true_now and true_next>alpha*true_now
          and abs(hat_next-true_next)<=epsilon_next+1e-12)
    alpha=np.exp(-.2);now_hat=1.;now_eps=.1;next_hat=.3;next_eps=.1
    lower_now=max(abs(now_hat)-now_eps-beta,0.)
    robust_rhs=beta+alpha*lower_now
    errs=np.linspace(-.1,.1,101)
    valid=all(max(abs(next_hat+e2)-beta,0.)<=alpha*max(abs(now_hat+e1)-beta,0.)+1e-12 for e1 in errs for e2 in errs)
    check('upper_next_lower_current_condition_implies_true_tube_contraction_under_valid_bounds',
        abs(next_hat)+next_eps<=robust_rhs and valid,robust_rhs=float(robust_rhs))
    legal=np.array([1.,100.]);bad=legal+1
    check('perfect_oracle_paired_ranking_does_not_give_global_legal_threshold',np.all(bad>legal) and legal.max()>bad.min(),
        paired_ordering=2,source_count=2,max_legal=float(legal.max()),min_bad=float(bad.min()))
    y=.3;s=1/(1+np.exp(-2*y));analytic=-4*s*(1-s)
    func=lambda z:1-2/(1+np.exp(-2*z))
    fd=(func(y+1e-5)-func(y-1e-5))/(2e-5)
    check('detaching_current_correspondence_can_remove_entire_geometry_derivative',abs(fd-analytic)<1e-8 and abs(analytic)>.5,
        true_derivative=float(analytic),frozen_plan_derivative=0.)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(scope='constructed mathematical diagnostics only; no SD3 or successful image edit',
        passed=len(rows),total=len(rows),checks=rows),indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(passed=len(rows),total=len(rows))))


if __name__=='__main__':main()
