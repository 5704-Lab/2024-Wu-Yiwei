"""Signal identities and conditional optimization checks, without image models.

Fact/keep operators below are declared toy operators. Passing does not establish
natural-language binding, calibrated uncertainty, or image editing quality.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize

rng=np.random.default_rng(20261024)
rows=[]
def check(name,passed,**metrics):
    rows.append(dict(name=name,passed=bool(passed),metrics=metrics))
    assert passed,(name,metrics)

# Actual FlowEdit descends noise time; use positive edit-progress increments.
vs=rng.normal(size=7);vt=rng.normal(size=7);h=.07;y=rng.normal(size=7)
check('flowedit_positive_progress_sign',np.allclose(y-h*(vt-vs),y+h*(vs-vt)))

# Decoder chain rule and the two components of a complex logarithmic derivative.
n,m=9,17
decoder=rng.normal(size=(n,5));bank=rng.normal(size=(m,n))+1j*rng.normal(size=(m,n))
y=rng.normal(size=5);v=rng.normal(size=5)
image=lambda p:np.tanh(decoder@p)
z=bank@image(y);dz=bank@((1-image(y)**2)*(decoder@v))
step=1e-6
zp=bank@image(y+step*v);zm=bank@image(y-step*v)
phase=np.imag(np.conj(z)*dz)/np.abs(z)**2
amp=np.real(np.conj(z)*dz)/np.abs(z)**2
phase_fd=np.angle(zp*np.conj(zm))/(2*step)
amp_fd=(np.log(np.abs(zp))-np.log(np.abs(zm)))/(2*step)
check('latent_decoder_filter_phase_chain_rule',np.max(np.abs(phase-phase_fd))<2e-7,max_error=float(np.max(np.abs(phase-phase_fd))))
check('log_amplitude_chain_rule',np.max(np.abs(amp-amp_fd))<2e-7,max_error=float(np.max(np.abs(amp-amp_fd))))
check('joint_amplitude_phase_reconstructs_complex_velocity',np.linalg.norm(z*(amp+1j*phase)-dz)<1e-11)
radial=.3*z
check('positive_global_gain_has_zero_phase_rate',np.max(np.abs(np.imag(np.conj(z)*radial)/np.abs(z)**2))<1e-14)
check('positive_global_gain_changes_amplitude',np.allclose(np.real(np.conj(z)*radial)/np.abs(z)**2,.3))

# A source-normalized complex observation is a genuine state observation.
source_scale=np.sqrt(np.abs(z)**2+.2**2)
complex_fd=(zp-zm)/(2*step)/source_scale
check('fixed_source_complex_observation_jvp',np.max(np.abs(complex_fd-dz/source_scale))<2e-7)
eps=.4
smooth=lambda w:w/np.sqrt(np.abs(w)**2+eps**2)
s=np.sqrt(np.abs(z)**2+eps**2)
ds=dz/s-z*np.real(np.conj(z)*dz)/s**3
check('epsilon_embedding_requires_radial_derivative',np.max(np.abs(ds-(smooth(zp)-smooth(zm))/(2*step)))<2e-7)
check('epsilon_embedding_is_not_gain_invariant',abs(smooth(np.array(2+0j))-smooth(np.array(1+0j)))>.01)

# Attenuated phase rates are generally not differentials of a scalar state.
# Closed polar rectangle: (r1,0)->(r1,a)->(r2,a)->(r2,0)->(r1,0).
eps=1.;r1=1.;r2=2.;angle=.7
weight=lambda r:r*r/(r*r+eps*eps)
loop_integral=(weight(r1)-weight(r2))*angle
check('regularized_phase_rate_has_nonzero_closed_loop',abs(loop_integral+.21)<1e-14,integral=loop_integral)
path=np.r_[r1*np.exp(1j*np.linspace(0,angle,101)),
           np.linspace(r1,r2,101)[1:]*np.exp(1j*angle),
           r2*np.exp(1j*np.linspace(angle,0,101))[1:],np.linspace(r2,r1,101)[1:]+0j]
true_loop=np.angle(path[1:]*np.conj(path[:-1])).sum()
check('true_phase_change_on_same_loop_is_zero',abs(true_loop)<1e-12,integral=float(true_loop))
dw=2*r1*eps**2/(r1*r1+eps**2)**2
check('regularized_phase_one_form_is_not_closed',dw>.1,exterior_derivative_coefficient=dw)
# Integrating the derivative of the actual smooth complex embedding closes.
def integrate_complex(fun,a,b):
    return quad(lambda t:float(np.real(fun(t))),a,b,epsabs=1e-12)[0]+1j*quad(lambda t:float(np.imag(fun(t))),a,b,epsabs=1e-12)[0]
u_theta=lambda r,t:1j*r*np.exp(1j*t)/np.sqrt(r*r+eps*eps)
u_radius=lambda r,t:eps*eps*np.exp(1j*t)/(r*r+eps*eps)**1.5
closed=(integrate_complex(lambda t:u_theta(r1,t),0,angle)
       +integrate_complex(lambda r:u_radius(r,angle),r1,r2)
       +integrate_complex(lambda t:u_theta(r2,t),angle,0)
       +integrate_complex(lambda r:u_radius(r,0),r2,r1))
check('actual_smooth_observation_differential_closes',abs(closed)<1e-12,error=float(abs(closed)))

# Spatial modulation does not commute with filtering, unlike a global gain.
length=128;p=np.arange(length);signal=1.+.4*np.cos(2*np.pi*p/length*5+.2)+.3*np.sin(2*np.pi*p/length*9)
kernel=np.exp(-.5*((p-length/2)/5)**2)*np.exp(1j*.5*(p-length/2))
kernel-=kernel.mean()
filter_signal=lambda s:np.fft.ifft(np.fft.fft(s)*np.fft.fft(kernel))
gain=.2*np.cos(2*np.pi*p/length*2)
zs=filter_signal(signal);dza=filter_signal(gain*signal)
commutator=dza-gain*zs
check('spatial_appearance_has_nonzero_filter_commutator',np.linalg.norm(commutator)>.01)
check('spatial_appearance_can_change_phase',np.linalg.norm(np.imag(np.conj(zs)*dza)/(np.abs(zs)**2+1e-12))>.01)
check('commutator_exactly_explains_spatial_phase',np.max(np.abs(np.imag(np.conj(zs)*dza)-np.imag(np.conj(zs)*commutator)))<1e-10)

# Shared amplitude/phase parameters can remove an actual ambiguity; independently
# choosing the two parameters recreates it. Added observations need not help.
v_phase=np.array([[1.]]);motion_phase=np.array([[1.]])
q_phase=np.eye(1)-v_phase@np.linalg.pinv(v_phase)
v_joint=np.array([[1.],[1.]]) # rows=(log-amplitude,phase)
motion_joint=np.array([[0.],[1.]])
q_joint=np.eye(2)-v_joint@np.linalg.pinv(v_joint)
check('phase_only_mechanism_geometry_ambiguity',np.linalg.norm(q_phase@motion_phase)<1e-12)
information=float((motion_joint.T@q_joint@motion_joint).item())
check('shared_amplitude_phase_increases_conditional_information',abs(information-.5)<1e-12,conditional_information=information)
independent=np.eye(2)
check('independent_amplitude_phase_coefficients_erase_the_gain',np.linalg.norm((np.eye(2)-independent@np.linalg.pinv(independent))@motion_joint)<1e-12)
check('joint_observation_does_not_guarantee_identification',np.linalg.norm(q_joint@v_joint)<1e-12)

# The same mechanism must also explain the declared task factor, not just phase.
A=np.array([[1.,1.]]);V=np.array([[1.]]);C=np.array([[1.,0.]]);Cref=np.array([[1.]])
wrong=np.array([0.,1.]);a=np.array([1.])
check('phase_fit_can_hide_wrong_fact',np.linalg.norm(A@wrong-V@a)==0 and np.linalg.norm(C@wrong-Cref@a)==1)

# Explicit colour/pose/background example for the three shared relation blocks.
P=np.array([[1.,1.,0.]]);R=np.array([[1.,0.,1.]]);F=np.array([[1.,0.,0.]])
shared=np.vstack([P,R,F]);a=.4
intended=np.linalg.solve(shared,np.ones(3)*a)
check('three_shared_relations_select_colour_without_pose_or_background',np.linalg.norm(intended-np.array([a,0.,0.]))<1e-12)
background_drift=intended+np.array([0.,0.,.3])
check('removing_amplitude_relation_leaves_background_free',np.allclose(np.vstack([P,F])@background_drift,a) and abs((R@background_drift).item()-a)>.2)
pose_drift=intended+np.array([0.,.3,0.])
check('removing_phase_relation_leaves_pose_free',np.allclose(np.vstack([R,F])@pose_drift,a) and abs((P@pose_drift).item()-a)>.2)

# Biphase preserves translation equivalence, not the absolute source position.
x=rng.normal(size=63);spectrum=np.fft.fft(x);shifted=np.fft.fft(np.roll(x,3))
triads=[(1,4),(2,7),(3,8)]
biphase=lambda z:np.array([z[i]*z[j]*np.conj(z[(i+j)%len(z)]) for i,j in triads])
unit=lambda z:z/np.abs(z)
check('bispectral_phase_is_translation_invariant',np.linalg.norm(unit(biphase(spectrum))-unit(biphase(shifted)))<1e-12)
check('absolute_anchor_detects_the_same_translation',abs(unit(spectrum[4])-unit(shifted[4]))>.5)
check('bispectral_phase_is_positive_gain_invariant',np.linalg.norm(unit(biphase(spectrum))-unit(biphase(2*spectrum)))<1e-12)
changed=spectrum.copy();changed[5]*=np.exp(.4j)
check('biphase_detects_one_nontranslation_phase_edit',np.linalg.norm(unit(biphase(spectrum))-unit(biphase(changed)))>.3)

# A jointly optimized state/mechanism block with multiple known facts/keeps.
A=rng.normal(size=(7,3));V=rng.normal(size=(7,2));e=rng.normal(size=7)*.1
G=np.diag([1.,2.,1.5]);Lambda=np.eye(2)*.3;lam=.7;proposal=np.array([.8,.6,.9])
H=np.block([[G+lam*A.T@A,-lam*A.T@V],[-lam*V.T@A,lam*(V.T@V+Lambda)]])
f=np.r_[G@proposal-lam*A.T@e,lam*V.T@e]
E=np.column_stack([np.eye(3),np.zeros((3,2))]);target=np.array([.4,.2,0.])
z0=np.linalg.solve(H,f);HE=np.linalg.solve(H,E.T)
mu=np.linalg.solve(E@HE,target-E@z0);sol=z0+HE@mu
check('coupled_multi_factor_constraints',np.linalg.norm(E@sol-target)<1e-12)
check('coupled_KKT_stationarity',np.linalg.norm(H@sol-f-E.T@mu)<1e-12)
opt=minimize(lambda z:.5*z@H@z-f@z,np.zeros(5),jac=lambda z:H@z-f,
    constraints=[dict(type='eq',fun=lambda z:E@z-target,jac=lambda z:E)],
    method='SLSQP',options=dict(ftol=1e-13,maxiter=300))
check('block_solution_agrees_with_independent_solver',opt.success and np.linalg.norm(sol-opt.x)<2e-6,error=float(np.linalg.norm(sol-opt.x)))
Schur=G+lam*A.T@(np.eye(7)-V@np.linalg.solve(V.T@V+Lambda,V.T))@A
check('profiled_block_Hessian_is_correct',np.linalg.norm(Schur-(H[:3,:3]-H[:3,3:]@np.linalg.solve(H[3:,3:],H[3:,:3])))<1e-11)

# No epsilon denominator can make an impossible fact reachable.
C=np.array([[1.,0.],[0.,0.]]);request=np.array([.2,.1])
check('unreachable_requested_fact_is_exposed',np.linalg.norm((np.eye(2)-C@np.linalg.pinv(C))@request)>.09)
task_and_phase=np.array([[1.,1.],[1.,1.]])
request_with_zero_phase=np.array([.2,0.])
incompatibility=np.linalg.norm((np.eye(2)-task_and_phase@np.linalg.pinv(task_and_phase))@request_with_zero_phase)
task_solution=np.array([.1,.1])
check('necessary_nonzero_phase_must_be_permitted',incompatibility>.1 and abs((task_and_phase@task_solution)[1]-.2)<1e-12,
      zero_phase_infeasibility=float(incompatibility))
e_source=.3;restore=-lam*e_source/(1+lam)
check('zero_raw_velocity_can_need_source_restoration',restore<0 and abs(e_source+restore)<abs(e_source))

# Protecting an observation with a squared energy derivative misses the first
# step at the source; a finite residual tube has no such first-order blind spot.
candidate=.2;budget=.01
check('source_squared_gradient_barrier_can_be_blind',2*0*candidate==0 and candidate*candidate>budget)
check('finite_residual_tube_detects_first_step',abs(candidate)>np.sqrt(budget))
point=.4;increments=np.linspace(-.4,.4,51)
error=np.abs(np.sin(point+increments)-np.sin(point)-np.cos(point)*increments)
check('finite_step_Taylor_remainder_bound',np.all(error<=.5*increments**2+1e-14))

# Different source/reference linearization points must not be conflated.
yy=2.;theta=.5;delta=1e-5;da=-2e-5
res=lambda p,q:p*p-q**3
actual=res(yy+delta,theta+da)-res(yy,theta)
correct=2*yy*delta-3*theta*theta*da
wrong=2*yy*delta-3*yy*yy*da
check('reference_Jacobian_is_evaluated_at_reference',abs(actual-correct)<1e-9 and abs(actual-wrong)>1e-4)

# Several wordings may be the same contaminated measurement in disguise.
proxies=np.array([[1.,2.],[2.,4.],[3.,6.]])
pose_only=np.array([0.,1.])
check('multiple_proxy_agreements_do_not_establish_binding',np.all(proxies@pose_only>0) and np.linalg.matrix_rank(proxies)==1)
# Equivariance/cycle consistency alone also permits a constant unwanted drift.
edit=lambda p:p+.2;geometry=lambda p:p+.3
check('commuting_edit_can_still_move_an_unrequested_anchor',abs(edit(geometry(0))-geometry(edit(0)))<1e-12 and edit(0)!=0)

# Local metric and derivatives must transform together under a latent change.
S=np.array([[2.,.4,0.],[0.,.7,.2],[0.,0.,1.3]]);inv=np.linalg.inv(S)
Ap=A@inv;Gp=inv.T@G@inv;pp=S@proposal
Hp=Gp+lam*Ap.T@(np.eye(7)-V@np.linalg.solve(V.T@V+Lambda,V.T))@Ap
linear=G@proposal-lam*A.T@(np.eye(7)-V@np.linalg.solve(V.T@V+Lambda,V.T))@e
linearp=Gp@pp-lam*Ap.T@(np.eye(7)-V@np.linalg.solve(V.T@V+Lambda,V.T))@e
check('latent_reparameterization_requires_metric_transform',np.linalg.norm(np.linalg.solve(Hp,linearp)-S@np.linalg.solve(Schur,linear))<1e-11)

path=Path(__file__)
result=dict(passed=sum(r['passed'] for r in rows),total=len(rows),checks=rows,
    scope='finite-dimensional identities, constructed signal controls and declared fact operators; no pretrained-image or semantic validation',
    script_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
path.with_suffix('.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(passed=result['passed'],total=result['total']),ensure_ascii=False))
