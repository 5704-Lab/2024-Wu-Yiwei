"""Independent identifiable manifolds and projection subproblem verification."""
import json
from pathlib import Path
import numpy as np
from scipy.special import i0e,i1e
from scipy.optimize import minimize,NonlinearConstraint
from numpy.polynomial.hermite import hermgauss

OUT=Path(__file__).parent/'outputs';OUT.mkdir(parents=True,exist_ok=True)
checks=[];rows=[]
for a in [.3,1.,2.]:
 for sigma in [.025,.05,.1]:
  s2=2*sigma;y=np.array([.7,.12]);pi=np.array([y[0],0.])
  den=lambda s:np.array([a*a/(a*a+s*s)*y[0],0.])
  single=den(sigma);dual=(4*single-den(s2))/3
  exact=np.array([(1-sigma*sigma*s2*s2/((a*a+sigma*sigma)*(a*a+s2*s2)))*y[0],0.])
  checks.append(dict(test='flat_gaussian',a=a,sigma=sigma,formula_error=float(np.linalg.norm(dual-exact)),less_tangent_shrink=bool(abs(dual[0]-pi[0])<abs(single[0]-pi[0])),normal_removed=bool(dual[1]==0)))
  rows.append(dict(a=a,sigma=sigma,single_tangent_bias=float(single[0]-pi[0]),dual_tangent_bias=float(dual[0]-pi[0])))

def circle_F(y,sigma,kappa,order):
 # Prior on the unit circle: rho(theta) proportional to exp(kappa*cos(theta)).
 # Observation likelihood gives a von Mises posterior with vector y/sigma^2+[kappa,0].
 nodes,w=hermgauss(order);grid=np.stack(np.meshgrid(nodes,nodes,indexing='ij'),axis=-1).reshape(-1,2)*np.sqrt(2)
 ws=np.outer(w,w).flatten()/np.pi
 z=y[None,:]+sigma*grid;eta=z/(sigma*sigma)+np.array([kappa,0.]);mag=np.linalg.norm(eta,axis=-1)
 m=(i1e(mag)/i0e(mag))[:,None]*eta/np.maximum(mag[:,None],1e-300)
 return np.sum(ws[:,None]*m,axis=0)

circles=[]
for kappa in [0.,2.]:
 for radial in [1.,1.08]:
  angle=1.1;normal=np.array([np.cos(angle),np.sin(angle)]);tangent=np.array([-np.sin(angle),np.cos(angle)]);y=radial*normal;fs=[circle_F(y,s,kappa,32) for s in [.025,.05,.1]]
  fp=[circle_F(y,s,kappa,48) for s in [.025,.05,.1]];dual=(4*fs[0]-fs[1])/3;pi=normal
  row=dict(kappa=kappa,radial=radial,quadrature_max_error=max(float(np.linalg.norm(a-b)) for a,b in zip(fs,fp)),single_error=float(np.linalg.norm(fs[0]-pi)),dual_error=float(np.linalg.norm(dual-pi)),single_tangent_bias=float(tangent@(fs[0]-pi)),dual_tangent_bias=float(tangent@(dual-pi)),single_normal_bias=float(normal@(fs[0]-pi)),dual_normal_bias=float(normal@(dual-pi)),second_order_scale_residual=float(np.linalg.norm((fs[2]-fs[1])-4*(fs[1]-fs[0]))/(np.linalg.norm(fs[2]-fs[1])+4*np.linalg.norm(fs[1]-fs[0]))))
  circles.append(row);checks.append(dict(test='circle',quadrature_pass=row['quadrature_max_error']<1e-10,dual_less_total_bias=row['dual_error']<row['single_error']))

rng=np.random.default_rng(7)
for i in range(6):
 r=rng.normal(size=5);radius=.2+i*.3;w=r*min(1,radius/np.linalg.norm(r))
 ref=minimize(lambda x:.5*np.sum((x-r)**2),np.zeros_like(r),jac=lambda x:x-r,method='SLSQP',constraints=[dict(type='ineq',fun=lambda x:radius**2-x@x,jac=lambda x:-2*x)],options=dict(ftol=1e-12,maxiter=1000))
 initial_success=bool(ref.success);initial_message=str(ref.message)
 if not ref.success:
  constraint=NonlinearConstraint(lambda x:x@x,-np.inf,radius**2,jac=lambda x:2*x,hess=lambda x,v:2*v[0]*np.eye(5))
  ref=minimize(lambda x:.5*np.sum((x-r)**2),np.zeros_like(r),jac=lambda x:x-r,hess=lambda x:np.eye(5),method='trust-constr',constraints=[constraint],options=dict(gtol=1e-12,xtol=1e-12,barrier_tol=1e-12,maxiter=1000))
 checks.append(dict(test='trust_region',initial_slsqp_success=initial_success,initial_message=initial_message,optimizer_success=bool(ref.success),solution_error=float(np.linalg.norm(w-ref.x)),feasible=bool(np.linalg.norm(w)<=radius+1e-12)))

y=rng.normal(size=5);x=rng.normal(size=5);qs=rng.normal(size=5);h=.02;delta_v=rng.normal(size=5);d=-h*delta_v;r=rng.normal(size=5);w=r*min(1,.05*np.linalg.norm(d)/np.linalg.norm(r));yn=y+d+w
query_error=float(np.linalg.norm((qs+yn-x)-(qs+y-x+d+w)));velocity_error=float(np.linalg.norm((delta_v-w/h)-(-(yn-y)/h)))
checks.append(dict(test='integration',query_error=query_error,velocity_error=velocity_error))
# Two equal-weight atoms +/-1: at the midpoint the mean is zero for all sigmas.
# Extrapolation cannot choose an unobserved branch; record as a structural counterexample.
counterexample=dict(support=[-1.,1.],observation=0.,posterior_mean=0.,dual_mean=0.,distance_to_support=1.,nearest_point_unique=False)
passed=all(c.get('formula_error',0)<1e-12 and c.get('less_tangent_shrink',True) and c.get('normal_removed',True) and c.get('quadrature_pass',True) and c.get('dual_less_total_bias',True) and c.get('optimizer_success',True) and c.get('solution_error',0)<1e-5 and c.get('feasible',True) for c in checks) and query_error<1e-12 and velocity_error<1e-12
result=dict(passed=passed,checks=checks,flat_rows=rows,circle_rows=circles,structural_counterexample=counterexample)
(OUT/'toy.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if not passed:raise RuntimeError('toy implementation gate failed')
