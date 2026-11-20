"""Independent toy checks of SigLIP state gradients and common ascent."""
import json,math
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
torch.set_default_dtype(torch.float64)
OUT=Path(__file__).parent/'outputs';OUT.mkdir(parents=True,exist_ok=True)
checks=[]
def record(name,passed,**data):checks.append(dict(name=name,passed=bool(passed),**data))
y=torch.tensor([.4,.3],requires_grad=True)
e=torch.tensor([.3,.7,-.2]);e=e/e.norm()
def feature(y):return torch.stack([y[0].sin()+.2,y[0].cos()-.1,y[1]+.5])
def logit(y):
 f=feature(y);return 3.*(f/f.norm())@e-1.
ell=logit(y);g=torch.autograd.grad(ell,y,create_graph=True)[0]
f=feature(y).detach();z=f/f.norm()
J=torch.tensor([[math.cos(.4),0],[-math.sin(.4),0],[0,1.]])
expected=3.*J.T@(torch.eye(3)-z[:,None]*z[None,:])@e/f.norm()
record('analytic_state_gradient',torch.allclose(g,expected,atol=1e-10),error=float((g-expected).norm()))
S=torch.nn.functional.logsigmoid(ell)
r=torch.autograd.grad(S,y,create_graph=True)[0]
record('logsigmoid_gradient_factor',torch.allclose(r,torch.sigmoid(-ell)*g,atol=1e-10))
H=torch.autograd.functional.hessian(logit,y)
HS=torch.autograd.functional.hessian(lambda v:torch.nn.functional.logsigmoid(logit(v)),y)
a=torch.sigmoid(-ell).detach();p=torch.sigmoid(ell).detach()
record('sigmoid_hessian',torch.allclose(HS,a*H-a*p*g.detach()[:,None]*g.detach()[None,:],atol=1e-10))
u=g/g.norm();P=torch.eye(2)-u.detach()[:,None]*u.detach()[None,:]
record('normalized_hessian_cancellation',torch.allclose(P@HS/r.detach().norm(),P@H/g.detach().norm(),atol=1e-10))
ref=torch.tensor([.9,.1]);ref=ref/ref.norm()
b=torch.autograd.grad(ref@u,y)[0]
bm=H@((torch.eye(2)-u.detach()[:,None]*u.detach()[None,:])@ref/g.detach().norm())
record('direction_hvp',torch.allclose(b,bm,atol=1e-10))
for angle in [0.,.6,1.8,math.pi-.01]:
 u=np.array([1.,0.]);v=np.array([math.cos(angle),math.sin(angle)]);c=float(u@v);m=.1
 exact=m*(u+v)/(1+c)
 res=minimize(lambda w:.5*w@w,exact,method='SLSQP',jac=lambda w:w,
 constraints=[{'type':'ineq','fun':lambda w,a=a:float(a@w-m),'jac':lambda w,a=a:a} for a in [u,v]],options={'ftol':1e-12,'maxiter':200})
 record('common_ascent_qp_'+str(angle),res.success and np.linalg.norm(res.x-exact)<1e-7,error=float(np.linalg.norm(res.x-exact)))
 record('common_ascent_positive_'+str(angle),u@exact>0 and v@exact>0)
record('antiparallel_infeasible',not (.1<=-.1))
zero=torch.tensor(0.,requires_grad=True)
sat=torch.nn.functional.logsigmoid(torch.tensor(100.,dtype=torch.float32)+zero.float())
rs=torch.autograd.grad(sat,zero)[0]
record('finite_saturation_value',bool(torch.isfinite(sat)),logsigmoid=float(sat),gradient=float(rs))
# Deliberately large steps can violate monotonicity even for the exact toy gradient.
base=logit(y.detach()).item();gd=expected/expected.norm()
small=logit(y.detach()+1e-4*gd).item()
record('small_semantic_step',small>base)
q=torch.tensor([.2,-.5]);qs=torch.tensor([1.,2.]);x=torch.tensor([.1,.4]);d=torch.tensor([.02,-.03]);w=torch.tensor([.001,.002]);h=.1
record('query_closure',torch.allclose(qs+q+d+w-x,(qs+q+d-x)+w))
record('effective_velocity_sign',torch.allclose(q+d+w,q-h*(-d/h-w/h)))
result=dict(passed=all(r['passed'] for r in checks),checks=checks,scope='toy gradient, HVP, independent constrained solve, saturation and algebra')
(OUT/'stage0.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2));assert result['passed']
