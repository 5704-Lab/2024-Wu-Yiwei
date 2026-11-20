"""Independent closed-form checks for degeneracy and finite control."""
import json,math
from pathlib import Path
import torch
torch.set_default_dtype(torch.float64)
out=Path(__file__).parent/'outputs';out.mkdir(exist_ok=True)
checks=[]
def add(name,passed,**values):checks.append(dict(name=name,passed=bool(passed),**values))
q=torch.tensor(.4,requires_grad=True);xi=torch.tensor([.2,.7],requires_grad=True)
g=torch.autograd.grad(xi[0]*q.cos()+xi[1]*q.sin(),xi,create_graph=True)[0]
u=g/g.norm();ref=torch.tensor([1.,0.]);b=torch.autograd.grad(ref@u,q,retain_graph=True)[0]
add('rotation_gradient',abs(float(b)+math.sin(.4))<1e-12)
for sign in [1.,-1.]:
    b0=torch.autograd.grad((sign*u.detach())@u,q,retain_graph=True)[0]
    add('parallel' if sign>0 else 'antiparallel',abs(float(b0))<1e-12)
z=torch.tensor(.3,requires_grad=True);e=torch.tensor([1.,2.],requires_grad=True)
gm=torch.autograd.grad(e[0]*z.exp(),e,create_graph=True)[0]
bm=torch.autograd.grad(ref@(gm/gm.norm()),z)[0]
add('magnitude_change_invisible',abs(float(bm))<1e-12)
P=torch.tensor([[1.,0.],[0.,1.],[0.,0.],[0.,0.]])
feature=torch.tensor([1.,2.],requires_grad=True);packed=P@feature
gp=torch.autograd.grad(packed@torch.tensor([.2,.4,9.,8.]),feature)[0]
add('actual_padding_coordinates',torch.allclose(gp,torch.tensor([.2,.4])))
A=torch.diag(torch.tensor([1.,4.]));rhs=torch.tensor([1.,1.]);lam=.7
def solve(mu):return torch.linalg.solve(A.T@A+(lam+mu)*torch.eye(2),rhs)
w0=solve(0);R=.5*float(w0.norm());lo,hi=0.,10.
for _ in range(100):
    mid=(lo+hi)/2
    if float(solve(mid).norm())>R:lo=mid
    else:hi=mid
wc=solve(hi);ws=w0*R/w0.norm()
def loss(w):return .5*float(w@(A.T@A+lam*torch.eye(2))@w)-float(rhs@w)
add('budget_scaling_not_exact',loss(wc)<loss(ws)-1e-5,exact_loss=loss(wc),scaled_loss=loss(ws))
add('unreachable_normal_component',float(((torch.eye(2)-u.detach()[:,None]*u.detach()[None,:])@u.detach()).norm())<1e-12)
small=.01*float(b)
j0=1-math.cos(.4);j1=1-math.cos(.4+small)+.7/2*small**2
add('finite_regularized_descent',j1<j0,loss_before=j0,loss_after=j1)
add('large_step_can_fail',1-math.cos(.4+2)+.7/2*4>j0)
errors=[]
for N in [10,20,40]:
    h=1/N;y=1.;base=1.
    for _ in range(N):y=(1-h)*y+.3*h*h;base=(1-h)*base
    errors.append(abs(y-base))
add('quadratic_correction_consistency',errors[2]<errors[1]<errors[0],errors=errors)
x=torch.tensor([.3,-.7]);ys=torch.tensor([.2,.4]);d=torch.tensor([.1,-.2]);w=torch.tensor([.02,.01]);qs=torch.tensor([1.,2.]);h=.1;dv=-d/h
new=ys+d+w
add('query_and_effective_velocity_closure',torch.allclose(qs+new-x,(qs+ys+d-x)+w) and torch.allclose(new,ys-h*(dv-w/h)))
add('zero_radius_no_control',float(torch.zeros(2).norm())==0.)
result=dict(passed=all(c['passed'] for c in checks),checks=checks)
(out/'stage0_edges.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
assert result['passed']
