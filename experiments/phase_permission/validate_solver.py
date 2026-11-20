"""Independent reduced convex-QP comparisons, reproducible without CUDA."""
import numpy as np
from scipy.optimize import minimize
from common import OUT,write,code_hashes
from real import reduced_qp

rng=np.random.default_rng(23)
rows=[]
for i in range(20):
    a=rng.normal(size=(3,3));h=a.T@a+np.eye(3);f=rng.normal(size=3);g=rng.normal(size=3)
    radius=.2 if i%2 else 3.;beta=radius*np.linalg.norm(g)*(.4 if i%3 else -.4)
    z,info=reduced_qp(h,f,g,beta,radius)
    ref=minimize(lambda x:.5*x@h@x+f@x,np.zeros(3),jac=lambda x:h@x+f,
        method='SLSQP',constraints=[dict(type='ineq',fun=lambda x:g@x-beta,jac=lambda x:g),
        dict(type='ineq',fun=lambda x:radius**2-x@x,jac=lambda x:-2*x)],options=dict(ftol=1e-12,maxiter=500))
    error=float(np.linalg.norm(z-ref.x))
    passed=error<2e-5 and g@z>=beta-1e-8 and np.linalg.norm(z)<=radius+1e-8 and info['stationarity']<1e-8
    rows.append(dict(case=i,passed=bool(passed),independent_success=bool(ref.success),error=error,qp=info))
    assert passed,rows[-1]
result=dict(passed=sum(r['passed'] for r in rows),total=len(rows),max_error=max(r['error'] for r in rows),rows=rows,code=code_hashes())
write(OUT/'solver_checks.json',result)
print({k:result[k] for k in ['passed','total','max_error']})
