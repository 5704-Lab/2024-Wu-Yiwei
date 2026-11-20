"""CPU geometry check of source-velocity-matched local lifting.

Uses only the frozen 2026-11-06 source trajectory. No model inference or edited
image is generated. The numerical reference is not a certified exact solution.
"""
import hashlib
import json
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent
RUN=HERE/'outputs/run_v1'
OUT=HERE/'noise_adaptation_20261109'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rms(value):
    return float(value.double().square().mean().sqrt())


def main():
    OUT.mkdir(exist_ok=True)
    data=torch.load(RUN/'inputs.pt',map_location='cpu',weights_only=True)
    ref=torch.load(RUN/'source_paths.pt',map_location='cpu',weights_only=True)['reference']
    x=data['inputs']['source']['img'].double()
    v0=ref['v'][0].double()
    effective_noise=x+v0
    rows=[]
    for i in [1,2,4,8,16,32]:
        t=.3*i/32
        q0=(1-t)*x
        q1=x+t*v0
        q1_noise=(1-t)*x+t*effective_noise
        target=ref['q'][i].double()
        e0,e1=rms(q0-target),rms(q1-target)
        rows.append(dict(time=t,zero_noise_query_error_rms=e0,
                         source_velocity_lift_error_rms=e1,error_reduction_factor=e0/e1,
                         coordinate_identity_rms=rms(q1-q1_noise)))
    # Verify exact noise-coordinate ODE algebra on a smooth scalar reference:
    # q'=a q, q(0)=x, q=e^(at)x, eps=(q-(1-t)x)/t.
    coordinate_errors=[]
    for t in [.01,.05,.2,.4]:
        time=torch.tensor(t,dtype=torch.float64,requires_grad=True)
        source=torch.tensor(1.3,dtype=torch.float64)
        a=.7
        q=torch.exp(a*time)*source
        eps=(q-(1-time)*source)/time
        eps_prime,=torch.autograd.grad(eps,time)
        residual=time*eps_prime-(a*q+source-eps)
        coordinate_errors.append(abs(float(residual.detach())))
    assert max(coordinate_errors)<1e-10
    assert all(row['coordinate_identity_rms']<1e-12 for row in rows)
    result=dict(date='2026-11-09',scope='CPU post-hoc query geometry only; no new model inference',
        effective_noise_rms=rms(effective_noise),local_lift='x+t*v_source(x,0)',
        reference='cached Heun32 learned-source trajectory on [0,.3]',
        rows=rows,coordinate_ode_max_residual=max(coordinate_errors),
        image_edit_benefit_verified=False,uniform_model_error_bound_certified=False,
        input_sha256={name:sha(RUN/name) for name in ['inputs.pt','source_paths.pt']},
        script_sha256=sha(Path(__file__)))
    (OUT/'query_geometry.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
