"""Show matched-range comparison and the separately costed Heun solver diagnostic."""
import hashlib
import json
from run import HERE
from summarize import figure


def main():
    folder=HERE/'outputs/bear_v1';heun=HERE/'outputs/heun_partial_v1'
    result=json.loads((heun/'complete.json').read_text())
    columns=[('Real source','source.png'),('FlowEdit: 33 steps','flowedit.png'),
             ('ODE Euler: 33 + 33','ode_partial.png'),
             ('ODE Heun: 33 + 33','../heun_partial_v1/ode_heun_partial.png')]
    figure(folder,'comparison_matched.png','Real-image comparison: FlowEdit and ordinary ODE inversion',columns[:3],
        'SD3 Medium | FP32 | Same sigma=0.850143 and CFG=3.5/13.5 | 132 branch evaluations per editing method')
    figure(folder,'comparison_with_solver.png','Same real source: FlowEdit versus direct partial ODE inversion',columns,
        'SD3 Medium | Same partial sigma=0.850143 | Source/target CFG=3.5/13.5 | Heun uses twice the velocity evaluations')
    figure(folder,'reconstruction_with_solver.png','Source reconstruction: does the numerical inverse preserve the input?',[
        ('Real source','source.png'),('VAE reconstruction','vae_reconstruction.png'),
        ('Euler partial reconstruction','ode_partial_reconstruction.png'),
        ('Heun partial reconstruction','../heun_partial_v1/ode_heun_partial_reconstruction.png')],
        'Same source prompt and CFG=3.5 in both directions; sigma=0.850143; Heun has twice the velocity evaluations')
    figure(folder,'head_detail_with_solver.png','Fixed source-coordinate crop: head, orientation and local artifacts',columns,
        'Same crop x=256:512, y=90:310; no registration or region-dependent editing.',crop=(256,90,512,310))
    files=[folder/'comparison_matched.png',folder/'comparison_with_solver.png',folder/'reconstruction_with_solver.png',folder/'head_detail_with_solver.png',
           heun/'protocol.json',heun/'complete.json',heun/'states.pt',HERE/'heun_check.py',HERE/'compare_heun.py']
    audit=dict(status='complete',heun_metrics=result['metrics'],
        files={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (folder/'solver_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(audit['heun_metrics'],indent=2))


if __name__=='__main__':main()
