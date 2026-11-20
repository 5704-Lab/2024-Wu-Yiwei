"""Real frozen-FLUX audit of clean-domain and source-coupling theory.

Protocol is registered by prepare.py. Input gradients use block checkpointing;
FP32 execution is partitioned over two GPUs, with unchanged stored weights.
All candidate states, failed gates and model calls are retained.
"""
import gc
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch.utils.checkpoint import checkpoint
from diffusers import AutoencoderKL

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs/run_v1'
sys.path.insert(0,str(ROOT/'experiments/sota_inversion_20261105'))
from common import setup,write,sha,load_model,base_info,decode_state,save_image,get_schedule
from flux.modules.layers import timestep_embedding


def rms(x):
    return float(x.detach().double().square().mean().sqrt())


def cosine(a,b):
    a,b=a.detach().double().flatten(),b.detach().double().flatten()
    return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-30))


class Execution:
    def __init__(self,model,inputs):
        self.model=model
        self.inputs=inputs
        self.dtype=torch.bfloat16
        self.calls=0
        self.vjps=0
        self.recomputed_blocks=0
        self.stage='init'
        self.start=time.monotonic()

    def info(self):
        info=base_info(0)
        info.update(inject=False,inverse=False,second_order=False,t=0.,feature_injection_mode='off')
        return info

    def partition_fp32(self):
        # Move the tail first, freeing GPU0 before widening its weights.
        for block in self.model.single_blocks[10:]:
            block.to(device='cuda:1',dtype=torch.float32)
        torch.cuda.empty_cache()
        for name,module in self.model.named_children():
            if name=='single_blocks':
                for block in module[:10]:
                    block.to(device='cuda:0',dtype=torch.float32)
            else:
                module.to(device='cuda:0',dtype=torch.float32)
        self.dtype=torch.float32
        gc.collect()
        torch.cuda.empty_cache()

    def velocity(self,q,t,condition='source',guidance=3.5,checkpoint_blocks=True):
        self.calls+=1
        m=self.model
        cond=self.inputs[condition]
        q=q.to(device='cuda:0',dtype=self.dtype)
        img=m.img_in(q)
        ts=torch.tensor([t],device='cuda:0',dtype=self.dtype)
        gs=torch.tensor([guidance],device='cuda:0',dtype=self.dtype)
        vec=m.time_in(timestep_embedding(ts,256))+m.guidance_in(timestep_embedding(gs,256))
        vec=vec+m.vector_in(cond['vec'].to(device='cuda:0',dtype=self.dtype))
        txt=m.txt_in(cond['txt'].to(device='cuda:0',dtype=self.dtype))
        ids=torch.cat((cond['txt_ids'],cond['img_ids']),dim=1).to('cuda:0')
        pe=m.pe_embedder(ids)
        for block in m.double_blocks:
            def double_fn(a,b,block=block):
                return block(img=a,txt=b,vec=vec,pe=pe,info=self.info())
            if torch.is_grad_enabled() and img.requires_grad and checkpoint_blocks:
                img,txt=checkpoint(double_fn,img,txt,use_reentrant=False,preserve_rng_state=False)
            else:
                img,txt=double_fn(img,txt)
        text_length=txt.shape[1]
        img=torch.cat((txt,img),dim=1)
        # Bind vec/pe into closures: later device transfers must not change the
        # tensors captured by an earlier block's backward recomputation.
        for i,block in enumerate(m.single_blocks):
            device=next(block.parameters()).device
            if img.device!=device:
                img=img.to(device)
            local_vec,local_pe=vec.to(device),pe.to(device)
            def single_fn(a,block=block,index=i,v=local_vec,p=local_pe):
                info=self.info();info.update(id=index,type='single')
                return block(a,vec=v,pe=p,info=info)[0]
            if torch.is_grad_enabled() and img.requires_grad and checkpoint_blocks:
                img=checkpoint(single_fn,img,use_reentrant=False,preserve_rng_state=False)
            else:
                img=single_fn(img)
        img=img[:,text_length:,:].to('cuda:0')
        result=m.final_layer(img,vec)
        if self.calls%5==0:
            print(json.dumps(dict(stage=self.stage,forward_calls=self.calls,vjps=self.vjps,
                                  seconds=round(time.monotonic()-self.start,2))),flush=True)
        return result


class CleanObjective:
    def __init__(self,execution,source,sigma=.5,nodes=2):
        self.e=execution
        self.x=source.detach()
        self.sigma=sigma
        self.nodes=nodes
        self.alpha=1/(1+sigma)
        self.t=sigma/(1+sigma)
        self.factor=1/(sigma*(1+sigma))

    @torch.no_grad()
    def field(self,y):
        q=self.alpha*y
        vs=self.e.velocity(q,self.t,'source').float()
        vt=self.e.velocity(q,self.t,'target').float()
        return self.factor*(vs-vt)

    @torch.no_grad()
    def value(self,y):
        if torch.equal(y,self.x):
            return 0.
        delta=y-self.x
        value=0.
        for j in range(self.nodes):
            z=self.x+(j+.5)/self.nodes*delta
            value+=float((self.field(z).double()*delta.double()).mean())/self.nodes
        return value

    def value_gradient(self,y):
        # At the source all Jacobian terms vanish exactly, and all quadrature
        # queries coincide. This shortcut is algebraically exact.
        if torch.equal(y,self.x):
            field=self.field(y)
            return 0.,field/y.numel(),field/y.numel()
        grad=torch.zeros_like(y)
        detached_grad=torch.zeros_like(y)
        total=0.
        for j in range(self.nodes):
            r=(j+.5)/self.nodes
            for name,sign in [('source',1.),('target',-1.)]:
                variable=y.detach().clone().requires_grad_(True)
                delta=variable-self.x
                q=self.alpha*(self.x+r*delta)
                v=self.e.velocity(q,self.t,name)
                coefficient=sign*self.factor/self.nodes
                loss=coefficient*(v.double()*delta.double()).mean()
                g,=torch.autograd.grad(loss,variable)
                self.e.vjps+=1
                self.e.recomputed_blocks+=57
                total+=float(loss.detach())
                grad+=g.detach()
                detached_grad+=coefficient*v.detach().float()/y.numel()
                del loss,g,v,variable,q,delta
        return total,grad,detached_grad


def gradient_audit(objective,source,initial,precision):
    e=objective.e;e.stage=precision+'_gradient_audit'
    probe=source+.08*initial/max(rms(initial),1e-30)
    value,g,detached=objective.value_gradient(probe)
    generator=torch.Generator(device='cpu').manual_seed(20261106)
    random=torch.randn(source.shape,generator=generator).to(source.device)
    directions={'gradient':g/max(rms(g),1e-30)}
    if precision=='fp32':
        directions['random']=random/max(rms(random),1e-30)
    rows=[]
    for name,u in directions.items():
        predicted=float((g.double()*u.double()).sum())
        for h in [.0005,.002,.008]:
            plus=objective.value(probe+h*u)
            minus=objective.value(probe-h*u)
            fd=(plus-minus)/(2*h)
            relative=abs(fd-predicted)/max(abs(fd),abs(predicted),1e-12)
            rows.append(dict(direction=name,perturbation_rms=h,autograd=predicted,
                             finite_difference=fd,relative_error=relative,passed=relative<=.05))
    result=dict(precision=precision,value=value,gradient_rms=rms(g),
                field_gradient_cosine=cosine(g,objective.field(probe)),
                detached_gradient_cosine=cosine(g,detached),
                detached_relative_error=rms(g-detached)/max(rms(g),1e-30),rows=rows,
                any_gradient_scale_passed=any(r['passed'] for r in rows if r['direction']=='gradient'))
    write(OUT/(precision+'_gradient_audit.json'),result)
    return result


def projection(y,x,radius):
    delta=y-x
    return x+delta*min(1.,radius/max(rms(delta),1e-30))


def clean_run(objective,source,cheap=False):
    name='clean_field_proposal' if cheap else 'clean_full_gradient'
    e=objective.e;e.stage=name
    y=source.clone();history=[y.cpu()];rows=[];all_trials=[]
    for k in range(8):
        if cheap:
            value=objective.value(y)
            g=objective.field(y)/y.numel()
        else:
            value,g,_=objective.value_gradient(y)
        norm=rms(g)
        if norm<1e-15:
            rows.append(dict(step=k,status='zero_direction'));break
        # Exact proximal step on the convex source ball. M=I/D. The
        # closed-form projection avoids a 65k-dimensional generic QCQP.
        accepted=False
        for factor in [1.,.5,.25,.125]:
            raw=factor*.08*g/norm
            candidate=projection(y+raw,source,.35)
            d=candidate-y
            linear=float((g.double()*d.double()).sum())
            measured=objective.value(candidate)
            gain=measured-value
            displacement=rms(candidate-source)
            passed=bool(math.isfinite(measured) and displacement<=.3500001 and linear>0
                        and gain>=max(.25*linear,1e-7))
            trial=dict(step=k,factor=factor,before=value,after=measured,gain=gain,
                       directional_gain=linear,source_rms=displacement,accepted=passed,
                       gradient_kind='field proxy' if cheap else 'full input gradient')
            all_trials.append(trial)
            if passed:
                y=candidate.detach();history.append(y.cpu());rows.append(dict(trial,status='accepted'))
                accepted=True;break
        if not accepted:
            rows.append(dict(step=k,status='no_accepted_candidate'));break
        write(OUT/(name+'_trace.json'),dict(steps=rows,trials=all_trials,
              completed_semantic_task=False,termination='running'))
        torch.save(dict(states=history),OUT/(name+'_partial.pt'))
        print(name,k,rows[-1],flush=True)
    write(OUT/(name+'_trace.json'),dict(steps=rows,trials=all_trials,
          completed_semantic_task=False,termination='registered_step_budget_or_stall'))
    return y,history


@torch.no_grad()
def flowedit(e,source,recipe):
    e.stage=recipe['name']
    schedule=get_schedule(28,source.shape[1],shift=True)
    y=source.clone();history=[y.cpu()]
    generator=torch.Generator(device='cpu').manual_seed(20261106)
    for i in range(4,28):
        t,tn=schedule[i:i+2]
        noise=torch.randn(source.shape,generator=generator).to(source.device)
        qs=(1-t)*source+t*noise;qt=qs+y-source
        vs=e.velocity(qs,t,'source',recipe['gs'])
        vt=e.velocity(qt,t,'target',recipe['gt'])
        y=y+(t-tn)*(vs-vt)
        history.append(y.cpu())
    return y,history


@torch.no_grad()
def noise_audit(e,source,objective):
    e.stage='random_query_audit'
    generator=torch.Generator(device='cpu').manual_seed(20261106)
    directions=[];rows=[]
    t=1/3
    for i in range(8):
        noise=torch.randn(source.shape,generator=generator).to(source.device)
        q=(1-t)*source+t*noise
        vs=e.velocity(q,t,'source');vt=e.velocity(q,t,'target')
        b=vs-vt;directions.append(b.cpu())
        rows.append(dict(candidate=i,noise_rms=rms(noise),
                         source_chord_residual_rms=rms(vs-(noise-source)),direction_rms=rms(b)))
    stack=torch.stack(directions)
    mean=stack.mean(0)
    variance=float((stack.double()-mean.double()).square().mean())
    deterministic=objective.field(source).cpu()
    write(OUT/'noise_audit.json',dict(rows=rows,pointwise_direction_variance=variance,
          mean_direction_rms=rms(mean),deterministic_vs_sample_mean_cosine=cosine(deterministic,mean),
          warning='8 candidates only; chord residual is not an ODE error certificate',
          deterministic_is_unbiased_estimate_of_flowedit=False))


def hermite(path,t):
    n=len(path['q'])-1;T=path['T'];dt=T/n
    index=min(int(t/dt+1e-8),n-1)
    r=(t-index*dt)/dt
    q0,q1=path['q'][index:index+2];v0,v1=path['v'][index:index+2]
    q=(2*r**3-3*r**2+1)*q0+(r**3-2*r**2+r)*dt*v0+(-2*r**3+3*r**2)*q1+(r**3-r**2)*dt*v1
    derivative=(6*r*r-6*r)/dt*q0+(3*r*r-4*r+1)*v0+(-6*r*r+6*r)/dt*q1+(3*r*r-2*r)*v1
    return q,derivative


@torch.no_grad()
def source_path(e,source,n,heun=False):
    T=.3;dt=T/n;q=source.clone();qs=[q];vs=[]
    for i in range(n):
        t=i*dt
        v=e.velocity(q,t,'source');vs.append(v)
        if heun:
            vp=e.velocity(q+dt*v,t+dt,'source')
            q=q+dt*(v+vp)/2
        else:
            q=q+dt*v
        qs.append(q)
    vs.append(e.velocity(q,T,'source'))
    return dict(T=T,q=qs,v=vs)


@torch.no_grad()
def path_edit(e,source,path,name):
    e.stage='path_edit_'+name
    y=source.clone();history=[y.cpu()];trace=[]
    for k in range(8):
        t=.3*(1-k/8);tn=.3*(1-(k+1)/8)
        q,_=hermite(path,t);qt=q+y-source
        vs=e.velocity(q,t,'source');vt=e.velocity(qt,t,'target')
        trace.append(dict(q=q.cpu(),qt=qt.cpu(),vs=vs.cpu(),vt=vt.cpu()))
        y=y+(t-tn)*(vs-vt);history.append(y.cpu())
    return y,history,trace


@torch.no_grad()
def source_audit(e,source):
    e.stage='source_path_reference'
    refs={n:source_path(e,source,n,True) for n in [16,32]}
    reference=refs[32]
    fine_discrepancy=max(rms(refs[16]['q'][i]-reference['q'][2*i]) for i in range(17))
    write(OUT/'source_reference.json',dict(reference='Heun32 numerical reference; not exact solution',
          heun16_vs_32_max_path_rms=fine_discrepancy,T=.3))
    paths={};rows=[]
    for n in [2,4,8]:
        e.stage='source_path_euler_'+str(n)
        path=source_path(e,source,n,False);paths[n]=path
        errors=[rms(hermite(path,.3*i/32)[0]-reference['q'][i]) for i in range(33)]
        residuals=[];secants=[]
        for i in range(n):
            t=.3*(i+.5)/n
            q,derivative=hermite(path,t)
            v=e.velocity(q,t,'source')
            residuals.append(rms(derivative-v))
            ri=round(t/.3*32)
            secants.append(rms(v-reference['v'][ri])/max(rms(q-reference['q'][ri]),1e-30))
        L=max(secants);delta=max(residuals)
        heuristic=delta*math.expm1(min(L*.3,50))/max(L,1e-30)
        rows.append(dict(euler_steps=n,max_path_rms_error_vs_heun32=max(errors),
                         terminal_rms_error_vs_heun32=errors[-1],midpoint_defect_rms_max=delta,
                         observed_secant_L=L,empirical_envelope=heuristic,
                         certified=False,reason='sampled residuals and secants are not uniform bounds'))
    write(OUT/'source_path_audit.json',dict(rows=rows,reference_discretization_discrepancy=fine_discrepancy))
    results={};histories={};traces={}
    for name,path in [('reference',reference),('euler2',paths[2]),('euler8',paths[8])]:
        y,history,trace=path_edit(e,source,path,name)
        results['path_'+name]=y.cpu();histories['path_'+name]=history;traces[name]=trace
    propagation=[]
    for name in ['euler2','euler8']:
        Ls=La=0.;deltas=[]
        for a,b in zip(traces[name],traces['reference']):
            delta=rms(a['q']-b['q']);dtarg=rms(a['qt']-b['qt'])
            deltas.append(delta)
            Ls=max(Ls,rms(a['vs']-b['vs'])/max(delta,1e-30))
            La=max(La,rms(a['vt']-b['vt'])/max(dtarg,1e-30))
        bound=0.;bound_history=[0.]
        for delta in deltas:
            bound=(1+.3/8*La)*bound+.3/8*(Ls+La)*delta
            bound_history.append(bound)
        actual=[rms(a-b) for a,b in zip(histories['path_'+name],histories['path_reference'])]
        propagation.append(dict(path=name,realized_pair_secant_Ls=Ls,realized_pair_secant_La=La,
                                realized_error_rms=actual,algebraic_bound_rms=bound_history,
                                all_realized_pairs_bounded=all(a<=b+1e-6 for a,b in zip(actual,bound_history)),
                                predictive_network_certificate=False))
    write(OUT/'source_edit_propagation.json',dict(rows=propagation))
    torch.save(dict(paths={str(k):{key:([v.cpu() for v in val] if isinstance(val,list) else val)
                                   for key,val in path.items()} for k,path in paths.items()},
                    reference={key:([v.cpu() for v in val] if isinstance(val,list) else val)
                               for key,val in reference.items()},edit_traces=traces),OUT/'source_paths.pt')
    return results,histories


def main():
    setup()
    if (OUT/'started.json').exists():
        raise RuntimeError('Run already started; retain outputs and use a fresh version')
    protocol=json.loads((OUT/'protocol.json').read_text())
    write(OUT/'started.json',dict(script_sha256=sha(__file__),unix_time=time.time(),
                                protocol_sha256=sha(OUT/'protocol.json')))
    payload=torch.load(OUT/'inputs.pt',map_location='cpu',weights_only=True)
    model=load_model(torch.device('cuda:0'),OUT/'weight_audit.json')
    e=Execution(model,payload['inputs']);x=payload['inputs']['source']['img'].to('cuda:0')
    # Check the checkpoint-compatible adapter against the unchanged native path.
    e.stage='native_adapter_parity'
    c=payload['inputs']['source']
    with torch.no_grad():
        q=(x*.7).bfloat16()
        native,_=model(img=q,img_ids=c['img_ids'].to('cuda:0'),txt=c['txt'].to('cuda:0'),
            txt_ids=c['txt_ids'].to('cuda:0'),timesteps=torch.tensor([.3],device='cuda:0',dtype=torch.bfloat16),
            y=c['vec'].to('cuda:0'),guidance=torch.tensor([3.5],device='cuda:0',dtype=torch.bfloat16),info=e.info())
        adapted=e.velocity(q,.3)
        parity=dict(bitwise_equal=bool(torch.equal(native,adapted)),rms_error=rms(native-adapted))
        write(OUT/'adapter_parity.json',parity)
        assert parity['bitwise_equal'],parity
    objective=CleanObjective(e,x)
    initial=objective.field(x)
    gradient_audit(objective,x,initial,'bf16')
    e.stage='partition_fp32';print('PARTITION_FP32',flush=True)
    del native,adapted,q;gc.collect();torch.cuda.empty_cache()
    e.partition_fp32()
    objective=CleanObjective(e,x)
    scale_rows=[]
    for sigma in [.2,.5,1.]:
        e.stage='scale_probe_'+str(sigma)
        field=CleanObjective(e,x,sigma).field(x)
        scale_rows.append(dict(sigma=sigma,query_time=sigma/(1+sigma),field_rms=rms(field),
                               finite=bool(torch.isfinite(field).all())))
    write(OUT/'scale_probe.json',dict(rows=scale_rows))
    initial=objective.field(x)
    fp32_audit=gradient_audit(objective,x,initial,'fp32')
    states={'source':x.cpu()};histories={};stages=[]
    def save():
        torch.save(dict(states=states,histories=histories),OUT/'checkpoint.pt')
        write(OUT/'progress.json',dict(stages_completed=stages,forward_calls=e.calls,vjps=e.vjps,
              recomputed_blocks=e.recomputed_blocks,seconds=time.monotonic()-e.start))
    for cheap in [False,True]:
        name='clean_field_proposal' if cheap else 'clean_full_gradient'
        y,history=clean_run(objective,x,cheap)
        states[name]=y.cpu();histories[name]=history;stages.append(name);save()
    with torch.no_grad():
        for recipe in protocol['baselines']:
            y,history=flowedit(e,x,recipe)
            states[recipe['name']]=y.cpu();histories[recipe['name']]=history
            stages.append(recipe['name']);save()
        noise_audit(e,x,objective);stages.append('noise_audit');save()
        astates,ahistories=source_audit(e,x)
        states.update(astates);histories.update(ahistories);stages.append('source_audit');save()
        e.stage='fixed_objective_endpoint_audit'
        scores={name:objective.value(state.to('cuda:0')) for name,state in states.items()}
        write(OUT/'endpoint_objective.json',dict(scores=scores,
              warning='fixed proxy scalar only; not semantic correctness or true likelihood'))
    peak={str(i):torch.cuda.max_memory_allocated(i) for i in [0,1]}
    forward_calls,vjps,recomputed=e.calls,e.vjps,e.recomputed_blocks
    seconds=time.monotonic()-e.start
    del objective,e,model,y,initial,field;gc.collect();torch.cuda.empty_cache()
    print('DECODE_STATES',flush=True)
    vae=AutoencoderKL.from_pretrained(ROOT/'pretrained_models/flux1-dev/vae',local_files_only=True,
        torch_dtype=torch.float32).eval().requires_grad_(False).to('cuda:0')
    images={}
    with torch.inference_mode():
        for name,state in states.items():
            image=decode_state(vae,state,512,'cuda:0')
            images[name]=image.cpu();save_image(image,OUT/(name+'.png'))
        for name in ['clean_full_gradient','clean_field_proposal']:
            for index,state in enumerate(histories[name]):
                image=decode_state(vae,state,512,'cuda:0')
                save_image(image,OUT/(name+f'_step{index:02d}.png'))
    torch.save(dict(states=states,histories=histories,images=images),OUT/'results.pt')
    write(OUT/'complete.json',dict(status='complete',forward_calls=forward_calls,
          additional_native_parity_forward=1,vjps=vjps,checkpoint_recomputed_blocks=recomputed,
          transformer_seconds=seconds,peak_allocated_bytes=peak,stages_completed=stages,
          fp32_gradient_audit_any_scale_passed=fp32_audit['any_gradient_scale_passed'],
          semantic_task_certified=False,script_sha256=sha(__file__)))
    print('REAL_THEORY_COMPLETE',flush=True)


if __name__=='__main__':
    main()
