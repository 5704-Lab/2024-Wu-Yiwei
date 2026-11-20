"""Create a reviewable, source-disjoint experiment plan; never run a model."""
import difflib
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path,value):
    text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if path.exists() and path.read_text()!=text:
        raise RuntimeError(f'Plan exists with different contents: {path}; use a new version')
    path.write_text(text)


def main():
    manifest=ROOT/'data/flowedit/manifests/all_pairs.jsonl'
    records={r['pair_id']:r for r in [json.loads(line) for line in manifest.read_text().splitlines()]}
    dev_ids=['bear__00','cat__00','corgi__00','bus__00']
    confirm_ids=['bikes__00','butterfly__00','clown_fish__00','dog_snow__00',
                 'iguana__00','parrot__00','lighthouse__00','groceries__00',
                 'cake__00','jump_kick__00','cat_and_dog__00','milk__00']
    dev=[records[k] for k in dev_ids];confirm=[records[k] for k in confirm_ids]
    assert not {r['source_sha256'] for r in dev}&{r['source_sha256'] for r in confirm}
    selected=dev+confirm
    for record in selected:
        path=ROOT/record['source_image']
        assert path.exists() and sha(path)==record['source_sha256'],record['pair_id']
    protocol=dict(date='2026-11-09',status='design_complete_not_executed_on_image_models',
        model='existing frozen FLUX.1-dev',primary_resolution=512,primary_dtype='float32',
        text_conditions='complete CLIP/T5; audit lengths and cache identity before each stage',
        source_target_guidance=[3.5,3.5],tf32=False,source_features_injected=False,
        editing_mask=False,extra_trainable_models=False,radial_objective=False,
        conditioning_identity_includes_guidance=True,
        references=dict(manifest_sha256=sha(manifest),reference_sha256=sha(HERE/'reference.py')),
        development_pair_ids=dev_ids,confirmation_pair_ids=confirm_ids,
        exposure='Existing benchmark with past project exposure; not claimed genuinely unseen data',
        source_disjoint_between_new_roles=True,
        times=dict(mechanism_horizons=[.15,.3,.9],confirmation_horizons=[.6,.9],
                   primary_confirmation_horizon=.9,grid='uniform within fixed horizon'),
        cycle_rms_primary=.02,cycle_rms_secondary=[.01,.04],
        adaptive_source_lift=dict(trials=['L1','Heun1','Heun2','Heun4','Heun8','Heun16'],
            forward_calls=[1,2,4,8,16,32],source_generation='Heun24',
            maximum_source_calls=351,maximum_target_calls=48,maximum_total_calls=399,
            selection='first source-cycle pass at fixed horizon; no target score used',
            horizon_shrink_on_failure=False,unaccepted_output_is_not_a_success=True),
        precision_gates=dict(native_adapter_relative_error_max=1e-6,
            same_condition_clean_rms_max=1e-7,pair_state_identity_rms_max=1e-7,
            source_reference_levels=[48,96],reference_discrepancy_rms_max=5e-4,
            maximum_reference_level=192,reference_is_not_exact_ODE=True),
        statistics=dict(unit='source image; seeds/targets within a source are not independent units',
            baseline_seeds=[20261109,20261110,20261111],paired_source_bootstrap_replicates=10000,
            primary_metric='joint full task completion and no major unrequested change/artifact',
            rejected_and_unassessable_cases='count as non-success in primary coverage denominator',
            equal_completion_subset='secondary only; report subset size and full-case outcomes',
            universal_semantic_guarantee_claim=False,
            zero_failures_12_sources_one_sided_95_failure_upper=1-.05**(1/12)),
        annotation=dict(before_generation='freeze per-pair allowed/protected factors and task rubric',
            raters=2,methods_blinded=True,adjudicate_disagreements=True,
            task_labels=['complete','partial','absent','unassessable'],
            protected_labels=['none','minor','major','unassessable'],
            category_morphology='allow target-appropriate morphology; do not equate it with pose drift',
            scorer='existing SigLIP2 offline; record absolute target/source logits separately',
            scorer_is_oracle=False,online_semantic_feedback=False),
        stage_gates=dict(
            math='all analytic checks pass',
            smoke='identity/pair accounting pass; retain rejected high-horizon candidates as failures',
            mechanism='reference numerically stable or mark reference_unresolved',
            confirmation='lock code/config/annotations; require development target completion at T>=.6 on at least 2/4 sources and source gate coverage>=3/4, otherwise stop and report mechanism-only',
            quality_claim='joint success and protected-factor outcomes at comparable full completion; source PSNR alone is insufficient'),
        costs='Transformer calls below exclude text/VAE and any offline derivative probes; report all actual calls/time',
        optional_extension='1024 resolution and genuinely new independent sources only after primary gates pass')
    jobs=[]
    def add(stage,pair,T,method,calls,seed=None,**extra):
        jobs.append(dict(job_id=f'{stage}_{pair["pair_id"]}_T{T}_{method}_{seed}',stage=stage,
            pair_id=pair['pair_id'],horizon=T,method=method,seed=seed,
            max_transformer_forward_calls=calls,status='planned_not_executed',**extra))

    # Geometry is shared by the mechanism runs at the same source/horizon.
    for pair in dev:
        for T in [.15,.3,.9]:
            add('geometry',pair,T,'source_geometry',821,
                nominal_calls=437,reference_call_count=288,maximum_reference_calls=672,
                measurements=['L1/L2/zero query error','continuous defects where defined',
                              'source roundtrip','reference grid discrepancy'])
            for method,setup,seeds in [('random_fresh',0,protocol['statistics']['baseline_seeds']),
                ('random_fixed',0,protocol['statistics']['baseline_seeds']),
                ('zero_noise',0,[None]),('independent_L1',1,[None]),
                ('independent_dense_Heun',2,[None]),('independent_source_reference',0,[None])]:
                for seed in seeds:
                    add('mechanism',pair,T,method,48+setup,seed,
                        integrator='Euler24',source_reference_setup_charged_to_geometry=True,
                        target_displacement_scale=1.)
    for pair in dev:
        for T in [.3,.9]:
            for lift,setup in [('L1',1),('Heun1',2),('source_reference',0)]:
                add('paired_ablation',pair,T,'paired_'+lift,48+setup,
                    integrator='Euler24',source_cycle_gate='diagnostic; failed images labeled rejected',
                    outputs=['raw_target','source_reconstruction','anchored_clean'],
                    raw_target_and_anchor_share_trajectories=True)
                add('identity',pair,T,'identity_'+lift,48+setup,
                    integrator='Euler24',target_condition='exact full source condition',
                    disable_noop_fastpath=True)
    # Integrator refinement is kept separate from the query ablation.
    for pair in dev[:2]:
        for T in [.3,.9]:
            for method,multiplier in [('euler',2),('heun',4)]:
                for n in [12,24,48]:
                    add('integrator',pair,T,f'paired_L1_{method}{n}',multiplier*n+1,
                        integrator=f'{method}{n}',target_displacement_scale=1.)
            add('discrete_inverse_reference',pair,T,'Euler24_inverse_reference',528,
                inverse_max_iterations_per_step=20,inverse_residual_tolerance_rms=1e-6,
                proof_requires_step_contraction=True,reference_is_not_sota_baseline=True)
    for pair in confirm:
        for T in [.6,.9]:
            for method,calls,seeds in [('flowedit_avg1',48,protocol['statistics']['baseline_seeds']),
                ('flowedit_avg2',96,protocol['statistics']['baseline_seeds']),
                ('independent_L1',49,[None]),('paired_L1',49,[None]),
                ('paired_source_guarded',399,[None])]:
                for seed in seeds:
                    add('confirmation',pair,T,method,calls,seed,
                        integrator='Heun24' if method=='paired_source_guarded' else 'Euler24',
                        max_calls_is_a_cap_not_a_speed_claim=True)
    # A small prefix is an execution suggestion, not extra duplicated jobs.
    smoke=[j['job_id'] for j in jobs if j['pair_id']=='bear__00' and j['horizon'] in [.3,.9]
           and j['stage'] in ['geometry','paired_ablation','identity']]
    counts=Counter();costs=Counter()
    for job in jobs:
        counts[job['stage']]+=1;costs[job['stage']]+=job['max_transformer_forward_calls']
    annotations=[]
    for record in selected:
        source=record['source_prompt'].split();target=record['target_prompt'].split()
        changed=[dict(source=' '.join(source[i:j]),target=' '.join(target[k:l]))
                 for tag,i,j,k,l in difflib.SequenceMatcher(a=source,b=target).get_opcodes() if tag!='equal']
        annotations.append(dict(pair_id=record['pair_id'],prompt_differences=changed,
            allowed_factors='must be reviewed and frozen before generation',
            protected_factors='all unrequested pose/view/layout/background changes, with prompt-specific exceptions',
            annotation_ready=False))
    write_once(HERE/'protocol.json',protocol)
    write_once(HERE/'selected_pairs.json',selected)
    write_once(HERE/'annotation_template.json',annotations)
    write_once(HERE/'plan.json',dict(status='not_executed',smoke_job_ids=smoke,
        counts_by_stage=dict(counts),max_calls_by_stage=dict(costs),
        total_planned_jobs=len(jobs),total_max_transformer_calls=sum(costs.values()),jobs=jobs))
    print(json.dumps(dict(jobs=len(jobs),counts_by_stage=dict(counts),
        max_calls_by_stage=dict(costs),maximum_calls=sum(costs.values()),models_executed=False)))


if __name__=='__main__':
    main()
