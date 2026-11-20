"""Versioned minimal repair of the v1 development gate; no model execution."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path,value):
    text=json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    if path.exists() and path.read_text()!=text:
        raise RuntimeError('Different revision exists; create a new revision')
    path.write_text(text)


def main():
    original=json.loads((PARENT/'plan.json').read_text())
    protocol=deepcopy(json.loads((PARENT/'protocol.json').read_text()))
    plan=deepcopy(original)
    new_ids=[];audit_ids=[]
    for pair in protocol['development_pair_ids']:
        main_id=f'development_main_{pair}_T0.9_paired_source_guarded_None'
        audit_id=f'main_map_stability_{pair}_T0.9_Heun48_96_None'
        new_ids.append(main_id);audit_ids.append(audit_id)
        plan['jobs'].append(dict(job_id=main_id,stage='development_main',pair_id=pair,
            horizon=.9,method='paired_source_guarded',seed=None,
            max_transformer_forward_calls=399,status='planned_not_executed',
            integrator='Heun24',role='actual main algorithm required by confirmation gate'))
        plan['jobs'].append(dict(job_id=audit_id,stage='main_map_stability',pair_id=pair,
            horizon=.9,method='Heun48_96',seed=None,
            max_transformer_forward_calls=576,status='planned_not_executed',
            depends_on=main_id,skip_if_main_rejected=True,
            fixed_input='same accepted qT from development_main',
            measurements=['source replay Heun24/48/96 discrepancy','target replay Heun24/48/96 discrepancy'],
            cost_formula='two conditions times (2*48 + 2*96)',
            role='empirical discretization audit; not a global inverse-Lipschitz certificate'))
    counts=Counter();costs=Counter()
    for row in plan['jobs']:
        counts[row['stage']]+=1;costs[row['stage']]+=row['max_transformer_forward_calls']
    plan.update(revision_date='2026-11-10',revision='v2_audit_draft_not_executed',
        parent_plan_sha256=sha(PARENT/'plan.json'),added_development_main_ids=new_ids,
        added_solver_audit_ids=audit_ids,counts_by_stage=dict(counts),max_calls_by_stage=dict(costs),
        total_planned_jobs=len(plan['jobs']),total_max_transformer_calls=sum(costs.values()))
    protocol.update(revision='v2_audit_draft',revision_date='2026-11-10',
        parent_protocol_sha256=sha(PARENT/'protocol.json'),
        reviewed_method_role='approximate partial-inversion diagnostic; no proven inversion-free preservation gain',
        machine_source_cycle_guarantee='correction amplitude only; edit reference error additionally requires conditioning',
        Heun_inverse_bound_certified_for_FLUX=False,
        solver_audit='On accepted development lifts, compare source and target Heun24/48/96 maps; numerical grid stability is not a rigorous error certificate')
    protocol['stage_gates']['confirmation']=(
        'Consume the four explicit development_main jobs at T=.9; require >=2/4 full target completions, '
        '>=3/4 source gate coverage, completed map-stability audit for every accepted development case, '
        'and code/config/annotations locked. Otherwise stop before confirmation.')
    protocol['stage_gates']['main_map_stability']=(
        'Compare fixed-qT source and target maps at 24/48/96 steps; require each successive '
        'difference <=5e-4 RMS or mark solver_consistency_failed. Do not silently change the main solver.')
    assert len(plan['jobs'])==432 and sum(costs.values())==48348
    assert {j['pair_id'] for j in plan['jobs'] if j['stage']=='development_main'}==set(protocol['development_pair_ids'])
    assert len({j['job_id'] for j in plan['jobs']})==len(plan['jobs'])
    write_once(HERE/'plan_v2.json',plan)
    write_once(HERE/'protocol_v2.json',protocol)
    write_once(HERE/'revision_summary.json',dict(
        status='reviewed_draft_no_models_executed',original_jobs=len(original['jobs']),
        revised_jobs=len(plan['jobs']),added_main_jobs=4,added_stability_jobs=4,
        original_max_calls=original['total_max_transformer_calls'],revised_max_calls=sum(costs.values()),
        added_max_calls=3900,original_plan_unchanged=True,original_protocol_unchanged=True,
        genuinely_certified_FLUX_inverse_conditions=False,script_sha256=sha(Path(__file__))))
    print(json.dumps(dict(jobs=len(plan['jobs']),max_calls=sum(costs.values()),new_model_calls=0)))


if __name__=='__main__':
    main()
