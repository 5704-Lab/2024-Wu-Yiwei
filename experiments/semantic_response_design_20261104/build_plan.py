"""Build a read-only pilot query plan; does NOT encode prompts or run SD3.

Existing saved states are reused as exploratory data, never called fresh test data.
"""
from pathlib import Path
import hashlib
import itertools
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def main():
    base=ROOT/'experiments/real_transport_20261104/outputs/trajectory_data_v1'
    manifest_path=base/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    wanted={'bear__00':'bear','brown_owl__00':'owl','cat_crown__00':'cat','corgi__00':'dog'}
    probes=[dict(tau=tau, noise_seed=seed) for tau in [.2, .35] for seed in [20261104, 20261105]]
    rows=[]
    for row in manifest['records']:
        pair=row['pair']
        if pair not in wanted: continue
        animal=wanted[pair]
        prompts=[]
        for edit, pose, position in itertools.product([0, 1], repeat=3):
            text=row['record']['target_prompt' if edit else 'source_prompt']
            if pose: text+=f' The {animal} has its head turned toward the camera.'
            if position: text+=f' The {animal} is positioned on the right side of the image.'
            prompts.append(dict(factors=[edit, pose, position], text=text,
                                meaning='factorial descriptive condition, not an observed-image fact'))
        controls=dict(
            empty='',
            no_edit=row['record']['source_prompt'],
            target_explicit_preserve=row['record']['target_prompt']+
                f' The {animal} keeps its original head orientation and its original position in the image.',
            reverse_pose=row['record']['target_prompt']+
                f' The {animal} has its head turned away from the camera.',
            reverse_position=row['record']['target_prompt']+
                f' The {animal} is positioned on the left side of the image.')
        states=[]
        for state in ['source', 'flow_step23', 'flow_step43']:
            path=base/pair/(state+'.pt')
            if not path.exists(): raise FileNotFoundError(path)
            states.append(dict(name=state, file=str(path.relative_to(ROOT)),
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               coordinate='actual scaled clean y; not a predicted endpoint',
                               source=state=='source'))
        rows.append(dict(pair=pair, original_source_split=row['split'],
                         split_in_this_plan='exploratory_reuse', source_record=row['record'],
                         prompts=prompts, controls=controls, states=states))
    assert len(rows)==4
    # 4 sources x (source + 2 actual current states) x 4 probes x (8 + empty).
    queries=len(rows)*3*len(probes)*9
    payload=dict(status='plan_only_no_model_queries', prompt_encoding='not_executed',
                 semantic_reader_gate='required_before_correction', sources=rows, probes=probes,
                 factorial_plus_empty_branch_queries=queries,
                 branch_query_count_scope='forwards only; excludes controls, gradients and finite candidates',
                 independent_noise_confirmation='new seeds must be generated and locked after development',
                 full_condition_requirement='re-encode both CLIP encoders, T5 and pooled projections; no token deletion in place',
                 manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    dest=HERE/'outputs/pilot_plan_v1.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    serialized=json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if dest.exists() and dest.read_text()!=serialized:
        raise RuntimeError('Different plan exists; version the plan before writing.')
    dest.write_text(serialized)
    print(json.dumps(dict(status=payload['status'], sources=len(rows),
                         branch_queries_planned=queries, output=str(dest)),ensure_ascii=False))


if __name__=='__main__': main()
