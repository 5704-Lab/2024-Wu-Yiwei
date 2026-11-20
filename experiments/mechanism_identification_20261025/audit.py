"""Independent artifact checks and compact figures for completed diagnostics."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs'


def load(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    audit={};checks=[];unique={}
    for tag in ['v1','v2','v3','v5']:
        folder=OUT/tag/'dataset';manifest=load(folder/'manifest.json')
        assert sha(folder/'snapshot_dataset.py')==manifest['script_sha256']
        for record in manifest['records']:
            for name,value in record['variants'].items():
                path=HERE/value['file'];actual=sha(path)
                assert actual==value['sha256'],path
                key=(record['scene'],name)
                if key in unique:assert unique[key]==actual,key
                unique[key]=actual
        checks.append(dict(tag=tag,kind='dataset_hashes',sources=len(manifest['records']),passed=True))
    for tag,protocol_name,snapshot in [
        ('v1','protocol_probe.json','snapshot_probe.py'),
        ('v2','protocol_probe.json','snapshot_probe.py'),
        ('v3','protocol_refine.json','snapshot_refine.py'),
        ('v4','protocol.json','snapshot_binding.py'),
        ('v5','protocol.json','snapshot_offset.py')]:
        protocol=load(OUT/tag/protocol_name)
        key=protocol.get('script_sha256') or protocol['code_hashes'][snapshot.removeprefix('snapshot_')]
        assert sha(OUT/tag/snapshot)==key,(tag,snapshot)
        for name,expected in protocol.get('code_hashes',{}).items():
            saved=OUT/tag/('snapshot_'+name)
            if saved.exists():assert sha(saved)==expected,(tag,name)
        checks.append(dict(tag=tag,kind='driver_and_available_dependency_snapshots',passed=True))
    for script in HERE.glob('*.py'):ast.parse(script.read_text())
    v1=load(OUT/'v1/analysis.json');v2=load(OUT/'v2/analysis.json')
    v3=load(OUT/'v3/refine_results.json');v4=load(OUT/'v4/analysis.json');v5=load(OUT/'v5/analysis.json')
    for row in v3['rows']:
        assert row['projected_gradient_inf']<1e-8,row
        assert row['minimum_gain']>=.04999 and row['maximum_gain']<=12.0001
    for row in v3['vae_rows']:
        assert row['identity_increment_max']==0
        assert row['exact_chord_endpoint_max']<1e-5,row
        assert all(c['relative_jvp_error']<.001 for c in row['jvp_finite_checks']),row
    for tag in ['v4','v5']:
        results=load(OUT/tag/'results.json')
        for row in results['rows']:
            assert row['fit_kkt' if tag=='v4' else 'kkt']<1e-8
            assert all(np.isfinite(v) for v in row.values() if isinstance(v,(float,int)))
        if tag=='v4':
            cal=[r for r in results['rows'] if r['split']=='calibration' and r['case']=='correct']
            for field in ['mechanism_rgb_mse','unexplained_phase','unexplained_complex']:
                assert abs(v4['thresholds'][field]-(1.25*max(r[field] for r in cal)+1e-10))<1e-12
        else:
            cal=[r for r in results['rows'] if r['split']=='calibration' and r['case']=='correct']
            for field in v5['thresholds']:
                assert abs(v5['thresholds'][field]-(1.25*max(r[field] for r in cal)+1e-10))<1e-12
    natural4=load(OUT/'v4/natural.json');natural5=load(OUT/'v5/natural.json')
    natural=[]
    for r4,r5 in zip(natural4['records'],natural5['records']):
        assert r4['pair']==r5['pair']
        d4={r['case']:r for r in r4['rows']};d5={r['case']:r for r in r5['rows']}
        for control in ['global_dark','global_white']:
            assert d5[control]['mechanism_rgb_mse']<1e-8
            assert d5[control]['unexplained_phase']<.001
        for name in d5:
            natural.append(dict(pair=r5['pair'],case=name,
                old_unexplained_phase=d4[name]['unexplained_phase'],new_unexplained_phase=d5[name]['unexplained_phase'],
                old_rgb_mse=d4[name]['mechanism_rgb_mse'],new_rgb_mse=d5[name]['mechanism_rgb_mse']))
    direct=load(OUT/'v4/direct_bank_analysis.json')['results']
    raw4=load(OUT/'v4/results.json')['rows'];threshold4=v4['thresholds']
    for rule in direct:
        for stored in rule['selected']:
            candidates=[r for r in raw4 if r['scene']==stored['scene'] and not r['case'].startswith('fit_')]
            def feasible(r):
                if r['completion_gain']<threshold4['completion']:return False
                if rule['rule']=='raw_patch':return r['patch_energy']<=threshold4['patch_energy']
                if r['mechanism_rgb_mse']>threshold4['mechanism_rgb_mse']:return False
                if rule['rule']!='mechanism_rgb' and r['unexplained_phase']>threshold4['unexplained_phase']:return False
                if rule['rule']=='plus_right_main' and abs(r['right_main_drift'])>threshold4['right_main']:return False
                if rule['rule']=='plus_right_conditional' and abs(r['right_conditional_drift'])>threshold4['right_conditional']:return False
                return True
            valid=[r for r in candidates if feasible(r)]
            chosen=min(valid,key=lambda r:r['rgb_mse']) if valid else next(r for r in candidates if r['case']=='source')
            assert chosen['case']==stored['case']
        assert rule['success']==sum(r['success'] for r in rule['selected'])
    audit.update(scope='Numerical/provenance audit only; not automatic editing validation',
                 audit_script_sha256=sha(Path(__file__)),
                 checks=checks,unique_synthetic_sources=len({i for i,_ in unique}),unique_rendered_images=len(unique),
                 source_groups_are_one_renderer=True,new_complete_sd3_trajectories=0,
                 v3_refined_coverage={split:dict(total=sum(r['split']==split for r in v3['rows']),
                     within_tolerance=sum(r['split']==split and r['within_reference_tolerance'] for r in v3['rows']))
                     for split in ['calibration','adaptation','heldout']},
                 vae_audit_scenes=[r['scene'] for r in v3['vae_rows']],natural_comparison=natural,
                 all_numerical_and_provenance_checks_passed=True)
    dest=OUT/'analysis';dest.mkdir(exist_ok=True)
    (dest/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    # Standalone scientific figure. Thresholds are independently calibrated for
    # each observable; this is not a claim that their raw scales are comparable.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results=load(OUT/'v5/results.json')['rows']
    cases=['correct','wrong_object','both_objects','turned','shifted','duplicate']
    fig,ax=plt.subplots(figsize=(10,4.8))
    for offset,(field,label,color) in zip([-.3,-.1,.1,.3],[('phase','Raw phase distance','#888888'),
        ('mechanism_rgb_mse','Unexplained RGB','#0072B2'),('unexplained_phase','Unexplained phase','#D55E00'),
        ('unexplained_complex','Unexplained complex response','#009E73')]):
        for k,case in enumerate(cases):
            values=np.array([r[field]/v5['thresholds'][field] for r in results if r['split']=='test' and r['case']==case])
            ax.scatter(k+offset+np.linspace(-.035,.035,len(values)),np.maximum(values,1e-7),s=17,alpha=.5,color=color,
                       label=label if k==0 else None)
            ax.plot([k+offset-.07,k+offset+.07],[np.median(values)]*2,lw=2,color=color)
    ax.axhline(1,color='black',linestyle='--',lw=1)
    ax.set_yscale('log');ax.set_xticks(range(len(cases)),['Legal','Wrong object','Both objects','Turn','Shift','Duplicate'])
    ax.set_ylabel('Residual / fixed legal-calibration budget')
    ax.set_title('V5: 8 renderer test scenes per case; value > 1 is rejected')
    ax.legend(fontsize=9);ax.grid(axis='y',alpha=.2);fig.tight_layout()
    fig.savefig(dest/'observation_comparison.png',dpi=180);fig.savefig(dest/'observation_comparison.svg');plt.close(fig)
    print(json.dumps({k:audit[k] for k in ['unique_synthetic_sources','unique_rendered_images','new_complete_sd3_trajectories','all_numerical_and_provenance_checks_passed']},ensure_ascii=False))


if __name__=='__main__':main()
