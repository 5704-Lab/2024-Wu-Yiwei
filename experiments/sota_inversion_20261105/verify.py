"""Verify provenance, actual state integrity, NFE and the predeclared fidelity gate."""
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
import torch
from PIL import Image
from common import ROOT,HERE,sha,write


def main():
    out=HERE/'outputs/comparison_v1';dest=HERE/'outputs/verification_v1'
    if dest.exists():raise RuntimeError('Verification output exists; use a fresh version')
    dest.mkdir(parents=True)
    protocol=json.loads((out/'protocol.json').read_text())
    complete=json.loads((out/'complete.json').read_text())
    audit=json.loads((out/'weight_layout_audit.json').read_text())
    measured=json.loads((out/'metrics.json').read_text())
    payload=torch.load(out/'results.pt',map_location='cpu',weights_only=True)
    inp=torch.load(HERE/'outputs/inputs_v1/inputs.pt',map_location='cpu',weights_only=True)
    assert sha(HERE/'outputs/inputs_v1/inputs.pt')==protocol['input_sha256']
    assert sha(HERE/'run_comparison.py')==protocol['script_sha256']==complete['script_sha256']
    assert sha(HERE/'common.py')==protocol['common_sha256']
    assert sha(ROOT/'vendor/selfix/src/flux/sampling.py')==protocol['official_solver_sha256']
    assert sha(ROOT/'vendor/selfix/src/flux/modules/layers.py')==protocol['official_layers_sha256']
    repo=ROOT/'vendor/selfix'
    commit=subprocess.check_output(['git','-c','safe.directory='+str(repo),'-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    assert commit==protocol['official_selfix_commit']
    subprocess.check_call(['git','-c','safe.directory='+str(repo),'-C',str(repo),'diff','--exit-code'])
    assert audit['layout_roundtrip_bitwise'] and audit['tensors_verified']==1160 and audit['original_module_strict_loading']
    assert torch.equal(payload['states']['source'],inp['inputs']['source']['img'])
    assert torch.equal(inp['inputs']['source']['img'],inp['inputs']['target']['img'])
    for name,state in payload['states'].items():
        assert state.shape==(1,4096,64) and bool(torch.isfinite(state).all()),name
    for name,image in payload['images'].items():
        assert image.shape==(1,3,1024,1024) and bool(torch.isfinite(image).all()),name
        assert 0<=float(image.min())<=float(image.max())<=1
    expected=dict(selfix_default_inversion=60,selfix_default_reconstruction=15,selfix_default_g2p0=15,
        selfix_high_fidelity_inversion=150,selfix_high_fidelity_reconstruction=15,
        selfix_high_fidelity_g2p0=15,selfix_high_fidelity_g3p5=15,flowedit_author=48,flowedit_guidance_matched=48)
    assert complete['nfe']==expected and complete['total_nfe']==sum(expected.values())==381
    gate=measured['reconstruction_gates']['selfix_high_fidelity']
    assert gate['passed'] and gate['psnr_db']>=28 and gate['ssim']>=.9
    assert gate['same_inverse_as_edits'] and gate['source_reconstruction_without_feature_injection']
    assert measured['script_sha256']==sha(HERE/'evaluate.py')
    references=json.loads((HERE/'outputs/text_input_audit.json').read_text())
    for encoder in ['clip','t5']:
        assert references[encoder]['cached_ids_match'] and not any(references[encoder]['truncated'])
    for name in ['comparison.png','all_recipes.png','reconstruction.png','guidance_controls.png']:
        im=Image.open(out/name);im.verify()
    report=ROOT/'docs/FlowEdit与SelFix高保真反演编辑_同FLUX真实图像对照_2026-11-05.md'
    for path in [report,HERE/'README.md']:
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if not target.startswith(('http','/','#')):assert (path.parent/target).exists(),target
    snapshot=dest/'code_snapshot';snapshot.mkdir()
    for path in HERE.glob('*.py'):shutil.copy2(path,snapshot/path.name)
    for name in ['sampling.py','model.py']:
        shutil.copy2(repo/'src/flux'/name,snapshot/('official_'+name))
    shutil.copy2(repo/'src/flux/modules/layers.py',snapshot/'official_layers.py')
    shutil.copy2(repo/'configs/pie_editing_selfix.yaml',snapshot/'official_editing.yaml')
    shutil.copy2(repo/'configs/pie_reconstruction_selfix.yaml',snapshot/'official_reconstruction.yaml')
    files=list(HERE.glob('*.py'))+[HERE/'sources.json',HERE/'README.md',report]
    files+=list(out.glob('*.json'))+list(out.glob('*.png'))+[out/'results.pt']
    files+=list((HERE/'outputs').glob('*.json'))
    write(dest/'artifact_sha256.json',{str(p.relative_to(ROOT)):sha(p) for p in files})
    summary=dict(status='passed',source_count=1,model='same frozen FLUX.1-dev',resolution=1024,
        official_commit=commit,official_solver_and_injection_unmodified=True,weight_layout_tensor_count=1160,
        all_saved_states_finite=True,input_source_unchanged=True,complete_text_conditions_verified=True,
        total_nfe=381,reconstruction_gate=gate,
        fidelity_reference='unchanged BF16 packed source decoded with the common VAE',
        vs_original=measured['metrics']['selfix_high_fidelity_reconstruction']['vs_original'],
        source_count_limit='one exploratory source; not a general superiority benchmark')
    write(dest/'result.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
