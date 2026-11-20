"""Separately registered lower-scale pilot after the coarse scaling audit."""
import hashlib,json
from pathlib import Path
import run_real as experiment

original_output=experiment.OUT
experiment.OUT=original_output/'refined';experiment.OUT.mkdir(exist_ok=True)
experiment.PAIRS=['bear__00','cat_crown__00']
experiment.CONFIG={**experiment.CONFIG,'sigmas':[.00625,.0125,.025],
 'edit_sigmas':[.00625,.0125],'extension_reason':'coarse quadratic scaling unsupported; lower-scale local diagnostic closer to quadratic',
 'cohort_scope':'two development-source refinement pilot; not heldout or confirmatory'}
(experiment.OUT/'toy.json').write_text((original_output/'toy.json').read_text())
(experiment.OUT/'extension_registration.json').write_text(json.dumps(dict(config=experiment.CONFIG,pairs=experiment.PAIRS,wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),original_full_protocol_sha256=hashlib.sha256((original_output/'protocol_full.json').read_bytes()).hexdigest(),refinement_evidence_sha256=hashlib.sha256((original_output/'scale_refinement.json').read_bytes()).hexdigest(),selection='bear and crown-cat fixed development cases; no per-image parameter choice'),indent=2)+'\n')
if __name__=='__main__':experiment.main()
