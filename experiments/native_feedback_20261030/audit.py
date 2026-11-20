"""Read locked calibration; report each independent source and each error type."""
import json
import torch
from probe import HERE, write
from analyze import metrics, summary


def main():
    torch.set_num_threads(4)
    out = HERE/'outputs/analysis_v1'
    limits = json.loads((out/'limits.json').read_text())
    all_results = {}
    for tag, minimum in [('native_v3', 8), ('native_fresh_v1', 0)]:
        dest = out/(tag+'_audit.json')
        if dest.exists():
            raise RuntimeError('Immutable audit exists')
        rows, supports = [], {}
        for folder in sorted((HERE/'outputs'/tag).glob('scene*')):
            scene = int(folder.name[5:])
            if scene < minimum:
                continue
            source = torch.load(folder/'source.pt', weights_only=True)
            for path in sorted(folder.glob('*.pt')):
                item = torch.load(path, weights_only=True)
                score, support = metrics(item, source)
                rows.append(dict(scene=scene, variant=path.stem, scores=score))
                supports[scene] = support
            print('AUDIT', tag, scene, flush=True)
        write(dest, dict(rows=rows, support=supports, results=summary(rows, limits),
            calibration_source='analysis_v1/limits.json', calibration_updated=False,
            independent_source_count=len(supports), not_bernoulli_independent_states=True))
        all_results[tag] = summary(rows, limits)
    print('AUDIT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
