"""Prepare prompt-only manifests without changing original FlowEdit assets."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import random
from collections import Counter
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "flowedit"
ALIASES = {"meditation1.png": "meditation2.png"}


def main():
    entries = yaml.safe_load((DATA / "flowedit.yaml").read_text())
    provenance = json.loads((DATA / "provenance.json").read_text())
    records, groups, mappings = [], {}, []
    for index, entry in enumerate(entries):
        requested = Path(entry["init_img"]).name
        resolved = ALIASES.get(requested, requested)
        path = DATA / "Images" / resolved
        if not path.is_file():
            raise FileNotFoundError(path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        groups.setdefault(sha, []).append(index)
        if requested != resolved:
            mappings.append({
                "requested": requested, "resolved": resolved,
                "evidence": "Upstream dataset.csv names meditation2; source prompt and inspected image both depict a man meditating.",
            })
        if len(entry["target_prompts"]) != len(entry["target_codes"]):
            raise ValueError(f"prompt/code mismatch at {index}")
        for target_index, (prompt, code) in enumerate(zip(entry["target_prompts"], entry["target_codes"])):
            records.append({
                "pair_id": f"{Path(resolved).stem}__{target_index:02d}",
                "upstream_entry": index,
                "source_image": path.relative_to(ROOT).as_posix(),
                "source_sha256": sha,
                "source_prompt": entry["source_prompt"],
                "target_prompt": prompt,
                "target_code": code,
            })
    ordered = sorted(groups)
    random.Random(20261005).shuffle(ordered)
    # Source groups, not edit pairs, are the split unit.
    counts = {"development": 24, "mechanism": 16, "heldout": len(ordered) - 40}
    split_by_sha, cursor = {}, 0
    for split, count in counts.items():
        for sha in ordered[cursor:cursor + count]:
            split_by_sha[sha] = split
        cursor += count
    for row in records:
        row["split"] = split_by_sha[row["source_sha256"]]
    output = DATA / "manifests"
    output.mkdir(exist_ok=True)
    for name, rows in [("all_pairs", records)] + [
        (split, [r for r in records if r["split"] == split]) for split in counts
    ]:
        (output / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        )
    # One deterministic target per development source for the initial compute audit.
    pilot = []
    for sha in ordered[:24]:
        options = [r for r in records if r["source_sha256"] == sha]
        pilot.append(sorted(options, key=lambda r: r["pair_id"])[0])
    (output / "stage1_pilot.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pilot)
    )
    summary = {
        "upstream_commit": provenance["commit"],
        "images": len(groups),
        "pairs": len(records),
        "source_groups_by_split": counts,
        "pairs_by_split": dict(Counter(r["split"] for r in records)),
        "stage1_pilot_pairs": len(pilot),
        "path_aliases": mappings,
        "split_seed": 20261005,
        "masks_used": False,
        "split_note": "Official FlowEdit lacks PIE categories; seeded source-hash grouping replaces the earlier PIE stratification plan.",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
