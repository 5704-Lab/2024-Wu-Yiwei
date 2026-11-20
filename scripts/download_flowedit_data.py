"""Download the original, commit-pinned FlowEdit Data directory."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
from pathlib import Path
import time

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "flowedit"
API = "https://api.github.com/repos/fallenshock/FlowEdit"


def get(url: str) -> requests.Response:
    for attempt in range(5):
        try:
            response = requests.get(url, timeout=(15, 90))
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    pin = DEST / "provenance.json"
    if pin.exists():
        commit = json.loads(pin.read_text())["commit"]
    else:
        commit = get(API + "/commits/main").json()["sha"]
    tree = get(API + f"/git/trees/{commit}?recursive=1").json()
    if tree.get("truncated"):
        raise RuntimeError("GitHub tree is truncated")
    files = [
        item for item in tree["tree"]
        if item["type"] == "blob" and item["path"].startswith("Data/")
    ]

    def download(item: dict) -> dict:
        relative = Path(item["path"]).relative_to("Data")
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = target.read_bytes() if target.exists() else b""
        if git_blob_sha(payload) != item["sha"]:
            payload = get(
                f"https://raw.githubusercontent.com/fallenshock/FlowEdit/{commit}/{item['path']}"
            ).content
            if git_blob_sha(payload) != item["sha"]:
                raise RuntimeError(f"Git blob checksum mismatch: {relative}")
            temporary = target.with_suffix(target.suffix + ".part")
            temporary.write_bytes(payload)
            temporary.replace(target)
        row = {
            "path": relative.as_posix(),
            "size_bytes": len(payload),
            "git_blob_sha": item["sha"],
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        if relative.parts[0] == "Images":
            with Image.open(target) as im:
                row.update(width=im.width, height=im.height, mode=im.mode)
                im.verify()
        return row

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for i, row in enumerate(pool.map(download, files), 1):
            records.append(row)
            if i % 10 == 0:
                print(f"verified {i}/{len(files)} files", flush=True)
    provenance = {
        "repository": "https://github.com/fallenshock/FlowEdit",
        "commit": commit,
        "upstream_directory": "Data",
        "files": records,
        "total_bytes": sum(row["size_bytes"] for row in records),
        "license_note": "Images retain original-source licensing; see upstream README.md and dataset.csv.",
    }
    pin.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "destination": str(DEST),
        "commit": commit,
        "files": len(records),
        "images": sum(r["path"].startswith("Images/") for r in records),
        "bytes": provenance["total_bytes"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
