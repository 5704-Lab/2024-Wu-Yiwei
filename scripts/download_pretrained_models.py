"""Download original SD3 Medium and FLUX.1-dev diffusers components from ModelScope.

Files are selected from a saved manifest, downloaded at immutable per-file
revisions, resumed when possible, and verified against upstream SHA256 values.
Duplicate fp16 variants and duplicate monolithic checkpoints are not required.
"""
from __future__ import annotations
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "pretrained_models"
MODELS = {
    "sd3-medium": "stabilityai/stable-diffusion-3-medium-diffusers",
    "flux1-dev": "black-forest-labs/FLUX.1-dev",
}
PARTS = {"scheduler", "tokenizer", "tokenizer_2", "tokenizer_3", "text_encoder", "text_encoder_2", "text_encoder_3", "transformer", "vae"}
LOCK = threading.Lock()


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result


def manifest(name, model):
    destination = BASE / name
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "download_manifest.json"
    if path.exists():
        return json.loads(path.read_text())
    response = requests.get(
        f"https://www.modelscope.cn/api/v1/models/{model}/repo/files",
        params={"Revision": "master", "Recursive": "true"}, timeout=(15, 90),
    )
    response.raise_for_status()
    body = response.json()
    if body.get("Code") != 200:
        raise RuntimeError(f"ModelScope listing failed for {model}: {body.get('Message')}")
    files = []
    for row in body["Data"]["Files"]:
        p = Path(row["Path"])
        if row["Type"] != "blob":
            continue
        if not (p.parts[0] in PARTS or row["Path"] in {"model_index.json", "README.md", "LICENSE", "LICENSE.md", "configuration.json"}):
            continue
        if ".fp16" in p.name:
            continue
        if not row.get("Sha256"):
            raise RuntimeError(f"Missing upstream SHA256 for {row['Path']}")
        files.append({
            "path": row["Path"], "bytes": row["Size"],
            "sha256": row["Sha256"], "revision": row["Revision"],
        })
    result = {
        "model_id": model, "source": f"https://www.modelscope.cn/models/{model}",
        "directory": str(destination), "listing_branch": "master",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": files, "total_bytes": sum(r["bytes"] for r in files),
        "selection": "Complete diffusers pipeline components, no quantization; duplicate fp16 filename variants and monolithic duplicate checkpoints excluded.",
    }
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def download_file(model, name, row):
    target = BASE / name / row["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == row["bytes"]:
        if digest(target).hexdigest() == row["sha256"]:
            return {"model": name, "path": row["path"], "bytes": row["bytes"], "verified": True, "reused": True}
    partial = target.with_suffix(target.suffix + ".partial")
    endpoint = f"https://www.modelscope.cn/api/v1/models/{model}/repo"
    error = ""
    for attempt in range(6):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > row["bytes"]:
            partial.unlink()
            offset = 0
        hasher = digest(partial) if offset else hashlib.sha256()
        if offset == row["bytes"]:
            if hasher.hexdigest() == row["sha256"]:
                partial.replace(target)
                break
            partial.unlink()
            offset, hasher = 0, hashlib.sha256()
        try:
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with requests.get(
                endpoint, params={"Revision": row["revision"], "FilePath": row["path"]},
                headers=headers, stream=True, timeout=(20, 120),
            ) as response:
                response.raise_for_status()
                if offset and response.status_code == 200:
                    offset, hasher = 0, hashlib.sha256()
                elif offset and response.status_code == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise RuntimeError("invalid resume range")
                mode = "ab" if offset else "wb"
                last_log, written = time.monotonic(), offset
                with partial.open(mode) as handle:
                    for chunk in response.iter_content(8 * 1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        hasher.update(chunk)
                        written += len(chunk)
                        if written > row["bytes"]:
                            raise RuntimeError("response exceeds manifest size")
                        if time.monotonic() - last_log > 30:
                            with LOCK:
                                print(f"{name}/{row['path']}: {written / 1e9:.2f}/{row['bytes'] / 1e9:.2f} GB", flush=True)
                            last_log = time.monotonic()
                if written != row["bytes"]:
                    raise RuntimeError(f"incomplete size {written}/{row['bytes']}")
                if hasher.hexdigest() != row["sha256"]:
                    partial.unlink()
                    raise RuntimeError("SHA256 mismatch")
            partial.replace(target)
            break
        except (requests.RequestException, RuntimeError, OSError) as exc:
            error = type(exc).__name__ + ": " + str(exc).split("https://")[0][:200]
            if attempt == 5:
                raise RuntimeError(f"{name}/{row['path']} failed: {error}") from None
            with LOCK:
                print(f"retry {name}/{row['path']}: {error}", flush=True)
            time.sleep(min(2 ** attempt, 16))
    record = {"model": name, "path": row["path"], "bytes": row["bytes"], "verified": True, "reused": False}
    with LOCK:
        print(f"verified {name}/{row['path']} ({row['bytes'] / 1e9:.3f} GB)", flush=True)
    return record


def main():
    BASE.mkdir(exist_ok=True)
    manifests = {name: manifest(name, model) for name, model in MODELS.items()}
    needed = sum(r["bytes"] for m in manifests.values() for r in m["files"]
                 if not (BASE / Path(m["directory"]).name / r["path"]).exists())
    if shutil.disk_usage(BASE).free < needed + 5 * 1024 ** 3:
        raise RuntimeError("Insufficient disk space for model downloads")
    print(json.dumps({name: {"files": len(m["files"]), "GB": round(m["total_bytes"] / 1e9, 3)} for name, m in manifests.items()}), flush=True)
    tasks = [
        (name, m["model_id"], row)
        for name, m in manifests.items()
        for row in m["files"]
    ]
    tasks.sort(key=lambda x: (x[2]["bytes"] > 10_000_000, list(MODELS).index(x[0]), -x[2]["bytes"]))
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(download_file, model, name, row) for name, model, row in tasks]
        for future in concurrent.futures.as_completed(futures):
            records.append(future.result())
            (BASE / "download_progress.json").write_text(json.dumps({
                "verified_files": len(records),
                "total_files": len(tasks),
                "verified_bytes": sum(r["bytes"] for r in records),
                "total_bytes": sum(m["total_bytes"] for m in manifests.values()),
                "records": records,
            }, indent=2) + "\n")
    for name, m in manifests.items():
        complete = {
            "model_id": m["model_id"], "source": m["source"],
            "complete": True, "files": len(m["files"]),
            "total_bytes": m["total_bytes"],
            "verification": "Every selected file matched the upstream SHA256 and byte size.",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        (BASE / name / "verification.json").write_text(json.dumps(complete, indent=2) + "\n")
    print("Both model pipelines downloaded and verified.", flush=True)


if __name__ == "__main__":
    main()
