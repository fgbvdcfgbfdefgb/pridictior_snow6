#!/usr/bin/env python3
"""
split_large_file.py / join_large_file.py (both live in this one file)
-----------------------------------------------------------------------
GitHub hard-rejects any single file over 100MB (without Git LFS). Our
monthly Parquet files comfortably stay under that, but large model
checkpoints (the "large" PredictorConfig is ~200M parameters, ~0.8GB in
fp32) do not. Rather than requiring Git LFS (which may not be mirrored by
Snowflake's Git integration in every account), this project just splits any
oversized file into <95MB chunks that git/GitHub are happy with, and
provides a trivial reassembly step.

Usage:
    # split anything over the limit into part files sitting next to it
    python scripts/split_large_file.py split checkpoints/ema_final.pt

    # produces: checkpoints/ema_final.pt.part000, .part001, ... + a .manifest
    # commit the .part* files (NOT the original) to git.

    # reassemble after cloning the repo (works identically inside Snowflake,
    # it's just local file concatenation, no network involved):
    python scripts/split_large_file.py join checkpoints/ema_final.pt

run_demo.py / online_trainer.py / the Snowflake notebook all auto-join any
split checkpoint the first time it's needed (see src/checkpoint_utils.py),
so you normally never have to call this manually -- it's here for the data
pipeline and for manual use.
"""
import argparse
import hashlib
import json
import os

CHUNK_SIZE = 90 * 1024 * 1024  # 90MB, safely under GitHub's 100MB hard limit


def split_file(path: str, chunk_size: int = CHUNK_SIZE):
    size = os.path.getsize(path)
    if size <= chunk_size:
        print(f"{path} is {size/1e6:.1f}MB, already under the limit -- nothing to split.")
        return
    n_parts = 0
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha256.update(chunk)
            part_path = f"{path}.part{n_parts:03d}"
            with open(part_path, "wb") as pf:
                pf.write(chunk)
            print(f"  wrote {part_path} ({len(chunk)/1e6:.1f} MB)")
            n_parts += 1
    manifest = {"original_name": os.path.basename(path), "n_parts": n_parts,
                "total_bytes": size, "sha256": sha256.hexdigest()}
    with open(f"{path}.manifest.json", "w") as mf:
        json.dump(manifest, mf, indent=2)
    os.remove(path)
    print(f"Split {path} into {n_parts} parts. Original removed "
          f"(commit the .part* + .manifest.json files instead).")


def join_file(path: str):
    manifest_path = f"{path}.manifest.json"
    if not os.path.exists(manifest_path):
        if os.path.exists(path):
            return  # never split, nothing to do
        raise FileNotFoundError(f"Neither {path} nor {manifest_path} exist.")
    manifest = json.load(open(manifest_path))
    sha256 = hashlib.sha256()
    with open(path, "wb") as out:
        for i in range(manifest["n_parts"]):
            part_path = f"{path}.part{i:03d}"
            with open(part_path, "rb") as pf:
                data = pf.read()
                sha256.update(data)
                out.write(data)
    ok = sha256.hexdigest() == manifest["sha256"]
    print(f"Joined -> {path} ({manifest['total_bytes']/1e6:.1f} MB). "
          f"Checksum {'OK' if ok else 'MISMATCH!!'}")
    if not ok:
        raise RuntimeError("Checksum mismatch after joining parts -- re-download/re-clone.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["split", "join"])
    ap.add_argument("path")
    args = ap.parse_args()
    if args.action == "split":
        split_file(args.path)
    else:
        join_file(args.path)
