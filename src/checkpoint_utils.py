"""
checkpoint_utils.py
--------------------
Thin convenience wrapper so training/demo/notebook code never has to think
about whether a checkpoint was committed whole or split into <100MB parts
(see scripts/split_large_file.py). Call `resolve_checkpoint(path)` before
`torch.load(...)`.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))


def resolve_checkpoint(path: str) -> str:
    """If `path` doesn't exist but `path.manifest.json` + `.part*` files do,
    reassemble it first. Returns `path` either way."""
    manifest_path = f"{path}.manifest.json"
    if not os.path.exists(path) and os.path.exists(manifest_path):
        from split_large_file import join_file
        join_file(path)
    return path
