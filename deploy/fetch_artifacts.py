"""Verify an already-downloaded local export. Hub fetching is a later integration."""

import argparse
from pathlib import Path

from src.common import read_json, sha256


def verify(root):
    root = Path(root).resolve()
    manifest = read_json(root / "manifest.json")
    for relative, expected in manifest["files"].items():
        target = (root / relative).resolve()
        if not target.is_relative_to(root) or not target.is_file() or sha256(target) != expected:
            raise ValueError(f"Missing, unsafe or modified artifact: {relative}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    print(verify(parser.parse_args().root)["model_version"])
