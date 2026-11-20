"""Spec 08: guard tracked source files, not generated local demo artifacts."""

import argparse
import subprocess
from pathlib import Path

MODEL_SUFFIXES = {".csv", ".parquet", ".pkl", ".pickle", ".joblib", ".onnx", ".pt", ".pth"}


def violations(entries):
    bad = []
    for name, size in entries:
        path = Path(name)
        reasons = []
        if path.parts[0] == "data" and path.name != ".gitkeep":
            reasons.append("tracked data")
        fixture = path.parts[:2] == ("tests", "fixtures")
        if path.suffix.lower() in MODEL_SUFFIXES and not fixture:
            reasons.append("data/model outside tests/fixtures")
        if size > 5 * 1024 * 1024:
            reasons.append("larger than 5 MB")
        if reasons:
            bad.append(f"{name}: {', '.join(reasons)}")
    return bad


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-untracked",
        action="store_true",
        help="also check pending source files, respecting gitignore",
    )
    args = parser.parse_args()
    command = ["git", "ls-files", "-z"]
    if args.include_untracked:
        command.extend(["--cached", "--others", "--exclude-standard"])
    names = sorted(set(subprocess.check_output(command).decode().split("\0")) - {""})
    if args.include_untracked:
        entries = [(name, Path(name).stat().st_size) for name in names if Path(name).is_file()]
    else:
        entries = [
            (name, int(subprocess.check_output(["git", "cat-file", "-s", f":{name}"])))
            for name in names
        ]
    bad = violations(entries)
    if bad:
        raise SystemExit("\n".join(bad))
    print(
        f"Data guard passed ({len(entries)} {'working-tree' if args.include_untracked else 'tracked'} files)."
    )


if __name__ == "__main__":
    main()
