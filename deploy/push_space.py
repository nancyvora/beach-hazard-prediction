"""Inspect the intended Space upload. Remote publishing is not part of the initial scaffold."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run:
        parser.error("Hub publishing is pending; use --dry-run to inspect the source package")
    names = ["Dockerfile", "requirements-serve.txt", "deploy/space_README.md"]
    names += [str(path) for path in Path("src").rglob("*.py")]
    missing = [name for name in names if not Path(name).is_file()]
    if missing:
        raise SystemExit(f"Missing source files: {missing}")
    print("DRY RUN ONLY; no files uploaded. Source package:")
    print("\n".join(sorted(names)))


if __name__ == "__main__":
    main()
