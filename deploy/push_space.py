"""Stage the Docker app and upload it to an existing Hugging Face Docker Space."""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def stage_source(destination, source_root=REPO_ROOT):
    """Copy only app sources; the Space gets its own README with Docker metadata."""
    destination = Path(destination)
    source_root = Path(source_root).resolve()
    files = {
        "Dockerfile": "Dockerfile",
        "requirements-serve.txt": "requirements-serve.txt",
        "README.md": "deploy/space_README.md",
    }
    for directory in ("src", "deploy"):
        for path in sorted((source_root / directory).rglob("*.py")):
            relative = path.relative_to(source_root).as_posix()
            files[relative] = relative
    if "src/api/main.py" not in files or "deploy/__init__.py" not in files:
        raise ValueError("Missing app source directories")
    # Validate all inputs before writing the package.
    for name in files.values():
        path = (source_root / name).resolve()
        if not path.is_relative_to(source_root) or not path.is_file():
            raise ValueError(f"Missing or unsafe source file: {name}")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Staging directory must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    for name, source in files.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / source, target)
    return sorted(files)


def upload_source(repo_id, token, source_root=REPO_ROOT, api=None):
    """Require an existing Docker Space; never create a paid resource implicitly."""
    if not repo_id or repo_id.count("/") != 1 or ":" in repo_id:
        raise ValueError("Set HF_SPACE_REPO to username/space-name, not a URL")
    if not token:
        raise ValueError("HF_TOKEN is missing; configure the GitHub Actions repository secret")
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
    info = api.space_info(repo_id=repo_id)
    if info.sdk != "docker":
        raise ValueError("The destination must be an existing Docker Space")
    with tempfile.TemporaryDirectory(prefix="beach-space-") as directory:
        stage_source(directory, source_root)
        return api.upload_folder(
            repo_id=repo_id,
            repo_type="space",
            folder_path=directory,
            commit_message="Deploy beach risk API source",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--repo-id", default=os.environ.get("HF_SPACE_REPO"))
    args = parser.parse_args()
    if args.dry_run:
        with tempfile.TemporaryDirectory(prefix="beach-space-") as directory:
            names = stage_source(directory)
            print("DRY RUN ONLY; no files uploaded. Source package:")
            print("\n".join(names))
        return
    try:
        commit = upload_source(args.repo_id, os.environ.get("HF_TOKEN"))
    except (ValueError, ImportError) as exc:
        parser.error(str(exc))
    print(f"Source uploaded: {commit.commit_url}")
    print("Model/history artifacts are not included; predictions need a separate artifact setup.")


if __name__ == "__main__":
    main()
