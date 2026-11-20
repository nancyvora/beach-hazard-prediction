from pathlib import Path
from types import SimpleNamespace

import pytest

from deploy.fetch_artifacts import verify
from deploy.push_space import upload_source
from scripts.check_repo_files import violations


def test_export_manifest_and_tampering(demo_bundle, tmp_path):
    import shutil

    settings, _, _ = demo_bundle
    root = settings.history.parent
    assert verify(root)["demo"] is True
    copied = tmp_path / "export"
    shutil.copytree(root, copied)
    (copied / "history_metadata.json").write_text("{}")
    with pytest.raises(ValueError, match="modified artifact"):
        verify(copied)


def test_data_guard_reports_every_offender():
    bad = violations(
        [
            ("data/raw/x.csv", 1),
            ("data/.gitkeep", 0),
            ("tests/fixtures/x.parquet", 10),
            ("src/x.parquet", 10),
            ("large.md", 6 * 1024 * 1024),
        ]
    )
    assert len(bad) == 3
    assert all(
        any(name in message for message in bad)
        for name in ["data/raw/x.csv", "src/x.parquet", "large.md"]
    )


def test_space_upload_stages_docker_app_without_local_data(tmp_path):
    sources = {
        "Dockerfile": "COPY src ./src\nCOPY deploy ./deploy\n",
        "requirements-serve.txt": "fastapi\n",
        "deploy/space_README.md": "---\nsdk: docker\napp_port: 7860\n---\n",
        "deploy/__init__.py": "",
        "deploy/fetch_artifacts.py": "# verify export\n",
        "src/api/main.py": "# app\n",
        "README.md": "GitHub README, not the Space README",
        ".env": "HF_TOKEN=not-a-real-token",
        "DSC200-Proposal.pdf": "class document",
        "data/raw/observations.csv": "raw data",
        "artifacts/model.pkl": "model bytes",
        "src/__pycache__/main.pyc": "bytecode",
    }
    for name, content in sources.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    calls = []

    class FakeHub:
        def space_info(self, *, repo_id):
            assert repo_id == "team/beach-risk"
            return SimpleNamespace(sdk="docker")

        def upload_folder(self, *, repo_id, repo_type, folder_path, commit_message):
            assert repo_id == "team/beach-risk" and repo_type == "space"
            root = Path(folder_path)
            uploaded = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
            assert uploaded == {
                "Dockerfile",
                "requirements-serve.txt",
                "README.md",
                "deploy/__init__.py",
                "deploy/fetch_artifacts.py",
                "src/api/main.py",
            }
            assert (root / "README.md").read_text() == sources["deploy/space_README.md"]
            calls.append(commit_message)
            return SimpleNamespace(commit_url="https://huggingface.co/spaces/team/beach-risk")

    result = upload_source("team/beach-risk", "fake-token", tmp_path, api=FakeHub())
    assert result.commit_url.endswith("team/beach-risk")
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("repo_id", "token", "sdk", "message"),
    [
        (None, "fake-token", "docker", "HF_SPACE_REPO"),
        ("https://huggingface.co/spaces/team/beach-risk", "fake-token", "docker", "not a URL"),
        ("team/beach-risk", None, "docker", "HF_TOKEN is missing"),
        ("team/beach-risk", "fake-token", "gradio", "Docker Space"),
    ],
)
def test_space_upload_rejects_missing_config_or_wrong_sdk(repo_id, token, sdk, message):
    class FakeHub:
        def space_info(self, *, repo_id):
            return SimpleNamespace(sdk=sdk)

        def upload_folder(self, **kwargs):
            pytest.fail("Invalid deployment must not upload")

    with pytest.raises(ValueError, match=message):
        upload_source(repo_id, token, api=FakeHub())
