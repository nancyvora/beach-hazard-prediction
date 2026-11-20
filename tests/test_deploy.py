import pytest

from deploy.fetch_artifacts import verify
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
