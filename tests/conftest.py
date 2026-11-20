"""All test artifacts are temporary; the production data directory is never read."""

import os

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import pytest

from src.data.cleaning import clean_results
from src.data.events import station_days
from src.demo import synthetic_raw


@pytest.fixture
def raw_rows():
    return synthetic_raw().iloc[:5].copy()


@pytest.fixture
def five_events(raw_rows):
    raw_rows["SampleDate"] = ["2024-01-01", "2024-01-03", "2024-01-05", "2024-01-06", "2024-01-07"]
    raw_rows["Result"] = [10, 100, 1000, 10000, 100000]
    raw_rows["Qualifier"] = "="
    clean, _ = clean_results(raw_rows, run_date="2026-11-19")
    return station_days(clean)


@pytest.fixture(scope="session")
def demo_bundle(tmp_path_factory):
    from src.demo import run

    root = tmp_path_factory.mktemp("offline-demo")
    previous = os.environ.get("MLFLOW_TRACKING_URI")
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{root / 'tracking.db'}"
    try:
        settings, smoke = run(root / "run", include_eda=False)
        yield settings, smoke, root / "run"
    finally:
        if previous is None:
            os.environ.pop("MLFLOW_TRACKING_URI", None)
        else:
            os.environ["MLFLOW_TRACKING_URI"] = previous


@pytest.fixture(autouse=True)
def block_external_network(monkeypatch):
    import socket

    def blocked(*args, **kwargs):
        raise AssertionError("Tests must not open network connections")

    monkeypatch.setattr(socket, "create_connection", blocked)
