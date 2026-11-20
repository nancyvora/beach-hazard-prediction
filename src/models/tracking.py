"""Use an explicit local SQLite store by default; never a team's remote store by accident."""

import os

import mlflow

from src.models.config import EXPERIMENT


def configure_tracking(paths, experiment=EXPERIMENT):
    paths.artifacts.mkdir(parents=True, exist_ok=True)
    uri = os.environ.get(
        "MLFLOW_TRACKING_URI", f"sqlite:///{(paths.artifacts / 'mlflow.db').resolve()}"
    )
    mlflow.set_tracking_uri(uri)
    client = mlflow.MlflowClient()
    existing = client.get_experiment_by_name(experiment)
    if existing is None:
        experiment_id = client.create_experiment(
            experiment, artifact_location=(paths.artifacts / "mlruns").resolve().as_uri()
        )
    else:
        experiment_id = existing.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return client, experiment_id
