"""Save exactly the calibrated pipeline that was evaluated."""

import subprocess
from importlib.metadata import version as dependency_version

import mlflow
from mlflow.models import infer_signature


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def register_model(wrapper, example, name, tags, alias="champion"):
    probabilities = wrapper.predict(None, example)
    packages = ["mlflow", "scikit-learn", "pandas", "numpy"]
    if wrapper.pipeline.named_steps["classifier"].__class__.__module__.startswith("lightgbm"):
        packages.append("lightgbm")
    info = mlflow.pyfunc.log_model(
        name="model",
        python_model=wrapper,
        input_example=example,
        signature=infer_signature(example, probabilities),
        code_paths=["src"],
        pip_requirements=[f"{package}=={dependency_version(package)}" for package in packages],
    )
    version = mlflow.register_model(info.model_uri, name)
    client = mlflow.MlflowClient()
    for key, value in tags.items():
        client.set_model_version_tag(name, version.version, key, str(value))
    client.set_registered_model_alias(name, alias, version.version)
    return str(version.version)
