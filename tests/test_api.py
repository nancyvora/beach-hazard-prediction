from dataclasses import replace
from datetime import date

import numpy as np
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.service import PredictionService
from src.features.build import build_features, model_input


def test_pipeline_to_api_offline(demo_bundle):
    settings, smoke, _ = demo_bundle
    with TestClient(create_app(settings)) as client:
        health = client.get("/health")
        response = client.get("/predict", params=smoke)
        assert health.status_code == response.status_code == 200
        body = response.json()
        assert body["demo"] and "SYNTHETIC" in body["disclaimer"]
        assert body["coverage"]["n_predicted"] == len(body["stations"])
        assert body["max_station_probability"] == max(s["probability"] for s in body["stations"])
        assert body["model_version"] == health.json()["model_version"]
        assert client.get("/docs").status_code == 200


def test_api_validation_and_supported_dates(demo_bundle):
    settings, smoke, _ = demo_bundle
    with TestClient(create_app(settings)) as client:
        for query in [
            {},
            {"beach": -1, "date": smoke["date"]},
            {"beach": 1, "date": "2024-02-30"},
            {**smoke, "extra": "bad"},
            {**smoke, "date": "2999-01-01"},
        ]:
            response = client.get("/predict", params=query)
            assert response.status_code == 422
            assert response.json()["detail"]["code"] == "invalid_request"
        assert client.get("/predict", params={**smoke, "beach": 999999}).status_code == 404
        assert (
            client.get("/predict", params={**smoke, "date": "2024-01-01"}).json()["detail"]["code"]
            == "unsupported_date"
        )
        assert client.get("/predict", params={**smoke, "date": "2024-01-02"}).status_code == 200


def test_missing_artifact_is_not_a_dummy_model(demo_bundle, tmp_path):
    settings, smoke, _ = demo_bundle
    missing = replace(settings, feature_report=tmp_path / "missing.json")
    with TestClient(create_app(missing)) as client:
        assert client.get("/health").status_code == 503
        assert client.get("/predict", params=smoke).status_code == 503


def test_changed_feature_report_is_rejected(demo_bundle, tmp_path):
    import json

    settings, smoke, _ = demo_bundle
    changed = tmp_path / "feature_report.json"
    report = json.loads(settings.feature_report.read_text())
    report["history_delay_days"] = 0
    changed.write_text(json.dumps(report))
    with TestClient(create_app(replace(settings, feature_report=changed))) as client:
        assert client.get("/health").status_code == 503
        assert client.get("/predict", params=smoke).status_code == 503


def test_changed_model_signature_is_rejected(demo_bundle, tmp_path):
    import json
    import shutil

    import yaml

    settings, _, _ = demo_bundle
    model = tmp_path / "model"
    shutil.copytree(settings.model_uri, model)
    metadata_path = model / "MLmodel"
    metadata = yaml.safe_load(metadata_path.read_text())
    signature = json.loads(metadata["signature"]["inputs"])
    signature[0]["name"] = "wrong_feature"
    metadata["signature"]["inputs"] = json.dumps(signature)
    metadata_path.write_text(yaml.safe_dump(metadata))
    with TestClient(create_app(replace(settings, model_uri=str(model)))) as client:
        assert client.get("/health").status_code == 503


def test_no_history_and_unseen_county_are_still_scored(demo_bundle):
    settings, smoke, _ = demo_bundle
    service = PredictionService(settings)
    station_ids = service.catalog.loc[service.catalog.beach_id.eq(smoke["beach"]), "station_id"]
    service.history = service.history.loc[~service.history.station_id.isin(station_ids)].copy()
    service.catalog["county"] = "Not seen in training"
    output = service.predict(smoke["beach"], date.fromisoformat(smoke["date"]))
    assert all(
        s["history_status"] == "none" and s["n_prev_samples"] == 0 for s in output["stations"]
    )
    assert all(np.isfinite(s["probability"]) for s in output["stations"])


def test_model_selection_comparison_uses_same_validation_rows(demo_bundle):
    import pandas as pd

    _, _, root = demo_bundle
    comparison = pd.read_csv(root / "reports/models/selection_comparison.csv")
    assert comparison.n_rows.nunique() == comparison.prevalence.nunique() == 1


def test_batch_serving_parity_and_invalid_scores(demo_bundle):
    import pandas as pd

    settings, smoke, _ = demo_bundle
    service = PredictionService(settings)
    requested = date.fromisoformat(smoke["date"])
    catalog = service.catalog.loc[service.catalog.beach_id.eq(smoke["beach"])].sort_values(
        "station_id"
    )
    query = catalog[["station_id"]].assign(date=pd.Timestamp(requested))
    features = build_features(service.history, query, catalog)
    direct = service.model.predict(model_input(features))
    output = service.predict(smoke["beach"], requested)
    np.testing.assert_allclose(direct, [s["probability"] for s in output["stations"]], atol=1e-9)
    service.model.predict = lambda frame: np.full(len(frame), np.nan)
    import pytest

    from src.api.service import ServiceError

    with pytest.raises(ServiceError, match="Invalid model"):
        service.predict(smoke["beach"], requested)
