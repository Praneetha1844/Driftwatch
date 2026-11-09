import numpy as np
import pytest
from fastapi.testclient import TestClient

from driftwatch.serving import create_app
from driftwatch.synthetic import make_dataset

FEATURES = ["tenure_months", "monthly_spend", "support_calls", "plan"]


class FakeModel:
    """Stands in for a trained pipeline: short-tenure customers get a high score."""

    version = "7"
    feature_names = FEATURES

    def predict_proba(self, frame) -> np.ndarray:
        return np.where(frame["tenure_months"].to_numpy() < 12, 0.9, 0.1)


def records(n: int, seed: int = 5) -> list[dict]:
    return make_dataset(n, seed=seed).drop(columns="churned").to_dict("records")


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(FakeModel()))


def test_health_reports_the_model_version(client) -> None:
    assert client.get("/health").json() == {"status": "ok", "model_version": "7"}


def test_predict_contract(client) -> None:
    rows = records(2)
    rows[0]["tenure_months"], rows[1]["tenure_months"] = 3.0, 30.0
    response = client.post("/predict", json={"records": rows})
    body = response.json()
    assert response.status_code == 200 and body["model_version"] == "7"
    assert [p["label"] for p in body["predictions"]] == [1, 0]
    assert body["predictions"][0]["probability"] == pytest.approx(0.9)


def test_missing_or_null_features_are_rejected(client) -> None:
    row = records(1)[0]
    del row["plan"]
    response = client.post("/predict", json={"records": [row]})
    assert response.status_code == 422
    assert response.json()["detail"]["missing_or_null_features"] == ["plan"]

    null_row = records(1)[0] | {"monthly_spend": None}
    assert client.post("/predict", json={"records": [null_row]}).status_code == 422


def test_empty_request_is_rejected(client) -> None:
    assert client.post("/predict", json={"records": []}).status_code == 422


def test_metrics_count_scored_and_rejected_rows(client) -> None:
    client.post("/predict", json={"records": records(3)})
    client.post("/predict", json={"records": [{"tenure_months": 1}]})
    text = client.get("/metrics").text
    assert 'driftwatch_predictions_total{model_version="7"} 3.0' in text
    assert "driftwatch_rejected_requests_total 1.0" in text
    assert 'driftwatch_model_info{model_version="7"} 1.0' in text


def test_two_apps_do_not_share_metric_registries() -> None:
    first, second = TestClient(create_app(FakeModel())), TestClient(create_app(FakeModel()))
    first.post("/predict", json={"records": records(2)})
    assert 'driftwatch_predictions_total{model_version="7"} 2.0' in first.get("/metrics").text
    assert 'driftwatch_predictions_total{model_version="7"}' not in second.get("/metrics").text


def test_drift_endpoint_requires_configuration(client) -> None:
    assert client.get("/drift").status_code == 503


def test_drift_endpoint_waits_for_enough_traffic(settings) -> None:
    app = create_app(FakeModel(), make_dataset(300, seed=1), settings)
    api = TestClient(app)
    api.post("/predict", json={"records": records(5)})
    body = api.get("/drift").json()
    assert body["status"] == "insufficient_data" and body["n_current"] == 5


def test_drift_endpoint_reports_and_exports_psi(settings) -> None:
    config = settings.model_copy(deep=True)
    config.drift.min_samples = 20
    api = TestClient(create_app(FakeModel(), make_dataset(400, seed=1), config))
    api.post("/predict", json={"records": records(60, seed=9)})
    body = api.get("/drift").json()
    assert body["status"] in {"ok", "warn", "alert"}
    assert {f["feature"] for f in body["features"]} == set(FEATURES)
    assert 'driftwatch_feature_psi{feature="tenure_months"}' in api.get("/metrics").text
