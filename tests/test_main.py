import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import FEATURE_COLUMNS, app, get_model, load_model

VALID_REQUEST = {
    "ticket_price_rs": 350,
    "days_since_release": 20,
    "star_rating": 4.2,
    "show_hour": 18,
    "screen_seats": 220,
    "distance_to_metro_km": 2.4,
    "language": "hindi",
    "location": "Mall",
}


class SyntheticModel:
    def __init__(self) -> None:
        self.received_features: pd.DataFrame | None = None

    def predict(self, features: pd.DataFrame) -> list[float]:
        self.received_features = features.copy()
        return [72.5]


@pytest.fixture
def api_client():
    model = SyntheticModel()
    app.dependency_overrides[get_model] = lambda: model
    with TestClient(app) as client:
        yield client, model
    app.dependency_overrides.clear()


def test_health_returns_ok(api_client) -> None:
    client, _ = api_client

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_uses_exact_fields_and_returns_prediction(api_client) -> None:
    client, model = api_client

    response = client.post("/predict", json=VALID_REQUEST)

    assert response.status_code == 200
    assert response.json() == {"prediction": 72.5}
    assert model.received_features is not None
    assert list(model.received_features.columns) == list(FEATURE_COLUMNS)
    assert model.received_features.iloc[0].to_dict() == VALID_REQUEST


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("days_since_release", 71),
        ("screen_seats", 39),
    ],
)
def test_predict_rejects_client_rule_violations(
    api_client, field: str, invalid_value: int
) -> None:
    client, model = api_client
    request = {**VALID_REQUEST, field: invalid_value}

    response = client.post("/predict", json=request)

    assert response.status_code == 422
    assert model.received_features is None


def test_predict_rejects_extra_fields(api_client) -> None:
    client, model = api_client

    response = client.post(
        "/predict",
        json={**VALID_REQUEST, "record_id": "R00001"},
    )

    assert response.status_code == 422
    assert model.received_features is None


def test_predict_reports_missing_model_as_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    app.dependency_overrides.clear()
    load_model.cache_clear()
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing-model.joblib"))
    try:
        with TestClient(app) as client:
            response = client.post("/predict", json=VALID_REQUEST)
    finally:
        load_model.cache_clear()

    assert response.status_code == 503
