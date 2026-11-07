from datetime import datetime, timedelta

import pytest

from backend.tests.test_api import DASH_DIR, fresh_client

pytestmark = pytest.mark.skipif(not (DASH_DIR / "recent_osisaf.npz").is_file(),
                                reason="需要随完整运行包提供的近期观测缓存")


@pytest.mark.parametrize("model", ["simvp", "ridge", "random_forest"])
def test_recent_forecast_runs_past_archive_and_matches_requested_lead(fresh_client, model):
    metadata = fresh_client.get("/api/recent").json()
    start = datetime.fromisoformat(metadata["latest_at"])
    req = {"source": "recent", "date": "latest", "model": model, "lead_days": 3,
           "cell": metadata["default_cell"]}
    response = fresh_client.post("/api/predict", json=req)
    assert response.status_code == 200, response.text
    forecast = response.json()
    assert forecast["data_time"]["observation_end_at"] == start.isoformat()
    assert forecast["data_time"]["forecast_valid_until"] == (start + timedelta(days=3)).isoformat()
    assert forecast["metrics"] is None
    assert [day["lead_hours"] for day in forecast["daily_forecasts"]] == [24, 48, 72]
    assert forecast["series_horizon"]["hours"] == list(range(73))
    assert forecast["series_24h"]["hours"] == list(range(25))
    response = fresh_client.get("/api/recent/field",
                                params={"model": model, "lead_days": 3, "date": metadata["latest_date"]})
    assert response.status_code == 200, response.text
    assert response.json()["valid_at"] == forecast["data_time"]["forecast_valid_until"]


def test_recent_uses_explicit_observation_batch(fresh_client):
    response = fresh_client.post("/api/predict", json={
        "source": "recent", "date": "2025-12-30", "model": "ridge", "cell": {"row": 65, "col": 60},
    })
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "recent_date_invalid"
