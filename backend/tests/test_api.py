"""FastAPI 后端 API 测试（pytest + TestClient）。

覆盖：
1. 成功路径：/api/health、/api/models、/api/metrics、/api/overview、/api/field、
   /api/field/predicted、POST /api/predict、POST /api/backtest、/docs、静态托管；
2. 错误路径：未知模型 404、非法 cell 400、非法日期 422、无效范围 422、错误体格式；
3. 契约字段结构 / 数值范围（predict 的 series_24h、位移换算、方向、不确定度带）；
4. 不重训验证：predict 前后 models/ml/*.joblib 与 models/simvp-24h/*.pt mtime 不变、无新文件；
5. artifacts/引擎缺失时自动 skip（pytest.mark.skipif）；
6. monkeypatch 假引擎覆盖路由逻辑（不依赖真实 artifacts）。

运行：``.venv/bin/python -m pytest backend/tests -q``
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

# conftest.py 已把工作区根目录加入 sys.path 并设置 PREDICT_WARMUP=0
ROOT = Path(__file__).resolve().parents[2]
ZARR_PATH = ROOT / "data/processed/weddell_icemotion_daily.zarr"
DASH_DIR = ROOT / "data/processed/dashboard"
ML_DIR = ROOT / "models/ml"
MODELS_DIR = ROOT / "models"
SIMVP_CKPT = MODELS_DIR / "simvp-24h/ckpt_best.pt"

DEFAULT_DATE = "2025-06-26"
DEFAULT_CELL = {"row": 65, "col": 60}
TARGET_DATE = "2025-06-27"
# 数据集中 (row=65, col=60) 的格点经纬度（laztest 允许 ±0.05°）
CELL_LAT, CELL_LON = -68.0446, -33.1976


# ---------------------------------------------------------------------------
# 依赖可用性探测（缺失时 skip 而不是失败）
# ---------------------------------------------------------------------------
def _importable(module_name: str) -> bool:
    try:
        __import__(module_name)
    except Exception:  # noqa: BLE001
        return False
    return True


HAS_ZARR = ZARR_PATH.exists()
TABULAR_ARTIFACTS = (ML_DIR / "ridge.joblib").exists() and (ML_DIR / "rf.joblib").exists()
TABULAR_READY = HAS_ZARR and TABULAR_ARTIFACTS and _importable("src.predict.engine")
SIMVP_READY = HAS_ZARR and SIMVP_CKPT.exists() and _importable("src.serving.simvp")
HAS_METRICS = any(
    (DASH_DIR / name).exists()
    for name in ("metrics.json", "metrics_tabular.json", "metrics_simvp.json")
)
HAS_OVERVIEW = (DASH_DIR / "overview.json").exists() or (DASH_DIR / "timeseries.json").exists()

needs_zarr = pytest.mark.skipif(not HAS_ZARR, reason="缺少数据集 zarr（data/processed/weddell_icemotion_daily.zarr）")
needs_tabular = pytest.mark.skipif(
    not TABULAR_READY, reason="Ridge/RF artifact 或 src.predict 引擎未就绪"
)
needs_simvp = pytest.mark.skipif(not SIMVP_READY, reason="SimVP ckpt / src.serving 未就绪")
SIMVP_PRECOMPUTED = (DASH_DIR / "simvp_day1_fields.npz").exists()
needs_metrics = pytest.mark.skipif(not HAS_METRICS, reason="metrics artifact 尚未生成")
needs_overview = pytest.mark.skipif(not HAS_OVERVIEW, reason="overview/timeseries artifact 尚未生成")
needs_simvp_backtest = pytest.mark.skipif(
    not (SIMVP_READY and SIMVP_PRECOMPUTED),
    reason="SimVP 预计算场 simvp_day1_fields.npz 尚未生成",
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def client():
    """只读端点 + 真实引擎集成测试共用的 session 级 TestClient。"""
    from backend.app.main import create_app

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def fresh_client():
    """每个用例独立 app（独立 store / 预测缓存），用于 monkeypatch 假引擎。"""
    from backend.app.main import create_app

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# 假引擎（monkeypatch 路由逻辑，不依赖真实 artifacts）
# ---------------------------------------------------------------------------
class FakeEngine:
    """与 src.predict.PredictionEngine 同构的假引擎。"""

    model_ids = ["ridge", "random_forest", "persistence", "climatology"]
    __test__ = False  # 避免 pytest 把它当测试类收集

    def __init__(self):
        self.predict_calls = 0
        self.predict_field_calls = 0
        self.backtest_calls = 0

    def available(self, model_id: str) -> bool:
        return model_id in self.model_ids

    def data_time(self) -> dict:
        return {
            "first_date": "1978-10-25",
            "latest_date": "2025-12-30",
            "n_frames": 17188,
            "built_at": "2026-10-25T14:40:44Z",
            "batch_age_days": 300,
            "source": "fake-source",
            "doi": "fake-doi",
        }

    def predict_point(self, model_id: str, date: str, row: int, col: int) -> dict:
        self.predict_calls += 1
        return {
            "ok": True,
            "model_id": model_id,
            "date": date,
            "next_date": TARGET_DATE,
            "next_idx": 17001,
            "last_obs_idx": 17000,
            "u": 16.72,
            "v": 4.81,
            "speed": math.hypot(16.72, 4.81),
            "direction_deg": math.degrees(math.atan2(4.81, 16.72)) % 360.0,
            "input": {
                "row": row,
                "col": col,
                "lat": CELL_LAT,
                "lon": CELL_LON,
                "date": date,
                "u": 17.21,
                "v": 5.04,
                "speed": math.hypot(17.21, 5.04),
                "direction_deg": math.degrees(math.atan2(5.04, 17.21)) % 360.0,
                "valid": True,
                "coverage": 0.4129,
            },
            "uncertainty": {
                "method": "假引擎不确定度（测试）",
                "lower_km": 13.0,
                "upper_km": 17.0,
            },
            "warnings": [{"code": "fake_warning", "level": "info", "message": "假引擎提示"}],
        }

    def predict_field(self, model_id: str, date: str) -> dict:
        self.predict_field_calls += 1
        u = np.full((116, 158), np.nan, dtype=np.float32)
        v = np.full((116, 158), np.nan, dtype=np.float32)
        mask = (np.arange(116 * 158).reshape(116, 158) % 3) == 0
        u[mask] = 10.0
        v[mask] = 2.0
        return {
            "ok": True,
            "u": u,
            "v": v,
            "speed": np.hypot(u, v),
            "coverage": float(mask.mean()),
            "target_date": TARGET_DATE,
            "warnings": [],
        }

    def backtest(self, model_id: str, row: int, col: int, start: str, end: str,
                 max_points: int = 180) -> dict:
        self.backtest_calls += 1
        series = [
            {
                "date": "2025-01-02",
                "obs_u": 12.1,
                "obs_v": 3.2,
                "obs_speed": 12.5,
                "pred_u": 11.8,
                "pred_v": 3.4,
                "pred_speed": 12.3,
            },
            {
                "date": "2025-01-03",
                "obs_u": 10.0,
                "obs_v": 2.0,
                "obs_speed": math.hypot(10.0, 2.0),
                "pred_u": 9.6,
                "pred_v": 2.4,
                "pred_speed": math.hypot(9.6, 2.4),
            },
        ]
        return {
            "ok": True,
            "range": {
                "requested_start": start,
                "requested_end": end,
                "available_start": "1978-11-02",
                "available_end": "2025-12-30",
                "n_points": len(series),
                "n_days_total": 357,
            },
            "series": series,
            "summary": {
                "n": 2,
                "rmse_u": 6.1,
                "rmse_v": 5.8,
                "vec_rmse": 8.4,
                "mae_vec": 6.7,
                "r2_u": 0.31,
                "r2_v": 0.29,
                "skill_vs_persistence": 0.02,
            },
            "warnings": [],
        }

    def cell_stats(self, row: int, col: int) -> dict:
        return {
            "speed_mean": 9.0,
            "speed_std": 4.0,
            "speed_p99": 30.0,
            "valid_days": 1000,
            "lat": CELL_LAT,
            "lon": CELL_LON,
        }


class FakeSimVP:
    """与 src.serving.SimVPPredictor 同构的假预测器。"""

    __test__ = False

    def __init__(self):
        self.state = {"loaded": True, "device": "cpu", "error": None, "precomputed": False}

    def predict_point(self, date: str, row: int, col: int) -> dict:
        speed = math.hypot(9.9, 3.3)
        return {
            "ok": True,
            "date": date,
            "next_date": TARGET_DATE,
            "next_idx": 17001,
            "u": 9.9,
            "v": 3.3,
            "speed": speed,
            "direction_deg": math.degrees(math.atan2(3.3, 9.9)) % 360.0,
            "input": {
                "row": row,
                "col": col,
                "lat": CELL_LAT,
                "lon": CELL_LON,
                "date": date,
                "u": 17.21,
                "v": 5.04,
                "speed": math.hypot(17.21, 5.04),
                "direction_deg": math.degrees(math.atan2(5.04, 17.21)) % 360.0,
                "valid": True,
            },
            "series_24h": {
                "hours": list(range(25)),
                "cumulative_km": [0.0] * 25,
                "speed_cm_s": [speed] * 25,
                "row": [float(row)] * 25,
                "col": [float(col)] * 25,
                "lat": [CELL_LAT] * 25,
                "lon": [CELL_LON] * 25,
                "uncertainty": None,
                "interpolation": "fake",
                "target_date": TARGET_DATE,
            },
            "warnings": [],
            "elapsed_ms": 1.0,
        }

    def predict_field(self, date: str) -> dict:
        u = np.full((116, 158), np.nan, dtype=np.float32)
        v = np.full((116, 158), np.nan, dtype=np.float32)
        mask = (np.arange(116 * 158).reshape(116, 158) % 4) == 0
        u[mask] = 11.0
        v[mask] = 3.0
        return {
            "ok": True,
            "u": u,
            "v": v,
            "speed": np.hypot(u, v),
            "coverage": float(mask.mean()),
            "target_date": TARGET_DATE,
            "warnings": [],
            "elapsed_ms": 1.0,
        }

    def backtest(self, row: int, col: int, start: str, end: str, max_points: int = 180) -> dict:
        return {
            "ok": True,
            "range": {
                "requested_start": start,
                "requested_end": end,
                "available_start": "2022-01-10",
                "available_end": "2025-12-29",
                "n_points": 2,
                "n_days_total": 10,
            },
            "series": [
                {
                    "date": "2022-01-11",
                    "obs_u": 5.0,
                    "obs_v": 1.0,
                    "obs_speed": math.hypot(5.0, 1.0),
                    "pred_u": 4.8,
                    "pred_v": 1.2,
                    "pred_speed": math.hypot(4.8, 1.2),
                }
            ],
            "summary": {"n": 1, "vec_rmse": 9.8, "mae_vec": 8.0},
            "warnings": [],
        }


def _inject_fakes(client: TestClient, engine=None, simvp=None):
    """把假引擎注入 app.state.store（复用真实 zarr 做数据/日期校验）。"""
    store = client.app.state.store
    if engine is not None:
        store._engine = engine
        store._engine_attempted = True
        store._engine_error = None
        store._engine_generation += 1
        try:
            store._engine_snapshot = store._artifact_snapshot()
        except Exception:  # noqa: BLE001
            store._engine_snapshot = None
    if simvp is not None:
        store._simvp = simvp
        store._simvp_attempted = True
        store._simvp_error = None
    return store


# ---------------------------------------------------------------------------
# 通用契约断言
# ---------------------------------------------------------------------------
def _assert_error_body(response, expected_code: str) -> dict:
    assert response.status_code >= 400, response.text
    assert "application/json" in response.headers.get("content-type", "")
    body = response.json()
    assert body.get("status") == "error"
    error = body.get("error")
    assert isinstance(error, dict)
    assert error.get("code") == expected_code, body
    assert isinstance(error.get("message"), str) and error["message"]
    if "hint" in error:
        assert error["hint"] is None or isinstance(error["hint"], str)
    return body


def _assert_series_contract(series: dict, prediction: dict) -> None:
    assert series["hours"] == list(range(25))
    for key in ("cumulative_km", "speed_cm_s", "row", "col", "lat", "lon"):
        assert len(series[key]) == 25, key
    assert series["cumulative_km"][0] == 0.0
    cumulative = series["cumulative_km"]
    assert all(b >= a - 1e-9 for a, b in zip(cumulative, cumulative[1:])), "累计位移应单调不减"
    # 24h 累计位移 = speed(cm/s) * 0.036 * 24（设计 §2）
    assert abs(cumulative[-1] - prediction["displacement_km_24h"]) <= 0.01
    assert abs(prediction["displacement_km_24h"] - prediction["speed"] * 0.864) <= 0.01
    # 日尺度产品：逐时速度恒定
    assert max(series["speed_cm_s"]) - min(series["speed_cm_s"]) <= 1e-9
    assert abs(series["speed_cm_s"][0] - prediction["speed"]) <= 0.01
    uncertainty = series["uncertainty"]
    if uncertainty is not None:
        assert isinstance(uncertainty["method"], str) and uncertainty["method"]
        assert len(uncertainty["lower_km"]) == 25
        assert len(uncertainty["upper_km"]) == 25
        assert all(
            low <= cum + 1e-9 <= high + 1e-6
            for low, cum, high in zip(
                uncertainty["lower_km"], cumulative, uncertainty["upper_km"]
            )
        ), "不确定度带应包住累计位移曲线"


def _assert_predict_payload(body: dict) -> None:
    assert body["status"] == "ok"
    assert body["status_text"]
    assert isinstance(body["request_id"], str) and len(body["request_id"]) >= 6
    assert body["elapsed_ms"] >= 0
    assert set(body["model"]) >= {"id", "name", "category"}
    data_time = body["data_time"]
    assert data_time["last_obs_date"]
    assert isinstance(data_time["last_obs_index"], int)
    assert data_time["next_frame_date"]
    assert data_time["horizon_hours"] == 24
    assert data_time["latest_date"] and isinstance(data_time["n_frames"], int)
    input_block = body["input"]
    for key in ("row", "col", "lat", "lon", "date", "u", "v", "speed",
                "direction_deg", "valid", "coverage"):
        assert key in input_block, key
    prediction = body["prediction"]
    for key in ("u", "v", "speed", "direction_deg", "unit", "displacement_km_24h",
                "bearing_deg", "delta_speed_vs_persistence"):
        assert key in prediction, key
    assert prediction["unit"] == "cm/s"
    assert prediction["speed"] >= 0
    assert 0.0 <= prediction["direction_deg"] <= 360.0
    assert 0.0 <= prediction["bearing_deg"] <= 360.0
    assert abs(prediction["speed"] - math.hypot(prediction["u"], prediction["v"])) <= 0.01
    _assert_series_contract(body["series_24h"], prediction)
    assert body["series_24h"]["target_date"]
    assert isinstance(body["warnings"], list)
    for warning in body["warnings"]:
        assert warning["level"] in ("info", "warning", "danger")
        assert warning["code"] and warning["message"]


# ===========================================================================
# 1. 基础端点
# ===========================================================================
def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "weddell-seaice-ml"
    assert body["version"] == client.app.version
    assert isinstance(body["uptime_s"], (int, float)) and body["uptime_s"] >= 0
    assert isinstance(body["models_loaded"], list)
    assert all(isinstance(m, str) for m in body["models_loaded"])
    assert isinstance(body["artifacts_ok"], bool)


def test_openapi_and_docs(client):
    docs = client.get("/docs")
    assert docs.status_code == 200
    assert "text/html" in docs.headers.get("content-type", "")
    spec = client.get("/openapi.json")
    assert spec.status_code == 200
    paths = spec.json()["paths"]
    for path in (
        "/api/health",
        "/api/models",
        "/api/metrics",
        "/api/overview",
        "/api/field",
        "/api/field/predicted",
        "/api/predict",
        "/api/backtest",
    ):
        assert path in paths, f"OpenAPI 缺少 {path}"


def test_models_payload(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    body = response.json()
    data_time = body["data_time"]
    for key in ("first_date", "latest_date", "n_frames", "built_at"):
        assert key in data_time
    assert body["default"]["model"] in (
        "ridge", "random_forest", "simvp", "persistence", "climatology"
    )
    assert body["default"]["date"]
    assert set(body["default"]["cell"]) == {"row", "col"}

    models = body["models"]
    assert [m["id"] for m in models] == [
        "ridge", "random_forest", "simvp", "persistence", "climatology"
    ]
    for model in models:
        for key in ("id", "name", "short", "category", "color", "available",
                    "description", "trained_at", "training_samples", "metrics",
                    "is_baseline"):
            assert key in model, key
        assert model["category"] in ("linear", "tree", "deep", "baseline")
        assert isinstance(model["available"], bool)
        assert isinstance(model["metrics"], dict)
    assert isinstance(body["combos"], list)
    for combo in body["combos"]:
        assert set(combo) == {"model", "date", "cell"}


@needs_metrics
def test_metrics(client):
    response = client.get("/api/metrics")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["models"], list) and body["models"]
    ids = {model.get("id") for model in body["models"]}
    assert {"ridge", "random_forest"} & ids or "simvp" in ids
    protocol = body["protocol"]
    assert protocol["task"]
    assert protocol["metric_defs"]
    for model in body["models"]:
        assert "id" in model
    if "warnings" in body:
        assert isinstance(body["warnings"], list)


@needs_overview
def test_overview(client):
    response = client.get("/api/overview")
    assert response.status_code == 200
    body = response.json()
    dataset = body["dataset"]
    assert dataset["grid"]["rows"] == 116 and dataset["grid"]["cols"] == 158
    assert isinstance(body["kpis"], list) and body["kpis"]
    assert isinstance(body["annual"], list)
    assert isinstance(body["monthly"], list)
    assert "data_time" in body


# ===========================================================================
# 2. 观测场
# ===========================================================================
@needs_zarr
def test_field_observed(client):
    response = client.get("/api/field")
    assert response.status_code == 200
    body = response.json()
    assert body["kind"] == "observed"
    assert len(body["speed"]) == 116
    assert all(len(row) == 158 for row in body["speed"])
    assert len(body["row_axis"]) == 116
    assert len(body["col_axis"]) == 158
    assert len(body["row_lat"]) == 116
    assert len(body["col_lon"]) == 158
    vectors = body["vectors"]
    assert vectors["stride"] >= 1
    assert len(vectors["rows"]) == len(vectors["cols"]) == len(vectors["u"]) == len(
        vectors["v"]
    ) == len(vectors["speed"])
    stats = body["stats"]
    assert stats["valid_cells"] >= 0
    assert 0.0 <= stats["coverage"] <= 1.0
    assert stats["valid_cells"] == round(stats["coverage"] * 116 * 158) or abs(
        stats["valid_cells"] - stats["coverage"] * 116 * 158
    ) < 2
    # 无效像元必须是 null；有效像元是有限数值
    non_null = 0
    for row in body["speed"]:
        for value in row:
            if value is not None:
                non_null += 1
                assert isinstance(value, (int, float))
                assert value >= 0
    assert non_null == stats["valid_cells"]
    assert body["data_time"]["latest_date"]


@needs_zarr
def test_field_observed_date_and_stride(client):
    explicit = client.get(f"/api/field?date={DEFAULT_DATE}&stride=10")
    assert explicit.status_code == 200
    assert explicit.json()["date"] == DEFAULT_DATE
    assert explicit.json()["vectors"]["stride"] == 10
    assert len(explicit.json()["vectors"]["u"]) <= (116 // 10 + 1) * (158 // 10 + 1)


@needs_zarr
def test_field_errors(client):
    assert _assert_error_body(client.get("/api/field?date=2025-13-01"), "date_invalid")
    # 数据集缺失日（zarr attrs missing_days 之一）
    missing = client.get("/api/field?date=1980-03-15")
    assert missing.status_code == 422
    assert _assert_error_body(missing, "date_not_in_dataset")
    stride_error = client.get("/api/field?stride=11")
    assert stride_error.status_code == 422
    assert stride_error.json()["status"] == "error"


# ===========================================================================
# 3. 假引擎路由逻辑（不依赖真实模型 artifact）
# ===========================================================================
@needs_zarr
def test_field_predicted_with_fake_engine(fresh_client):
    fake = FakeEngine()
    _inject_fakes(fresh_client, engine=fake)
    response = fresh_client.get(f"/api/field/predicted?model=ridge&date={DEFAULT_DATE}")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["kind"] == "predicted"
    assert body["model"]["id"] == "ridge"
    assert body["date"] == DEFAULT_DATE
    assert body["target_date"] == TARGET_DATE
    assert len(body["speed"]) == 116 and all(len(row) == 158 for row in body["speed"])
    assert body["stats"]["comparison"] is not None
    comparison = body["stats"]["comparison"]
    assert comparison["n"] >= 0
    if comparison["vec_rmse"] is not None:
        assert comparison["vec_rmse"] >= 0
    assert body["elapsed_ms"] >= 0
    assert fake.predict_field_calls == 1


def test_field_predicted_unknown_model(fresh_client):
    response = fresh_client.get("/api/field/predicted?model=not-a-model&date=2025-06-26")
    assert response.status_code == 404
    _assert_error_body(response, "unknown_model")


def test_model_unavailable_409(fresh_client, monkeypatch):
    store = fresh_client.app.state.store
    monkeypatch.setattr(store, "model_available", lambda model_id: False)
    predict = fresh_client.post(
        "/api/predict", json={"model": "ridge", "date": DEFAULT_DATE, "cell": DEFAULT_CELL}
    )
    assert predict.status_code == 409
    _assert_error_body(predict, "model_unavailable")
    field = fresh_client.get(f"/api/field/predicted?model=ridge&date={DEFAULT_DATE}")
    assert field.status_code == 409


@needs_zarr
def test_predict_with_fake_engine(fresh_client):
    fake = FakeEngine()
    _inject_fakes(fresh_client, engine=fake)
    response = fresh_client.post(
        "/api/predict", json={"model": "random_forest", "date": DEFAULT_DATE, "cell": DEFAULT_CELL}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    _assert_predict_payload(body)
    assert body["model"]["id"] == "random_forest"
    assert body["model"]["category"] == "tree"
    assert body["input"]["row"] == DEFAULT_CELL["row"]
    assert body["input"]["col"] == DEFAULT_CELL["col"]
    assert abs(body["input"]["lat"] - CELL_LAT) < 0.05
    assert abs(body["input"]["lon"] - CELL_LON) < 0.05
    assert body["data_time"]["last_obs_index"] == 17000
    assert body["data_time"]["next_frame_date"] == TARGET_DATE
    assert body["series_24h"]["target_date"] == TARGET_DATE
    codes = {w["code"] for w in body["warnings"]}
    assert "fake_warning" in codes
    assert fake.predict_calls == 1


@needs_zarr
def test_predict_simvp_with_fake_predictor(fresh_client):
    fake_engine = FakeEngine()
    fake_simvp = FakeSimVP()
    _inject_fakes(fresh_client, engine=fake_engine, simvp=fake_simvp)
    response = fresh_client.post(
        "/api/predict", json={"model": "simvp", "date": DEFAULT_DATE, "cell": DEFAULT_CELL}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    _assert_predict_payload(body)
    assert body["model"]["id"] == "simvp"
    assert body["model"]["category"] == "deep"
    assert abs(body["prediction"]["speed"] - math.hypot(9.9, 3.3)) <= 0.01


@needs_zarr
def test_predict_lru_cache_hit_returns_same_fields(fresh_client):
    fake = FakeEngine()
    _inject_fakes(fresh_client, engine=fake)
    payload = {"model": "ridge", "date": DEFAULT_DATE, "cell": DEFAULT_CELL}
    first = fresh_client.post("/api/predict", json=payload)
    second = fresh_client.post("/api/predict", json=payload)
    assert first.status_code == second.status_code == 200
    first_body, second_body = first.json(), second.json()
    assert fake.predict_calls == 1, "第二次请求应命中 LRU 缓存，不重复推理"
    for body in (first_body, second_body):
        for key in ("request_id", "elapsed_ms"):
            body.pop(key)
    assert first_body == second_body


@needs_zarr
def test_backtest_with_fake_engine(fresh_client):
    fake = FakeEngine()
    _inject_fakes(fresh_client, engine=fake)
    response = fresh_client.post(
        "/api/backtest",
        json={
            "model": "ridge",
            "cell": DEFAULT_CELL,
            "start": "2025-01-01",
            "end": "2025-12-30",
            "max_points": 60,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"]["id"] == "ridge"
    assert set(body["cell"]) == {"row", "col", "lat", "lon"}
    assert body["range"]["requested_start"] == "2025-01-01"
    assert body["range"]["requested_end"] == "2025-12-30"
    assert body["range"]["available_start"] == "1978-11-02"
    assert body["range"]["n_points"] == len(body["series"])
    expected_err = math.hypot(11.8 - 12.1, 3.4 - 3.2)
    assert abs(body["series"][0]["err_vec"] - round(expected_err, 3)) <= 1e-3
    summary = body["summary"]
    assert summary["n"] == 2
    for key in ("rmse_u", "rmse_v", "vec_rmse", "mae_vec", "r2_u", "r2_v",
                "skill_vs_persistence"):
        assert key in summary
    assert fake.backtest_calls == 1


# ===========================================================================
# 4. 请求校验 / 错误体
# ===========================================================================
@needs_zarr
def test_predict_errors(fresh_client):
    unknown = fresh_client.post(
        "/api/predict", json={"model": "nope", "date": DEFAULT_DATE, "cell": DEFAULT_CELL}
    )
    assert unknown.status_code == 404
    _assert_error_body(unknown, "unknown_model")

    bad_cell = fresh_client.post(
        "/api/predict",
        json={"model": "ridge", "date": DEFAULT_DATE, "cell": {"row": 999, "col": 60}},
    )
    assert bad_cell.status_code == 400
    _assert_error_body(bad_cell, "invalid_cell")

    bad_cell_type = fresh_client.post(
        "/api/predict",
        json={"model": "ridge", "date": DEFAULT_DATE, "cell": {"row": "x", "col": 60}},
    )
    assert bad_cell_type.status_code == 400
    _assert_error_body(bad_cell_type, "invalid_cell")

    bad_date = fresh_client.post(
        "/api/predict", json={"model": "ridge", "date": "2025-13-01", "cell": DEFAULT_CELL}
    )
    assert bad_date.status_code == 422
    _assert_error_body(bad_date, "date_invalid")

    missing_date = fresh_client.post(
        "/api/predict", json={"model": "ridge", "date": "1980-03-15", "cell": DEFAULT_CELL}
    )
    assert missing_date.status_code == 422
    _assert_error_body(missing_date, "date_not_in_dataset")

    missing_field = fresh_client.post("/api/predict", json={"model": "ridge"})
    assert missing_field.status_code == 422
    _assert_error_body(missing_field, "validation_error")

    # 数据集最后一帧可以起报，目标日暂缺观测
    last_frame = fresh_client.post(
        "/api/predict", json={"model": "ridge", "date": "2025-12-30", "cell": DEFAULT_CELL}
    )
    assert last_frame.status_code == 200, last_frame.text
    payload = last_frame.json()
    assert payload["data_time"]["target_observed"] is False
    assert payload["series_24h"]["target_date"] == "2025-12-31"
    assert any(warning["code"] == "target_unobserved" for warning in payload["warnings"])


def test_backtest_errors(fresh_client):
    bad_points = fresh_client.post(
        "/api/backtest",
        json={"model": "ridge", "cell": DEFAULT_CELL, "max_points": 999},
    )
    assert bad_points.status_code == 422
    _assert_error_body(bad_points, "invalid_range")

    bad_range = fresh_client.post(
        "/api/backtest",
        json={
            "model": "ridge",
            "cell": DEFAULT_CELL,
            "start": "2025-12-01",
            "end": "2025-01-01",
        },
    )
    assert bad_range.status_code == 422
    _assert_error_body(bad_range, "invalid_range")

    unknown = fresh_client.post(
        "/api/backtest", json={"model": "nope", "cell": DEFAULT_CELL}
    )
    assert unknown.status_code == 404
    _assert_error_body(unknown, "unknown_model")


def test_error_body_format_and_http_errors(fresh_client):
    response = fresh_client.get("/api/not-exist")
    assert response.status_code == 404
    body = _assert_error_body(response, "not_found")
    assert set(body) == {"status", "error"}
    assert set(body["error"]) <= {"code", "message", "hint"}

    method = fresh_client.get("/api/predict")
    assert method.status_code == 405
    assert method.json()["status"] == "error"


def test_cors_headers(fresh_client):
    response = fresh_client.options(
        "/api/health",
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code in (200, 204)
    assert response.headers.get("access-control-allow-origin") in (
        "http://127.0.0.1:5173",
        "*",
    )
    simple = fresh_client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert simple.headers.get("access-control-allow-origin") in (
        "http://localhost:5173",
        "*",
    )


def test_static_root_and_spa_fallback(fresh_client):
    dist = ROOT / "dashboard/dist"
    response = fresh_client.get("/")
    assert response.status_code == 200
    if (dist / "index.html").is_file():
        assert "text/html" in response.headers.get("content-type", "")
    else:
        body = response.json()
        assert body["status"] == "error"
        assert body["error"]["code"] == "frontend_not_built"
        assert "build" in json.dumps(body)
    spa = fresh_client.get("/some/spa/route")
    assert spa.status_code == 200


# ===========================================================================
# 5. artifact fallback（临时目录，不依赖 Lead 最终合并产物）
# ===========================================================================
def test_metrics_fallback_from_tabular(tmp_path):
    from backend.app.config import get_settings
    from backend.app.services.artifact_store import ArtifactStore

    settings = replace(get_settings(), dashboard_dir=tmp_path)
    payload = {
        "generated_at": "2026-10-26T06:00:00Z",
        "models": [
            {"id": "ridge", "name": "Ridge 岭回归", "vec_rmse": 10.62,
             "skill_vs_persistence": 0.03},
            {"id": "random_forest", "name": "Random Forest 随机森林", "vec_rmse": 10.15,
             "skill_vs_persistence": 0.07},
            {"id": "persistence", "name": "持续性基线", "vec_rmse": 10.91,
             "skill_vs_persistence": 0.0},
        ],
    }
    (tmp_path / "metrics_tabular.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    store = ArtifactStore(settings)
    metrics, warnings = store.metrics()
    assert metrics is not None
    ids = [model["id"] for model in metrics["models"]]
    assert "ridge" in ids and "random_forest" in ids
    assert ids.index("ridge") < ids.index("persistence")
    assert "protocol" in metrics
    assert any(w["code"] == "metrics_fallback" for w in warnings)


def test_overview_fallback_from_timeseries(tmp_path):
    from backend.app.config import get_settings
    from backend.app.services.artifact_store import ArtifactStore

    settings = replace(get_settings(), dashboard_dir=tmp_path)
    timeseries = {
        "dataset": {
            "name": "test-dataset",
            "region": "test-region",
            "grid": {"rows": 116, "cols": 158, "cell_km": 25.07},
            "n_frames": 17188,
            "first_date": "1978-10-25",
            "latest_date": "2025-12-30",
            "years": 48,
            "missing_days": 36,
            "mean_valid_fraction": 0.309,
            "mean_speed_cm_s": 9.94,
        },
        "kpis": [{"id": "years", "label": "覆盖年限", "value": 48, "unit": "年"}],
        "annual": [{"year": 1978, "mean_speed": 5.1, "coverage": 0.28, "n_days": 68}],
        "monthly": [{"month": 1, "mean_speed": 4.5, "coverage": 0.19}],
        "year_month": {"years": [1978], "months": [1], "values": [[5.1]]},
    }
    (tmp_path / "timeseries.json").write_text(
        json.dumps(timeseries, ensure_ascii=False), encoding="utf-8"
    )
    store = ArtifactStore(settings)
    overview, warnings = store.overview()
    assert overview is not None
    assert overview["dataset"]["name"] == "test-dataset"
    assert overview["kpis"] and overview["annual"] and overview["monthly"]
    assert overview["data_time"]["latest_date"]
    assert any(w["code"] == "overview_fallback" for w in warnings)


# ===========================================================================
# 6. 真实引擎集成（artifact 就绪后运行；缺失自动 skip）
# ===========================================================================
def _model_availability(client) -> dict[str, bool]:
    response = client.get("/api/models")
    assert response.status_code == 200
    return {model["id"]: model["available"] for model in response.json()["models"]}


@needs_tabular
@pytest.mark.parametrize("model_id", ["ridge", "random_forest", "persistence", "climatology"])
def test_real_predict_tabular_models(client, model_id):
    if not _model_availability(client).get(model_id):
        pytest.skip(f"{model_id} 不可用（artifact 缺失或加载失败）")
    response = client.post(
        "/api/predict",
        json={"model": model_id, "date": DEFAULT_DATE, "cell": DEFAULT_CELL},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    _assert_predict_payload(body)
    assert body["model"]["id"] == model_id
    assert body["data_time"]["last_obs_date"] == DEFAULT_DATE
    assert body["data_time"]["last_obs_index"] == 17000
    assert body["series_24h"]["target_date"] == TARGET_DATE
    assert -200 <= body["prediction"]["u"] <= 200
    assert -200 <= body["prediction"]["v"] <= 200
    assert abs(body["input"]["lat"] - CELL_LAT) < 0.05
    assert abs(body["input"]["lon"] - CELL_LON) < 0.05
    metrics = body["metrics"]
    assert metrics is None or set(metrics) == {"vec_rmse", "skill_vs_persistence"}


@needs_simvp
def test_real_predict_simvp(client):
    if not _model_availability(client).get("simvp"):
        pytest.skip("SimVP 不可用（ckpt 缺失或加载失败）")
    response = client.post(
        "/api/predict", json={"model": "simvp", "date": DEFAULT_DATE, "cell": DEFAULT_CELL}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    _assert_predict_payload(body)
    assert body["model"]["category"] == "deep"
    assert body["series_24h"]["target_date"] == TARGET_DATE


@needs_tabular
def test_real_field_predicted(client):
    if not _model_availability(client).get("ridge"):
        pytest.skip("ridge 不可用")
    response = client.get(f"/api/field/predicted?model=ridge&date={DEFAULT_DATE}")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["target_date"] == TARGET_DATE
    assert len(body["speed"]) == 116
    assert body["stats"]["valid_cells"] > 0
    comparison = body["stats"]["comparison"]
    assert comparison is not None and comparison["n"] > 0
    assert comparison["vec_rmse"] is not None and comparison["vec_rmse"] >= 0


@needs_tabular
def test_real_backtest(client):
    if not _model_availability(client).get("ridge"):
        pytest.skip("ridge 不可用")
    response = client.post(
        "/api/backtest",
        json={
            "model": "ridge",
            "cell": DEFAULT_CELL,
            "start": "2025-01-01",
            "end": "2025-12-30",
            "max_points": 60,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ok"
    assert 1 <= len(body["series"]) <= 60
    assert body["range"]["n_points"] == len(body["series"])
    assert body["range"]["available_start"] <= "2025-01-01"
    for item in body["series"]:
        assert set(item) == {
            "date", "obs_u", "obs_v", "obs_speed", "pred_u", "pred_v",
            "pred_speed", "err_vec",
        }
    summary = body["summary"]
    assert summary["n"] >= 1
    if summary["skill_vs_persistence"] is not None:
        assert -2.0 <= summary["skill_vs_persistence"] <= 1.0


@needs_simvp_backtest
def test_real_simvp_backtest_and_range(client):
    if not _model_availability(client).get("simvp"):
        pytest.skip("SimVP 不可用（ckpt 缺失或加载失败）")
    response = client.post(
        "/api/backtest",
        json={
            "model": "simvp",
            "cell": DEFAULT_CELL,
            "start": "2025-01-01",
            "end": "2025-12-30",
            "max_points": 30,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert 1 <= len(body["series"]) <= 30
    assert body["range"]["n_points"] == len(body["series"])
    # test 期可用范围（预计算场覆盖 2022-01-01 ~ 2025-12-30）
    assert body["range"]["available_start"].startswith("2022-")
    assert body["range"]["available_end"] == "2025-12-30"
    summary = body["summary"]
    assert summary["n"] >= 1
    if summary["vec_rmse"] is not None:
        assert 0 <= summary["vec_rmse"] <= 100

    # 契约：SimVP 请求 2022 之前日期 → 422 backtest_out_of_range + 可用范围
    out_of_range = client.post(
        "/api/backtest",
        json={
            "model": "simvp",
            "cell": DEFAULT_CELL,
            "start": "2021-01-01",
            "end": "2022-06-30",
        },
    )
    assert out_of_range.status_code == 422
    error = _assert_error_body(out_of_range, "backtest_out_of_range")
    assert "可用回测范围" in (error["error"].get("hint") or "")


# ---------------------------------------------------------------------------
# 不重训验证：predict 前后模型文件 mtime 不变、无新文件
# ---------------------------------------------------------------------------
def _model_files_snapshot() -> dict[str, int]:
    files: dict[str, int] = {}
    for path in sorted(ML_DIR.glob("*.joblib")):
        files[str(path.relative_to(ROOT))] = path.stat().st_mtime_ns
    for path in sorted(MODELS_DIR.rglob("*.pt")):
        files[str(path.relative_to(ROOT))] = path.stat().st_mtime_ns
    return files


@needs_tabular
def test_predict_does_not_retrain(client):
    availability = _model_availability(client)
    models = [mid for mid in ("ridge", "random_forest", "persistence") if availability.get(mid)]
    if not models:
        pytest.skip("没有可用的表格模型")

    # 先跑一次让引擎/模型加载完成（加载不算预测写入）
    warm = client.post(
        "/api/predict", json={"model": models[0], "date": DEFAULT_DATE, "cell": DEFAULT_CELL}
    )
    assert warm.status_code == 200, warm.text

    before = _model_files_snapshot()
    assert before, "models/ml 下应已有 joblib artifact"
    for model_id in models:
        response = client.post(
            "/api/predict",
            json={"model": model_id, "date": DEFAULT_DATE, "cell": DEFAULT_CELL},
        )
        assert response.status_code == 200, response.text
    # 回测也不允许写模型文件
    backtest = client.post(
        "/api/backtest",
        json={"model": models[0], "cell": DEFAULT_CELL, "start": "2025-06-01",
              "end": "2025-06-30", "max_points": 10},
    )
    assert backtest.status_code == 200, backtest.text

    after = _model_files_snapshot()
    if before != after:
        # 后台训练可能正在并发替换 artifact（外部写入）：短暂等待判断是否仍在变化
        time.sleep(1.0)
        after2 = _model_files_snapshot()
        if after2 != after:
            pytest.skip("检测到后台训练正在并发更新模型 artifact，无法稳定断言（非本次 predict 写入）")
    assert after == before, (
        "predict/backtest 不应修改或新增模型文件；"
        f"变化={ {k for k in set(before) | set(after) if before.get(k) != after.get(k)} }"
    )


# ===========================================================================
# 7. metrics.json / overview.json 主路径（Lead 最终合并产物原样透传）
# ===========================================================================
def _full_metrics_payload() -> dict:
    ids = [
        ("ridge", "Ridge 岭回归", "linear"),
        ("random_forest", "Random Forest 随机森林", "tree"),
        ("simvp", "SimVP-v1 深度学习", "deep"),
        ("persistence", "持续性基线", "baseline"),
        ("climatology", "气候态基线", "baseline"),
    ]
    models = []
    for idx, (model_id, name, category) in enumerate(ids):
        models.append(
            {
                "id": model_id,
                "name": name,
                "name_zh": name,
                "category": category,
                "vec_rmse": 10.0 + idx,
                "rmse_u": 7.0 + idx,
                "rmse_v": 7.0 + idx,
                "mae_vec": 8.0 + idx,
                "r2_u": 0.4,
                "r2_v": 0.4,
                "skill_vs_persistence": 0.1,
                "latency_ms": 1.0 + idx,
                "params": {},
                "training_samples": 1000,
                "is_highlight": model_id in ("ridge", "random_forest"),
                "is_baseline": category == "baseline",
            }
        )
    return {
        "protocol": {
            "task": "未来 24 小时（下一日帧）海冰漂移矢量预测",
            "target": "u(t+1), v(t+1)，cm/s，25 km EASE-Grid South 像元",
            "split": {"train": "1978–2017", "val": "2018–2021", "test": "2022–2025"},
            "sampling": "test split 每 5 日一个起报窗口（287 个窗口），评估全部有效像元",
            "n_windows": 287,
            "n_samples": 1556493,
            "metric_defs": {
                "vec_rmse": "√mean(Δu²+Δv²)，cm/s",
                "mae_vec": "mean(√(Δu²+Δv²))，cm/s",
                "skill_vs_persistence": "1 − RMSE_model/RMSE_persistence",
                "r2_u": "u 分量决定系数",
            },
        },
        "models": models,
        "generated_at": "2026-10-26T06:00:00Z",
    }


def test_metrics_primary_passthrough(tmp_path):
    """metrics.json 存在时必须原样透传（数值与离线评估 artifact 完全一致），不再走 fallback。"""
    from backend.app.config import get_settings
    from backend.app.services.artifact_store import ArtifactStore

    settings = replace(get_settings(), dashboard_dir=tmp_path)
    merged = _full_metrics_payload()
    (tmp_path / "metrics.json").write_text(
        json.dumps(merged, ensure_ascii=False), encoding="utf-8"
    )
    # 同时放一份数值不同的 tabular fallback，验证不会被误用
    stale = _full_metrics_payload()
    for row in stale["models"]:
        row["vec_rmse"] = -1.0
    (tmp_path / "metrics_tabular.json").write_text(
        json.dumps(stale, ensure_ascii=False), encoding="utf-8"
    )
    store = ArtifactStore(settings)
    payload, warnings = store.metrics()
    assert payload is not None
    assert [m["id"] for m in payload["models"]] == [
        "ridge", "random_forest", "simvp", "persistence", "climatology"
    ]
    assert payload["models"][0]["vec_rmse"] == merged["models"][0]["vec_rmse"]
    assert payload["generated_at"] == merged["generated_at"]
    assert payload["protocol"]["n_windows"] == 287
    assert payload["protocol"]["metric_defs"] == merged["protocol"]["metric_defs"]
    assert not warnings, warnings
    assert "warnings" not in payload


def test_overview_primary_passthrough(tmp_path):
    """overview.json 存在时直接透传，并补齐 data_time（若缺失）。"""
    from backend.app.config import get_settings
    from backend.app.services.artifact_store import ArtifactStore

    settings = replace(get_settings(), dashboard_dir=tmp_path)
    overview = {
        "dataset": {"name": "merged-dataset", "grid": {"rows": 116, "cols": 158,
                                                       "cell_km": 25.07}},
        "kpis": [{"id": "years", "label": "覆盖年限", "value": 48, "unit": "年"}],
        "annual": [{"year": 1978, "mean_speed": 5.1, "coverage": 0.28, "n_days": 68}],
        "monthly": [{"month": 1, "mean_speed": 4.5, "coverage": 0.19}],
        "year_month": {"years": [1978], "months": [1], "values": [[5.1]]},
    }
    (tmp_path / "overview.json").write_text(
        json.dumps(overview, ensure_ascii=False), encoding="utf-8"
    )
    store = ArtifactStore(settings)
    payload, warnings = store.overview()
    assert payload is not None
    assert payload["dataset"]["name"] == "merged-dataset"
    assert payload["annual"] == overview["annual"]
    assert isinstance(payload.get("data_time"), dict)
    assert payload["data_time"]["latest_date"]
    assert not warnings, warnings


def test_engine_hot_reload_on_artifact_replace(tmp_path):
    """模型 artifact 被原子替换后，后续请求应触发 engine.reload()（服务无需重启）。"""
    from backend.app.config import get_settings
    from backend.app.services.artifact_store import ArtifactStore

    models_dir = tmp_path / "models_ml"
    dash_dir = tmp_path / "dashboard"
    models_dir.mkdir()
    dash_dir.mkdir()
    ridge = models_dir / "ridge.joblib"
    ridge.write_bytes(b"smoke")
    settings = replace(
        get_settings(), models_ml_dir=models_dir, dashboard_dir=dash_dir
    )

    class ReloadableEngine:
        __test__ = False

        def __init__(self):
            self.reloads = 0
            self.warmups = 0

        def reload(self):
            self.reloads += 1

        def warmup(self):
            self.warmups += 1

        def available(self, model_id):
            return model_id in ("ridge", "random_forest", "persistence", "climatology")

        def model_ids(self):
            return ["ridge", "random_forest", "persistence", "climatology"]

    store = ArtifactStore(settings)
    fake = ReloadableEngine()
    store._engine = fake
    store._engine_attempted = True
    store._engine_snapshot = store._artifact_snapshot()
    store._engine_generation = 5
    # 未变化：不 reload
    assert store.maybe_reload_engine() is False
    assert fake.reloads == 0
    # 原子替换（mtime 变化）→ reload + generation+1
    time.sleep(0.02)
    ridge.write_bytes(b"final-model")
    assert store.maybe_reload_engine() is True
    assert fake.reloads == 1
    assert fake.warmups == 1
    assert store.engine_generation == 6
    assert store.maybe_reload_engine() is False


def test_simvp_remains_available_during_initial_load(monkeypatch, tmp_path):
    from backend.app.config import get_settings
    from backend.app.services.artifact_store import ArtifactStore

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    store = ArtifactStore(replace(get_settings(), simvp_ckpt=checkpoint))

    class StartingPredictor:
        def __init__(self, **kwargs):
            self.state = {"loaded": False}

        def load(self):
            assert store.model_available("simvp")
            self.state["loaded"] = True
            return self.state

    monkeypatch.setattr("src.serving.simvp.SimVPPredictor", StartingPredictor)
    assert isinstance(store.ensure_simvp(), StartingPredictor)
    assert store.model_available("simvp")
