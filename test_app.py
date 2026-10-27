"""Runnable assert-based self-check for Timestral Webapp & AI Agent API (Ponytail verification)."""

import time
from fastapi.testclient import TestClient
from app import app, _MODEL_STATE, _evict_if_idle_unlocked, IDLE_TTL_SECONDS

client = TestClient(app)


def main() -> None:
    # 1. Check Zero-Load /health before any inference (must NOT load a model into RAM)
    r_health = client.get("/health")
    assert r_health.status_code == 200, r_health.text
    h = r_health.json()
    assert h["status"] == "ok"
    assert h["default_model"] == "mistralai/Timestral-5M"
    assert h["loaded_in_memory"] is False, "GET /health must be zero-load when idle!"
    assert "sys_ram_total_mb" in h["device_telemetry"]
    assert "process_rss_mb" in h["device_telemetry"]
    docs = h["documentation_endpoints"]
    for key in ("swagger_ui", "doc", "redoc", "openapi_json", "readme", "llms_txt", "agent_tools_schema", "visual_examples_page"):
        assert key in docs, f"Missing {key} in /health documentation_endpoints"

    # 2. Check /readme, /llms.txt, /doc, /openapi.json, /docs, /redoc
    for path in ("/readme", "/llms.txt", "/doc", "/openapi.json", "/docs", "/redoc", "/"):
        resp = client.get(path)
        assert resp.status_code == 200, f"GET {path} failed with {resp.status_code}"

    # 3. Check Zero-Shot Forecast (/api/v1/forecast) -> defaults to 5M for responsiveness
    r_fc = client.post(
        "/api/v1/forecast",
        json={"context": [10.0, 12.5, None, 15.0, 18.2, 20.1, 22.0, 24.5] * 6, "prediction_length": 12, "threshold": 21.0},
    )
    assert r_fc.status_code == 200, r_fc.text
    fc = r_fc.json()
    assert fc["model_size"] == "5M"
    assert fc["telemetry"]["model_state"]["loaded_in_memory"] is True
    assert fc["telemetry"]["model_state"]["active_model_size"] == "5M"
    assert len(fc["median"]) == 12
    assert len(fc["quantile_levels"]) == 21

    # 4. Check On-Demand Model Switch to 209M (/api/v1/backtest)
    r_bt209 = client.post("/api/v1/backtest", json={"dataset_id": "grid_load_mw", "holdout_length": 24, "model_size": "209M"})
    assert r_bt209.status_code == 200, r_bt209.text
    bt209 = r_bt209.json()
    assert bt209["model_size"] == "209M"
    assert bt209["telemetry"]["model_state"]["active_model_size"] == "209M"

    # 5. Check Manual Unload (POST /api/v1/model/unload) & Zero-Load /health afterwards
    r_un = client.post("/api/v1/model/unload")
    assert r_un.status_code == 200
    assert r_un.json()["unloaded"] is True
    assert r_un.json()["telemetry"]["model_state"]["loaded_in_memory"] is False
    assert client.get("/health").json()["loaded_in_memory"] is False

    # 6. Check Idle TTL Automatic Eviction (300s simulation)
    client.post("/api/v1/backtest", json={"dataset_id": "grid_load_mw", "holdout_length": 12, "model_size": "5M"})
    assert _MODEL_STATE["forecaster"] is not None
    evicted = _evict_if_idle_unlocked(now=time.time() + IDLE_TTL_SECONDS + 5)
    assert evicted is True and _MODEL_STATE["forecaster"] is None

    # 7. Check Multi-Model Comparison (5M vs 209M cached on NVMe)
    r_cmp = client.post(
        "/api/v1/backtest/compare-models",
        json={"dataset_id": "grid_load_mw", "holdout_length": 24, "only_cached": True},
    )
    assert r_cmp.status_code == 200
    cmp_list = r_cmp.json()["comparisons"]
    assert len(cmp_list) >= 2

    # 8. Check all 5 additional native variants (using default 5M)
    assert client.post("/api/v1/anomalies", json={"dataset_id": "web_traffic_rps", "eval_window": 24}).status_code == 200
    assert client.post("/api/v1/scenario", json={"prediction_length": 24}).status_code == 200
    assert client.post("/api/v1/risk", json={"dataset_id": "grid_load_mw", "prediction_length": 24}).status_code == 200
    assert client.post("/api/v1/missing-stress", json={"dataset_id": "weather_temp_c", "missing_rate": 0.25, "holdout_length": 24}).status_code == 200
    assert client.post("/api/v1/multivariate", json={"holdout_length": 24}).status_code == 200

    # 9. Check Crypto Trading Workbench (/crypto, candles, and backtest with IS vs OOS KPIs)
    r_crypto_page = client.get("/crypto")
    assert r_crypto_page.status_code == 200, "/crypto page failed to serve"
    r_candles = client.get("/api/v1/crypto/candles?symbol=BTCUSDT&interval=1h&limit=50")
    assert r_candles.status_code == 200
    assert len(r_candles.json()["candles"]) >= 30

    r_cbt = client.post(
        "/api/v1/crypto/backtest",
        json={"symbol": "BTCUSDT", "interval": "1h", "model_size": "5M", "in_sample_ratio": 0.70},
    )
    assert r_cbt.status_code == 200, r_cbt.text
    cbt = r_cbt.json()
    assert "profit_factor" in cbt["kpis"]["out_of_sample"]
    assert "win_rate_pct" in cbt["kpis"]["out_of_sample"]
    assert "max_drawdown_pct" in cbt["kpis"]["out_of_sample"]
    assert "sharpe_ratio" in cbt["kpis"]["out_of_sample"]
    assert "action" in cbt["live_signal"]
    assert len(cbt["equity_curve"]) > 1, "Equity curve must be continuous mark-to-market!"

    # 10. Check Manual Investment Simulator (/api/v1/crypto/simulate-investment)
    r_sim = client.post(
        "/api/v1/crypto/simulate-investment",
        json={"symbol": "BTCUSDT", "interval": "1h", "model_size": "5M", "investment_amount": 1000.0, "leverage": 2.0},
    )
    assert r_sim.status_code == 200, r_sim.text
    sim = r_sim.json()
    assert "win_probability_pct" in sim
    assert "scenarios" in sim
    assert "base_case_q50" in sim["scenarios"]
    assert "bullish_target_q90" in sim["scenarios"]
    assert "bearish_risk_q10" in sim["scenarios"]

    print("ALL CHECKS PASSED: zero-load /health, 5M default, on-demand 209M, Pi5 telemetry, manual & TTL unload, and Crypto Workbench verified.")


if __name__ == "__main__":
    main()
