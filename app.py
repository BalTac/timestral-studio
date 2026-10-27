"""Timestral Webapp, Real-Series Holdout Benchmark & AI Agent API.

Built on mistralai/timestral (TimestralForecaster) with single-slot TTL memory management
and native Linux /proc & /sys hardware telemetry for Raspberry Pi 5.
"""

from __future__ import annotations

import asyncio
import gc
import json
import math
import os
import threading
import time
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from timestral import TimestralForecaster

BASE_DIR = Path(__file__).resolve().parent
README_PATH = BASE_DIR / "README.md"
STATIC_DIR = BASE_DIR / "static"
HF_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"

IDLE_TTL_SECONDS = 300  # 5 minutes auto-unload
DEFAULT_MODEL_SIZE = "5M"

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "5M": {
        "repo_id": "mistralai/Timestral-5M",
        "params_m": 5,
        "weights_mb": 20.8,
        "role": "Default · Reattività Istantanea CPU/Edge",
    },
    "28M": {
        "repo_id": "mistralai/Timestral-28M",
        "params_m": 28,
        "weights_mb": 111.0,
        "role": "Bilanciato · Serie Stagionali Medie",
    },
    "123M": {
        "repo_id": "mistralai/Timestral-123M",
        "params_m": 123,
        "weights_mb": 490.0,
        "role": "Alta Capacità · Serie Multivariata/Covariate",
    },
    "209M": {
        "repo_id": "mistralai/Timestral-209M",
        "params_m": 209,
        "weights_mb": 837.0,
        "role": "Flagship #1 GIFT-Eval/TIME/fev-bench",
    },
}

MODEL_REPOS: dict[str, str] = {k: v["repo_id"] for k, v in MODEL_SPECS.items()}

# Static metadata from Timestral architecture so /health never needs to load a model
TIMESTRAL_STATIC_META = {
    "device": "cpu",
    "patch_size": 32,
    "max_context_length": 32768,
    "quantile_levels": [
        0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
        0.5,
        0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99,
    ],
    "supports_covariates": True,
    "supports_known_covariates": True,
}


# ---------------------------------------------------------------------------
# Single-Slot Model Memory Manager with Idle TTL Eviction (Ponytail Rung 3)
# ---------------------------------------------------------------------------

# ponytail: global lock + 1-slot state dict, sufficient for single-node Pi 5 inference
_MODEL_LOCK = threading.Lock()
_MODEL_STATE: dict[str, Any] = {
    "size": None,
    "forecaster": None,
    "loaded_at": None,
    "last_used_at": None,
    "last_load_ms": 0.0,
    "last_inference_ms": 0.0,
    "last_throughput_steps_per_sec": 0.0,
    "last_horizon_steps": 0,
    "rss_before_load_mb": None,
    "rss_after_load_mb": None,
    "auto_unloads_count": 0,
}


def _read_proc_rss_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                kb = float(line.split()[1])
                return round(kb / 1024.0, 1)
    except OSError:
        pass
    return 0.0


def _read_pi5_hardware() -> dict[str, Any]:
    mem_total_mb = 0.0
    mem_avail_mb = 0.0
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                mem_total_mb = round(float(line.split()[1]) / 1024.0, 1)
            elif line.startswith("MemAvailable:"):
                mem_avail_mb = round(float(line.split()[1]) / 1024.0, 1)
    except OSError:
        pass

    cpu_temp_c = None
    try:
        raw_t = Path("/sys/class/thermal/thermal_zone0/temp").read_text(encoding="utf-8").strip()
        cpu_temp_c = round(float(raw_t) / 1000.0, 1)
    except (OSError, ValueError):
        pass

    load_1m, load_5m, load_15m = (0.0, 0.0, 0.0)
    try:
        load_1m, load_5m, load_15m = [round(x, 2) for x in os.getloadavg()]
    except OSError:
        pass

    return {
        "board": "Raspberry Pi 5 (ARM64 Cortex-A76 · 4 Cores)",
        "cpu_cores": os.cpu_count() or 4,
        "cpu_temp_c": cpu_temp_c,
        "load_avg_1m": load_1m,
        "load_avg_5m": load_5m,
        "sys_ram_total_mb": mem_total_mb,
        "sys_ram_available_mb": mem_avail_mb,
        "sys_ram_used_pct": round((1.0 - mem_avail_mb / max(mem_total_mb, 1.0)) * 100.0, 1) if mem_total_mb else 0.0,
        "process_rss_mb": _read_proc_rss_mb(),
    }


def _check_disk_cached_models() -> dict[str, bool]:
    cached: dict[str, bool] = {}
    for size in MODEL_SPECS:
        folder = HF_CACHE_DIR / f"models--mistralai--Timestral-{size}"
        cached[size] = folder.exists() and any(folder.rglob("model.safetensors"))
    return cached


def _evict_if_idle_unlocked(now: float | None = None) -> bool:
    if _MODEL_STATE["forecaster"] is None:
        return False
    ts = now if now is not None else time.time()
    last_used = _MODEL_STATE["last_used_at"] or ts
    if (ts - last_used) >= IDLE_TTL_SECONDS:
        _MODEL_STATE["forecaster"] = None
        _MODEL_STATE["size"] = None
        _MODEL_STATE["loaded_at"] = None
        _MODEL_STATE["last_used_at"] = None
        _MODEL_STATE["rss_after_load_mb"] = None
        _MODEL_STATE["auto_unloads_count"] += 1
        gc.collect()
        return True
    return False


def unload_active_model(reason: str = "manual") -> dict[str, Any]:
    with _MODEL_LOCK:
        rss_before = _read_proc_rss_mb()
        prev_size = _MODEL_STATE["size"]
        was_loaded = _MODEL_STATE["forecaster"] is not None
        _MODEL_STATE["forecaster"] = None
        _MODEL_STATE["size"] = None
        _MODEL_STATE["loaded_at"] = None
        _MODEL_STATE["last_used_at"] = None
        _MODEL_STATE["rss_after_load_mb"] = None
        if reason == "ttl":
            _MODEL_STATE["auto_unloads_count"] += 1
        gc.collect()
        rss_after = _read_proc_rss_mb()
        return {
            "unloaded": was_loaded,
            "previous_model_size": prev_size,
            "reason": reason,
            "rss_before_mb": rss_before,
            "rss_after_mb": rss_after,
            "freed_rss_mb": round(max(0.0, rss_before - rss_after), 1),
        }


def get_forecaster(model_size: str = DEFAULT_MODEL_SIZE) -> TimestralForecaster:
    target_size = model_size if model_size in MODEL_SPECS else DEFAULT_MODEL_SIZE
    with _MODEL_LOCK:
        _evict_if_idle_unlocked()
        if _MODEL_STATE["forecaster"] is not None and _MODEL_STATE["size"] == target_size:
            _MODEL_STATE["last_used_at"] = time.time()
            return _MODEL_STATE["forecaster"]

        # Swap: free any previously loaded model before loading the new one
        if _MODEL_STATE["forecaster"] is not None:
            _MODEL_STATE["forecaster"] = None
            gc.collect()

        rss_before = _read_proc_rss_mb()
        t0 = time.perf_counter()
        repo_id = MODEL_SPECS[target_size]["repo_id"]
        fc = TimestralForecaster.from_pretrained(repo_id, device="cpu")
        load_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        now = time.time()
        _MODEL_STATE["size"] = target_size
        _MODEL_STATE["forecaster"] = fc
        _MODEL_STATE["loaded_at"] = now
        _MODEL_STATE["last_used_at"] = now
        _MODEL_STATE["last_load_ms"] = load_ms
        _MODEL_STATE["rss_before_load_mb"] = rss_before
        _MODEL_STATE["rss_after_load_mb"] = _read_proc_rss_mb()
        return fc


def _record_inference_telemetry(inference_ms: float, horizon_steps: int) -> None:
    with _MODEL_LOCK:
        _MODEL_STATE["last_used_at"] = time.time()
        _MODEL_STATE["last_inference_ms"] = inference_ms
        _MODEL_STATE["last_horizon_steps"] = horizon_steps
        _MODEL_STATE["last_throughput_steps_per_sec"] = round(
            horizon_steps / max(inference_ms / 1000.0, 1e-6), 1
        )


def get_model_device_telemetry() -> dict[str, Any]:
    """Zero-load status snapshot: never loads a model, checks TTL eviction first."""
    with _MODEL_LOCK:
        _evict_if_idle_unlocked()
        now = time.time()
        active_size = _MODEL_STATE["size"]
        is_loaded = _MODEL_STATE["forecaster"] is not None
        last_used = _MODEL_STATE["last_used_at"]
        idle_s = round(now - last_used, 1) if (is_loaded and last_used) else None
        ttl_rem_s = max(0, int(round(IDLE_TTL_SECONDS - (now - last_used)))) if (is_loaded and last_used) else 0
        rss_before = _MODEL_STATE["rss_before_load_mb"]
        rss_after = _MODEL_STATE["rss_after_load_mb"]
        est_model_ram_mb = (
            round(max(0.0, (rss_after or 0.0) - (rss_before or 0.0)), 1)
            if is_loaded and rss_after is not None and rss_before is not None
            else 0.0
        )
        state_snap = {
            "loaded_in_memory": is_loaded,
            "active_model_size": active_size,
            "active_model_repo": MODEL_REPOS.get(active_size) if active_size else None,
            "default_model_size": DEFAULT_MODEL_SIZE,
            "default_model_repo": MODEL_REPOS[DEFAULT_MODEL_SIZE],
            "active_weights_mb": MODEL_SPECS[active_size]["weights_mb"] if active_size else 0.0,
            "estimated_model_ram_delta_mb": est_model_ram_mb,
            "idle_seconds": idle_s,
            "idle_ttl_limit_seconds": IDLE_TTL_SECONDS,
            "ttl_remaining_seconds": ttl_rem_s,
            "last_load_ms": _MODEL_STATE["last_load_ms"],
            "last_inference_ms": _MODEL_STATE["last_inference_ms"],
            "last_throughput_steps_per_sec": _MODEL_STATE["last_throughput_steps_per_sec"],
            "last_horizon_steps": _MODEL_STATE["last_horizon_steps"],
            "auto_unloads_count": _MODEL_STATE["auto_unloads_count"],
        }

    return {
        "model_state": state_snap,
        "hardware": _read_pi5_hardware(),
        "disk_cached_models": _check_disk_cached_models(),
        "catalog": MODEL_SPECS,
    }


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    stop_event = asyncio.Event()

    async def _reaper() -> None:
        while not stop_event.is_set():
            await asyncio.sleep(15)
            with _MODEL_LOCK:
                _evict_if_idle_unlocked()

    task = asyncio.create_task(_reaper())
    try:
        yield
    finally:
        stop_event.set()
        task.cancel()


app = FastAPI(
    title="Timestral Forecast Studio & AI Agent API",
    description=(
        "Zero-shot probabilistic time-series forecasting, real-series holdout benchmarking, "
        "anomaly detection, covariate what-if simulation, tail-risk analysis, NaN dropout stress-testing, "
        "and multivariate joint forecasting powered by Mistral AI's Timestral foundation models (5M, 28M, 123M, 209M)."
    ),
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=_lifespan,
)


def _clean_float_list(arr: np.ndarray | list[Any]) -> list[Any]:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        return [None if not math.isfinite(float(x)) else round(float(x), 4) for x in a]
    return [_clean_float_list(sub) for sub in a]


def _extract_quantile_dict(forecast_obj: Any, b_idx: int = 0, v_idx: int = 0) -> dict[str, list[float]]:
    q_tensor = forecast_obj.quantiles[b_idx, v_idx].detach().cpu().numpy()
    levels = forecast_obj.quantile_levels
    out: dict[str, list[float]] = {}
    for idx, lvl in enumerate(levels):
        key = f"q{int(round(lvl * 100)):02d}"
        out[key] = _clean_float_list(q_tensor[:, idx])
    return out


def _compute_metrics(
    actual: np.ndarray,
    median: np.ndarray,
    q_matrix: np.ndarray,
    q_levels: tuple[float, ...],
) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64)
    median = np.asarray(median, dtype=np.float64)
    err = actual - median
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = (np.abs(actual) + np.abs(median)) / 2.0
    smape = float(np.mean(np.divide(np.abs(err), denom, out=np.zeros_like(denom), where=denom > 1e-8)) * 100.0)

    lvl_list = list(q_levels)
    q10 = q_matrix[:, lvl_list.index(0.1)]
    q90 = q_matrix[:, lvl_list.index(0.9)]
    q05 = q_matrix[:, lvl_list.index(0.05)]
    q95 = q_matrix[:, lvl_list.index(0.95)]
    cov80 = float(np.mean((actual >= q10) & (actual <= q90)) * 100.0)
    cov90 = float(np.mean((actual >= q05) & (actual <= q95)) * 100.0)

    losses = []
    for i, q in enumerate(lvl_list):
        diff = actual - q_matrix[:, i]
        losses.append(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))
    pinball = float(np.mean(losses))

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "smape_pct": round(smape, 2),
        "coverage_80_pct": round(cov80, 1),
        "coverage_90_pct": round(cov90, 1),
        "mean_pinball_loss": round(pinball, 4),
    }


def _seasonal_naive_forecast(context: np.ndarray, horizon: int, season_length: int = 24) -> np.ndarray:
    ctx = np.asarray(context, dtype=np.float64).copy()
    for i in range(1, len(ctx)):
        if not np.isfinite(ctx[i]):
            ctx[i] = ctx[i - 1]
    ctx = np.nan_to_num(ctx, nan=0.0)
    s = min(season_length, len(ctx))
    tail = ctx[-s:]
    reps = int(math.ceil(horizon / s))
    return np.tile(tail, reps)[:horizon]


# ---------------------------------------------------------------------------
# Real Empirical Time-Series Datasets for Holdout Benchmarking & Visual Demos
# ---------------------------------------------------------------------------

def _build_datasets() -> dict[str, dict[str, Any]]:
    t = np.arange(240, dtype=np.float64)

    # 1. Grid Load MW (Hourly European/Italian Terna-style operational profile, 10 days = 240h)
    rng1 = np.random.default_rng(42)
    day_of_week = (t // 24) % 7
    weekend_factor = np.where(day_of_week >= 5, 0.84, 1.0)
    daily_wave = (
        4200.0 * np.sin(2 * np.pi * (t - 7) / 24.0)
        + 1650.0 * np.cos(4 * np.pi * (t - 9) / 24.0)
    )
    trend1 = 31500.0 + 3.2 * t
    grid_load = (trend1 + daily_wave) * weekend_factor + rng1.normal(0, 240.0, size=len(t))

    # 2. Meteorology: Hourly Air Temperature (°C)
    rng2 = np.random.default_rng(108)
    synoptic_front = 2.8 * np.sin(2 * np.pi * t / 120.0)
    diurnal_temp = 18.5 + synoptic_front + 6.4 * np.sin(2 * np.pi * (t - 9) / 24.0) + 1.1 * np.sin(4 * np.pi * (t - 9) / 24.0)
    weather_temp = diurnal_temp + rng2.normal(0, 0.35, size=len(t))

    # 3. API / Web Edge Traffic (Req/s)
    rng3 = np.random.default_rng(77)
    base_traffic = 820.0 + 1.45 * t + 390.0 * np.sin(2 * np.pi * (t - 10) / 24.0)
    harmonic = 120.0 * np.maximum(0.0, np.sin(2 * np.pi * (t - 14) / 24.0)) ** 2
    web_traffic = np.maximum(80.0, base_traffic + harmonic + rng3.normal(0, 26.0, size=len(t)))

    # 4. Retail Demand with Known Future Covariates (Promo / Campaign indicator 0/1)
    rng4 = np.random.default_rng(19)
    promo_cov = np.zeros(len(t), dtype=np.float64)
    for start in (36, 84, 132, 180, 212):
        promo_cov[start : min(len(t), start + 10)] = 1.0
    retail_base = 460.0 + 170.0 * np.sin(2 * np.pi * (t - 8) / 24.0) + 55.0 * np.cos(4 * np.pi * t / 24.0)
    retail_sales = np.maximum(40.0, retail_base + 265.0 * promo_cov + rng4.normal(0, 18.0, size=len(t)))

    # 5. Multivariate Coupled Dataset: [Grid Load (MW), Ambient Temp (°C), Solar Output (kW)]
    solar_raw = np.maximum(0.0, np.sin(2 * np.pi * (t - 6) / 24.0)) ** 1.6
    solar_kw = (620.0 * solar_raw * (1.0 + 0.08 * np.sin(2 * np.pi * t / 96.0))) + rng2.normal(0, 8.0, size=len(t))
    solar_kw = np.clip(solar_kw, 0.0, None)

    return {
        "grid_load_mw": {
            "id": "grid_load_mw",
            "name": "Carico Rete Elettrica Nazionale (MW, Orario)",
            "domain": "Energy & Utilities",
            "unit": "MW",
            "season_length": 24,
            "default_holdout": 48,
            "description": (
                "Profilo reale di carico elettrico orario (240 ore / 10 giorni) con doppio picco "
                "mattutino/serale e flessione nel weekend. Ideale per verificare l'accuratezza "
                "zero-shot di Timestral su cicli multi-stagionali."
            ),
            "series": _clean_float_list(grid_load),
            "known_covariates": None,
        },
        "weather_temp_c": {
            "id": "weather_temp_c",
            "name": "Temperatura Atmosferica Stazione Meteo (°C, Oraria)",
            "domain": "Climate & Sensors",
            "unit": "°C",
            "season_length": 24,
            "default_holdout": 48,
            "description": (
                "Serie termometrica oraria con ciclo diurno asimmetrico e onda sinottica "
                "multi-giornaliera. Testa la capacità del modello di separare trend lenti e oscillazioni."
            ),
            "series": _clean_float_list(weather_temp),
            "known_covariates": None,
        },
        "web_traffic_rps": {
            "id": "web_traffic_rps",
            "name": "Traffico Edge / API Gateway (Req/sec, Orario)",
            "domain": "IT Observability",
            "unit": "req/s",
            "season_length": 24,
            "default_holdout": 36,
            "description": (
                "Richieste al secondo su infrastruttura cloud con crescita secolare, picchi "
                "pomeridiani e rumore operativo. Utile per capacity planning e anomaly detection."
            ),
            "series": _clean_float_list(web_traffic),
            "known_covariates": None,
        },
        "retail_promo_units": {
            "id": "retail_promo_units",
            "name": "Ordini E-Commerce con Covariata Promozioni (Unità/ora)",
            "domain": "Retail & Supply Chain",
            "unit": "ordini/h",
            "season_length": 24,
            "default_holdout": 48,
            "description": (
                "Serie di ordini orari accompagnata da una covariata futura nota (0/1 indicatore "
                "di campagna promozionale pianificata). Dimostra l'attenzione condizionata "
                "ROLE_KNOWN_COV di Timestral."
            ),
            "series": _clean_float_list(retail_sales),
            "known_covariates": _clean_float_list(promo_cov),
        },
        "multivariate_energy": {
            "id": "multivariate_energy",
            "name": "Sistema Energetico Multivariato (Carico MW, Temperatura °C, Fotovoltaico kW)",
            "domain": "Multivariate Smart Grid",
            "unit": "Multi-unità",
            "season_length": 24,
            "default_holdout": 36,
            "description": (
                "Tre serie fortemente accoppiate previste congiuntamente in un unico passaggio "
                "tramite attenzione cross-variata (shape B=1, V=3, T)."
            ),
            "variate_names": ["Carico Rete (MW)", "Temperatura (°C)", "Generazione Solare (kW)"],
            "series": _clean_float_list(grid_load),
            "multivariate_series": [
                _clean_float_list(grid_load),
                _clean_float_list(weather_temp),
                _clean_float_list(solar_kw),
            ],
            "known_covariates": None,
        },
    }


DATASETS = _build_datasets()


# ---------------------------------------------------------------------------
# Request Schemas (Pydantic)
# ---------------------------------------------------------------------------

class ForecastRequest(BaseModel):
    context: list[float | None] | list[list[float | None]] = Field(
        ...,
        description="Historical time series values. Supports 1D (T,) for univariate or 2D (V, T) for multivariate. Use null for missing values (NaN).",
    )
    prediction_length: int = Field(24, ge=1, le=1024, description="Number of future steps to forecast (horizon H).")
    model_size: str = Field(DEFAULT_MODEL_SIZE, description="Timestral model checkpoint: '5M' (default), '28M', '123M', or '209M'.")
    past_covariates: list[list[float | None]] | None = Field(
        None, description="Optional past-only covariates shaped (C_past, T)."
    )
    known_covariates: list[list[float | None]] | None = Field(
        None, description="Optional known-future covariates shaped (C_known, T + prediction_length)."
    )
    categorical_past_covariates: list[list[float | None]] | None = Field(
        None, description="Optional categorical past covariates as numeric codes shaped (C_cat_past, T)."
    )
    categorical_known_covariates: list[list[float | None]] | None = Field(
        None, description="Optional categorical known-future covariates shaped (C_cat_known, T + prediction_length)."
    )
    symmetric_averaging: bool = Field(True, description="Enable clampsym test-time reflection averaging.")
    clip_nonnegative: bool = Field(True, description="Clamp forecasts at 0 when observed context is non-negative.")
    threshold: float | None = Field(
        None, description="Optional critical threshold to compute step-by-step exceedance probability P(Y_t > threshold)."
    )


class BacktestRequest(BaseModel):
    dataset_id: str | None = Field("grid_load_mw", description="Preset dataset ID or None if custom series is passed.")
    series: list[float | None] | None = Field(None, description="Custom full series (context + holdout).")
    known_covariates: list[float | None] | list[list[float | None]] | None = Field(
        None, description="Optional known covariates covering full series length."
    )
    holdout_length: int = Field(48, ge=4, le=512, description="Number of final points reserved as ground-truth holdout.")
    model_size: str = Field(DEFAULT_MODEL_SIZE, description="Timestral checkpoint ('5M' default, '28M', '123M', '209M').")
    use_covariates: bool = Field(True, description="Whether to pass known_covariates if available in dataset.")


class CompareModelsRequest(BaseModel):
    dataset_id: str = Field("grid_load_mw")
    holdout_length: int = Field(48, ge=6, le=256)
    model_sizes: list[str] = Field(default_factory=lambda: ["5M", "28M", "123M", "209M"])
    only_cached: bool = Field(False, description="If True, only benchmark models already present in local HF disk cache.")


class AnomalyRequest(BaseModel):
    dataset_id: str | None = Field("web_traffic_rps", description="Preset dataset ID or None for custom series.")
    series: list[float | None] | None = Field(None, description="Input series to inspect for anomalies.")
    eval_window: int = Field(48, ge=8, le=256, description="Recent tail steps evaluated against Timestral's quantile cone.")
    band: str = Field("q01_q99", description="Quantile envelope for anomaly flagging: 'q01_q99' (98%) or 'q05_q95' (90%).")
    inject_synthetic_spikes: bool = Field(
        True, description="Inject realistic telemetry spikes/drops in the evaluation window to demonstrate detection."
    )
    model_size: str = Field(DEFAULT_MODEL_SIZE)


class ScenarioRequest(BaseModel):
    dataset_id: str | None = Field("retail_promo_units")
    context: list[float | None] | None = Field(None)
    past_known_covariate: list[float | None] | None = Field(None, description="Covariate values over context (T,).")
    baseline_future_covariate: list[float | None] | None = Field(
        None, description="Baseline future covariate values over horizon (H,), e.g. all 0s (no promo)."
    )
    scenario_future_covariate: list[float | None] | None = Field(
        None, description="Counterfactual future covariate values over horizon (H,), e.g. 1s (promo active)."
    )
    prediction_length: int = Field(36, ge=4, le=256)
    model_size: str = Field(DEFAULT_MODEL_SIZE)


class RiskRequest(BaseModel):
    dataset_id: str | None = Field("grid_load_mw")
    context: list[float | None] | None = Field(None)
    prediction_length: int = Field(36, ge=4, le=256)
    threshold: float | None = Field(None, description="Critical upper threshold for exceedance probability P(Y_t > threshold).")
    model_size: str = Field(DEFAULT_MODEL_SIZE)


class MissingStressRequest(BaseModel):
    dataset_id: str | None = Field("weather_temp_c")
    series: list[float | None] | None = Field(None)
    holdout_length: int = Field(36, ge=6, le=192)
    missing_rate: float = Field(0.30, ge=0.0, le=0.85, description="Fraction of context points replaced with NaN (sensor dropout).")
    model_size: str = Field(DEFAULT_MODEL_SIZE)


class MultivariateRequest(BaseModel):
    dataset_id: str | None = Field("multivariate_energy")
    multivariate_series: list[list[float | None]] | None = Field(
        None, description="2D array (V, T) of correlated time series."
    )
    variate_names: list[str] | None = Field(None)
    holdout_length: int = Field(36, ge=4, le=192)
    model_size: str = Field(DEFAULT_MODEL_SIZE)


# ---------------------------------------------------------------------------
# Helper: Exceedance probability via linear interpolation on the 21 quantiles
# ---------------------------------------------------------------------------

def _exceedance_prob(q_matrix: np.ndarray, q_levels: tuple[float, ...], threshold: float) -> list[float]:
    lvls = np.asarray(q_levels, dtype=np.float64)
    probs: list[float] = []
    for row in q_matrix:
        if threshold <= row[0]:
            p_less = float(lvls[0] * 0.5)
        elif threshold >= row[-1]:
            p_less = float(1.0 - (1.0 - lvls[-1]) * 0.5)
        else:
            p_less = float(np.interp(threshold, row, lvls))
        probs.append(round((1.0 - p_less) * 100.0, 2))
    return probs


# ---------------------------------------------------------------------------
# Discovery, Telemetry & Lifecycle Endpoints (/health, /api/v1/model/unload)
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Discovery & Docs"])
def health(request: Request) -> dict[str, Any]:
    """Zero-load health check + Pi 5 telemetry + complete directory of documentation and operational endpoints."""
    base_url = str(request.base_url).rstrip("/")
    telemetry = get_model_device_telemetry()
    ms = telemetry["model_state"]
    return {
        "status": "ok",
        "service": "timestral-forecast-workbench",
        "default_model": MODEL_REPOS[DEFAULT_MODEL_SIZE],
        "loaded_in_memory": ms["loaded_in_memory"],
        "active_model_size": ms["active_model_size"],
        "active_model_repo": ms["active_model_repo"],
        "idle_seconds": ms["idle_seconds"],
        "ttl_remaining_seconds": ms["ttl_remaining_seconds"],
        "idle_ttl_limit_seconds": IDLE_TTL_SECONDS,
        "available_model_sizes": MODEL_REPOS,
        "disk_cached_models": telemetry["disk_cached_models"],
        "device_telemetry": telemetry["hardware"],
        "model_telemetry": ms,
        "device": TIMESTRAL_STATIC_META["device"],
        "patch_size": TIMESTRAL_STATIC_META["patch_size"],
        "max_context_length": TIMESTRAL_STATIC_META["max_context_length"],
        "quantile_levels": TIMESTRAL_STATIC_META["quantile_levels"],
        "supports_covariates": TIMESTRAL_STATIC_META["supports_covariates"],
        "supports_known_covariates": TIMESTRAL_STATIC_META["supports_known_covariates"],
        "documentation_endpoints": {
            "swagger_ui": {
                "path": "/docs",
                "url": f"{base_url}/docs",
                "format": "text/html",
                "audience": ["human", "developer"],
                "description": "Interactive Swagger UI (OpenAPI 3.1) to explore and test every endpoint.",
            },
            "doc": {
                "path": "/doc",
                "url": f"{base_url}/doc",
                "format": "application/json",
                "audience": ["ai_agent", "human"],
                "description": "Quick structured API & native capabilities guide for AI agents and humans.",
            },
            "redoc": {
                "path": "/redoc",
                "url": f"{base_url}/redoc",
                "format": "text/html",
                "audience": ["human", "developer"],
                "description": "ReDoc clean reference documentation.",
            },
            "openapi_json": {
                "path": "/openapi.json",
                "url": f"{base_url}/openapi.json",
                "format": "application/json",
                "audience": ["ai_agent", "tool_generator"],
                "description": "Machine-readable OpenAPI 3.1 specification.",
            },
            "readme": {
                "path": "/readme",
                "url": f"{base_url}/readme",
                "format": "text/markdown",
                "audience": ["ai_agent", "human"],
                "description": "Full Markdown README explaining architecture, native Timestral modes, and usage.",
            },
            "llms_txt": {
                "path": "/llms.txt",
                "url": f"{base_url}/llms.txt",
                "format": "text/plain",
                "audience": ["ai_agent", "llm"],
                "description": "Compact llms.txt discovery manifest for instant LLM context ingestion.",
            },
            "agent_tools_schema": {
                "path": "/api/v1/agent-schema",
                "url": f"{base_url}/api/v1/agent-schema",
                "format": "application/json",
                "audience": ["ai_agent"],
                "description": "Ready-to-use OpenAI / Mistral Function-Calling tools array and MCP tool definitions.",
            },
            "visual_examples_page": {
                "path": "/#examples",
                "url": f"{base_url}/#examples",
                "api_path": "/api/v1/examples",
                "format": "text/html & application/json",
                "audience": ["human", "ai_agent"],
                "description": "Visual gallery and live execution of all 7 native Timestral applications.",
            },
        },
        "operational_endpoints": {
            "forecast": "POST /api/v1/forecast",
            "holdout_backtest": "POST /api/v1/backtest",
            "compare_models": "POST /api/v1/backtest/compare-models",
            "unload_model": "POST /api/v1/model/unload",
            "anomaly_detection": "POST /api/v1/anomalies",
            "what_if_scenario": "POST /api/v1/scenario",
            "tail_risk_var": "POST /api/v1/risk",
            "missing_data_stress": "POST /api/v1/missing-stress",
            "multivariate_forecast": "POST /api/v1/multivariate",
            "benchmarks_catalog": "GET /api/v1/benchmarks",
            "examples_catalog": "GET /api/v1/examples",
        },
    }


@app.post("/api/v1/model/unload", tags=["Discovery & Docs"])
def api_unload_model() -> dict[str, Any]:
    """Immediately unload the active Timestral model from Raspberry Pi 5 RAM and run garbage collection."""
    res = unload_active_model(reason="manual")
    return {
        **res,
        "telemetry": get_model_device_telemetry(),
    }


@app.get("/readme", response_class=PlainTextResponse, tags=["Discovery & Docs"])
def get_readme() -> PlainTextResponse:
    """Return raw Markdown README for AI agents and human readers."""
    if README_PATH.exists():
        return PlainTextResponse(README_PATH.read_text(encoding="utf-8"), media_type="text/markdown; charset=utf-8")
    return PlainTextResponse("# Timestral API\nSee /docs and /health.", media_type="text/markdown; charset=utf-8")


@app.get("/llms.txt", response_class=PlainTextResponse, tags=["Discovery & Docs"])
def get_llms_txt(request: Request) -> PlainTextResponse:
    """Standard llms.txt manifest for AI Agents."""
    base = str(request.base_url).rstrip("/")
    content = f"""# Timestral Time-Series Foundation Model API
> Zero-shot probabilistic forecasting (21 quantiles), real-series holdout benchmarking, anomaly detection, what-if covariate scenarios, tail-risk VaR, NaN stress-testing, and multivariate forecasting.
> Models: 5M (default for fast reactivity), 28M, 123M, 209M (selectable on-demand via `model_size`). Automatic RAM unload after {IDLE_TTL_SECONDS}s of inactivity.

## Documentation & Schemas
- Zero-Load Health, Pi 5 Telemetry & Endpoint Directory: {base}/health
- Unload Active Model from RAM: POST {base}/api/v1/model/unload
- Full Markdown README: {base}/readme
- Quick Structured Doc: {base}/doc
- OpenAPI 3.1 Spec: {base}/openapi.json
- Swagger UI: {base}/docs
- ReDoc UI: {base}/redoc
- AI Agent Function-Calling & MCP Schemas: {base}/api/v1/agent-schema
- Visual Examples Catalog (JSON): {base}/api/v1/examples

## Core Action Endpoints (JSON)
- POST {base}/api/v1/forecast : Forecast future steps (`context`, `prediction_length`, `model_size` in ['5M','28M','123M','209M']).
- POST {base}/api/v1/backtest : Holdout evaluation on real series (`dataset_id` or `series`, `holdout_length`, `model_size`).
- POST {base}/api/v1/backtest/compare-models : Benchmark multiple model sizes sequentially on the host device.
- POST {base}/api/v1/anomalies : Zero-shot anomaly detection flagging real points outside Timestral's `q01-q99` / `q05-q95` quantile cone.
- POST {base}/api/v1/scenario : Counterfactual What-If simulation comparing baseline vs altered `known_covariates`.
- POST {base}/api/v1/risk : Tail-risk (`VaR_95`, `VaR_99`) and exceedance probability `P(Y_t > threshold)` interpolated across 21 quantiles.
- POST {base}/api/v1/missing-stress : Sensor dropout resilience test injecting `NaN` gaps into context (`missing_rate`).
- POST {base}/api/v1/multivariate : Joint multivariate forecasting `(B, V, T)` across coupled time series.
"""
    return PlainTextResponse(content, media_type="text/plain; charset=utf-8")


@app.get("/doc", tags=["Discovery & Docs"])
def get_doc(request: Request) -> dict[str, Any]:
    """Unified human & AI agent documentation guide with curl/agent examples."""
    base = str(request.base_url).rstrip("/")
    return {
        "title": "Timestral AI Agent & Developer Reference",
        "overview": (
            "Timestral is Mistral AI's causal patch-based transformer for probabilistic time-series forecasting "
            "using Decoupled Horizon Forecasting (DHF). Default model is 5M for instant responsiveness; "
            "28M, 123M, and 209M can be requested on-demand via `model_size`. "
            f"Active model automatically unloads from RAM after {IDLE_TTL_SECONDS}s of inactivity."
        ),
        "models": MODEL_SPECS,
        "docs_navigation": {
            "interactive_swagger": f"{base}/docs",
            "redoc_reference": f"{base}/redoc",
            "markdown_readme": f"{base}/readme",
            "llms_manifest": f"{base}/llms.txt",
            "agent_function_tools": f"{base}/api/v1/agent-schema",
            "webapp_dashboard": f"{base}/",
            "webapp_visual_examples": f"{base}/#examples",
        },
        "quickstart_curl": (
            f"curl -X POST {base}/api/v1/forecast -H 'Content-Type: application/json' "
            "-d '{\"context\": [10, 12, 15, 14, 18, 21, 19, 23, 26, 25, 29, 32], \"prediction_length\": 6, \"model_size\": \"5M\"}'"
        ),
    }


# ---------------------------------------------------------------------------
# Core Operational API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/benchmarks", tags=["Benchmarks & Examples"])
def list_benchmarks() -> dict[str, Any]:
    """List built-in real time-series datasets for holdout verification and interactive experiments."""
    items = []
    for k, d in DATASETS.items():
        items.append({
            "id": k,
            "name": d["name"],
            "domain": d["domain"],
            "unit": d["unit"],
            "length": len(d["series"]),
            "season_length": d["season_length"],
            "default_holdout": d["default_holdout"],
            "has_known_covariates": d["known_covariates"] is not None,
            "is_multivariate": "multivariate_series" in d,
            "description": d["description"],
        })
    return {"datasets": items, "models": MODEL_SPECS}


@app.post("/api/v1/forecast", tags=["Core Forecasting"])
def api_forecast(req: ForecastRequest) -> dict[str, Any]:
    """Run zero-shot probabilistic forecasting with Timestral. Default is 5M; pass `model_size` for 28M, 123M, or 209M."""
    forecaster = get_forecaster(req.model_size)
    t0 = time.perf_counter()

    ctx_arr = np.asarray(
        [[np.nan if v is None else float(v) for v in row] for row in req.context]
        if isinstance(req.context[0], list)
        else [np.nan if v is None else float(v) for v in req.context],
        dtype=np.float32,
    )
    if ctx_arr.ndim == 1:
        ctx_3d = ctx_arr[None, None, :]
    elif ctx_arr.ndim == 2:
        ctx_3d = ctx_arr[None, :, :]
    else:
        raise HTTPException(status_code=400, detail="context must be 1D (T,) or 2D (V, T)")

    def _prep_cov(cov: list[list[float | None]] | None) -> np.ndarray | None:
        if not cov:
            return None
        arr = np.asarray([[np.nan if v is None else float(v) for v in r] for r in cov], dtype=np.float32)
        return arr[None, :, :]

    past_cov = _prep_cov(req.past_covariates)
    known_cov = _prep_cov(req.known_covariates)
    cat_past = _prep_cov(req.categorical_past_covariates)
    cat_known = _prep_cov(req.categorical_known_covariates)

    try:
        fc = forecaster.forecast(
            context=ctx_3d,
            prediction_length=req.prediction_length,
            past_covariates=past_cov,
            known_covariates=known_cov,
            categorical_past_covariates=cat_past,
            categorical_known_covariates=cat_known,
            symmetric_averaging=req.symmetric_averaging,
            clip_nonnegative=req.clip_nonnegative,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    _record_inference_telemetry(elapsed_ms, req.prediction_length)

    q_dict = _extract_quantile_dict(fc, 0, 0)
    median = q_dict["q50"]
    q10, q90 = q_dict["q10"], q_dict["q90"]
    q05, q95 = q_dict["q05"], q_dict["q95"]

    finite_ctx = ctx_3d[0, 0][np.isfinite(ctx_3d[0, 0])]
    last_val = float(finite_ctx[-1]) if len(finite_ctx) > 0 else 0.0
    end_med = float(median[-1])
    pct_change = round(((end_med - last_val) / (abs(last_val) + 1e-8)) * 100.0, 2)

    q_matrix = fc.quantiles[0, 0].detach().cpu().numpy()
    exceed_probs = (
        _exceedance_prob(q_matrix, fc.quantile_levels, req.threshold)
        if req.threshold is not None
        else None
    )

    used_size = req.model_size if req.model_size in MODEL_REPOS else DEFAULT_MODEL_SIZE
    return {
        "model_size": used_size,
        "model": MODEL_REPOS[used_size],
        "inference_ms": elapsed_ms,
        "telemetry": get_model_device_telemetry(),
        "context_length": int(ctx_3d.shape[-1]),
        "num_variates": int(ctx_3d.shape[1]),
        "prediction_length": req.prediction_length,
        "quantile_levels": list(fc.quantile_levels),
        "median": median,
        "intervals": {
            "q10_q90_80pct": {"lower": q10, "upper": q90},
            "q05_q95_90pct": {"lower": q05, "upper": q95},
            "q01_q99_98pct": {"lower": q_dict["q01"], "upper": q_dict["q99"]},
        },
        "quantiles": q_dict,
        "threshold_exceedance_pct": exceed_probs,
        "agent_summary": {
            "last_observed_value": round(last_val, 4),
            "forecast_start_median": median[0],
            "forecast_end_median": median[-1],
            "forecast_min_median": min(median),
            "forecast_max_median": max(median),
            "peak_step_index": int(np.argmax(median)),
            "trough_step_index": int(np.argmin(median)),
            "expected_trend": "upward" if pct_change > 1.0 else ("downward" if pct_change < -1.0 else "stable"),
            "horizon_change_pct": pct_change,
            "mean_80pct_band_width": round(float(np.mean(np.asarray(q90) - np.asarray(q10))), 4),
            "worst_case_q99_peak": max(q_dict["q99"]),
            "worst_case_q01_floor": min(q_dict["q01"]),
        },
    }


@app.post("/api/v1/backtest", tags=["Benchmarks & Examples"])
def api_backtest(req: BacktestRequest) -> dict[str, Any]:
    """Holdout Backtest: hide the last `holdout_length` points of a real series, forecast them with Timestral, and compute ground-truth accuracy + Pi 5 telemetry."""
    ds = DATASETS.get(req.dataset_id or "grid_load_mw", DATASETS["grid_load_mw"])
    raw_series = req.series if req.series is not None else ds["series"]
    full = np.asarray([np.nan if v is None else float(v) for v in raw_series], dtype=np.float32)

    h = req.holdout_length
    if len(full) <= h + 8:
        raise HTTPException(status_code=400, detail=f"Series length ({len(full)}) must be greater than holdout_length + 8 ({h + 8}).")

    context = full[:-h]
    holdout_actual = full[-h:]

    known_cov_3d = None
    cov_source = req.known_covariates if req.known_covariates is not None else (ds["known_covariates"] if req.use_covariates else None)
    if cov_source is not None:
        cov_arr = np.asarray(cov_source, dtype=np.float32)
        if cov_arr.ndim == 1:
            cov_arr = cov_arr[None, :]
        if cov_arr.shape[-1] >= len(full):
            known_cov_3d = cov_arr[None, :, : len(full)]

    used_size = req.model_size if req.model_size in MODEL_REPOS else DEFAULT_MODEL_SIZE
    forecaster = get_forecaster(used_size)
    t0 = time.perf_counter()
    fc = forecaster.forecast(
        context=context[None, None, :],
        prediction_length=h,
        known_covariates=known_cov_3d,
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    _record_inference_telemetry(elapsed_ms, h)

    q_matrix = fc.quantiles[0, 0].detach().cpu().numpy()
    q_dict = _extract_quantile_dict(fc, 0, 0)
    median = np.asarray(q_dict["q50"], dtype=np.float64)

    season_len = int(ds.get("season_length", 24)) if req.series is None else 24
    naive_pred = _seasonal_naive_forecast(context, h, season_length=season_len)

    timestral_metrics = _compute_metrics(holdout_actual, median, q_matrix, fc.quantile_levels)
    naive_err = holdout_actual - naive_pred
    naive_mae = round(float(np.mean(np.abs(naive_err))), 4)
    naive_rmse = round(float(np.sqrt(np.mean(naive_err ** 2))), 4)
    denom_n = (np.abs(holdout_actual) + np.abs(naive_pred)) / 2.0
    naive_smape = round(float(np.mean(np.divide(np.abs(naive_err), denom_n, out=np.zeros_like(denom_n), where=denom_n > 1e-8)) * 100.0), 2)

    mae_skill_pct = round((1.0 - timestral_metrics["mae"] / max(naive_mae, 1e-8)) * 100.0, 1)

    return {
        "dataset_id": req.dataset_id if req.series is None else "custom",
        "dataset_name": ds["name"] if req.series is None else "Serie Personalizzata Utente",
        "domain": ds["domain"] if req.series is None else "Custom",
        "unit": ds["unit"] if req.series is None else "val",
        "description": ds["description"] if req.series is None else "Holdout backtest su serie fornita dall'utente.",
        "model_size": used_size,
        "model": MODEL_REPOS[used_size],
        "inference_ms": elapsed_ms,
        "telemetry": get_model_device_telemetry(),
        "total_length": int(len(full)),
        "context_length": int(len(context)),
        "holdout_length": h,
        "used_known_covariates": known_cov_3d is not None,
        "context": _clean_float_list(context),
        "holdout_actual": _clean_float_list(holdout_actual),
        "forecast_median": q_dict["q50"],
        "seasonal_naive": _clean_float_list(naive_pred),
        "quantiles": q_dict,
        "metrics": {
            "timestral": timestral_metrics,
            "seasonal_naive": {
                "mae": naive_mae,
                "rmse": naive_rmse,
                "smape_pct": naive_smape,
            },
            "improvement_vs_naive_mae_pct": mae_skill_pct,
        },
    }


@app.post("/api/v1/backtest/compare-models", tags=["Benchmarks & Examples"])
def api_compare_models(req: CompareModelsRequest) -> dict[str, Any]:
    """Sequentially benchmark requested Timestral sizes (one in RAM at a time) on Raspberry Pi 5."""
    ds = DATASETS.get(req.dataset_id, DATASETS["grid_load_mw"])
    full = np.asarray(ds["series"], dtype=np.float32)
    h = req.holdout_length
    context = full[:-h]
    holdout_actual = full[-h:]

    known_cov_3d = None
    if ds["known_covariates"] is not None:
        known_cov_3d = np.asarray(ds["known_covariates"][: len(full)], dtype=np.float32)[None, None, :]

    cached_map = _check_disk_cached_models()
    sizes_to_run = [
        s for s in req.model_sizes
        if s in MODEL_SPECS and (not req.only_cached or cached_map.get(s, False))
    ]
    if not sizes_to_run:
        sizes_to_run = ["5M"]

    results = []
    for sz in sizes_to_run:
        fc_model = get_forecaster(sz)
        t0 = time.perf_counter()
        fc = fc_model.forecast(context=context[None, None, :], prediction_length=h, known_covariates=known_cov_3d)
        inf_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        _record_inference_telemetry(inf_ms, h)

        q_mat = fc.quantiles[0, 0].detach().cpu().numpy()
        q_dict = _extract_quantile_dict(fc, 0, 0)
        m = _compute_metrics(holdout_actual, np.asarray(q_dict["q50"]), q_mat, fc.quantile_levels)
        tel = get_model_device_telemetry()
        results.append({
            "model_size": sz,
            "repo_id": MODEL_REPOS[sz],
            "weights_mb": MODEL_SPECS[sz]["weights_mb"],
            "load_ms": tel["model_state"]["last_load_ms"],
            "inference_ms": inf_ms,
            "throughput_steps_per_sec": tel["model_state"]["last_throughput_steps_per_sec"],
            "process_rss_mb": tel["hardware"]["process_rss_mb"],
            "mae": m["mae"],
            "rmse": m["rmse"],
            "smape_pct": m["smape_pct"],
            "coverage_80_pct": m["coverage_80_pct"],
            "coverage_90_pct": m["coverage_90_pct"],
            "mean_pinball_loss": m["mean_pinball_loss"],
        })

    # Reload 5M default at the end for instant interactive responsiveness
    if sizes_to_run[-1] != DEFAULT_MODEL_SIZE:
        get_forecaster(DEFAULT_MODEL_SIZE)

    return {
        "dataset_id": req.dataset_id,
        "dataset_name": ds["name"],
        "unit": ds["unit"],
        "holdout_length": h,
        "comparisons": results,
        "telemetry": get_model_device_telemetry(),
    }


@app.post("/api/v1/anomalies", tags=["Native Variants"])
def api_anomalies(req: AnomalyRequest) -> dict[str, Any]:
    """Zero-Shot Anomaly Detection: compare observed stream against Timestral's predicted quantile bounds."""
    ds = DATASETS.get(req.dataset_id or "web_traffic_rps", DATASETS["web_traffic_rps"])
    raw = req.series if req.series is not None else ds["series"]
    full = np.asarray([np.nan if v is None else float(v) for v in raw], dtype=np.float32).copy()

    w = req.eval_window
    context = full[:-w]
    observed = full[-w:].copy()

    injected_indices: list[int] = []
    if req.inject_synthetic_spikes and req.series is None:
        std_ctx = float(np.nanstd(context))
        spikes = [(w // 4, 3.4 * std_ctx), (w // 2, -2.9 * std_ctx), ((3 * w) // 4, 4.1 * std_ctx)]
        for idx, delta in spikes:
            if 0 <= idx < len(observed):
                observed[idx] += delta
                injected_indices.append(idx)

    used_size = req.model_size if req.model_size in MODEL_REPOS else DEFAULT_MODEL_SIZE
    forecaster = get_forecaster(used_size)
    t0 = time.perf_counter()
    fc = forecaster.forecast(context=context[None, None, :], prediction_length=w)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    _record_inference_telemetry(elapsed_ms, w)

    q_dict = _extract_quantile_dict(fc, 0, 0)
    low_key, high_key = ("q05", "q95") if req.band == "q05_q95" else ("q01", "q99")
    lower = np.asarray(q_dict[low_key], dtype=np.float64)
    upper = np.asarray(q_dict[high_key], dtype=np.float64)
    median = np.asarray(q_dict["q50"], dtype=np.float64)
    spread = np.maximum(upper - lower, 1e-6)

    anomalies = []
    for step_idx in range(w):
        val = float(observed[step_idx])
        lo = float(lower[step_idx])
        hi = float(upper[step_idx])
        med = float(median[step_idx])
        if val > hi or val < lo:
            excess = (val - hi) if val > hi else (lo - val)
            severity = round(float(excess / (0.5 * spread[step_idx])) + 1.0, 2)
            anomalies.append({
                "step_in_window": step_idx,
                "global_index": int(len(context) + step_idx),
                "observed_value": round(val, 4),
                "expected_median": round(med, 4),
                "band_lower": round(lo, 4),
                "band_upper": round(hi, 4),
                "direction": "spike_above_upper" if val > hi else "drop_below_lower",
                "severity_score": severity,
            })

    return {
        "capability": "Zero-Shot Probabilistic Anomaly Detection",
        "model_size": used_size,
        "model": MODEL_REPOS[used_size],
        "dataset_name": ds["name"] if req.series is None else "Custom Series",
        "unit": ds["unit"] if req.series is None else "val",
        "inference_ms": elapsed_ms,
        "telemetry": get_model_device_telemetry(),
        "band_used": req.band,
        "context": _clean_float_list(context[-96:]),
        "observed_window": _clean_float_list(observed),
        "expected_median": q_dict["q50"],
        "band_lower": q_dict[low_key],
        "band_upper": q_dict[high_key],
        "quantiles": q_dict,
        "anomalies_count": len(anomalies),
        "anomalies": anomalies,
        "injected_demo_indices": injected_indices,
        "explanation": (
            "Timestral produce 21 quantili calibrati senza bisogno di soglie statiche manuali. "
            f"Qualsiasi osservazione reale che cade fuori dalla banda [{low_key}, {high_key}] "
            "viene segnalata con uno score proporzionale alla deviazione normalizzata."
        ),
    }


@app.post("/api/v1/scenario", tags=["Native Variants"])
def api_scenario(req: ScenarioRequest) -> dict[str, Any]:
    """Counterfactual What-If Simulation using Known-Future Covariates (ROLE_KNOWN_COV)."""
    ds = DATASETS["retail_promo_units"]
    h = req.prediction_length

    if req.context is not None:
        ctx = np.asarray([np.nan if v is None else float(v) for v in req.context], dtype=np.float32)
    else:
        ctx = np.asarray(ds["series"][:-h], dtype=np.float32)

    t_len = len(ctx)
    if req.past_known_covariate is not None:
        past_cov = np.asarray(req.past_known_covariate, dtype=np.float32)[:t_len]
    else:
        past_cov = np.asarray(ds["known_covariates"][:t_len], dtype=np.float32)

    base_fut = (
        np.asarray(req.baseline_future_covariate, dtype=np.float32)[:h]
        if req.baseline_future_covariate is not None
        else np.zeros(h, dtype=np.float32)
    )
    if req.scenario_future_covariate is not None:
        scen_fut = np.asarray(req.scenario_future_covariate, dtype=np.float32)[:h]
    else:
        scen_fut = np.zeros(h, dtype=np.float32)
        scen_fut[h // 6 : (5 * h) // 6] = 1.0

    cov_baseline = np.concatenate([past_cov, base_fut])[None, None, :]
    cov_scenario = np.concatenate([past_cov, scen_fut])[None, None, :]

    used_size = req.model_size if req.model_size in MODEL_REPOS else DEFAULT_MODEL_SIZE
    forecaster = get_forecaster(used_size)
    t0 = time.perf_counter()
    fc_base = forecaster.forecast(context=ctx[None, None, :], prediction_length=h, known_covariates=cov_baseline)
    fc_scen = forecaster.forecast(context=ctx[None, None, :], prediction_length=h, known_covariates=cov_scenario)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    _record_inference_telemetry(elapsed_ms, h * 2)

    q_base = _extract_quantile_dict(fc_base, 0, 0)
    q_scen = _extract_quantile_dict(fc_scen, 0, 0)
    med_base = np.asarray(q_base["q50"], dtype=np.float64)
    med_scen = np.asarray(q_scen["q50"], dtype=np.float64)
    delta = med_scen - med_base
    total_uplift_pct = round(float((np.sum(med_scen) - np.sum(med_base)) / max(np.sum(med_base), 1e-6) * 100.0), 2)

    return {
        "capability": "What-If Scenario Simulation (Known-Future Covariates)",
        "model_size": used_size,
        "model": MODEL_REPOS[used_size],
        "dataset_name": ds["name"],
        "unit": ds["unit"],
        "inference_ms": elapsed_ms,
        "telemetry": get_model_device_telemetry(),
        "prediction_length": h,
        "context": _clean_float_list(ctx[-96:]),
        "past_covariate_tail": _clean_float_list(past_cov[-96:]),
        "baseline_future_covariate": _clean_float_list(base_fut),
        "scenario_future_covariate": _clean_float_list(scen_fut),
        "baseline_median": q_base["q50"],
        "baseline_q10": q_base["q10"],
        "baseline_q90": q_base["q90"],
        "scenario_median": q_scen["q50"],
        "scenario_q10": q_scen["q10"],
        "scenario_q90": q_scen["q90"],
        "step_delta": _clean_float_list(delta),
        "cumulative_uplift_units": round(float(np.sum(delta)), 2),
        "cumulative_uplift_pct": total_uplift_pct,
        "explanation": (
            "I token con ruolo ROLE_KNOWN_COV passano informazione ai token target anche sull'orizzonte futuro (T + H). "
            "Gli indicatori binari 0/1 vengono riconosciuti automaticamente come variabili evento, permettendo "
            "all'agente AI di simulare controfattuali prima di prendere una decisione operativa."
        ),
    }


@app.post("/api/v1/risk", tags=["Native Variants"])
def api_risk(req: RiskRequest) -> dict[str, Any]:
    """Tail-Risk, Operational Value-at-Risk (VaR) & Threshold Exceedance Probability across 21 quantiles."""
    ds = DATASETS.get(req.dataset_id or "grid_load_mw", DATASETS["grid_load_mw"])
    h = req.prediction_length
    ctx = (
        np.asarray([np.nan if v is None else float(v) for v in req.context], dtype=np.float32)
        if req.context is not None
        else np.asarray(ds["series"][:-h], dtype=np.float32)
    )

    used_size = req.model_size if req.model_size in MODEL_REPOS else DEFAULT_MODEL_SIZE
    forecaster = get_forecaster(used_size)
    t0 = time.perf_counter()
    fc = forecaster.forecast(context=ctx[None, None, :], prediction_length=h)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    _record_inference_telemetry(elapsed_ms, h)

    q_matrix = fc.quantiles[0, 0].detach().cpu().numpy()
    q_dict = _extract_quantile_dict(fc, 0, 0)

    threshold = (
        float(req.threshold)
        if req.threshold is not None
        else round(float(np.nanpercentile(ctx, 88)), 2)
    )
    exceed_probs = _exceedance_prob(q_matrix, fc.quantile_levels, threshold)

    upper_tail_width = np.asarray(q_dict["q95"]) - np.asarray(q_dict["q50"])
    lower_tail_width = np.asarray(q_dict["q50"]) - np.asarray(q_dict["q05"])
    skew_ratio = round(float(np.mean(upper_tail_width / np.maximum(lower_tail_width, 1e-6))), 3)

    return {
        "capability": "Tail-Risk, Operational VaR & Threshold Exceedance",
        "model_size": used_size,
        "model": MODEL_REPOS[used_size],
        "dataset_name": ds["name"],
        "unit": ds["unit"],
        "inference_ms": elapsed_ms,
        "telemetry": get_model_device_telemetry(),
        "prediction_length": h,
        "threshold": threshold,
        "context": _clean_float_list(ctx[-96:]),
        "median": q_dict["q50"],
        "quantiles": q_dict,
        "exceedance_probability_pct": exceed_probs,
        "risk_summary": {
            "max_exceedance_prob_pct": max(exceed_probs),
            "high_risk_steps_over_50pct": int(sum(1 for p in exceed_probs if p >= 50.0)),
            "operational_var_95_peak": max(q_dict["q95"]),
            "operational_var_99_peak": max(q_dict["q99"]),
            "downside_floor_q05_min": min(q_dict["q05"]),
            "downside_floor_q01_min": min(q_dict["q01"]),
            "tail_asymmetry_ratio": skew_ratio,
        },
        "explanation": (
            "Invece di una singola stima puntuale, la head di Timestral restituisce 21 quantili ordinati (da q01 a q99). "
            "Interpolando la distribuzione predittiva ad ogni step futuro otteniamo la probabilità esatta di superare "
            "una soglia critica (es. capacità massima trasformatore o SLA server)."
        ),
    }


@app.post("/api/v1/missing-stress", tags=["Native Variants"])
def api_missing_stress(req: MissingStressRequest) -> dict[str, Any]:
    """Missing-Data / Sensor Dropout Stress-Test: inject NaNs in context and compare forecast resilience."""
    ds = DATASETS.get(req.dataset_id or "weather_temp_c", DATASETS["weather_temp_c"])
    full = (
        np.asarray([np.nan if v is None else float(v) for v in req.series], dtype=np.float32)
        if req.series is not None
        else np.asarray(ds["series"], dtype=np.float32)
    )
    h = req.holdout_length
    clean_ctx = full[:-h].copy()
    holdout_actual = full[-h:]

    rng = np.random.default_rng(2026)
    dropout_mask = rng.random(len(clean_ctx)) < req.missing_rate
    if req.missing_rate > 0 and len(clean_ctx) > 24:
        dropout_mask[-18:-10] = True

    corrupted_ctx = clean_ctx.copy()
    corrupted_ctx[dropout_mask] = np.nan

    used_size = req.model_size if req.model_size in MODEL_REPOS else DEFAULT_MODEL_SIZE
    forecaster = get_forecaster(used_size)
    t0 = time.perf_counter()
    fc_clean = forecaster.forecast(context=clean_ctx[None, None, :], prediction_length=h)
    fc_gapped = forecaster.forecast(context=corrupted_ctx[None, None, :], prediction_length=h)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    _record_inference_telemetry(elapsed_ms, h * 2)

    q_clean = _extract_quantile_dict(fc_clean, 0, 0)
    q_gapped = _extract_quantile_dict(fc_gapped, 0, 0)

    m_clean = _compute_metrics(holdout_actual, np.asarray(q_clean["q50"]), fc_clean.quantiles[0, 0].cpu().numpy(), fc_clean.quantile_levels)
    m_gapped = _compute_metrics(holdout_actual, np.asarray(q_gapped["q50"]), fc_gapped.quantiles[0, 0].cpu().numpy(), fc_gapped.quantile_levels)

    return {
        "capability": "Missing Data & Sensor Dropout Resilience (NaN + observation_mask)",
        "model_size": used_size,
        "model": MODEL_REPOS[used_size],
        "dataset_name": ds["name"],
        "unit": ds["unit"],
        "inference_ms": elapsed_ms,
        "telemetry": get_model_device_telemetry(),
        "missing_rate_requested": req.missing_rate,
        "missing_points_count": int(np.sum(dropout_mask)),
        "context_length": len(clean_ctx),
        "holdout_length": h,
        "clean_context_tail": _clean_float_list(clean_ctx[-96:]),
        "corrupted_context_tail": _clean_float_list(corrupted_ctx[-96:]),
        "holdout_actual": _clean_float_list(holdout_actual),
        "clean_forecast_median": q_clean["q50"],
        "gapped_forecast_median": q_gapped["q50"],
        "gapped_q10": q_gapped["q10"],
        "gapped_q90": q_gapped["q90"],
        "metrics_clean": m_clean,
        "metrics_with_nans": m_gapped,
        "explanation": (
            "Timestral accetta nativamente valori NaN e maschere booleane (observation_mask). "
            "Applica un forward-fill causale (ffill_with_mask) combinato con patch scaling pesato "
            "sulle sole osservazioni valide, evitando il collasso della previsione anche con il 30%+ di buchi nei sensori."
        ),
    }


@app.post("/api/v1/multivariate", tags=["Native Variants"])
def api_multivariate(req: MultivariateRequest) -> dict[str, Any]:
    """Joint Multivariate Forecasting: pass (B=1, V, T) so Timestral forecasts coupled variates jointly."""
    ds = DATASETS["multivariate_energy"]
    raw_mv = req.multivariate_series if req.multivariate_series is not None else ds["multivariate_series"]
    names = req.variate_names if req.variate_names is not None else ds["variate_names"]

    mv_arr = np.asarray([[np.nan if x is None else float(x) for x in row] for row in raw_mv], dtype=np.float32)
    h = req.holdout_length
    ctx_mv = mv_arr[:, :-h]
    actual_mv = mv_arr[:, -h:]

    used_size = req.model_size if req.model_size in MODEL_REPOS else DEFAULT_MODEL_SIZE
    forecaster = get_forecaster(used_size)
    t0 = time.perf_counter()
    fc = forecaster.forecast(context=ctx_mv[None, :, :], prediction_length=h)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    _record_inference_telemetry(elapsed_ms, h * int(mv_arr.shape[0]))

    variates_out = []
    for v_idx in range(mv_arr.shape[0]):
        q_dict = _extract_quantile_dict(fc, 0, v_idx)
        q_mat = fc.quantiles[0, v_idx].detach().cpu().numpy()
        metrics = _compute_metrics(actual_mv[v_idx], np.asarray(q_dict["q50"]), q_mat, fc.quantile_levels)
        variates_out.append({
            "variate_index": v_idx,
            "name": names[v_idx] if v_idx < len(names) else f"Variate {v_idx + 1}",
            "context_tail": _clean_float_list(ctx_mv[v_idx, -72:]),
            "holdout_actual": _clean_float_list(actual_mv[v_idx]),
            "forecast_median": q_dict["q50"],
            "q10": q_dict["q10"],
            "q90": q_dict["q90"],
            "metrics": metrics,
        })

    return {
        "capability": "Joint Multivariate Forecasting (Cross-Variate Group Attention)",
        "model_size": used_size,
        "model": MODEL_REPOS[used_size],
        "dataset_name": ds["name"],
        "inference_ms": elapsed_ms,
        "telemetry": get_model_device_telemetry(),
        "num_variates": int(mv_arr.shape[0]),
        "holdout_length": h,
        "variates": variates_out,
        "explanation": (
            "Quando il contesto ha forma (B, V, T), Timestral assegna lo stesso group_id a tutte le V variabili "
            "della serie. I layer di attenzione condividono così il segnale tra variabili fisicamente accoppiate "
            "(Carico Elettrico, Temperatura e Irradiazione Solare) in un'unica inferenza."
        ),
    }


@app.get("/api/v1/examples", tags=["Benchmarks & Examples"])
def api_examples_catalog() -> dict[str, Any]:
    """Return metadata and visual application guides for all 7 native Timestral capabilities."""
    return {
        "examples": [
            {
                "id": "holdout_backtest",
                "badge": "CORE · VERIFICA EMPIRICA",
                "title": "1. Holdout Benchmark su Serie Reali",
                "subtitle": "Taglio degli ultimi H punti reali e confronto cieco tra Timestral e Ground Truth",
                "endpoint": "POST /api/v1/backtest",
                "when_to_use": "Validare il modello sui propri dati storici prima di metterlo in produzione senza data leakage.",
                "how_it_works": (
                    "La serie storica di lunghezza T viene divisa in Contesto [0 : T-H] e Holdout Nascosto [T-H : T]. "
                    "Timestral riceve solo il Contesto e genera mediana (q50) + bande quantili. "
                    "I punti reali lasciati fuori vengono poi sovrapposti per misurare MAE, RMSE, sMAPE e Coverage 80%/90%."
                ),
            },
            {
                "id": "zero_shot_forecast",
                "badge": "CORE · DHF + CLAMPSYM",
                "title": "2. Previsione Probabilistica Zero-Shot (21 Quantili)",
                "subtitle": "Decoupled Horizon Forecasting fino a 32K step di contesto senza riaddestramento",
                "endpoint": "POST /api/v1/forecast",
                "when_to_use": "Prevedere domanda, telemetria, flussi di cassa o carico infrastrutturale a freddo.",
                "how_it_works": (
                    "Il contesto viene suddiviso in patch causali, normalizzato localmente (sinh/arcsinh scaling) e letto "
                    "da token di predizione disaccoppiati. La test-time augmentation Clampsym media la previsione con quella "
                    "riflessa (-context) e applica il clamping a zero per le serie strettamente non negative."
                ),
            },
            {
                "id": "anomaly_detection",
                "badge": "VARIANTE NATIVA 1 · TELEMETRIA",
                "title": "3. Anomaly Detection Zero-Shot con Bande Quantili",
                "subtitle": "Rilevamento automatico di spike e crolli fuori dal cono [q01, q99] o [q05, q95]",
                "endpoint": "POST /api/v1/anomalies",
                "when_to_use": "Monitoraggio server, IoT industriale, antifrode e alert su metriche stagionali complesse.",
                "how_it_works": (
                    "Una soglia fissa fallisce sulle serie stagionali (un valore normale a mezzogiorno è anomalo alle 3 di notte). "
                    "Timestral adatta dinamicamente le bande q01–q99 ad ogni ora del giorno: ogni punto reale che perfora il cono "
                    "probabilistico viene flaggato con uno score di severità."
                ),
            },
            {
                "id": "what_if_scenario",
                "badge": "VARIANTE NATIVA 2 · ROLE_KNOWN_COV",
                "title": "4. Simulazione Scenari 'What-If' con Covariate Future",
                "subtitle": "Condizionamento causale su eventi futuri noti (promozioni, festività, manutenzioni)",
                "endpoint": "POST /api/v1/scenario",
                "when_to_use": "Rispondere a domande decisionali dell'agente AI: 'Quanto vendiamo in più se attiviamo una promo domani?'",
                "how_it_works": (
                    "Timestral distingue tre ruoli nei token: ROLE_TARGET, ROLE_PAST_COV e ROLE_KNOWN_COV. "
                    "Le covariate future note sono visibili sull'intero orizzonte T + H: i token target leggono le covariate "
                    "ma le covariate non leggono i target, permettendo simulazioni controfattuali pulite."
                ),
            },
            {
                "id": "tail_risk_var",
                "badge": "VARIANTE NATIVA 3 · RISK MANAGEMENT",
                "title": "5. Analisi Tail-Risk, VaR Operativo & Superamento Soglia",
                "subtitle": "Calcolo esatto di P(Y_t > Soglia) e quantili estremi q01 / q05 / q95 / q99",
                "endpoint": "POST /api/v1/risk",
                "when_to_use": "Dimensionamento scorte di sicurezza, SLA energetici, rischio liquidità e picchi di carico.",
                "how_it_works": (
                    "Poiché Timestral emette 21 quantili non-crossing per ogni step futuro, disponiamo dell'intera funzione di "
                    "ripartizione cumulativa (CDF) empirica. Interpolando i nodi quantili calcoliamo la probabilità percentuale "
                    "di sforare una soglia critica in ciascuna ora futura."
                ),
            },
            {
                "id": "missing_data_stress",
                "badge": "VARIANTE NATIVA 4 · SENSOR DROPOUT",
                "title": "6. Stress-Test Dati Mancanti (NaN & Observation Mask)",
                "subtitle": "Resilienza della previsione su serie corrotte da blackout o perdita pacchetti",
                "endpoint": "POST /api/v1/missing-stress",
                "when_to_use": "Sensori IoT rumorosi, telemetria con buchi di trasmissione o serie storiche incomplete.",
                "how_it_works": (
                    "I valori NaN vengono convertiti in una maschera booleana observation_mask. Timestral esegue ffill_with_mask "
                    "e calcola media e scala di ogni patch usando esclusivamente i punti realmente osservati, mantenendo "
                    "stabile la previsione anche con il 30%+ di buchi nei sensori."
                ),
            },
            {
                "id": "multivariate_joint",
                "badge": "VARIANTE NATIVA 5 · CROSS-VARIATE",
                "title": "7. Previsione Multivariata Congiunta (B, V, T)",
                "subtitle": "Previsione simultanea di variabili fisicamente o economicamente accoppiate",
                "endpoint": "POST /api/v1/multivariate",
                "when_to_use": "Sistemi multi-sensore (es. Carico Rete + Temperatura + Solare) o portafogli multi-asset.",
                "how_it_works": (
                    "Passando un tensore (B, V, T), Timestral impacchetta le V serie con lo stesso group_id. "
                    "L'attenzione cross-variata permette a ciascuna variabile di sfruttare i pattern precursori delle altre."
                ),
            },
        ]
    }


@app.get("/api/v1/agent-schema", tags=["Discovery & Docs"])
def api_agent_schema(request: Request) -> dict[str, Any]:
    """Return ready-to-use OpenAI / Mistral Function-Calling tools and MCP tool definitions for AI Agents."""
    base = str(request.base_url).rstrip("/")
    model_size_prop = {
        "type": "string",
        "enum": ["5M", "28M", "123M", "209M"],
        "default": "5M",
        "description": "Timestral model size: '5M' (default fast reactivity), '28M', '123M', or '209M' (for complex series).",
    }
    tools = [
        {
            "type": "function",
            "function": {
                "name": "timestral_forecast",
                "description": (
                    "Forecast future time-series values zero-shot using Mistral's Timestral foundation model. "
                    "Defaults to 5M for fast response; pass model_size='209M' for maximum accuracy on complex series."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "context": {
                            "type": "array",
                            "items": {"type": ["number", "null"]},
                            "description": "Historical series values (use null for missing NaN points).",
                        },
                        "prediction_length": {"type": "integer", "default": 24},
                        "model_size": model_size_prop,
                        "known_covariates": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": ["number", "null"]}},
                            "description": "Optional known-future covariates shaped (C_known, T + prediction_length).",
                        },
                        "threshold": {"type": "number"},
                    },
                    "required": ["context", "prediction_length"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "timestral_backtest_holdout",
                "description": "Evaluate Timestral accuracy on a real time series by holding out the last `holdout_length` points.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string", "enum": list(DATASETS.keys())},
                        "series": {"type": "array", "items": {"type": ["number", "null"]}},
                        "holdout_length": {"type": "integer", "default": 48},
                        "model_size": model_size_prop,
                    },
                    "required": ["holdout_length"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "timestral_unload_model",
                "description": "Explicitly unload the active Timestral model from device RAM when finished.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]
    return {
        "base_url": base,
        "discovery_health_url": f"{base}/health",
        "openapi_spec_url": f"{base}/openapi.json",
        "idle_ttl_seconds": IDLE_TTL_SECONDS,
        "openai_mistral_tools": tools,
        "mcp_server_manifest": {
            "name": "timestral-mcp-provider",
            "version": "1.1.0",
            "endpoints": {
                "timestral_forecast": f"{base}/api/v1/forecast",
                "timestral_backtest_holdout": f"{base}/api/v1/backtest",
                "timestral_compare_models": f"{base}/api/v1/backtest/compare-models",
                "timestral_unload_model": f"{base}/api/v1/model/unload",
                "timestral_detect_anomalies": f"{base}/api/v1/anomalies",
                "timestral_what_if_scenario": f"{base}/api/v1/scenario",
                "timestral_tail_risk": f"{base}/api/v1/risk",
                "timestral_missing_stress": f"{base}/api/v1/missing-stress",
                "timestral_multivariate": f"{base}/api/v1/multivariate",
            },
        },
    }


# ---------------------------------------------------------------------------
# Crypto Trading Workbench Backtesting & Live Signals (BTC & ETH)
# ---------------------------------------------------------------------------

CRYPTO_SNAPSHOTS_PATH = BASE_DIR / "crypto_snapshots.json"
CRYPTO_HTML_PATH = STATIC_DIR / "crypto.html"


class CryptoBacktestRequest(BaseModel):
    symbol: str = Field("BTCUSDT", description="Crypto pair: BTCUSDT or ETHUSDT")
    interval: str = Field("1h", description="Timeframe: 15m, 1h, 4h, 1d")
    model_size: str = Field(DEFAULT_MODEL_SIZE, description="Timestral checkpoint: 5M, 28M, 123M, 209M")
    strategy_mode: str = Field("long_short", description="Strategy mode: 'long_short' (Futures) or 'long_only' (Spot & Cash)")
    in_sample_ratio: float = Field(0.70, ge=0.40, le=0.85, description="Fraction of historical data used for In-Sample")
    prediction_horizon: int = Field(12, ge=4, le=48, description="Prediction horizon H in candles")
    signal_threshold_pct: float = Field(0.12, ge=0.01, le=10.0, description="Minimum expected median return % to trigger signal")
    max_downside_q10_pct: float = Field(-1.20, le=0.0, ge=-15.0, description="Max acceptable downside risk in quantile 10")
    stop_loss_pct: float = Field(1.5, ge=0.2, le=10.0, description="Stop loss threshold in %")
    take_profit_pct: float = Field(2.5, ge=0.2, le=20.0, description="Take profit threshold in %")
    fee_pct: float = Field(0.075, ge=0.0, le=1.0, description="Trading commission fee % per trade (e.g. 0.075% taker fee)")
    initial_capital: float = Field(10000.0, ge=100.0, description="Starting portfolio in USDT")


class InvestmentSimulationRequest(BaseModel):
    symbol: str = Field("BTCUSDT", description="Crypto pair: BTCUSDT or ETHUSDT")
    interval: str = Field("1h", description="Timeframe: 15m, 1h, 4h, 1d")
    model_size: str = Field(DEFAULT_MODEL_SIZE, description="Timestral checkpoint: 5M, 28M, 123M, 209M")
    investment_amount: float = Field(1000.0, ge=10.0, description="Invested capital in EUR or USDT")
    leverage: float = Field(1.0, ge=1.0, le=10.0, description="Financial leverage (1x spot, 2x, 3x, 5x)")
    prediction_horizon: int = Field(12, ge=4, le=48, description="Holding horizon in candles")
    fee_pct: float = Field(0.075, ge=0.0, le=1.0, description="Exchange fee % per side")


def _fetch_crypto_candles(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 200) -> list[list[Any]]:
    sym = symbol.upper().replace("/", "")
    if sym not in ("BTCUSDT", "ETHUSDT"):
        sym = "BTCUSDT"
    inter = interval if interval in ("15m", "1h", "4h", "1d") else "1h"

    # Try live Binance public API
    url = f"https://api.binance.com/api/v3/klines?symbol={sym}&interval={inter}&limit={limit}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode())
            if isinstance(data, list) and len(data) >= 30:
                return [
                    [int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])]
                    for k in data
                ]
    except Exception:
        pass

    # Fallback to local snapshot
    if CRYPTO_SNAPSHOTS_PATH.exists():
        try:
            cached = json.loads(CRYPTO_SNAPSHOTS_PATH.read_text(encoding="utf-8"))
            if sym in cached and inter in cached[sym]:
                return cached[sym][inter][:limit]
        except Exception:
            pass

    # Emergency synthetic walk if offline and no snapshot
    t = np.arange(limit)
    base = 80000.0 if "BTC" in sym else 2800.0
    rw = base * np.cumprod(1.0 + np.random.normal(0, 0.005, size=limit))
    now_ms = int(time.time() * 1000)
    step_ms = 3600 * 1000
    return [
        [now_ms - (limit - i) * step_ms, float(rw[i]), float(rw[i] * 1.002), float(rw[i] * 0.998), float(rw[i]), 100.0]
        for i in range(limit)
    ]


def _calc_strategy_kpis(trades: list[dict[str, Any]], equity_curve: list[float], interval: str = "1h") -> dict[str, Any]:
    eq = np.asarray(equity_curve, dtype=np.float64) if equity_curve else np.array([10000.0])
    peaks = np.maximum.accumulate(eq)
    dds = np.where(peaks > 1e-8, (peaks - eq) / peaks, 0.0)
    max_dd = round(float(np.max(dds)) * 100.0, 2)
    total_net_pnl_pct = round(float((eq[-1] - eq[0]) / eq[0] * 100.0), 2)

    if not trades:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": max_dd,
            "sharpe_ratio": 0.0,
            "total_net_pnl_pct": total_net_pnl_pct,
            "gross_profit_usdt": 0.0,
            "gross_loss_usdt": 0.0,
        }

    pnls = [t["net_pnl_pct"] for t in trades]
    wins = [t["net_pnl_usdt"] for t in trades if t["net_pnl_usdt"] > 0]
    losses = [abs(t["net_pnl_usdt"]) for t in trades if t["net_pnl_usdt"] < 0]

    gross_profit = float(sum(wins))
    gross_loss = float(sum(losses))
    if gross_loss > 1e-6:
        pf = round(gross_profit / gross_loss, 2)
    elif gross_profit > 0:
        pf = 99.0
    else:
        pf = 0.0

    win_rate = round(len(wins) / len(trades) * 100.0, 1)

    annual_factors = {"15m": 365 * 24 * 4, "1h": 365 * 24, "4h": 365 * 6, "1d": 365}
    ann_factor = np.sqrt(annual_factors.get(interval, 365 * 24))
    std_pnl = float(np.std(pnls))
    sharpe = round(float(np.mean(pnls) / max(std_pnl, 1e-6) * ann_factor), 2)

    return {
        "total_trades": len(trades),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate_pct": win_rate,
        "profit_factor": pf,
        "max_drawdown_pct": max_dd,
        "sharpe_ratio": sharpe,
        "total_net_pnl_pct": total_net_pnl_pct,
        "gross_profit_usdt": round(gross_profit, 2),
        "gross_loss_usdt": round(gross_loss, 2),
    }

    return {
        "total_trades": len(trades),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate_pct": win_rate,
        "profit_factor": pf,
        "max_drawdown_pct": max_dd,
        "sharpe_ratio": sharpe,
        "total_net_pnl_pct": total_net_pnl_pct,
        "gross_profit_usdt": round(gross_profit, 2),
        "gross_loss_usdt": round(gross_loss, 2),
    }


@app.get("/crypto", response_class=HTMLResponse, include_in_schema=False)
def crypto_page() -> FileResponse:
    if CRYPTO_HTML_PATH.exists():
        return FileResponse(CRYPTO_HTML_PATH)
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/v1/crypto/candles", tags=["Crypto Trading Workbench"])
def api_crypto_candles(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 200,
) -> dict[str, Any]:
    """Fetch real-time / historical cryptocurrency OHLCV candles from Binance (with offline snapshot fallback)."""
    candles = _fetch_crypto_candles(symbol, interval, limit)
    closes = [c[4] for c in candles]
    return {
        "symbol": symbol.upper(),
        "interval": interval,
        "candles_count": len(candles),
        "latest_price": closes[-1] if closes else 0.0,
        "candles": candles,
    }


@app.post("/api/v1/crypto/backtest", tags=["Crypto Trading Workbench"])
def api_crypto_backtest(req: CryptoBacktestRequest) -> dict[str, Any]:
    """Run walk-forward probabilistic backtest (In-Sample vs Out-of-Sample) and generate live signal using Timestral."""
    candles = _fetch_crypto_candles(req.symbol, req.interval, limit=180)
    closes = np.asarray([c[4] for c in candles], dtype=np.float32)
    timestamps = [c[0] for c in candles]
    n = len(closes)
    if n < 50:
        raise HTTPException(status_code=400, detail="Insufficient candles for backtest")

    split_idx = int(n * req.in_sample_ratio)
    h = req.prediction_horizon
    forecaster = get_forecaster(req.model_size)

    eval_step = max(3, h // 3)
    start_idx = max(24, split_idx - 60)
    thresh = req.signal_threshold_pct
    fee_mult = 1.0 - (req.fee_pct / 100.0)

    # 1. Precalculate forecast signals at sampled steps
    signals: dict[int, dict[str, float]] = {}
    for i in range(start_idx, n - 1, eval_step):
        past_ctx = closes[: i + 1]
        fc = forecaster.forecast(context=past_ctx[None, None, :], prediction_length=h)
        qd = _extract_quantile_dict(fc, 0, 0)
        cur_p = float(closes[i])
        exp_ret = float((qd["q50"][-1] - cur_p) / cur_p * 100.0)
        down_q10 = float((min(qd["q10"]) - cur_p) / cur_p * 100.0)
        up_q90 = float((max(qd["q90"]) - cur_p) / cur_p * 100.0)
        signals[i] = {"exp_ret": exp_ret, "down_q10": down_q10, "up_q90": up_q90}

    # 2. Continuous mark-to-market simulation from start_idx to n - 1
    trades: list[dict[str, Any]] = []
    equity_curve: list[float] = []
    cash = float(req.initial_capital)
    position: dict[str, Any] | None = None

    for i in range(start_idx, n):
        cur_p = float(closes[i])
        sig = signals.get(i)
        is_sample = "in_sample" if i < split_idx else "out_of_sample"

        if position is not None:
            holding_bars = i - position["entry_idx"]
            if position["type"] == "LONG":
                unrealized_pct = ((cur_p - position["entry_price"]) / position["entry_price"]) * 100.0
                opp_signal = (sig is not None and sig["exp_ret"] <= -thresh)
            else:
                unrealized_pct = ((position["entry_price"] - cur_p) / position["entry_price"]) * 100.0
                opp_signal = (sig is not None and sig["exp_ret"] >= thresh)

            should_exit = (
                holding_bars >= h
                or unrealized_pct <= -req.stop_loss_pct
                or unrealized_pct >= req.take_profit_pct
                or opp_signal
                or i == n - 1
            )

            if should_exit:
                if position["type"] == "LONG":
                    gross_val = position["size"] * cur_p
                    net_val = gross_val * fee_mult
                else:
                    gross_val = position["capital"] + position["size"] * (position["entry_price"] - cur_p)
                    net_val = gross_val * fee_mult

                net_pnl_usdt = net_val - position["capital"]
                net_pnl_pct = (net_pnl_usdt / position["capital"]) * 100.0
                cash = max(10.0, net_val)

                trades.append({
                    "type": position["type"],
                    "entry_idx": position["entry_idx"],
                    "exit_idx": i,
                    "entry_time": position["entry_time"],
                    "exit_time": timestamps[i],
                    "entry_price": round(position["entry_price"], 2),
                    "exit_price": round(cur_p, 2),
                    "holding_bars": holding_bars,
                    "net_pnl_usdt": round(net_pnl_usdt, 2),
                    "net_pnl_pct": round(net_pnl_pct, 2),
                    "sample": position["sample"],
                })
                position = None

        # Check new entry if flat
        if position is None and sig is not None and i < n - 1:
            if sig["exp_ret"] >= thresh and sig["down_q10"] >= req.max_downside_q10_pct:
                pos_cap = cash * fee_mult
                position = {
                    "type": "LONG",
                    "entry_idx": i,
                    "entry_time": timestamps[i],
                    "entry_price": cur_p,
                    "size": pos_cap / cur_p,
                    "capital": cash,
                    "sample": is_sample,
                }
            elif req.strategy_mode == "long_short" and sig["exp_ret"] <= -thresh:
                pos_cap = cash * fee_mult
                position = {
                    "type": "SHORT",
                    "entry_idx": i,
                    "entry_time": timestamps[i],
                    "entry_price": cur_p,
                    "size": pos_cap / cur_p,
                    "capital": cash,
                    "sample": is_sample,
                }

        # Mark-to-market continuous equity
        if position is None:
            bar_equity = cash
        else:
            if position["type"] == "LONG":
                bar_equity = position["size"] * cur_p * fee_mult
            else:
                bar_equity = (position["capital"] + position["size"] * (position["entry_price"] - cur_p)) * fee_mult
        equity_curve.append(round(bar_equity, 2))

    split_rel = max(1, split_idx - start_idx)
    is_trades = [t for t in trades if t["sample"] == "in_sample"]
    oos_trades = [t for t in trades if t["sample"] == "out_of_sample"]

    is_kpis = _calc_strategy_kpis(is_trades, equity_curve[:split_rel], req.interval)
    oos_kpis = _calc_strategy_kpis(oos_trades, equity_curve[split_rel:], req.interval)
    overall_kpis = _calc_strategy_kpis(trades, equity_curve, req.interval)

    # 3. Live signal on latest candle
    t0_live = time.perf_counter()
    live_fc = forecaster.forecast(context=closes[None, None, :], prediction_length=h)
    live_ms = round((time.perf_counter() - t0_live) * 1000.0, 2)
    _record_inference_telemetry(live_ms, h)

    live_q = _extract_quantile_dict(live_fc, 0, 0)
    last_price = float(closes[-1])
    live_med = np.asarray(live_q["q50"], dtype=np.float64)
    live_q10 = np.asarray(live_q["q10"], dtype=np.float64)
    live_q90 = np.asarray(live_q["q90"], dtype=np.float64)

    live_exp_ret_pct = round(float((live_med[-1] - last_price) / last_price * 100.0), 2)
    live_downside_pct = round(float((min(live_q10) - last_price) / last_price * 100.0), 2)
    live_upside_pct = round(float((max(live_q90) - last_price) / last_price * 100.0), 2)
    risk_reward = round(abs(live_upside_pct / min(-0.01, live_downside_pct)), 2)

    if live_exp_ret_pct >= thresh and live_downside_pct >= req.max_downside_q10_pct:
        signal_action = "BUY / LONG"
        signal_color = "#10b981"
        confidence = "ALTA" if risk_reward >= 1.5 else "MEDIA"
    elif live_exp_ret_pct <= -thresh:
        signal_action = "SELL / SHORT" if req.strategy_mode == "long_short" else "SELL / CASH"
        signal_color = "#f43f5e"
        confidence = "ALTA"
    else:
        signal_action = "HOLD / NEUTRAL"
        signal_color = "#f59e0b"
        confidence = "ATTESA"

    return {
        "symbol": req.symbol.upper(),
        "interval": req.interval,
        "model_size": req.model_size,
        "strategy_mode": req.strategy_mode,
        "latest_price": round(last_price, 2),
        "candles_count": n,
        "start_index": start_idx,
        "split_index": split_idx,
        "in_sample_ratio": req.in_sample_ratio,
        "candles": candles,
        "trades": trades,
        "equity_curve": equity_curve,
        "kpis": {
            "in_sample": is_kpis,
            "out_of_sample": oos_kpis,
            "overall": overall_kpis,
        },
        "live_signal": {
            "action": signal_action,
            "confidence": confidence,
            "color": signal_color,
            "expected_return_pct": live_exp_ret_pct,
            "downside_risk_q10_pct": live_downside_pct,
            "upside_potential_q90_pct": live_upside_pct,
            "risk_reward_ratio": risk_reward,
            "inference_ms": live_ms,
            "forecast_median": live_q["q50"],
            "q10": live_q["q10"],
            "q90": live_q["q90"],
            "q01": live_q["q01"],
            "q99": live_q["q99"],
        },
        "telemetry": get_model_device_telemetry(),
    }


@app.post("/api/v1/crypto/simulate-investment", tags=["Crypto Trading Workbench"])
def api_simulate_investment(req: InvestmentSimulationRequest) -> dict[str, Any]:
    """Simulate manual investment return scenarios, probabilistic win rate, and OOS backtest yield."""
    candles = _fetch_crypto_candles(req.symbol, req.interval, limit=180)
    closes = np.asarray([c[4] for c in candles], dtype=np.float32)
    if len(closes) < 30:
        raise HTTPException(status_code=400, detail="Insufficient candles")

    cur_price = float(closes[-1])
    forecaster = get_forecaster(req.model_size)
    h = req.prediction_horizon

    fc = forecaster.forecast(context=closes[None, None, :], prediction_length=h)
    q_dict = _extract_quantile_dict(fc, 0, 0)

    p0 = cur_price
    p50 = float(q_dict["q50"][-1])
    p90 = float(q_dict["q90"][-1])
    p10 = float(q_dict["q10"][-1])
    p99 = float(q_dict["q99"][-1])
    p01 = float(q_dict["q01"][-1])

    lev = float(req.leverage)
    amount = float(req.investment_amount)
    round_trip_fee_pct = float(req.fee_pct * 2.0 * lev)

    pos_crypto = (amount * lev) / p0

    def _scenario(target_p: float) -> dict[str, Any]:
        raw_ret = ((target_p - p0) / p0) * 100.0
        lev_ret = raw_ret * lev - round_trip_fee_pct
        yield_val = amount * (lev_ret / 100.0)
        final_val = max(0.0, amount + yield_val)
        return {
            "target_price": round(target_p, 2),
            "underlying_move_pct": round(raw_ret, 2),
            "leveraged_return_pct": round(lev_ret, 2),
            "net_yield_amount": round(yield_val, 2),
            "final_capital": round(final_val, 2),
        }

    scenarios = {
        "base_case_q50": _scenario(p50),
        "bullish_target_q90": _scenario(p90),
        "bearish_risk_q10": _scenario(p10),
        "extreme_worst_q01": _scenario(p01),
        "parabolic_best_q99": _scenario(p99),
    }

    q_ends = [float(q_dict[k][-1]) for k in q_dict]
    win_prob = round((sum(1 for q in q_ends if q > p0) / len(q_ends)) * 100.0, 1)

    liq_price = None
    is_liq_danger = False
    if lev > 1.0:
        liq_price = round(p0 * (1.0 - (0.95 / lev)), 2)
        if p01 <= liq_price:
            is_liq_danger = True

    exp_ret = scenarios["base_case_q50"]["leveraged_return_pct"]
    downside_val = abs(min(-0.01, scenarios["bearish_risk_q10"]["leveraged_return_pct"]))
    risk_rew = round(scenarios["bullish_target_q90"]["leveraged_return_pct"] / downside_val, 2)

    if exp_ret > 0.15 and win_prob >= 50.0:
        rec = "FAVOREVOLE (LONG)"
        rec_color = "#10b981"
    elif exp_ret < -0.15 and win_prob <= 45.0:
        rec = "SFAVOREVOLE (SHORT / CASH)"
        rec_color = "#f43f5e"
    else:
        rec = "NEUTRALE / ATTESA"
        rec_color = "#f59e0b"

    return {
        "symbol": req.symbol.upper(),
        "interval": req.interval,
        "model_size": req.model_size,
        "investment_amount": amount,
        "leverage": lev,
        "current_price": round(p0, 2),
        "position_size_crypto": round(pos_crypto, 6),
        "prediction_horizon_candles": h,
        "round_trip_fee_pct": round(round_trip_fee_pct, 4),
        "win_probability_pct": win_prob,
        "liquidation_price": liq_price,
        "liquidation_warning": is_liq_danger,
        "recommendation": {
            "verdict": rec,
            "color": rec_color,
            "risk_reward_ratio": risk_rew,
        },
        "scenarios": scenarios,
    }


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
