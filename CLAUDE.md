# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Volf** is a volatility forecasting benchmark framework for agricultural commodities (wheat, corn, soybeans). It systematically evaluates HAR (Heterogeneous Autoregressive) econometric models and ML baselines (Random Forest, XGBoost) across multiple forecast horizons and feature combinations.

## Setup

```bash
uv venv
uv sync
```

Python 3.10.12 is required (see `.python-version`). All scripts are run via `uv run python -m`.

## Running Benchmarks

All benchmarks are configuration-driven. Pass a JSON config from `config/` to the relevant script:

```bash
# HAR models (OLS, Lasso, BSR variants)
uv run python -m scripts.benchmark.har --config config/wheat/har_mean.json

# Random Forest
uv run python -m scripts.benchmark.random_forest --config config/wheat/random_forest_mean.json

# XGBoost
uv run python -m scripts.benchmark.xgboost --config config/wheat/xgboost_mean.json

# Clark-West statistical significance test (requires prior benchmark run)
uv run python -m scripts.benchmark.clark_west --config config/wheat/clark_west_har_mean.json

# SHAP feature importance
uv run python -m scripts.benchmark.shap --config config/wheat/shap_har_mean_top.json
uv run python -m scripts.benchmark.tree_shap --config config/wheat/shap_rf.json
uv run python -m scripts.benchmark.tree_shap --config config/wheat/shap_xgb.json
```

## Linting

```bash
uv run ruff check .
uv run ruff format .
```

Ruff is configured in `ruff.toml` with strict rules, 92-char line length, and Python 3.10 target.

## Architecture

### Core Flow

1. A JSON config defines: input data path, target horizons, walk-forward window parameters, feature selection method, and model hyperparameters.
2. The benchmark script loads the config via Dynaconf, instantiates a walk-forward experiment, and trains models over each rolling/expanding window step.
3. Results (metrics + checkpoints) are written to `data/benchmark/{commodity}/{model}/{target_mode}/target_horizon_{h}/`.

### Source Structure

- **`src/model/`** — Model implementations: `har/` (OLS, Lasso, BSR), `rf/`, `xgb/`, `common/` (shared preprocessing)
- **`src/benchmark/`** — Benchmark runners, checkpoint save/load (`checkpoints.py`), SHAP for tree models (`tree_shap.py`)
- **`src/dataset/`** — Data pipelines: `news/` (BigQuery), `climate/` (NOAA), `google_trend/`, `production_by_state/`
- **`src/variable_selection/`** — Feature selection: `lasso.py`, `bsr.py` (Bayesian Spike & Regression)
- **`src/metrics/statistical.py`** — Evaluation metrics (R², adjusted R², MSE, etc.)
- **`scripts/`** — Entry-point scripts for training (`benchmark/`) and data collection (`dataset/`)
- **`config/`** — JSON config files organized by commodity (`wheat/`, `soybean/`)
- **`data/`** — DVC-tracked data: raw (`ag/v4.csv`) and benchmark outputs

### Feature Hierarchy

Models are evaluated on progressively richer feature sets:
- `har` → core RV lags (weekly, monthly, seasonal)
- `har_endo` → + endogenous commodity features
- `har_endo_exo` → + cross-commodity features
- `har_endo_exo_climate` → + climate indices (ENSO, NAO, drought, precip)
- `har_endo_exo_climate_news` → + news sentiment (FRBSF, EPU)
- `har_endo_exo_climate_news_macro` → + macroeconomic indicators (DJIA, WTI, dollar)

### HAR Model Variants

- **OLS**: baseline HAR via `statsmodels`
- **Lasso**: regularized HAR via scikit-learn
- **BSR**: Bayesian Spike & Regression for variable selection
- Walk-forward modes: `expanding` (grows training set) or `rolling` (fixed window)

### Configuration System

Each JSON config contains one or more named run configs, each specifying:
- `walk_forward`: window type, sizes, step
- `selection`: method (`none`, `lasso`, `bsr`) and its hyperparameters
- `model`: estimator-specific hyperparameters (standardize, log_transform, add_constant, etc.)

### External Dependencies

Requires these env vars (in `.env`):
- `GOOGLE_APPLICATION_CREDENTIALS`, `GOOGLE_CLOUD_PROJECT` — BigQuery for news features
- `NOAA_TOKEN`, `NOAA_BASE_URL` — NOAA API for climate data
