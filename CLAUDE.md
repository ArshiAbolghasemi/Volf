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

All benchmarks are configuration-driven. Pass a JSON config from `config/` to the relevant script. Configs are organized by commodity: `config/wheat/`, `config/corn/`, `config/soybean/`.

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

Most config keys can be overridden on the CLI. Useful flags (see `scripts/benchmark/har.py`):
`--target_horizons 1,4,8` (overrides the config list), `--parallel_jobs N`, `--no_cache` /
`--cache_overwrite` (force retrain), `--cache_dir`, `--print_hyperparams`, `--log_level DEBUG`.

### Caching vs. checkpoints (two separate mechanisms)

- **Result cache** (`.cache/benchmark/`, `src/benchmark/{family}/cache.py`): memoizes per-run
  training output keyed by a dataset signature + config hash. Enabled by default; edit code/config
  and the cache key changes automatically. Use `--no_cache` or `--cache_overwrite` to force retrain.
- **Best-result checkpoints** (`src/benchmark/checkpoints.py`): the final saved artifacts (best
  model per horizon) written under `data/benchmark/...`. Clark-West and SHAP scripts read these,
  so a benchmark run must precede them.

## Linting

```bash
uv run ruff check .
uv run ruff format .
```

Ruff is configured in `ruff.toml` with strict rules, 92-char line length, and Python 3.10 target.

## Architecture

### Core Flow

1. A JSON config defines: input data path, target column/horizons, walk-forward window parameters, feature selection method, and model hyperparameters. The top-level config carries multiple named `run_configs` (e.g. `ols_expanding`, `lasso_rolling`), each a model+window+selection variant run over every feature set and horizon.
2. The benchmark script loads/parses the config into dataclasses, instantiates a walk-forward experiment, and trains models over each rolling/expanding window step.
3. Results (metrics + checkpoints) are written to `data/benchmark/{commodity}/{model}/{target_mode}/target_horizon_{h}/`. `target_mode` is `point` or `mean` (set by config `target_mode`, reflected in the `_mean` config filenames); the script auto-rewrites the `--output` path into this layout and splits the summary CSV per horizon.

### Two-layer model/benchmark split

The model logic and the benchmark harness are deliberately separated:

- **`src/model/{har,rf,xgb}/`** — pure estimators and walk-forward experiment logic (`experiment.py`, `types.py`); no benchmarking/IO concerns. `common/preprocessing.py` is shared.
- **`src/benchmark/{har,rf,xgb}/`** — the harness around each model family, with a consistent file layout per family: `runner.py` (orchestrates the multi-horizon/multi-feature sweep + grid search), `features.py` (builds the feature-set hierarchy), `cache.py` (result cache), `types.py` (benchmark config dataclasses). HAR additionally has `clark_west.py` and `shap.py`.

### Source Structure

- **`src/benchmark/`** — also holds `checkpoints.py` (best-result save/load) and `tree_shap.py` (SHAP for RF/XGB)
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
