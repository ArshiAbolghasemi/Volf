# Volf

Volatility forecasting framework for agricultural commodities (wheat, corn, soybeans) using HAR-based econometric models and ML baselines across multiple horizons.

## Models used

- **HAR family**: `ols`, `lasso`, `bsr` (each with expanding/rolling walk-forward variants)
- **ML baselines**: `random_forest`, `xgboost`

For full model descriptions and methodology:
- [`docs/model/har.md`](docs/model/har.md)
- [`docs/model/rf.md`](docs/model/rf.md)
- [`docs/model/xgb.md`](docs/model/xgb.md)

## Feature combinations used

The benchmark runs combinations of:

- Core HAR RV features: weekly, monthly, seasonal
- Endogenous commodity features (`har_endo`)
- Exogenous cross-commodity features (`har_endo_exo`)
- News features (`..._news`)
- Macroeconomic features (`..._macro`)
- Climate features (`..._climate`)
- Full combined set (`har_endo_exo_climate_news_macro`)

For detailed feature/data descriptions:
- [`docs/dataset/news.md`](docs/dataset/news.md)
- [`src/benchmark/utils.py`](src/benchmark/utils.py)

## Setup

```bash
uv venv
uv sync
```

## Training workflow (scripts)

### HAR
- `uv run python -m scripts.benchmark.har --config config/wheat/har_mean.json`
- Runs HAR benchmark for configured horizons/feature sets/model variants.
- Saves summary tables (`har.csv`) and checkpoints for each model-feature-horizon run.

### Random Forest
- `uv run python -m scripts.benchmark.random_forest --config config/wheat/random_forest_mean.json`
- Runs RF benchmark under the same multi-horizon setup.

### XGBoost
- `uv run python -m scripts.benchmark.xgboost --config config/wheat/xgboost_mean.json`
- Runs XGBoost benchmark under the same multi-horizon setup.

## Post-training analysis

### Clark-West significance test
- `uv run python -m scripts.benchmark.clark_west --config config/wheat/clark_west_har_mean.json`
- Tests whether augmented feature sets significantly improve forecast accuracy using saved checkpoints.

### SHAP interpretation
- `uv run python -m scripts.benchmark.shap --config config/wheat/shap_har_mean_top.json`
- Computes SHAP for selected best runs and exports top-10 feature-importance histogram.

## HAR mean benchmark snapshot (best by horizon)

Source: [`data/benchmark/wheat/har/mean/target_horizon_*/har.csv`](data/benchmark/wheat/har/mean)

| Horizon | Best Model | Best Feature Set | Test R² | Test MSE |
|---|---|---|---:|---:|
| 4 | `lasso_rolling` | `har_endo_exo_climate_macro` | 0.7044 | 0.00000384 |
| 8 | `lasso_rolling` | `har_endo_exo_macro` | 0.5865 | 0.00000450 |
| 12 | `bsr_rolling` | `har_endo_exo` | 0.6095 | 0.00000487 |
| 16 | `bsr_rolling` | `har_endo_exo_climate_news_macro` | 0.5832 | 0.00000426 |
