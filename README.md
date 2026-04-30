# Volf

Volf is a forecasting framework for **realized volatility (RV)** in agricultural commodities (wheat, corn, soybeans), built to compare:

- Econometric models (HAR with OLS/LASSO/BSR selection)
- Machine-learning baselines (Random Forest, XGBoost)
- Multiple forecast horizons
- Multiple feature-set designs (market, cross-commodity, climate, news, macro)

The project is designed for systematic benchmarking, model comparison, and interpretation.

---

## 1) End-to-End Workflow Overview

The project workflow is organized in five stages:

1. **Build/collect datasets** from multiple domains (market RV, climate, news, macro)
2. **Construct feature sets** from core HAR variables and optional exogenous blocks
3. **Train and evaluate models** with walk-forward validation across horizons
4. **Compare forecast accuracy** and statistical significance (Clark-West)
5. **Interpret top models** with SHAP feature importance

This gives both predictive performance and explainability for different market horizons.

---

## 2) Data Layers Used by the Project

The modeling stack combines heterogeneous data sources:

- **RV market data**: weekly, monthly, seasonal RV and related RV descriptors
- **Cross-commodity features**: information from other crops (spillover effects)
- **Climate features**: temperature, precipitation/drought, wind, climate indices
- **News sentiment and attention**:
  - FRBSF news sentiment
  - EPU (economic policy uncertainty)
  - `Text_Climate_Anomaly` (weekly climate-attention signal from Google Trends)
- **Macro/financial features**: equity, oil, dollar, uncertainty proxies

The project’s motivation is that commodity volatility is jointly driven by market dynamics, climate shocks, and macro-information flow.

---

## 3) Core Modeling Structure

### HAR foundation

All model families are evaluated against a HAR-style structure with core volatility components:

- weekly RV
- monthly RV
- seasonal RV

This captures short/medium/long memory in volatility, and serves as the baseline state representation.

### Target modes

Two target formulations are supported:

- **Point target**: forecast a single future RV point
- **Mean target**: forecast average RV across a horizon window

For mean target with horizon \(h\):

\[
y_t^{(h)}=\frac{1}{h}\sum_{i=1}^{h} RV_{t+i}
\]

This is especially useful for medium/long-horizon risk planning.

---

## 4) Feature-Set Hierarchy (What is Compared)

The benchmark is not a single model; it is a structured comparison across progressively richer feature sets:

1. `har` (core weekly/monthly/seasonal RV only)
2. `har_endo` (core + endogenous commodity features)
3. `har_endo_exo` (add cross-commodity exogenous features)
4. Add blocks individually:
   - news
   - macro
   - climate
5. Add block combinations:
   - news+macro
   - climate+news
   - climate+macro
6. Full combined set:
   - `har_endo_exo_climate_news_macro`

This allows clean ablation-style analysis of what information actually improves forecasts.

---

## 5) Training Procedure

Each model is trained using **walk-forward validation** (time-series safe):

- **Expanding window**: training sample grows over time
- **Rolling window**: fixed-size moving training sample

For each horizon and feature set, models are trained and evaluated on repeated out-of-sample windows.  
Typical horizons in project experiments include short and longer-term settings (e.g., 4/8/12/16 weeks in mean-target studies).

---

## 6) Model Families and Selection Approaches

### HAR-based models

- **OLS**: baseline linear HAR fit
- **LASSO**: sparse selection for high-dimensional feature sets
- **BSR** (Bayesian subset/shrinkage style): adaptive probabilistic selection

These are run under expanding and rolling schemes to test stability vs adaptivity.

### Machine-learning baselines

- **Random Forest**
- **XGBoost**

Used to benchmark non-linear alternatives against HAR-family models.

---

## 7) Benchmarking Logic

The benchmark grid spans:

- target commodity
- target mode (point/mean)
- forecast horizon
- model type (lasso/bsr/...)
- windowing strategy (expanding/rolling)
- feature set

Outputs are consolidated per horizon into summary tables (e.g., `har.csv`) with metrics such as MSE, MAE, QLIKE, R², and model metadata.

---

## 8) Clark-West: Significance of Improvement

After performance ranking, **Clark-West tests** are applied to nested model pairs to answer:

> Is the richer feature set genuinely improving forecast accuracy, or is the difference noise?

Procedure:

1. Select base vs augmented feature-set pairs (nested comparisons)
2. Load saved out-of-sample predictions from checkpoints
3. Compute Clark-West adjusted loss-difference statistics
4. Report p-values and one-sided significance (augmented better or not)

This is the formal statistical layer over raw metric comparisons.

---

## 9) SHAP: Interpreting Why a Model Wins

For selected top-performing models/horizons, SHAP is applied to explain feature contribution.

Current interpretation output focuses on:

- **Top-10 SHAP feature importance** histogram
- Feature names on Y-axis
- Importance measured by mean absolute SHAP value

This gives practical attribution: which inputs drive the best-performing forecast setup at each horizon.

---

## 10) Saved Checkpoints (Why They Matter)

Each benchmark run stores per-model/per-feature-set checkpoint artifacts, including:

- train/test predictions
- selected features
- model metadata and diagnostics

These checkpoints enable:

- reproducible Clark-West testing without retraining
- SHAP analysis aligned with exact benchmark settings
- post-hoc analysis across horizons and feature sets

---

## 11) Main Scripts to Run

Below are the key entry points and what they do.

- `python -m scripts.benchmark.har --config config/wheat/har_mean.json`  
  Runs HAR-family benchmark for configured horizons/feature sets and writes result tables + checkpoints.

- `python -m scripts.benchmark.random_forest --config config/wheat/random_forest_mean.json`  
  Runs RF benchmark under the same benchmarking philosophy.

- `python -m scripts.benchmark.xgboost --config config/wheat/xgboost_mean.json`  
  Runs XGBoost benchmark under the same benchmarking philosophy.

- `python -m scripts.benchmark.clark_west --config config/wheat/clark_west_har_mean.json`  
  Runs Clark-West significance tests using saved checkpoints.

- `python -m scripts.benchmark.shap --config config/wheat/shap_har_mean_top.json`  
  Runs SHAP analysis for selected best models/horizons and exports top-10 importance histograms.

---

## 12) Practical Interpretation Strategy

A recommended analysis flow:

1. Run benchmark and rank by out-of-sample metrics per horizon
2. Choose the strongest candidates per horizon
3. Apply Clark-West to confirm significance of richer feature sets
4. Apply SHAP to explain which features create the gain
5. Compare short-horizon vs long-horizon winners to understand regime behavior

This turns the project into a full decision pipeline: **performance → significance → explanation**.

---

## 13) HAR Mean Results Snapshot (Best per Horizon)

From the HAR mean-target benchmark (`data/benchmark/wheat/har/mean/target_horizon_*/har.csv`), the best out-of-sample performers by horizon are:

| Horizon | Best Model Type | Best Feature Set | Test R² | Test MSE |
|---|---|---|---:|---:|
| 4 | `lasso_rolling` | `har_endo_exo_climate_macro` | 0.7044 | 0.00000384 |
| 8 | `lasso_rolling` | `har_endo_exo_macro` | 0.5865 | 0.00000450 |
| 12 | `bsr_rolling` | `har_endo_exo` | 0.6095 | 0.00000487 |
| 16 | `bsr_rolling` | `har_endo_exo_climate_news_macro` | 0.5832 | 0.00000426 |

### Quick reading

- Short-to-mid horizons (4, 8) are led by **LASSO rolling** variants.
- Longer horizons (12, 16) are led by **BSR rolling** variants.
- Feature-set winners shift by horizon, indicating that the useful information block changes with forecast distance.
