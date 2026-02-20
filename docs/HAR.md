# HAR Benchmark and Training Workflow

## 1) Scope
This document explains the implemented HAR benchmark pipeline for agricultural RV forecasting (currently centered on Wheat RV), including:

- Feature-set construction
- Variable selection (LASSO, BSR, or none)
- HAR model training with walk-forward validation (rolling/expanding)
- Metrics and outputs
- Full data dictionary for benchmark CSV output

Primary benchmark entrypoints:

- `scripts/benchmark/har.py`
- `src/benchmark/har_benchmark.py`
- `src/model/har.py`
- `src/variable_selection/lasso.py`
- `src/variable_selection/bsr.py`
- `src/metrics/statistical.py`

## 2) HAR Model Structure
The baseline HAR-style target equation used in training is:

`RV_t = beta0 + beta1 * RV_weekly + beta2 * RV_monthly + beta3 * RV_seasonal + sum(alpha_j * X_j) + u_t`

Where:

- `RV_weekly`, `RV_monthly`, `RV_seasonal` are core HAR columns (always forced in selection)
- `X_j` are optional extra features (endogenous, exogenous, climate, news, macro)

Notes:

- `RV_monthly` and `RV_seasonal` are expected to be precomputed in the dataset.
- Target is shifted by `target_horizon` during design-matrix creation.

Mathematical HAR lag construction (typical form):

- Weekly lag: `RV_{t-1}`
- Monthly average lag: `RV_{t-1:t-4} = (1/4) * sum_{i=1..4} RV_{t-i}`
- Seasonal average lag: `RV_{t-1:t-12} = (1/12) * sum_{i=1..12} RV_{t-i}`

Then:

`RV_t = beta0 + beta1 * RV_{t-1} + beta2 * RV_{t-1:t-4} + beta3 * RV_{t-1:t-12} + sum(alpha_j * X_{j,t-1}) + u_t`

## 3) Benchmark Feature-Set Approaches
From `build_wheat_feature_sets`, benchmark runs these sets:

1. `har`: only core HAR columns
2. `har_endo`: add wheat endogenous columns (`wheat_*` excluding core)
3. `har_endo_exo`: add corn/soybeans exogenous RV-related columns
4. `har_endo_exogenous_climate`: add climate columns
5. `har_endo_exogenous_climate_news`: add news columns
6. `har__all`: add macro columns as well

The benchmark automatically keeps only columns that exist in the input dataset.

## 4) Model-Type Approaches in Benchmark
Default model-types (run configs):

1. `ols_expanding`
2. `ols_rolling`
3. `lasso_expanding`
4. `lasso_rolling`
5. `bsr_expanding`
6. `bsr_rolling`

Each model-type defines:

- Walk-forward strategy (rolling or expanding)
- Selection method (`none`, `lasso`, `bsr`)
- Model configuration (standardization, log-target transform, etc.)
- Selection refit cadence via `refit_every_windows`

## 5) Time-Series Training Workflow
Implemented in `run_har_experiment_from_xy`:

1. Build and clean `X, y` from HAR design matrix.
2. Optionally log-transform RV feature columns (`log_transform_rv_features=True`).
3. Generate walk-forward windows:
- Expanding: train starts at first sample, grows over time.
- Rolling: fixed-length trailing train window.
4. For each window:
- Split train/test by time (no shuffle).
- Feature selection on train only (or reuse selected features if refit window not reached).
- Enforce feature budget:
  - `max_selected_features`
  - `min_train_feature_ratio`
- Fit OLS on transformed target if `target_transform="log"`.
- Predict on train and test window.
5. Aggregate all window predictions to global train/test prediction series.
6. Compute train and test metrics.

This avoids leakage because selection, scaling, and fitting are done using window-train data only.

### 5.1 Expanding vs Rolling Window Formulas
Let total observations be indexed `t = 1, ..., T`.
Let `n0` be initial train size, `h` test size, and `s` step.

For each window `k`:

- Test start index: `tau_k = n0 + (k-1)*s`
- Test set: `I_test(k) = {tau_k, ..., tau_k + h - 1}`

Expanding train:

- `I_train_exp(k) = {1, ..., tau_k - 1}`

Rolling train (window length `w`):

- `I_train_roll(k) = {max(1, tau_k - w), ..., tau_k - 1}`

One-step forecasting case used in benchmark defaults:

- `h = 1`, so each window predicts only `t = tau_k`.

## 6) Variable Selection Details

### 6.1 LASSO (`lasso_time_series_feature_selection`)
- Uses `LassoCV` with `TimeSeriesSplit`.
- Uses feature standardization inside pipeline (`StandardScaler`).
- Core HAR columns are forced (always included in final set).
- Convergence warning handling:
  - If warning appears and retry is enabled, refit with larger `max_iter` and `eps`.

Practical behavior in HAR loop:

- Inner LASSO progress bars are disabled to avoid noisy logs.
- Selection is not necessarily recomputed every window; controlled by `refit_every_windows`.

LASSO optimization objective (on train data):

`min_{beta0,beta} (1/(2n)) * sum_{t in train} (y_t - beta0 - x_t' beta)^2 + lambda * sum_j |beta_j|`

Where:

- `lambda` is selected by time-series CV (`TimeSeriesSplit`)
- Core HAR columns are force-kept even if LASSO coefficient is near zero

### 6.2 BSR (`backward_stepwise_feature_selection`)
- p-value backward elimination with OLS.
- Can use HAC/Newey-West covariance (`hac_maxlags`) for robust inference.
- Forced core columns are never dropped.
- Drops worst candidate iteratively if `p > alpha` until stopping condition.

Practical behavior in HAR loop:

- To avoid expensive nested windowing, HAR calls BSR in `window_type="full"` on each current train window.
- `refit_every_windows` controls how often BSR reruns.

BSR elimination rule (on train window):

1. Fit OLS on forced + current candidate set.
2. Compute candidate p-values `p_j`.
3. Find worst feature `j* = argmax_j p_j`.
4. If `p_{j*} > alpha` and feature-count constraints allow, drop `j*`.
5. Repeat until all candidate p-values are `<= alpha` or stop condition reached.

If HAC is enabled, p-values are computed from HAC/Newey-West covariance estimate.

## 7) Model Transform and Scaling
- Target transform:
  - Default in HAR model config is log target (`target_transform="log"`).
  - Prediction inverse-transformed by `exp`, clipped by `prediction_floor`.
- Feature transform:
  - RV-like features can be log-transformed via `log_transform_rv_features`.
  - Optional standardization by `standardize_features` is fit on train window only and applied to train/test.

## 8) Walk-Forward Modes

### Expanding
- Train window starts at index `0`.
- End point increases each step.
- Suitable when older data remains relevant.

### Rolling
- Train window length is fixed (`rolling_window_size`).
- Slides forward through time.
- Suitable under regime changes where stale history should be dropped.

## 9) Metrics
Computed by `evaluate_statistical_metrics`:

- `mse`: mean squared error
- `mae`: mean absolute error
- `qlike`: volatility-forecast loss, `mean(log(h) + rv/h)`
- `r2`: standard R-squared on original scale
- `r2log`: R-squared on log scale
- `n_obs`: number of aligned observations used in metric computation

Let aligned true/predicted sequences be `{y_t}` and `{yhat_t}`, `t=1..n`.
Let `eps > 0` be a small floor.

- MSE:
`MSE = (1/n) * sum_t (y_t - yhat_t)^2`

- MAE:
`MAE = (1/n) * sum_t |y_t - yhat_t|`

- QLIKE:
`QLIKE = (1/n) * sum_t [ log(max(yhat_t, eps)) + max(y_t, eps)/max(yhat_t, eps) ]`

- R-squared:
`R2 = 1 - [sum_t (y_t - yhat_t)^2] / [sum_t (y_t - ybar)^2]`
where `ybar = (1/n) * sum_t y_t`

- Log-scale R-squared:
`R2_log = 1 - [sum_t (log(max(y_t,eps)) - log(max(yhat_t,eps)))^2] / [sum_t (log(max(y_t,eps)) - m)^2]`
where `m = (1/n) * sum_t log(max(y_t,eps))`

## 10) Caching and Runtime
Benchmark caching is implemented in `src/benchmark/har_benchmark.py` under `.cache/benchmark/`:

- Cache key includes dataset signature + run config + feature config + model/feature-set names.
- Stored artifacts include predictions, coefficients, and metadata in parquet.
- Cache reduces repeated rerun cost dramatically.

## 11) Output CSV
The benchmark summary CSV is generated from `benchmark_results_to_frame`.
Default script output path is:

- `data/benchmark/har.csv`

## 12) CSV Column Dictionary
Each row is one `(model_type, feature_set)` experiment result.

| Column | Description |
|---|---|
| `model_type` | Benchmark run profile name, e.g. `lasso_expanding`, `bsr_rolling`. |
| `feature_set` | Feature-set name, e.g. `har`, `har_endo_exo`, `har__all`. |
| `n_selected` | Number of final selected features reported in experiment result. |
| `selected_features` | Comma-separated selected feature names. |
| `train_mse` | Train aggregated MSE over walk-forward windows. |
| `train_mae` | Train aggregated MAE over walk-forward windows. |
| `train_qlike` | Train aggregated QLIKE loss. |
| `train_r2` | Train aggregated R-squared on original scale. |
| `train_r2log` | Train aggregated R-squared on log scale. |
| `test_mse` | Test aggregated MSE over walk-forward windows. |
| `test_mae` | Test aggregated MAE over walk-forward windows. |
| `test_qlike` | Test aggregated QLIKE loss. |
| `test_r2` | Test aggregated R-squared on original scale. |
| `test_r2log` | Test aggregated R-squared on log scale. |
| `target_col_raw` | Original target column from dataset (e.g. `wheat_weekly_rv`). |
| `target_col_model` | Internal model target column name (usually `RV_target`). |
| `target_horizon` | Forecast horizon used when shifting target. |
| `core_columns` | Comma-separated HAR core columns forced in selection. |
| `extra_feature_cols` | Comma-separated extra columns for current feature set. |
| `window_type` | Walk-forward mode used by model: `expanding` or `rolling`. |
| `initial_train_size` | First train window size for walk-forward. |
| `window_test_size` | Number of observations predicted per window step. |
| `window_step` | Step size used to move walk-forward windows. |
| `rolling_window_size` | Fixed train window length when `window_type=rolling`; null otherwise. |
| `n_windows` | Number of generated walk-forward windows used for evaluation. |
| `selection_method` | Feature selection method used: `none`, `lasso`, or `bsr`. |
| `model_add_constant` | Whether OLS intercept (`const`) was included. |
| `model_standardize_features` | Whether feature standardization was applied in each window. |
| `model_target_transform` | Target transform mode (`none` or `log`). |
| `model_prediction_floor` | Lower clipping bound used for prediction/target transforms. |
| `model_log_transform_rv_features` | Whether RV-like feature columns were log-transformed. |
| `model_feature_floor` | Lower clipping bound for RV feature log transform. |
| `model_max_selected_features` | Hard cap on selected feature count after selection. |
| `model_min_train_feature_ratio` | Minimum ratio of train observations per selected feature used in feature-budget control. |
| `lasso_best_alpha` | Best alpha found by LASSO CV for this run (if method is LASSO). |
| `bsr_alpha` | BSR significance threshold alpha (if method is BSR). |
| `bsr_window_type` | BSR internal window type from selection info (HAR currently enforces full-window BSR per train window). |
| `bsr_window_size` | BSR rolling window size if relevant. |
| `bsr_step` | BSR step parameter if relevant. |

## 13) Interpretation Guidance
- Compare models primarily on test metrics (`test_mse`, `test_mae`, `test_qlike`, `test_r2log`).
- Use train-test gap to detect overfitting.
- Prefer lower complexity (`n_selected`) when test performance is comparable.
- For heavy feature sets, `lasso_*` and `bsr_*` with moderate `refit_every_windows` often provide better runtime-accuracy tradeoff than refitting every window.
