# HAR Benchmark and Training Workflow

## 1) Scope

This document explains the implemented HAR benchmark pipeline for agricultural realized volatility (RV) forecasting (currently centered on wheat RV), including:

* Feature-set construction
* Variable selection (LASSO, BSR)
* HAR model training with walk-forward validation (rolling/expanding)
* Metrics and outputs
* Full data dictionary for benchmark CSV output

---

## 2) HAR Model Structure

The baseline HAR-style target equation used in training is:

$$
RV_t = \beta_0 + \beta_1 RV_{\text{weekly},t} +
\beta_2 RV_{\text{monthly},t}
+
\beta_3 RV_{\text{seasonal},t}
+
\sum_{j} \alpha_j X_{j,t}
+
u_t
$$

Where:

* $RV_{\text{weekly}}$, $RV_{\text{monthly}}$, $RV_{\text{seasonal}}$ are **core HAR columns** (always forced in selection)
* $X_j$ are optional extra features (endogenous, exogenous, climate, news, macro)
* $u_t$ is the error term

**Notes**

* $RV_{\text{monthly}}$ and $RV_{\text{seasonal}}$ are precomputed in the dataset.
* The target is shifted by the forecast horizon during design-matrix creation.

---

### HAR Lag Construction (Canonical Form)

Weekly lag:
$$
RV_{t-1}
$$

Monthly average lag:
$$
RV_{t-1:t-4}

=

\frac{1}{4}
\sum_{i=1}^{4} RV_{t-i}
$$

Seasonal average lag:
$$
RV_{t-1:t-12}

=

\frac{1}{12}
\sum_{i=1}^{12} RV_{t-i}
$$

Final HAR regression form:

$$
RV_t

=

\beta_0
+
\beta_1 RV_{t-1}
+
\beta_2 RV_{t-1:t-4}
+
\beta_3 RV_{t-1:t-12}
+
\sum_j \alpha_j X_{j,t-1}
+
u_t
$$

---

## 3) Benchmark Feature-Set Approaches

From `build_wheat_feature_sets`, the benchmark evaluates:

1. **har** — core HAR columns only
2. **har_endo** — add wheat endogenous features
3. **har_endo_exo** — add corn and soybean exogenous RV features
4. **har_endo_exogenous_climate** — add climate variables
5. **har_endo_exogenous_climate_news** — add news variables
6. **har__all** — add macroeconomic variables

Only columns present in the input dataset are retained.

---

## 4) Model-Type Approaches

Benchmark run profiles:

1. `ols_expanding`
2. `ols_rolling`
3. `lasso_expanding`
4. `lasso_rolling`
5. `bsr_expanding`
6. `bsr_rolling`

Each profile defines:

* Walk-forward strategy (expanding or rolling)
* Selection method (`none`, `lasso`, `bsr`)
* Model configuration (standardization, target transform)
* Selection refit cadence via `refit_every_windows`

---

## 5) Time-Series Training Workflow

Implemented in `run_har_experiment_from_xy`:

1. Build and clean $(X, y)$ from the HAR design matrix
2. Optionally apply log transform to RV features
3. Generate walk-forward windows
4. For each window:

   * Time-ordered train/test split
   * Feature selection on **train only**
   * Feature-budget enforcement
   * Model fitting
   * Train and test prediction
5. Aggregate predictions across windows
6. Compute global train/test metrics

This procedure avoids leakage by strictly isolating training data inside each window.

---

### 5.1 Expanding vs Rolling Window Formulation

Let observations be indexed by $t = 1, \dots, T$.

Let:

* $n_0$ = initial train size
* $h$ = test window size
* $s$ = step size

For window $k$:

Test start index:
$$
\tau_k = n_0 + (k-1)s
$$

Test set:
$$
\mathcal{I}_{\text{test}}^{(k)}

=

\{\tau_k, \dots, \tau_k + h - 1\}
$$

**Expanding train window**
$$
\mathcal{I}_{\text{train}}^{(k)}

=

\{1, \dots, \tau_k - 1\}
$$

**Rolling train window** (length $w$)
$$
\mathcal{I}_{\text{train}}^{(k)}

=

\{\max(1, \tau_k - w), \dots, \tau_k - 1\}
$$

**Benchmark default**
$$
h = 1
$$

---

## 6) Variable Selection

### 6.1 LASSO Selection

Optimization problem on training data:

$$
\min_{\beta_0, \boldsymbol{\beta}}
;
\frac{1}{2n}
\sum_{t \in \mathcal{I}_{\text{train}}}
\left(
y_t - \beta_0 - \mathbf{x}_t^\top \boldsymbol{\beta}
\right)^2
+
\lambda
\sum_j |\beta_j|
$$

Where:

* $\lambda$ is chosen via time-series cross-validation
* Core HAR features are **force-kept**, even if $\beta_j \approx 0$

---

### 6.2 Backward Stepwise Regression (BSR)

Iterative elimination procedure:

1. Fit OLS on forced + candidate features
2. Compute p-values $p_j$
3. Identify worst feature:
   $$
   j^\star = \arg\max_j p_j
   $$
4. Drop $j^\star$ if:
   $$
   p_{j^\star} > \alpha
   $$
5. Repeat until all remaining $p_j \le \alpha$

If HAC/Newey–West is enabled, p-values are computed using the HAC covariance estimator.

---

## 7) Transformations and Scaling

### Target Transform

Log transform:
$$
y_t^{(\log)} = \log(\max(y_t, \varepsilon))
$$

Inverse transform:
$$
\hat{y}_t = \exp(\hat{y}_t^{(\log)})
$$

### Feature Transform

Optional RV feature log transform:
$$
x_{t}^{(\log)} = \log(\max(x_t, \varepsilon))
$$

Standardization (train-only):
$$
x_{t,j}^{(\text{std})}

=

\frac{x_{t,j} - \mu_j}{\sigma_j}
$$

---

## 8) Walk-Forward Modes

### Expanding

* Training set grows over time
* Assumes historical relevance persists

### Rolling

* Fixed-length training window
* Suitable under regime shifts

---

## 9) Metrics

Let aligned sequences be $\{y_t\}_{t=1}^n$ and $\{\hat{y}_t\}_{t=1}^n$.

### Mean Squared Error

$$
\text{MSE}

=

\frac{1}{n}
\sum_{t=1}^{n}
(y_t - \hat{y}_t)^2
$$

### Mean Absolute Error

$$
\text{MAE}

=

\frac{1}{n}
\sum_{t=1}^{n}
|y_t - \hat{y}_t|
$$

### QLIKE

$$
\text{QLIKE}

=

\frac{1}{n}
\sum_{t=1}^{n}
\left[
\log(\max(\hat{y}_t, \varepsilon))
+
\frac{\max(y_t, \varepsilon)}{\max(\hat{y}_t, \varepsilon)}
\right]
$$

### R-squared

$$
R^2
=
1
-
\frac{\sum_{t=1}^{n}(y_t - \hat{y}_t)^2}
{\sum_{t=1}^{n}(y_t - \bar{y})^2},
\qquad
\bar{y} = \frac{1}{n}\sum_{t=1}^{n} y_t
$$

### Log-scale R-squared

$$
R^2_{\log}

=

1
-
\frac{
\sum_{t=1}^{n}
\left[
\log(\max(y_t,\varepsilon)) - \log(\max(\hat{y}_t,\varepsilon))
\right]^2
}{
\sum_{t=1}^{n}
\left[
\log(\max(y_t,\varepsilon)) - m
\right]^2
}
$$

where
$$
m
=

\frac{1}{n}
\sum_{t=1}^{n}
\log(\max(y_t,\varepsilon))
$$

---

## 10) Caching and Runtime

Benchmark caching is implemented under:

```
.cache/benchmark/
```

Cache key includes dataset signature, run config, feature config, and model name. Cached artifacts include predictions, coefficients, and metadata.

---

## 11) Output CSV

Default output path:

```
data/benchmark/har.csv
```

---

## 12) CSV Column Dictionary

*(unchanged — descriptive, no math required)*

---

## 13) Interpretation Guidance

* Focus on **test metrics**, especially $\text{QLIKE}$ and $R^2_{\log}$
* Monitor train–test gaps for overfitting
* Prefer parsimonious models when performance is similar
* Moderate refit cadence offers strong runtime–accuracy tradeoffs
