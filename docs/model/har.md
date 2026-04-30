# HAR Model Implementation and Training Procedure

## Overview

The Heterogeneous Autoregressive (HAR) model is a parsimonious yet powerful framework for modeling and forecasting realized volatility. Introduced by Corsi (2009), the HAR model captures the long-memory properties of volatility by incorporating multiple time horizons (weekly, monthly, seasonal), reflecting the heterogeneous behavior of market participants operating at different frequencies.

## Model Specification

### Basic HAR Structure

The standard HAR model for realized volatility is specified as:

$$
RV_t = \beta_0 + \beta_w RV_t^{(w)} + \beta_m RV_t^{(m)} + \beta_s RV_t^{(s)} + \epsilon_t
$$

where:
- $RV_t$ is the target realized volatility at time $t$
- $RV_t^{(w)}$ is the weekly realized volatility
- $RV_t^{(m)}$ is the monthly realized volatility (average over ~4 weeks)
- $RV_t^{(s)}$ is the seasonal realized volatility (average over ~13 weeks)
- $\epsilon_t$ is the error term with $\mathbb{E}[\epsilon_t] = 0$

### Core HAR Features

For each commodity (wheat, corn, soybeans), the core HAR features are:

- **Weekly RV**: `{commodity}_weekly_rv` - Short-term volatility component
- **Monthly RV**: `{commodity}_monthly_rv` - Medium-term volatility component  
- **Seasonal RV**: `{commodity}_seasonal_rv` - Long-term volatility component

These three components form the foundation of the HAR model, capturing volatility dynamics at different time scales.

### Extended HAR Model

The model can be extended to include additional predictors:

$$
RV_t = \beta_0 + \beta_w RV_t^{(w)} + \beta_m RV_t^{(m)} + \beta_s RV_t^{(s)} + \sum_{j=1}^{p} \gamma_j X_{j,t} + \epsilon_t
$$

where $X_{j,t}$ represents additional predictors from various categories.

## Feature Set Hierarchy

The implementation uses a hierarchical feature set structure with 10 progressively complex configurations:

### 1. HAR (Core Only)
Contains only the three core HAR components:
- `{commodity}_weekly_rv`
- `{commodity}_monthly_rv`
- `{commodity}_seasonal_rv`

### 2. HAR-Endo (Endogenous Features)
Core HAR + additional endogenous features from the same commodity:
- All transformations and lags of the target commodity
- Examples: lagged returns, volatility ratios, momentum indicators

### 3. HAR-Endo-Exo (+ Exogenous Commodities)
HAR-Endo + features from other commodities:
- When forecasting wheat: includes corn and soybeans features
- When forecasting corn: includes wheat and soybeans features
- When forecasting soybeans: includes wheat and corn features
- Captures cross-commodity spillover effects

### 4. HAR-Endo-Exo-News (+ News Sentiment)
HAR-Endo-Exo + news and sentiment indicators:
- `frbsf_sentiment`: Federal Reserve Bank of San Francisco sentiment index
- `Text_Climate_Anomaly`: Climate-related news sentiment
- `epu_index`: Economic Policy Uncertainty index

### 5. HAR-Endo-Exo-Macro (+ Macroeconomic)
HAR-Endo-Exo + macroeconomic indicators:
- `DJIA_Index`: Dow Jones Industrial Average
- `WTI_Index`: West Texas Intermediate crude oil price
- `Broad_Dollar_index`: US Dollar strength
- `Stock_Uncertainty`: Market uncertainty measure

### 6. HAR-Endo-Exo-Climate (+ Climate Variables)
HAR-Endo-Exo + climate and weather variables:

**Climate Indices:**
- `ssta_elino`: El Niño sea surface temperature anomaly
- `ssta_lanina`: La Niña sea surface temperature anomaly
- `SOI_index`: Southern Oscillation Index
- `NAO_index`: North Atlantic Oscillation index

**Temperature Extremes (Planting & Harvesting):**
- `tmax_hot_in_planting/harvesting`: Hot temperature days
- `tmax_very_hot_in_planting/harvesting`: Very hot temperature days
- `tmin_cold_in_planting/harvesting`: Cold temperature days
- `tmin_very_cold_in_planting/harvesting`: Very cold temperature days

**Wind Conditions:**
- `awnd_moderate_high_wind_in_planting/harvesting`: Moderate-high wind events
- `awnd_extreme_high_wind_in_planting/harvesting`: Extreme wind events

**Precipitation (SPI - Standardized Precipitation Index):**
- `spi_7d/1m/3m_very_wet_in_planting/harvesting`: Very wet conditions
- `spi_7d/1m/3m_extreme_wet_in_planting/harvesting`: Extreme wet conditions
- `spi_7d/1m/3m_very_dry_in_planting/harvesting`: Very dry conditions
- `spi_7d/1m/3m_extreme_dry_in_planting/harvesting`: Extreme dry conditions

**Drought Indices (PDSI - Palmer Drought Severity Index):**
- `pdsi_very_wet_in_planting/harvesting`: Very wet soil conditions
- `pdsi_extreme_wet_in_planting/harvesting`: Extreme wet soil conditions
- `pdsi_extreme_drought_in_planting/harvesting`: Extreme drought
- `pdsi_severe_drought_in_planting/harvesting`: Severe drought

**Atmospheric CO2:**
- `co2_extreme_in_planting/harvesting`: Extreme CO2 concentration levels

### 7. HAR-Endo-Exo-Climate-News
Combines climate variables with news sentiment indicators.

### 8. HAR-Endo-Exo-Climate-Macro
Combines climate variables with macroeconomic indicators.

### 9. HAR-Endo-Exo-News-Macro
Combines news sentiment with macroeconomic indicators (without climate).

### 10. HAR-Endo-Exo-Climate-News-Macro (Full Model)
Includes all available features: endogenous, exogenous, climate, news, and macroeconomic indicators.

## Target Construction Methods

### Point Target

Point target predicts the realized volatility at a specific future time point:

$$
y_t^{(h)} = RV_{t+h}
$$

where $h$ is the forecast horizon (1, 2, or 4 weeks ahead).

**Characteristics:**
- Direct prediction of specific future value
- Captures volatility at exact time point
- More volatile predictions
- Suitable for tactical trading decisions

### Mean Target

Mean target predicts the average realized volatility over the forecast horizon:

$$
y_t^{(h)} = \frac{1}{h}\sum_{i=1}^{h} RV_{t+i}
$$

**Mathematical Formulation:**

For $h=1$ (1-week ahead):
$$
y_t^{(1)} = RV_{t+1}
$$

For $h=2$ (2-week ahead mean):
$$
y_t^{(2)} = \frac{1}{2}(RV_{t+1} + RV_{t+2})
$$

For $h=4$ (4-week ahead mean):
$$
y_t^{(4)} = \frac{1}{4}(RV_{t+1} + RV_{t+2} + RV_{t+3} + RV_{t+4})
$$

**Characteristics:**
- Smoother target through averaging
- Reduces impact of single-period volatility spikes
- Lower prediction variance
- Suitable for risk management and strategic planning

**Impact on Model Performance:**
- Mean targets typically show better statistical metrics (higher R², lower MSE)
- This is due to reduced noise, not necessarily better practical performance
- Choice depends on forecasting objective

## Multi-Target Training Procedure

### Target Commodities

The system supports three agricultural commodities:
- **Wheat**: `wheat_weekly_rv` as target
- **Corn**: `corn_weekly_rv` as target
- **Soybeans**: `soybeans_weekly_rv` as target

### Forecast Horizons

For each commodity, multiple forecast horizons are evaluated:
- **h=1**: 1-week ahead forecast
- **h=2**: 2-week ahead forecast
- **h=4**: 4-week ahead forecast

### Walk-Forward Validation

#### Expanding Window Strategy
Training data accumulates over time, capturing long-term patterns and structural changes.
- Default initial training size: 104 weeks (~2 years)
- Test size: 1 week
- Step size: 1 week

#### Rolling Window Strategy
Fixed-size training window focuses on recent market dynamics.
- Default rolling window size: 104 weeks
- Test size: 1 week
- Step size: 1 week

## Variable Selection Methods

### LASSO (Least Absolute Shrinkage and Selection Operator)

LASSO performs variable selection by adding an L1 penalty to the regression objective:

$$
\min_{\beta} \left\{ \frac{1}{2n}\sum_{i=1}^{n}(y_i - \beta_0 - \mathbf{x}_i^T\boldsymbol{\beta})^2 + \lambda\sum_{j=1}^{p}|\beta_j| \right\}
$$

where:
- $\lambda \geq 0$ is the regularization parameter
- The L1 penalty $\sum_{j=1}^{p}|\beta_j|$ induces sparsity
- As $\lambda$ increases, more coefficients are shrunk to exactly zero

**Properties:**
- Automatic variable selection through coefficient shrinkage
- Optimal $\lambda$ selected via time-series cross-validation (5 folds)
- Refit every 4 walk-forward windows to adapt to changing conditions
- Requires feature standardization

### BASR (Bayesian Adaptive Shrinkage Regression)

BASR uses hierarchical Bayesian priors for adaptive coefficient shrinkage:

$$
\begin{align}
RV_t &\sim \mathcal{N}(\beta_0 + \mathbf{x}_t^T\boldsymbol{\beta}, \sigma^2) \\
\beta_j &\sim \mathcal{N}(0, \tau_j^2) \\
\tau_j^2 &\sim \text{InverseGamma}(a, b)
\end{align}
$$

where $\tau_j^2$ is the variance hyperparameter for coefficient $\beta_j$, allowing adaptive shrinkage.

**Properties:**
- Different shrinkage for each coefficient based on posterior distributions
- Significance level $\alpha = 0.05$ for feature selection
- Features selected if posterior inclusion probability > 0.95
- Refit every 8 walk-forward windows (more stable than LASSO)
- No feature standardization required

### Comparison: LASSO vs BASR

| Aspect | LASSO | BASR |
|--------|-------|------|
| Framework | Frequentist | Bayesian |
| Shrinkage | Uniform (same λ) | Adaptive (different τⱼ) |
| Uncertainty | Bootstrap/asymptotic | Posterior distribution |
| Computation | Fast (convex optimization) | Slower (Bayesian inference) |
| Selection | Hard thresholding | Soft (posterior probabilities) |
| Refit Frequency | Every 4 windows | Every 8 windows |
| Standardization | Required | Not required |

## Default Model Configurations

### OLS Models (No Selection)
- **ols_expanding**: Expanding window, all features, no standardization
- **ols_rolling**: Rolling window (104 weeks), all features, no standardization

### LASSO Models
- **lasso_expanding**: Expanding window, LASSO selection, standardized features
- **lasso_rolling**: Rolling window, LASSO selection, standardized features

### BASR Models
- **bsr_expanding**: Expanding window, Bayesian selection, no standardization
- **bsr_rolling**: Rolling window, Bayesian selection, no standardization

## Benchmark Execution

### Comprehensive Benchmark Structure

For each commodity target:
- **10 feature sets** × **6 model configurations** = **60 experiments**
- Across **3 horizons** (1, 2, 4 weeks) = **180 total experiments per commodity**

### Evaluation Metrics

- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination
- **R²log**: R² on log-transformed values
- **QLIKE**: Quasi-likelihood loss function
- **MAPE**: Mean Absolute Percentage Error

### Caching System

Results are cached based on:
- Model configuration (window type, selection method, hyperparameters)
- Feature set composition
- Target horizon and mode (point/mean)
- Data signature (hash of input data)

This enables efficient re-runs and incremental experiments.

## Practical Recommendations

### Feature Set Selection

- **Start with HAR**: Establish baseline with core features
- **Add Endo**: Capture commodity-specific dynamics
- **Add Exo**: Incorporate cross-commodity effects
- **Add Climate**: For agricultural commodities, climate is crucial
- **Add News/Macro**: For market-driven volatility components

### Model Selection

- **OLS**: Fast baseline, interpretable coefficients
- **LASSO**: When feature space is large relative to sample size
- **BASR**: When uncertainty quantification is important

### Window Strategy

- **Expanding**: When long-term patterns are stable
- **Rolling**: When recent data is more relevant (regime changes)

### Target Mode

- **Point**: For specific date forecasting and tactical decisions
- **Mean**: For risk management over periods and strategic planning

## References

1. Corsi, F. (2009). "A Simple Approximate Long-Memory Model of Realized Volatility." *Journal of Financial Econometrics*, 7(2), 174-196.

2. Tibshirani, R. (1996). "Regression Shrinkage and Selection via the Lasso." *Journal of the Royal Statistical Society: Series B*, 58(1), 267-288.

3. George, E. I., & McCulloch, R. E. (1993). "Variable Selection via Gibbs Sampling." *Journal of the American Statistical Association*, 88(423), 881-889.

4. Andersen, T. G., Bollerslev, T., & Diebold, F. X. (2007). "Roughing It Up: Including Jump Components in the Measurement, Modeling, and Forecasting of Return Volatility." *The Review of Economics and Statistics*, 89(4), 701-720.
