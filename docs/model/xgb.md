# XGBoost Implementation and Training Procedure

## Overview

XGBoost (eXtreme Gradient Boosting) is an advanced implementation of gradient boosting that has become one of the most powerful machine learning algorithms. In volatility forecasting, XGBoost excels at capturing complex non-linear patterns, handling high-dimensional feature spaces, and providing robust predictions through sophisticated regularization techniques.

## Model Architecture

### Gradient Boosting Framework

XGBoost minimizes a regularized objective function:

$$
\mathcal{L}(\phi) = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)
$$

where:
- $l(y_i, \hat{y}_i)$ is the loss function (squared error for regression)
- $\Omega(f_k) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$ is the regularization term
- $T$ is the number of leaves, $w_j$ are leaf weights
- $\gamma$ is the complexity penalty, $\lambda$ is the L2 regularization parameter

### Additive Training with Second-Order Approximation

At iteration $t$, XGBoost uses second-order Taylor expansion:

$$
\mathcal{L}^{(t)} \approx \sum_{i=1}^{n} \left[ l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)
$$

where:
- $g_i = \frac{\partial l}{\partial \hat{y}^{(t-1)}}$ is the first-order gradient
- $h_i = \frac{\partial^2 l}{\partial (\hat{y}^{(t-1)})^2}$ is the second-order gradient (Hessian)

This second-order information leads to more accurate optimization compared to traditional gradient boosting.

## Core Features

For each commodity (wheat, corn, soybeans), the core HAR features are:

- **Weekly RV**: `{commodity}_weekly_rv` - Short-term volatility component
- **Monthly RV**: `{commodity}_monthly_rv` - Medium-term volatility component  
- **Seasonal RV**: `{commodity}_seasonal_rv` - Long-term volatility component

These three components capture volatility dynamics at different time scales.

## Feature Set Hierarchy

The implementation uses 10 progressively complex feature configurations:

### 1. HAR (Core Only)
Three core HAR components only.

### 2. HAR-Endo (Endogenous Features)
Core + additional endogenous features from the same commodity.

### 3. HAR-Endo-Exo (+ Exogenous Commodities)
HAR-Endo + features from other commodities for cross-commodity spillover effects.

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
HAR-Endo-Exo + comprehensive climate and weather variables:

**Climate Indices:** El Niño/La Niña sea surface temperature anomalies, Southern Oscillation Index, North Atlantic Oscillation

**Temperature Extremes:** Hot/very hot and cold/very cold temperature days during planting and harvesting seasons

**Wind Conditions:** Moderate-high and extreme wind events during critical growing periods

**Precipitation (SPI - Standardized Precipitation Index):** Very wet/dry and extreme wet/dry conditions at 7-day, 1-month, and 3-month time scales

**Drought Indices (PDSI - Palmer Drought Severity Index):** Very wet, extreme wet, severe drought, and extreme drought soil moisture conditions

**Atmospheric CO2:** Extreme concentration levels during planting and harvesting

### 7. HAR-Endo-Exo-Climate-News
Combines climate variables with news sentiment indicators.

### 8. HAR-Endo-Exo-Climate-Macro
Combines climate variables with macroeconomic indicators.

### 9. HAR-Endo-Exo-News-Macro
Combines news sentiment with macroeconomic indicators (without climate).

### 10. HAR-Endo-Exo-Climate-News-Macro (Full Model)
Includes all available features from all categories.

## Target Construction Methods

### Point Target

Point target predicts the realized volatility at a specific future time point:

$$
y_t^{(h)} = RV_{t+h}
$$

**XGBoost Gradient Behavior:**
- Larger gradient magnitudes when target is volatile
- More aggressive tree splits to capture spikes
- Requires more boosting rounds to converge
- Benefits from higher learning rate (0.01-0.03)

### Mean Target

Mean target predicts the average realized volatility over the forecast horizon:

$$
y_t^{(h)} = \frac{1}{h}\sum_{i=1}^{h} RV_{t+i}
$$

**Explicit Formulations:**

1-week ahead: $y_t^{(1)} = RV_{t+1}$

2-week ahead mean: $y_t^{(2)} = \frac{1}{2}(RV_{t+1} + RV_{t+2})$

4-week ahead mean: $y_t^{(4)} = \frac{1}{4}(RV_{t+1} + RV_{t+2} + RV_{t+3} + RV_{t+4})$

**XGBoost Gradient Behavior:**
- Smaller gradient magnitudes due to averaging
- More stable tree construction
- Faster convergence with fewer boosting rounds
- Benefits from lower learning rate (0.005-0.01)

**Feature Importance Shift:**
XGBoost's gradient-based learning amplifies the shift toward long-term features with mean targets:
- Point targets: Recent volatility (weekly) dominates importance
- Mean targets: Long-term patterns (seasonal, monthly) gain prominence

## Multi-Target Training Procedure

### Target Commodities
- Wheat: `wheat_weekly_rv`
- Corn: `corn_weekly_rv`
- Soybeans: `soybeans_weekly_rv`

### Forecast Horizons
- h=1: 1-week ahead
- h=2: 2-week ahead
- h=4: 4-week ahead

### Walk-Forward Validation

**Expanding Window:**
- Training data accumulates over time
- Default initial training size: 104 weeks (~2 years)
- Captures long-term patterns and structural changes

**Rolling Window:**
- Fixed-size training window (104 weeks)
- Focuses on recent market dynamics
- Better for regime changes and non-stationary environments

## Feature Importance Analysis

XGBoost provides multiple importance metrics:

### Gain (Default)
Total loss reduction from splits using feature $j$:

$$
\text{Gain}_j = \sum_{t: v_t = j} \Delta \text{Loss}_t
$$

Most informative metric for understanding feature contribution to model performance.

### Weight
Number of times feature $j$ is used for splitting:

$$
\text{Weight}_j = \sum_{t: v_t = j} 1
$$

Indicates feature frequency in tree construction.

### Cover
Total number of samples affected by splits on feature $j$:

$$
\text{Cover}_j = \sum_{t: v_t = j} n_t
$$

Measures feature's impact across the dataset.

## Default Model Configurations

### XGB-Expanding
- Window type: Expanding
- n_estimators: 1000
- max_depth: 6
- learning_rate: 0.01
- min_child_weight: 1.0
- subsample: 0.8
- colsample_bytree: 0.8
- gamma: 0.0
- reg_alpha: 0.0 (L1)
- reg_lambda: 1.0 (L2)
- Target transform: log

### XGB-Rolling
- Window type: Rolling (104 weeks)
- Same boosting parameters as XGB-Expanding
- Adapts to recent patterns

### XGB-Expanding-Regularized
- Window type: Expanding
- max_depth: 4 (shallower trees)
- learning_rate: 0.01
- min_child_weight: 3.0 (more regularization)
- subsample: 0.7
- colsample_bytree: 0.7
- gamma: 0.1 (complexity penalty)
- reg_alpha: 0.1 (L1 regularization)
- reg_lambda: 2.0 (stronger L2)
- Better for high-dimensional feature sets

## Regularization Strategies

### Tree Complexity Control
- **max_depth**: Limits tree depth to prevent overfitting
- **min_child_weight**: Requires minimum samples for splits
- **gamma**: Minimum loss reduction required for split

### Sampling Strategies
- **subsample**: Row subsampling (0.6-0.9)
- **colsample_bytree**: Column subsampling (0.6-0.9)

### Shrinkage and Regularization
- **learning_rate**: Step size shrinkage (0.001-0.1)
- **reg_alpha**: L1 regularization (promotes sparsity)
- **reg_lambda**: L2 regularization (smooths weights)

## Hyperparameter Tuning

Grid search can optimize:
- **Window parameters**: initial_train_size, test_size, step
- **Boosting parameters**: n_estimators, max_depth, learning_rate
- **Regularization**: min_child_weight, gamma, reg_alpha, reg_lambda
- **Optimization metric**: R², MSE, MAE

Typical search space:
- n_estimators: [500, 1000, 2000]
- max_depth: [3, 6, 9]
- learning_rate: [0.001, 0.01, 0.1]
- min_child_weight: [1, 3, 5]

## Benchmark Execution

### Comprehensive Benchmark Structure

For each commodity target:
- **10 feature sets** × **2-3 model configurations** = **20-30 experiments**
- Across **3 horizons** (1, 2, 4 weeks) = **60-90 total experiments per commodity**

### Evaluation Metrics

- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination
- **R²log**: R² on log-transformed values
- **QLIKE**: Quasi-likelihood loss
- **MAPE**: Mean Absolute Percentage Error
- **Directional Accuracy**: Percentage of correct directional predictions
- **Theil's U**: Ratio of model RMSE to naive forecast RMSE (U < 1 indicates model beats naive forecast)
