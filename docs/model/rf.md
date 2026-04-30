# Random Forest Implementation and Training Procedure

## Overview

Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mean prediction of individual trees for regression tasks. In volatility forecasting, Random Forest offers a flexible, non-parametric approach that can capture complex non-linear relationships and interactions between predictors without requiring explicit functional form specification.

## Model Architecture

### Ensemble Structure

Random Forest builds $B$ decision trees on bootstrap samples:

$$
\hat{f}_{RF}(\mathbf{x}) = \frac{1}{B}\sum_{b=1}^{B} \hat{f}_b(\mathbf{x})
$$

where each tree $\hat{f}_b$ is trained on a bootstrap sample with random feature subsampling at each split.

### Key Mechanisms

- **Bootstrap Aggregating (Bagging)**: Each tree trained on random sample with replacement
- **Random Feature Selection**: At each split, only subset of features considered ($m = \sqrt{p}$ for regression)
- **Tree Averaging**: Final prediction is mean across all trees, reducing variance

## Core Features

For each commodity (wheat, corn, soybeans), the core HAR features are:

- **Weekly RV**: `{commodity}_weekly_rv` - Short-term volatility component
- **Monthly RV**: `{commodity}_monthly_rv` - Medium-term volatility component  
- **Seasonal RV**: `{commodity}_seasonal_rv` - Long-term volatility component

These three components capture volatility dynamics at different time scales and form the foundation for all feature sets.

## Feature Set Hierarchy

The implementation uses 10 progressively complex feature configurations:

### 1. HAR (Core Only)
Three core HAR components only.

### 2. HAR-Endo (Endogenous Features)
Core + additional endogenous features from the same commodity (lags, transformations, ratios).

### 3. HAR-Endo-Exo (+ Exogenous Commodities)
HAR-Endo + features from other commodities to capture cross-commodity spillover effects.

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

**Climate Indices:** El Niño/La Niña (SSTA), Southern Oscillation Index (SOI), North Atlantic Oscillation (NAO)

**Temperature Extremes:** Hot/very hot and cold/very cold days during planting and harvesting seasons

**Wind Conditions:** Moderate-high and extreme wind events

**Precipitation (SPI):** Very wet/dry and extreme wet/dry conditions at 7-day, 1-month, and 3-month scales

**Drought Indices (PDSI):** Very wet, extreme wet, severe drought, and extreme drought conditions

**Atmospheric CO2:** Extreme concentration levels during growing seasons

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

**Random Forest Behavior:**
- Higher variance in predictions
- More sensitive to individual tree predictions
- Captures short-term volatility spikes
- Requires deeper trees to model complexity

### Mean Target

Mean target predicts the average realized volatility over the forecast horizon:

$$
y_t^{(h)} = \frac{1}{h}\sum_{i=1}^{h} RV_{t+i}
$$

**Explicit Formulations:**

1-week ahead: $y_t^{(1)} = RV_{t+1}$

2-week ahead mean: $y_t^{(2)} = \frac{1}{2}(RV_{t+1} + RV_{t+2})$

4-week ahead mean: $y_t^{(4)} = \frac{1}{4}(RV_{t+1} + RV_{t+2} + RV_{t+3} + RV_{t+4})$

**Random Forest Behavior:**
- Lower variance in predictions (smoother)
- More stable across different tree samples
- Averages out short-term noise
- Can use shallower trees effectively

**Feature Importance Shift:**
- Point targets: Recent volatility features most important
- Mean targets: Long-term patterns (seasonal, monthly) gain importance

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
- Captures long-term patterns

**Rolling Window:**
- Fixed-size training window (104 weeks)
- Adapts to recent market conditions
- Better for regime changes

## Feature Importance Analysis

Random Forest provides built-in feature importance through Mean Decrease in Impurity (MDI):

$$
\text{Importance}_j = \frac{1}{B}\sum_{b=1}^{B}\sum_{t \in T_b} \mathbb{1}(v_t = j) \cdot \Delta i_t
$$

where:
- $v_t$ is the feature used at node $t$
- $\Delta i_t$ is the decrease in impurity at node $t$
- $T_b$ is the set of nodes in tree $b$

**Interpretation:**
- Higher importance = feature contributes more to reducing prediction error
- Aggregated across all trees and all splits
- Provides ranking of feature relevance

## Default Model Configurations

### RF-Expanding
- Window type: Expanding
- n_estimators: 500 trees
- max_depth: None (unlimited)
- min_samples_split: 5
- min_samples_leaf: 2
- max_features: "sqrt"
- Target transform: log

### RF-Rolling
- Window type: Rolling (104 weeks)
- Same tree parameters as RF-Expanding
- Adapts to recent patterns

### RF-Expanding-Shallow (Regularized)
- Window type: Expanding
- max_depth: 20 (limited)
- min_samples_split: 10
- min_samples_leaf: 4
- More regularization to prevent overfitting

## Hyperparameter Tuning

Grid search can optimize:
- **Window parameters**: initial_train_size, test_size, step
- **Tree parameters**: max_depth, min_samples_split, min_samples_leaf
- **Optimization metric**: R², MSE, MAE

Search space typically includes:
- max_depth: [10, 20, 30, None]
- min_samples_split: [2, 5, 10]
- min_samples_leaf: [1, 2, 4]

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
