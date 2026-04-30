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
- $\epsilon_t$ is the error term

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

### Mean Target

Mean target predicts the average realized volatility over the forecast horizon:

$$
y_t^{(h)} = \frac{1}{h}\sum_{i=1}^{h} RV_{t+i}
$$

### Walk-Forward Validation

#### Expanding Window Strategy

At each iteration, you **keep all past data** and **expand the training set forward**.

##### Setup

- Total time series: ${y_1, y_2, \dots, y_T}$
- Initial training size: $n_0$
- Forecast horizon $test size$: $h$
- Step size: $s$

##### At iteration $k$:

**Training set:**  
$$
\mathcal{D}_{\text{train}}^{(k)} = {y_1, y_2, \dots, y_{n_0 + k s}}  
$$

**Test set:**  
$$
\mathcal{D}_{\text{test}}^{(k)} = {y_{n_0 + k s + 1}, \dots, y_{n_0 + k s + h}}  
$$


- Each step:
	- Training set **grows by $s$** points
    - Test set **moves forward by $s$**

#### Rolling Window Strategy

At each iteration, you **keep a fixed-size window** and **slide it forward**.

##### Setup

- Window size: $w$
- Step size: $s$
- Forecast horizon: $h$

##### At iteration $k$:

**Training set:**  
$$
\mathcal{D}_{\text{train}}^{(k)} = {y_{1 + k s}, \dots, y_{w + k s}}  
$$

**Test set:**  
$$
\mathcal{D}_{\text{test}}^{(k)} = {y_{w + k s + 1}, \dots, y_{w + k s + h}}  
$$

## Variable Selection Methods

#### LASSO (Least Absolute Shrinkage and Selection Operator)

LASSO performs variable selection by adding an L1 penalty to the regression objective:

$$  
\min_{\beta} \left\{ \frac{1}{2n}\sum_{i=1}^{n}(y_i - \beta_0 - \mathbf{x}_i^T\boldsymbol{\beta})^2 + \lambda\sum_{j=1}^{p}|\beta_j| \right\}  
$$

where:

- $\lambda \geq 0$ is the regularization parameter
- The L1 penalty $\sum_{j=1}^{p}|\beta_j|$ induces sparsity
- As $\lambda$ increases, more coefficients are shrunk to exactly zero

**Properties:**

- Performs automatic variable selection via coefficient shrinkage
- $\lambda$ is selected using time-series cross-validation (5 folds)
- Model is refit every 4 walk-forward windows to adapt to changing conditions
- Feature selection is applied only at these refit points (i.e., once per batch of 4 windows), not at every step
- Requires feature standardization

### ### BSR (Backward Stepwise Regression)

BSR performs variable selection by iteratively removing features from a full model based on a chosen criterion:

$$  
\mathcal{F}^{(k+1)} = \mathcal{F}^{(k)} \setminus \{x_{j^\star}\}  
\quad \text{where} \quad  
j^\star = \arg\min_j \ \mathcal{C}\big(\mathcal{F}^{(k)} \setminus \{x_j\}\big)
$$

where:

- $\mathcal{F}^{(k)}$ is the feature set at iteration $k$
- $\mathcal{C}(\cdot)$ is a selection criterion (e.g., AIC, BIC, CV error)
- At each step, the feature whose removal improves the criterion the most is dropped

**Properties:**

- Starts from the full model and removes one feature at a time
- Produces sparse models through sequential elimination
- Criterion (e.g., AIC/BIC or time-series CV) guides feature removal
- Feature selection is applied only at refit points (e.g., every 4 walk-forward windows), not at every step
- Computationally more expensive than LASSO due to repeated model fitting

