# Heterogeneous Autoregressive (HAR) Model

## Overview

The Heterogeneous Autoregressive (HAR) model is a parsimonious yet powerful framework for modeling and forecasting realized volatility. Introduced by Corsi (2009), the HAR model captures the long-memory properties of volatility by incorporating multiple time horizons, reflecting the heterogeneous behavior of market participants operating at different frequencies (daily, weekly, monthly).

## Model Specification

### Basic HAR Model

The standard HAR model for realized volatility is specified as:

$$
RV_t = \beta_0 + \beta_d RV_{t-1} + \beta_w RV_{t-1}^{(w)} + \beta_m RV_{t-1}^{(m)} + \epsilon_t
$$

where:
- $RV_t$ is the realized volatility at time $t$
- $RV_{t-1}$ is the daily lagged realized volatility
- $RV_{t-1}^{(w)} = \frac{1}{5}\sum_{i=1}^{5} RV_{t-i}$ is the weekly average (5-day)
- $RV_{t-1}^{(m)} = \frac{1}{22}\sum_{i=1}^{22} RV_{t-i}$ is the monthly average (22-day)
- $\epsilon_t$ is the error term with $\mathbb{E}[\epsilon_t] = 0$

### Extended HAR Model

The model can be extended to include additional components:

$$
RV_t = \beta_0 + \beta_d RV_{t-1} + \beta_w RV_{t-1}^{(w)} + \beta_m RV_{t-1}^{(m)} + \sum_{j=1}^{p} \gamma_j X_{j,t} + \epsilon_t
$$

where $X_{j,t}$ represents additional predictors such as:
- Jump components
- Leverage effects
- Trading volume
- Market microstructure variables

## Theoretical Foundation

### Long Memory and Cascade of Volatilities

The HAR model approximates long-memory processes through a cascade of volatility components at different frequencies. This is based on the Heterogeneous Market Hypothesis, which posits that:

1. **Daily traders** react to short-term information (daily component)
2. **Weekly traders** focus on medium-term trends (weekly component)
3. **Monthly traders** consider long-term fundamentals (monthly component)

The aggregation of these heterogeneous agents creates the observed long-memory behavior in volatility.

### Relationship to ARFIMA Models

The HAR model can be viewed as a restricted approximation to ARFIMA (Autoregressive Fractionally Integrated Moving Average) models:

$$
(1-L)^d RV_t = \epsilon_t
$$

where $d \in (0, 0.5)$ is the fractional differencing parameter. The HAR structure provides a simpler, more interpretable alternative while maintaining forecasting accuracy.

## Implementation

### Data Preparation

```python
import numpy as np
import pandas as pd

def prepare_har_features(rv_series, daily_lag=1, weekly_lag=5, monthly_lag=22):
    """
    Prepare HAR model features from realized volatility series.
    
    Parameters:
    -----------
    rv_series : pd.Series
        Realized volatility time series
    daily_lag : int
        Number of days for daily component (default: 1)
    weekly_lag : int
        Number of days for weekly component (default: 5)
    monthly_lag : int
        Number of days for monthly component (default: 22)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with HAR features
    """
    df = pd.DataFrame(index=rv_series.index)
    
    # Daily component
    df['RV_daily'] = rv_series.shift(daily_lag)
    
    # Weekly component (average of past 5 days)
    df['RV_weekly'] = rv_series.rolling(window=weekly_lag).mean().shift(1)
    
    # Monthly component (average of past 22 days)
    df['RV_monthly'] = rv_series.rolling(window=monthly_lag).mean().shift(1)
    
    # Target variable
    df['RV_target'] = rv_series
    
    return df.dropna()
```

### Basic HAR Estimation

```python
from sklearn.linear_model import LinearRegression

class HARModel:
    """
    Heterogeneous Autoregressive (HAR) model for realized volatility.
    """
    
    def __init__(self, daily_lag=1, weekly_lag=5, monthly_lag=22):
        self.daily_lag = daily_lag
        self.weekly_lag = weekly_lag
        self.monthly_lag = monthly_lag
        self.model = LinearRegression()
        self.coefficients_ = None
        self.intercept_ = None
        
    def fit(self, rv_series):
        """
        Fit the HAR model.
        
        Parameters:
        -----------
        rv_series : pd.Series
            Realized volatility time series
        """
        # Prepare features
        data = prepare_har_features(
            rv_series, 
            self.daily_lag, 
            self.weekly_lag, 
            self.monthly_lag
        )
        
        X = data[['RV_daily', 'RV_weekly', 'RV_monthly']]
        y = data['RV_target']
        
        # Fit model
        self.model.fit(X, y)
        self.coefficients_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        
        return self
    
    def predict(self, rv_series):
        """
        Generate predictions.
        
        Parameters:
        -----------
        rv_series : pd.Series
            Realized volatility time series
        
        Returns:
        --------
        np.ndarray
            Predicted realized volatility
        """
        data = prepare_har_features(
            rv_series, 
            self.daily_lag, 
            self.weekly_lag, 
            self.monthly_lag
        )
        
        X = data[['RV_daily', 'RV_weekly', 'RV_monthly']]
        return self.model.predict(X)
    
    def summary(self):
        """Print model summary."""
        print("HAR Model Coefficients:")
        print(f"Intercept: {self.intercept_:.6f}")
        print(f"β_daily:   {self.coefficients_[0]:.6f}")
        print(f"β_weekly:  {self.coefficients_[1]:.6f}")
        print(f"β_monthly: {self.coefficients_[2]:.6f}")
```

## Variable Selection Approaches

### 1. LASSO (Least Absolute Shrinkage and Selection Operator)

LASSO performs variable selection by adding an L1 penalty to the regression objective:

$$
\min_{\beta} \left\{ \frac{1}{2n}\sum_{i=1}^{n}(y_i - \beta_0 - \mathbf{x}_i^T\boldsymbol{\beta})^2 + \lambda\sum_{j=1}^{p}|\beta_j| \right\}
$$

where:
- $\lambda \geq 0$ is the regularization parameter
- The L1 penalty $\sum_{j=1}^{p}|\beta_j|$ induces sparsity
- As $\lambda$ increases, more coefficients are shrunk to exactly zero

#### Implementation

```python
from sklearn.linear_model import LassoCV

class HAR_LASSO:
    """
    HAR model with LASSO variable selection.
    """
    
    def __init__(self, daily_lag=1, weekly_lag=5, monthly_lag=22, 
                 cv=5, alphas=None):
        self.daily_lag = daily_lag
        self.weekly_lag = weekly_lag
        self.monthly_lag = monthly_lag
        
        if alphas is None:
            alphas = np.logspace(-4, 1, 100)
        
        self.model = LassoCV(cv=cv, alphas=alphas, random_state=42)
        self.selected_features_ = None
        
    def fit(self, rv_series, additional_features=None):
        """
        Fit HAR-LASSO model.
        
        Parameters:
        -----------
        rv_series : pd.Series
            Realized volatility time series
        additional_features : pd.DataFrame, optional
            Additional predictors for variable selection
        """
        # Prepare base HAR features
        data = prepare_har_features(
            rv_series, 
            self.daily_lag, 
            self.weekly_lag, 
            self.monthly_lag
        )
        
        X = data[['RV_daily', 'RV_weekly', 'RV_monthly']]
        
        # Add additional features if provided
        if additional_features is not None:
            X = pd.concat([X, additional_features.loc[X.index]], axis=1)
        
        y = data['RV_target']
        
        # Fit LASSO with cross-validation
        self.model.fit(X, y)
        
        # Identify selected features (non-zero coefficients)
        self.selected_features_ = X.columns[self.model.coef_ != 0].tolist()
        
        print(f"Optimal λ: {self.model.alpha_:.6f}")
        print(f"Selected features: {self.selected_features_}")
        
        return self
    
    def predict(self, rv_series, additional_features=None):
        """Generate predictions."""
        data = prepare_har_features(
            rv_series, 
            self.daily_lag, 
            self.weekly_lag, 
            self.monthly_lag
        )
        
        X = data[['RV_daily', 'RV_weekly', 'RV_monthly']]
        
        if additional_features is not None:
            X = pd.concat([X, additional_features.loc[X.index]], axis=1)
        
        return self.model.predict(X)
```

#### LASSO Properties

1. **Automatic variable selection**: Sets irrelevant coefficients to exactly zero
2. **Bias-variance tradeoff**: Introduces bias to reduce variance
3. **Computational efficiency**: Convex optimization problem
4. **Cross-validation**: Optimal $\lambda$ selected via CV

### 2. BASR (Bayesian Adaptive Shrinkage Regression)

BASR is a Bayesian approach that adaptively shrinks coefficients based on their posterior distributions. It uses hierarchical priors to achieve automatic variable selection.

#### Model Specification

The Bayesian HAR model with adaptive shrinkage:

$$
\begin{align}
RV_t &\sim \mathcal{N}(\beta_0 + \mathbf{x}_t^T\boldsymbol{\beta}, \sigma^2) \\
\beta_j &\sim \mathcal{N}(0, \tau_j^2) \\
\tau_j^2 &\sim \text{InverseGamma}(a, b) \\
\sigma^2 &\sim \text{InverseGamma}(c, d)
\end{align}
$$

where:
- $\tau_j^2$ is the variance hyperparameter for coefficient $\beta_j$
- The hierarchical prior allows adaptive shrinkage
- Small $\tau_j^2$ implies strong shrinkage toward zero

#### Horseshoe Prior

A popular choice for BASR is the horseshoe prior:

$$
\begin{align}
\beta_j &\sim \mathcal{N}(0, \lambda_j^2 \tau^2) \\
\lambda_j &\sim \text{Cauchy}^+(0, 1) \\
\tau &\sim \text{Cauchy}^+(0, 1)
\end{align}
$$

The horseshoe prior provides:
- Strong shrinkage for small signals
- Minimal shrinkage for large signals
- Heavy tails to accommodate outliers

#### Implementation

```python
import pymc as pm
import arviz as az

class HAR_BASR:
    """
    HAR model with Bayesian Adaptive Shrinkage Regression.
    """
    
    def __init__(self, daily_lag=1, weekly_lag=5, monthly_lag=22, 
                 prior='horseshoe'):
        self.daily_lag = daily_lag
        self.weekly_lag = weekly_lag
        self.monthly_lag = monthly_lag
        self.prior = prior
        self.trace = None
        self.model = None
        
    def fit(self, rv_series, additional_features=None, 
            draws=2000, tune=1000, chains=4):
        """
        Fit HAR-BASR model using MCMC.
        
        Parameters:
        -----------
        rv_series : pd.Series
            Realized volatility time series
        additional_features : pd.DataFrame, optional
            Additional predictors
        draws : int
            Number of MCMC samples
        tune : int
            Number of tuning steps
        chains : int
            Number of MCMC chains
        """
        # Prepare features
        data = prepare_har_features(
            rv_series, 
            self.daily_lag, 
            self.weekly_lag, 
            self.monthly_lag
        )
        
        X = data[['RV_daily', 'RV_weekly', 'RV_monthly']].values
        
        if additional_features is not None:
            X_add = additional_features.loc[data.index].values
            X = np.hstack([X, X_add])
        
        y = data['RV_target'].values
        n_features = X.shape[1]
        
        # Build Bayesian model
        with pm.Model() as self.model:
            # Horseshoe prior for adaptive shrinkage
            if self.prior == 'horseshoe':
                # Global shrinkage parameter
                tau = pm.HalfCauchy('tau', beta=1)
                
                # Local shrinkage parameters
                lambda_j = pm.HalfCauchy('lambda', beta=1, shape=n_features)
                
                # Coefficients with horseshoe prior
                beta = pm.Normal('beta', mu=0, 
                                sigma=lambda_j * tau, 
                                shape=n_features)
            
            elif self.prior == 'laplace':
                # Laplace prior (Bayesian LASSO)
                scale = pm.HalfCauchy('scale', beta=1)
                beta = pm.Laplace('beta', mu=0, b=scale, shape=n_features)
            
            # Intercept
            beta_0 = pm.Normal('beta_0', mu=0, sigma=10)
            
            # Likelihood variance
            sigma = pm.HalfNormal('sigma', sigma=1)
            
            # Linear predictor
            mu = beta_0 + pm.math.dot(X, beta)
            
            # Likelihood
            likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y)
            
            # Sample from posterior
            self.trace = pm.sample(draws=draws, tune=tune, 
                                  chains=chains, return_inferencedata=True)
        
        return self
    
    def predict(self, rv_series, additional_features=None):
        """
        Generate posterior predictive samples.
        
        Returns:
        --------
        dict
            Dictionary with 'mean', 'lower', 'upper' prediction intervals
        """
        data = prepare_har_features(
            rv_series, 
            self.daily_lag, 
            self.weekly_lag, 
            self.monthly_lag
        )
        
        X = data[['RV_daily', 'RV_weekly', 'RV_monthly']].values
        
        if additional_features is not None:
            X_add = additional_features.loc[data.index].values
            X = np.hstack([X, X_add])
        
        # Posterior predictive
        with self.model:
            pm.set_data({'X': X})
            posterior_pred = pm.sample_posterior_predictive(
                self.trace, var_names=['y']
            )
        
        # Extract predictions
        y_pred = posterior_pred.posterior_predictive['y'].values
        
        return {
            'mean': y_pred.mean(axis=(0, 1)),
            'lower': np.percentile(y_pred, 2.5, axis=(0, 1)),
            'upper': np.percentile(y_pred, 97.5, axis=(0, 1))
        }
    
    def summary(self):
        """Print posterior summary."""
        return az.summary(self.trace, var_names=['beta_0', 'beta', 'sigma'])
```

#### BASR Properties

1. **Adaptive shrinkage**: Different shrinkage for each coefficient
2. **Uncertainty quantification**: Full posterior distributions
3. **Automatic relevance determination**: Identifies important features
4. **Robust to outliers**: Heavy-tailed priors
5. **Interpretability**: Posterior inclusion probabilities

### Comparison: LASSO vs. BASR

| Aspect | LASSO | BASR |
|--------|-------|------|
| Framework | Frequentist | Bayesian |
| Shrinkage | Uniform (same λ) | Adaptive (different τⱼ) |
| Uncertainty | Bootstrap/asymptotic | Posterior distribution |
| Computation | Fast (convex optimization) | Slower (MCMC) |
| Selection | Hard thresholding | Soft (posterior probabilities) |
| Tuning | Cross-validation | Prior specification |

## Model Evaluation

### In-Sample Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_har_model(y_true, y_pred):
    """
    Evaluate HAR model performance.
    
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # QLIKE (Quasi-Likelihood)
    qlike = np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'QLIKE': qlike
    }
```

### Out-of-Sample Forecasting

```python
def rolling_forecast(model, rv_series, window_size=252, horizon=1):
    """
    Perform rolling window forecasting.
    
    Parameters:
    -----------
    model : HAR model instance
        Fitted HAR model
    rv_series : pd.Series
        Realized volatility series
    window_size : int
        Size of rolling window
    horizon : int
        Forecast horizon
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with forecasts and actuals
    """
    forecasts = []
    actuals = []
    dates = []
    
    for i in range(window_size, len(rv_series) - horizon):
        # Training window
        train_data = rv_series.iloc[i-window_size:i]
        
        # Fit model
        model.fit(train_data)
        
        # Forecast
        test_data = rv_series.iloc[i:i+horizon+22]  # Need extra for features
        pred = model.predict(test_data)
        
        forecasts.append(pred[0])
        actuals.append(rv_series.iloc[i+horizon])
        dates.append(rv_series.index[i+horizon])
    
    return pd.DataFrame({
        'forecast': forecasts,
        'actual': actuals
    }, index=dates)
```

## Extensions and Variations

### HAR-RV-J (with Jumps)

$$
RV_t = \beta_0 + \beta_d^c C_{t-1} + \beta_d^j J_{t-1} + \beta_w RV_{t-1}^{(w)} + \beta_m RV_{t-1}^{(m)} + \epsilon_t
$$

where $C_t$ is the continuous component and $J_t$ is the jump component.

### HAR-RSV (with Signed Volatility)

$$
RV_t = \beta_0 + \beta_d RV_{t-1} + \beta_w RV_{t-1}^{(w)} + \beta_m RV_{t-1}^{(m)} + \beta_r r_{t-1} + \epsilon_t
$$

where $r_t$ is the return, capturing leverage effects.

### HAR-GARCH

Combines HAR structure with GARCH dynamics:

$$
\begin{align}
RV_t &= \mu_t + \epsilon_t \\
\mu_t &= \beta_0 + \beta_d RV_{t-1} + \beta_w RV_{t-1}^{(w)} + \beta_m RV_{t-1}^{(m)} \\
\epsilon_t &\sim \mathcal{N}(0, h_t) \\
h_t &= \omega + \alpha \epsilon_{t-1}^2 + \beta h_{t-1}
\end{align}
$$

## References

1. Corsi, F. (2009). "A Simple Approximate Long-Memory Model of Realized Volatility." *Journal of Financial Econometrics*, 7(2), 174-196.

2. Andersen, T. G., Bollerslev, T., & Diebold, F. X. (2007). "Roughing It Up: Including Jump Components in the Measurement, Modeling, and Forecasting of Return Volatility." *The Review of Economics and Statistics*, 89(4), 701-720.

3. Tibshirani, R. (1996). "Regression Shrinkage and Selection via the Lasso." *Journal of the Royal Statistical Society: Series B*, 58(1), 267-288.

4. Carvalho, C. M., Polson, N. G., & Scott, J. G. (2010). "The Horseshoe Estimator for Sparse Signals." *Biometrika*, 97(2), 465-480.

5. Audrino, F., & Knaus, S. D. (2016). "Lassoing the HAR Model: A Model Selection Perspective on Realized Volatility Dynamics." *Econometric Reviews*, 35(8-10), 1485-1521.
