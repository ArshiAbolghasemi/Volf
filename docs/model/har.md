# HAR Model Implementation and Training Procedure

## Overview

This document describes the Heterogeneous Autoregressive (HAR) model implementation used in the Volf project for volatility forecasting. The HAR model captures long-memory properties of volatility by incorporating multiple time horizons (daily, weekly, monthly), reflecting heterogeneous market participant behavior.

## Model Specification

### Basic HAR Structure

$$
RV_t = \beta_0 + \beta_d RV_{t-1} + \beta_w RV_{t-1}^{(w)} + \beta_m RV_{t-1}^{(m)} + \epsilon_t
$$

where:
- $RV_t$ is the realized volatility at time $t$
- $RV_{t-1}$ is the daily lagged realized volatility (weekly component)
- $RV_{t-1}^{(w)} = \frac{1}{5}\sum_{i=1}^{5} RV_{t-i}$ is the monthly average
- $RV_{t-1}^{(m)} = \frac{1}{22}\sum_{i=1}^{22} RV_{t-i}$ is the seasonal average
- $\epsilon_t$ is the error term

### Extended HAR with Additional Features

$$
RV_t = \beta_0 + \beta_d RV_{t-1} + \beta_w RV_{t-1}^{(w)} + \beta_m RV_{t-1}^{(m)} + \sum_{j=1}^{p} \gamma_j X_{j,t} + \epsilon_t
$$

## Implementation Architecture

### Configuration Structure

The implementation uses a hierarchical configuration system:

```python
@dataclass
class HARFeatureConfig:
    target_col: str                              # Target variable name
    core_columns: list[str]                      # Core HAR features (weekly, monthly, seasonal)
    target_horizon: int = 1                      # Forecast horizon
    extra_feature_cols: list[str] | None = None  # Additional predictors
    target_col_name: str = "RV_target"
    target_mode: Literal["point", "mean"] = "point"  # Point forecast or mean
    target_floor: float = 1e-10                  # Minimum target value

@dataclass
class HARWalkForwardConfig:
    window_type: Literal["expanding", "rolling"] = "expanding"
    initial_train_size: int = 104                # Initial training window (weeks)
    test_size: int = 1                           # Test window size
    step: int = 1                                # Step size for walk-forward
    rolling_window_size: int | None = None       # Fixed window size for rolling

@dataclass
class HARSelectionConfig:
    method: Literal["lasso", "bsr", "none"] = "lasso"
    lasso: LassoSelectionConfig | None = None
    bsr: BSRSelectionConfig | None = None
    refit_every_windows: int = 1                 # Refit selection every N windows

@dataclass
class HARModelConfig:
    add_constant: bool = True
    standardize_features: bool = False
    target_transform: Literal["none", "log"] = "log"
    prediction_floor: float = 1e-10
    log_transform_rv_features: bool = True
    feature_floor: float = 1e-10
```

## Multi-Target Training Procedure

### 1. Target Definition

The system supports multiple target variables with different horizons:

```python
# Example targets
targets = {
    "wheat_weekly_rv": [1, 2, 4],      # 1-week, 2-week, 4-week ahead
    "corn_weekly_rv": [1, 2, 4],
    "soybeans_weekly_rv": [1, 2, 4]
}
```

### 2. Feature Set Construction

For each target, multiple feature sets are automatically constructed:

```python
def build_target_feature_sets(target_col: str, data: pd.DataFrame):
    """
    Builds hierarchical feature sets for a given target.
    
    Feature Sets:
    - har: Core HAR features only (weekly, monthly, seasonal)
    - har_endo: Core + endogenous features (same commodity)
    - har_endo_exo: Core + endo + exogenous (other commodities)
    - har_endo_exo_news: + news sentiment features
    - har_endo_exo_macro: + macroeconomic indicators
    - har_endo_exo_climate: + climate variables
    - har_endo_exo_climate_news: + climate + news
    - har_endo_exo_climate_macro: + climate + macro
    - har_endo_exo_news_macro: + news + macro
    - har_endo_exo_climate_news_macro: All features
    """
    
    # Core HAR features (always included)
    core = [f"{prefix}_weekly_rv", f"{prefix}_monthly_rv", f"{prefix}_seasonal_rv"]
    
    # Endogenous features (same commodity, different transformations)
    endo = [col for col in data.columns if col.startswith(f"{prefix}_")]
    
    # Exogenous features (other commodities)
    exo = [col for col in data.columns 
           if col.startswith(("wheat_", "corn_", "soybeans_")) 
           and not col.startswith(f"{prefix}_")]
    
    # Climate features
    climate = ["ssta_elino", "ssta_lanina", "SOI_index", "NAO_index", 
               "tmax_hot_in_planting", "tmin_cold_in_harvesting", ...]
    
    # News sentiment
    news = ["frbsf_sentiment", "Text_Climate_Anomaly", "epu_index"]
    
    # Macroeconomic
    macro = ["DJIA_Index", "WTI_Index", "Broad_Dollar_index", "Stock_Uncertainty"]
    
    return feature_sets
```

### 3. Walk-Forward Validation

Two windowing approaches are implemented:

#### Expanding Window

```
Training:  [--------------------]
Test:                            [*]
           
Training:  [------------------------]
Test:                                [*]

Training:  [----------------------------]
Test:                                    [*]
```

- Training window grows with each iteration
- Captures all historical information
- Default: `initial_train_size=104` weeks (~2 years)

#### Rolling Window

```
Training:  [----------]
Test:                  [*]
           
Training:      [----------]
Test:                      [*]

Training:          [----------]
Test:                          [*]
```

- Fixed-size training window
- Adapts to recent patterns
- Default: `rolling_window_size=104` weeks

### 4. Variable Selection Methods

#### LASSO (Least Absolute Shrinkage and Selection Operator)

**Objective Function:**

$$
\min_{\beta} \left\{ \frac{1}{2n}\sum_{i=1}^{n}(y_i - \beta_0 - \mathbf{x}_i^T\boldsymbol{\beta})^2 + \lambda\sum_{j=1}^{p}|\beta_j| \right\}
$$

**Implementation:**

```python
class LassoSelectionConfig:
    n_splits: int = 5              # Cross-validation folds
    alphas: list[float] | None     # Regularization parameters to try
    max_iter: int = 10000
    tol: float = 1e-4

# Selection procedure
def select_features_lasso(X, y, config):
    """
    1. Perform time-series cross-validation
    2. Test multiple alpha values
    3. Select alpha with best CV score
    4. Identify features with non-zero coefficients
    5. Return selected feature subset
    """
    lasso_cv = LassoCV(
        alphas=config.alphas,
        cv=TimeSeriesSplit(n_splits=config.n_splits),
        max_iter=config.max_iter
    )
    lasso_cv.fit(X, y)
    
    selected = X.columns[lasso_cv.coef_ != 0].tolist()
    return selected, lasso_cv.alpha_
```

**Refit Strategy:**
- `refit_every_windows=4`: Reselect features every 4 walk-forward windows
- Reduces computational cost while maintaining adaptivity

#### BASR (Bayesian Adaptive Shrinkage Regression)

**Model Specification:**

$$
\begin{align}
RV_t &\sim \mathcal{N}(\beta_0 + \mathbf{x}_t^T\boldsymbol{\beta}, \sigma^2) \\
\beta_j &\sim \mathcal{N}(0, \tau_j^2) \\
\tau_j^2 &\sim \text{InverseGamma}(a, b)
\end{align}
$$

**Implementation:**

```python
class BSRSelectionConfig:
    alpha: float = 0.05            # Significance level for selection
    window_type: str = "expanding"
    window_size: int | None = None
    step: int = 1

# Selection procedure
def select_features_bsr(X, y, config):
    """
    1. Fit Bayesian regression with adaptive priors
    2. Compute posterior inclusion probabilities
    3. Select features with P(inclusion) > 1 - alpha
    4. Return selected feature subset
    """
    # Bayesian Subset Regression
    model = BayesianSubsetRegression(alpha=config.alpha)
    model.fit(X, y)
    
    # Features with significant posterior probability
    selected = [col for col, prob in model.inclusion_probs.items() 
                if prob > (1 - config.alpha)]
    
    return selected, model.inclusion_probs
```

**Refit Strategy:**
- `refit_every_windows=8`: Reselect features every 8 windows
- More stable than LASSO, requires less frequent refitting

### 5. Model Training Pipeline

```python
def run_har_experiment(data, feature_config, run_config):
    """
    Complete HAR training pipeline.
    
    Steps:
    1. Prepare features and target
    2. Initialize walk-forward validator
    3. For each window:
       a. Extract training data
       b. Perform feature selection (if enabled)
       c. Apply transformations (log, standardization)
       d. Fit OLS regression
       e. Generate predictions
       f. Store results
    4. Aggregate predictions across windows
    5. Compute evaluation metrics
    """
    
    # Feature preparation
    X, y = prepare_har_features(data, feature_config)
    
    # Walk-forward validation
    wf_config = run_config.walk_forward
    validator = WalkForwardValidator(
        window_type=wf_config.window_type,
        initial_train_size=wf_config.initial_train_size,
        test_size=wf_config.test_size,
        step=wf_config.step
    )
    
    predictions = []
    selected_features_history = []
    
    for window_idx, (train_idx, test_idx) in enumerate(validator.split(X)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        # Feature selection (periodic refitting)
        if should_refit_selection(window_idx, run_config.selection):
            selected_features = select_features(
                X_train, y_train, run_config.selection
            )
        
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        
        # Apply transformations
        if run_config.model.log_transform_rv_features:
            X_train_selected = log_transform(X_train_selected)
            X_test_selected = log_transform(X_test_selected)
        
        if run_config.model.target_transform == "log":
            y_train = np.log(y_train)
        
        # Standardization
        if run_config.model.standardize_features:
            scaler = StandardScaler()
            X_train_selected = scaler.fit_transform(X_train_selected)
            X_test_selected = scaler.transform(X_test_selected)
        
        # Fit OLS
        model = OLS(y_train, X_train_selected, 
                   add_constant=run_config.model.add_constant)
        results = model.fit()
        
        # Predict
        y_pred = results.predict(X_test_selected)
        
        # Inverse transform
        if run_config.model.target_transform == "log":
            y_pred = np.exp(y_pred)
        
        # Apply floor
        y_pred = np.maximum(y_pred, run_config.model.prediction_floor)
        
        predictions.append({
            'window': window_idx,
            'y_true': y_test,
            'y_pred': y_pred,
            'selected_features': selected_features
        })
    
    return aggregate_results(predictions)
```

## Grid Search for Hyperparameter Tuning

### Grid Search Configuration

```python
@dataclass
class HARGridSearchConfig:
    enabled: bool = False
    initial_train_sizes: list[int] | None = None    # [52, 104, 156]
    test_sizes: list[int] | None = None             # [1, 2, 4]
    steps: list[int] | None = None                  # [1, 2]
    metric: str = "test_mse"                        # Optimization metric
    max_candidates: int | None = None               # Limit search space
```

### Grid Search Procedure

```python
def grid_search_har(data, base_config, grid_config):
    """
    1. Generate all parameter combinations
    2. For each combination:
       a. Create run configuration
       b. Train model with walk-forward validation
       c. Compute validation metric
    3. Select best configuration
    4. Retrain with best parameters
    """
    
    # Generate candidates
    candidates = []
    for train_size in grid_config.initial_train_sizes:
        for test_size in grid_config.test_sizes:
            for step in grid_config.steps:
                candidates.append({
                    'initial_train_size': train_size,
                    'test_size': test_size,
                    'step': step
                })
    
    # Evaluate each candidate
    results = []
    for candidate in candidates:
        run_config = create_config_from_candidate(base_config, candidate)
        result = run_har_experiment(data, feature_config, run_config)
        metric_value = result.metrics['test'][grid_config.metric]
        results.append((candidate, metric_value, result))
    
    # Select best
    best_candidate, best_metric, best_result = min(
        results, key=lambda x: x[1]  # Minimize metric
    )
    
    return best_result, best_candidate
```

## Benchmark Execution

### Multi-Horizon Multi-Model Benchmark

```python
def benchmark_multi_horizon(data, config):
    """
    Run comprehensive benchmark across:
    - Multiple target horizons (1, 2, 4 weeks)
    - Multiple feature sets (10 combinations)
    - Multiple models (OLS, LASSO, BSR)
    - Multiple windowing strategies (expanding, rolling)
    
    Total experiments per target: 10 feature sets × 6 models = 60 runs
    """
    
    results = {}
    
    for horizon in config.target_horizons:
        results[horizon] = {}
        
        # Build feature sets for this horizon
        feature_sets = build_wheat_feature_sets(data)
        
        for feature_set_name, extra_cols in feature_sets.items():
            results[horizon][feature_set_name] = {}
            
            for model_name, run_config in config.model_configs.items():
                # Create feature config
                feature_config = HARFeatureConfig(
                    target_col=config.target_col,
                    core_columns=config.core_columns,
                    target_horizon=horizon,
                    extra_feature_cols=extra_cols
                )
                
                # Run experiment with caching
                result = run_with_cache(
                    data=data,
                    feature_config=feature_config,
                    run_config=run_config,
                    cache_key=f"{horizon}_{feature_set_name}_{model_name}"
                )
                
                results[horizon][feature_set_name][model_name] = result
    
    return results
```

### Default Model Configurations

```python
DEFAULT_MODELS = {
    "ols_expanding": HARRunConfig(
        walk_forward=HARWalkForwardConfig(
            window_type="expanding",
            initial_train_size=104,
            test_size=1,
            step=1
        ),
        selection=HARSelectionConfig(method="none"),
        model=HARModelConfig(standardize_features=False)
    ),
    
    "ols_rolling": HARRunConfig(
        walk_forward=HARWalkForwardConfig(
            window_type="rolling",
            initial_train_size=104,
            rolling_window_size=104,
            test_size=1,
            step=1
        ),
        selection=HARSelectionConfig(method="none"),
        model=HARModelConfig(standardize_features=False)
    ),
    
    "lasso_expanding": HARRunConfig(
        walk_forward=HARWalkForwardConfig(
            window_type="expanding",
            initial_train_size=104,
            test_size=1,
            step=1
        ),
        selection=HARSelectionConfig(
            method="lasso",
            lasso=LassoSelectionConfig(n_splits=5),
            refit_every_windows=4
        ),
        model=HARModelConfig(standardize_features=True)
    ),
    
    "lasso_rolling": HARRunConfig(
        walk_forward=HARWalkForwardConfig(
            window_type="rolling",
            initial_train_size=104,
            rolling_window_size=104,
            test_size=1,
            step=1
        ),
        selection=HARSelectionConfig(
            method="lasso",
            lasso=LassoSelectionConfig(n_splits=5),
            refit_every_windows=4
        ),
        model=HARModelConfig(standardize_features=True)
    ),
    
    "bsr_expanding": HARRunConfig(
        walk_forward=HARWalkForwardConfig(
            window_type="expanding",
            initial_train_size=104,
            test_size=1,
            step=1
        ),
        selection=HARSelectionConfig(
            method="bsr",
            bsr=BSRSelectionConfig(alpha=0.05),
            refit_every_windows=8
        ),
        model=HARModelConfig(standardize_features=False)
    ),
    
    "bsr_rolling": HARRunConfig(
        walk_forward=HARWalkForwardConfig(
            window_type="rolling",
            initial_train_size=104,
            rolling_window_size=104,
            test_size=1,
            step=1
        ),
        selection=HARSelectionConfig(
            method="bsr",
            bsr=BSRSelectionConfig(alpha=0.05),
            refit_every_windows=8
        ),
        model=HARModelConfig(standardize_features=False)
    )
}
```

## Evaluation Metrics

```python
def compute_metrics(y_true, y_pred):
    """
    Compute comprehensive evaluation metrics.
    """
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'r2log': r2_score(np.log(y_true), np.log(y_pred)),
        'qlike': np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }
```

## Caching System

```python
def cache_key(model_name, feature_set_name, feature_config, run_config, data_signature):
    """
    Generate unique cache key based on:
    - Model configuration
    - Feature set
    - Data signature (hash of input data)
    - All hyperparameters
    """
    config_dict = {
        'model': model_name,
        'features': feature_set_name,
        'target_horizon': feature_config.target_horizon,
        'window_type': run_config.walk_forward.window_type,
        'initial_train_size': run_config.walk_forward.initial_train_size,
        'selection_method': run_config.selection.method,
        'data_sig': data_signature
    }
    return hashlib.md5(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()
```

## Usage Example

```python
from src.benchmark.har import benchmark_multi_horizon

# Configuration
config = WheatHARBenchmarkConfig(
    target_col="wheat_weekly_rv",
    target_horizons=[1, 2, 4],
    core_columns=["wheat_weekly_rv", "wheat_monthly_rv", "wheat_seasonal_rv"],
    model_names=["ols_expanding", "lasso_expanding", "bsr_expanding"],
    feature_set_names=["har", "har_endo_exo", "har_endo_exo_climate_news_macro"],
    use_cache=True,
    cache_dir="cache/har"
)

# Run benchmark
results = benchmark_multi_horizon(data, config)

# Convert to DataFrame
df = benchmark_multi_horizon_results_to_frame(results)

# Results structure:
# - target_horizon: 1, 2, 4
# - model_type: ols_expanding, lasso_expanding, bsr_expanding
# - feature_set: har, har_endo_exo, etc.
# - test_mse, test_rmse, test_mae, test_r2, test_r2log
# - n_selected_features
# - lasso_best_alpha (if applicable)
# - bsr_alpha (if applicable)
```

## References

1. Corsi, F. (2009). "A Simple Approximate Long-Memory Model of Realized Volatility." *Journal of Financial Econometrics*, 7(2), 174-196.

2. Tibshirani, R. (1996). "Regression Shrinkage and Selection via the Lasso." *Journal of the Royal Statistical Society: Series B*, 58(1), 267-288.

3. George, E. I., & McCulloch, R. E. (1993). "Variable Selection via Gibbs Sampling." *Journal of the American Statistical Association*, 88(423), 881-889.
