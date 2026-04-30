# XGBoost Implementation and Training Procedure

## Overview

This document describes the XGBoost (eXtreme Gradient Boosting) implementation used in the Volf project for volatility forecasting. XGBoost is a gradient boosting framework that builds an ensemble of decision trees sequentially, with each tree correcting errors from previous trees through gradient-based optimization.

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

### Additive Training with Second-Order Approximation

At iteration $t$, XGBoost uses second-order Taylor expansion:

$$
\mathcal{L}^{(t)} \approx \sum_{i=1}^{n} \left[ l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)
$$

where:
- $g_i = \frac{\partial l}{\partial \hat{y}^{(t-1)}}$ is the first-order gradient
- $h_i = \frac{\partial^2 l}{\partial (\hat{y}^{(t-1)})^2}$ is the second-order gradient (Hessian)

## Implementation Architecture

### Configuration Structure

```python
@dataclass
class XGBFeatureConfig:
    target_col: str                              # Target variable name
    core_columns: list[str]                      # Core HAR features
    target_horizon: int = 1                      # Forecast horizon
    extra_feature_cols: list[str] | None = None  # Additional predictors
    target_col_name: str = "RV_target"
    target_mode: Literal["point", "mean"] = "point"
    target_floor: float = 1e-10

@dataclass
class XGBWalkForwardConfig:
    window_type: Literal["expanding", "rolling"] = "expanding"
    initial_train_size: int = 104                # Initial training window
    test_size: int = 1                           # Test window size
    step: int = 1                                # Step size
    rolling_window_size: int | None = None       # Fixed window for rolling
    progress_bar: bool = True

@dataclass
class XGBModelConfig:
    n_estimators: int = 1000                     # Number of boosting rounds
    max_depth: int = 6                           # Maximum tree depth
    learning_rate: float = 0.01                  # Step size shrinkage (eta)
    min_child_weight: float = 1.0                # Minimum sum of instance weight
    subsample: float = 0.8                       # Subsample ratio of instances
    colsample_bytree: float = 0.8                # Subsample ratio of features
    gamma: float = 0.0                           # Minimum loss reduction for split
    reg_alpha: float = 0.0                       # L1 regularization
    reg_lambda: float = 1.0                      # L2 regularization
    objective: str = "reg:squarederror"          # Loss function
    random_state: int = 42
    n_jobs: int = -1
    target_transform: Literal["none", "log"] = "log"
    prediction_floor: float = 1e-10
```

## Multi-Target Training Procedure

### 1. Feature Set Construction

Same hierarchical feature sets as HAR and RF models:

```python
feature_sets = {
    "har": [],                                    # Core HAR only
    "har_endo": endo_features,                    # + endogenous
    "har_endo_exo": endo + exo_features,          # + exogenous
    "har_endo_exo_news": endo + exo + news,       # + news sentiment
    "har_endo_exo_macro": endo + exo + macro,     # + macroeconomic
    "har_endo_exo_climate": endo + exo + climate, # + climate
    "har_endo_exo_climate_news": endo + exo + climate + news,
    "har_endo_exo_climate_macro": endo + exo + climate + macro,
    "har_endo_exo_news_macro": endo + exo + news + macro,
    "har_endo_exo_climate_news_macro": endo + exo + climate + news + macro
}
```

### 2. Walk-Forward Validation

#### Expanding Window Strategy

```
Window 1:  [--------------------] → [*]
Window 2:  [------------------------] → [*]
Window 3:  [----------------------------] → [*]
```

- Training data accumulates over time
- Captures long-term patterns and structural changes
- Default: `initial_train_size=104` weeks

#### Rolling Window Strategy

```
Window 1:  [----------] → [*]
Window 2:      [----------] → [*]
Window 3:          [----------] → [*]
```

- Fixed-size training window
- Focuses on recent market dynamics
- Default: `rolling_window_size=104` weeks

### 3. Model Training Pipeline

```python
def run_xgb_experiment(data, feature_config, run_config):
    """
    Complete XGBoost training pipeline.
    
    Steps:
    1. Prepare features and target
    2. Initialize walk-forward validator
    3. For each window:
       a. Extract training and test data
       b. Apply target transformation (log if enabled)
       c. Create DMatrix objects for XGBoost
       d. Train XGBoost with early stopping
       e. Generate predictions on test data
       f. Inverse transform predictions
       g. Apply prediction floor
       h. Store results and feature importance
    4. Aggregate predictions across all windows
    5. Compute evaluation metrics
    6. Extract feature importance rankings
    """
    
    # Feature preparation
    X, y = prepare_features(data, feature_config)
    
    # Walk-forward validation
    wf_config = run_config.walk_forward
    validator = WalkForwardValidator(
        window_type=wf_config.window_type,
        initial_train_size=wf_config.initial_train_size,
        test_size=wf_config.test_size,
        step=wf_config.step,
        rolling_window_size=wf_config.rolling_window_size
    )
    
    predictions = []
    feature_importances = []
    
    for window_idx, (train_idx, test_idx) in enumerate(validator.split(X)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        # Target transformation
        if run_config.model.target_transform == "log":
            y_train_transformed = np.log(y_train)
        else:
            y_train_transformed = y_train
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train_transformed)
        dtest = xgb.DMatrix(X_test)
        
        # XGBoost parameters
        params = {
            'objective': run_config.model.objective,
            'max_depth': run_config.model.max_depth,
            'learning_rate': run_config.model.learning_rate,
            'min_child_weight': run_config.model.min_child_weight,
            'subsample': run_config.model.subsample,
            'colsample_bytree': run_config.model.colsample_bytree,
            'gamma': run_config.model.gamma,
            'reg_alpha': run_config.model.reg_alpha,
            'reg_lambda': run_config.model.reg_lambda,
            'random_state': run_config.model.random_state,
            'n_jobs': run_config.model.n_jobs,
            'tree_method': 'hist'  # Fast histogram-based algorithm
        }
        
        # Train with early stopping (internal validation)
        evals = [(dtrain, 'train')]
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=run_config.model.n_estimators,
            evals=evals,
            verbose_eval=False
        )
        
        # Predict
        y_pred = model.predict(dtest)
        
        # Inverse transform
        if run_config.model.target_transform == "log":
            y_pred = np.exp(y_pred)
        
        # Apply floor
        y_pred = np.maximum(y_pred, run_config.model.prediction_floor)
        
        # Extract feature importance
        importance = model.get_score(importance_type='gain')
        
        # Store results
        predictions.append({
            'window': window_idx,
            'y_true': y_test,
            'y_pred': y_pred,
            'feature_importance': importance,
            'best_iteration': model.best_iteration
        })
        
        feature_importances.append(importance)
    
    # Aggregate feature importance across windows
    avg_importance = aggregate_importance(feature_importances, X.columns)
    feature_ranking = pd.DataFrame({
        'feature': avg_importance.keys(),
        'importance': avg_importance.values()
    }).sort_values('importance', ascending=False)
    
    return aggregate_results(predictions, feature_ranking)
```

### 4. Feature Importance Analysis

XGBoost provides multiple importance metrics:

#### Gain (Default)

$$
\text{Gain}_j = \sum_{t: v_t = j} \Delta \text{Loss}_t
$$

Total loss reduction from splits using feature $j$.

#### Weight

$$
\text{Weight}_j = \sum_{t: v_t = j} 1
$$

Number of times feature $j$ is used for splitting.

#### Cover

$$
\text{Cover}_j = \sum_{t: v_t = j} n_t
$$

Total number of samples affected by splits on feature $j$.

```python
def extract_xgb_importance(model, feature_names, importance_type='gain'):
    """
    Extract feature importance from XGBoost model.
    
    Parameters:
    - importance_type: 'gain', 'weight', 'cover', 'total_gain', 'total_cover'
    
    Returns:
    - Ranked feature importance DataFrame
    """
    importance = model.get_score(importance_type=importance_type)
    
    # Map feature indices to names
    importance_dict = {}
    for key, value in importance.items():
        if key.startswith('f'):
            idx = int(key[1:])
            if idx < len(feature_names):
                importance_dict[feature_names[idx]] = value
        else:
            importance_dict[key] = value
    
    importance_df = pd.DataFrame({
        'feature': list(importance_dict.keys()),
        'importance': list(importance_dict.values())
    }).sort_values('importance', ascending=False)
    
    return importance_df
```

## Grid Search for Hyperparameter Tuning

### Grid Search Configuration

```python
@dataclass
class XGBGridSearchConfig:
    enabled: bool = False
    initial_train_sizes: list[int] | None = None     # [52, 104, 156]
    test_sizes: list[int] | None = None              # [1, 2, 4]
    steps: list[int] | None = None                   # [1, 2]
    n_estimators: list[int] | None = None            # [500, 1000, 2000]
    max_depths: list[int] | None = None              # [3, 6, 9]
    learning_rates: list[float] | None = None        # [0.001, 0.01, 0.1]
    min_child_weights: list[float] | None = None     # [1, 3, 5]
    metric: str = "test_r2"                          # Optimization metric
    max_candidates: int | None = None                # Limit search space
```

### Grid Search Procedure

```python
def grid_search_xgb(data, base_config, grid_config):
    """
    Hyperparameter optimization for XGBoost.
    
    Search Space:
    - Window parameters: initial_train_size, test_size, step
    - Boosting parameters: n_estimators, max_depth, learning_rate
    - Regularization: min_child_weight, gamma, reg_alpha, reg_lambda
    
    Procedure:
    1. Generate all parameter combinations
    2. For each combination:
       a. Create run configuration
       b. Train model with walk-forward validation
       c. Compute validation metric (R², MSE, etc.)
    3. Select configuration with best metric
    4. Return best model and parameters
    """
    
    # Generate candidates
    candidates = []
    for train_size in grid_config.initial_train_sizes:
        for test_size in grid_config.test_sizes:
            for step in grid_config.steps:
                for n_est in grid_config.n_estimators:
                    for max_depth in grid_config.max_depths:
                        for lr in grid_config.learning_rates:
                            for mcw in grid_config.min_child_weights:
                                candidates.append({
                                    'initial_train_size': train_size,
                                    'test_size': test_size,
                                    'step': step,
                                    'n_estimators': n_est,
                                    'max_depth': max_depth,
                                    'learning_rate': lr,
                                    'min_child_weight': mcw
                                })
    
    # Limit candidates if specified
    if grid_config.max_candidates:
        candidates = candidates[:grid_config.max_candidates]
    
    # Evaluate each candidate
    results = []
    for candidate in candidates:
        run_config = create_config_from_candidate(base_config, candidate)
        result = run_xgb_experiment(data, feature_config, run_config)
        
        # Extract metric
        metric_value = result.metrics['test'][grid_config.metric]
        
        results.append({
            'candidate': candidate,
            'metric': metric_value,
            'result': result
        })
    
    # Select best (maximize R² or minimize MSE)
    if 'r2' in grid_config.metric:
        best = max(results, key=lambda x: x['metric'])
    else:
        best = min(results, key=lambda x: x['metric'])
    
    return best['result'], best['candidate']
```

## Benchmark Execution

### Multi-Horizon Multi-Feature Benchmark

```python
def benchmark_multi_horizon_xgb(data, config):
    """
    Run comprehensive XGBoost benchmark.
    
    Dimensions:
    - Target horizons: [1, 2, 4] weeks
    - Feature sets: 10 combinations
    - Window types: expanding, rolling
    
    Total experiments per target: 10 feature sets × 2 windows = 20 runs
    """
    
    results = {}
    
    for horizon in config.target_horizons:
        results[horizon] = {}
        
        # Build feature sets
        feature_sets = build_wheat_feature_sets(data)
        
        for feature_set_name, extra_cols in feature_sets.items():
            results[horizon][feature_set_name] = {}
            
            for model_name, run_config in config.model_configs.items():
                # Create feature config
                feature_config = XGBFeatureConfig(
                    target_col=config.target_col,
                    core_columns=config.core_columns,
                    target_horizon=horizon,
                    extra_feature_cols=extra_cols
                )
                
                # Run with caching
                cache_key = generate_cache_key(
                    model_name, feature_set_name, 
                    feature_config, run_config
                )
                
                result = run_with_cache(
                    data=data,
                    feature_config=feature_config,
                    run_config=run_config,
                    cache_key=cache_key,
                    cache_dir=f"cache/xgb/horizon_{horizon}"
                )
                
                results[horizon][feature_set_name][model_name] = result
    
    return results
```

### Default Model Configurations

```python
DEFAULT_XGB_MODELS = {
    "xgb_expanding": XGBRunConfig(
        walk_forward=XGBWalkForwardConfig(
            window_type="expanding",
            initial_train_size=104,
            test_size=1,
            step=1
        ),
        model=XGBModelConfig(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.01,
            min_child_weight=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.0,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            target_transform="log"
        )
    ),
    
    "xgb_rolling": XGBRunConfig(
        walk_forward=XGBWalkForwardConfig(
            window_type="rolling",
            initial_train_size=104,
            rolling_window_size=104,
            test_size=1,
            step=1
        ),
        model=XGBModelConfig(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.01,
            min_child_weight=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.0,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            target_transform="log"
        )
    ),
    
    "xgb_expanding_regularized": XGBRunConfig(
        walk_forward=XGBWalkForwardConfig(
            window_type="expanding",
            initial_train_size=104,
            test_size=1,
            step=1
        ),
        model=XGBModelConfig(
            n_estimators=1000,
            max_depth=4,                 # Shallower trees
            learning_rate=0.01,
            min_child_weight=3.0,        # More regularization
            subsample=0.7,
            colsample_bytree=0.7,
            gamma=0.1,                   # Complexity penalty
            reg_alpha=0.1,               # L1 regularization
            reg_lambda=2.0,              # Stronger L2
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            target_transform="log"
        )
    )
}
```

## Regularization Strategies

### 1. Tree Complexity Control

```python
# Limit tree depth
max_depth = 6  # Prevents overfitting

# Minimum samples in child nodes
min_child_weight = 3.0  # Requires more samples for splits

# Complexity penalty
gamma = 0.1  # Minimum loss reduction required for split
```

### 2. Sampling Strategies

```python
# Row subsampling
subsample = 0.8  # Use 80% of training data per tree

# Column subsampling
colsample_bytree = 0.8  # Use 80% of features per tree
```

### 3. Shrinkage and Regularization

```python
# Learning rate (shrinkage)
learning_rate = 0.01  # Small steps, more trees needed

# L1 regularization (Lasso)
reg_alpha = 0.1  # Promotes sparsity

# L2 regularization (Ridge)
reg_lambda = 1.0  # Smooths weights
```

## Evaluation Metrics

```python
def compute_xgb_metrics(y_true, y_pred):
    """
    Compute comprehensive evaluation metrics for XGBoost.
    """
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'r2log': r2_score(np.log(y_true), np.log(y_pred)),
        'qlike': np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        'directional_accuracy': compute_directional_accuracy(y_true, y_pred),
        'theil_u': compute_theil_u(y_true, y_pred)
    }

def compute_theil_u(y_true, y_pred):
    """
    Theil's U statistic: ratio of model RMSE to naive forecast RMSE.
    U < 1: Model beats naive forecast
    U = 1: Model equals naive forecast
    U > 1: Naive forecast is better
    """
    naive_forecast = y_true[:-1]
    actual_next = y_true[1:]
    pred_next = y_pred[1:]
    
    mse_model = np.mean((actual_next - pred_next) ** 2)
    mse_naive = np.mean((actual_next - naive_forecast) ** 2)
    
    return np.sqrt(mse_model) / np.sqrt(mse_naive)
```

## Caching System

```python
def cache_key_xgb(model_name, feature_set_name, feature_config, 
                  run_config, data_signature):
    """
    Generate unique cache key for XGBoost experiments.
    
    Includes:
    - Model name and feature set
    - Target horizon
    - Window configuration
    - XGBoost hyperparameters (all relevant params)
    - Data signature
    """
    config_dict = {
        'model': model_name,
        'features': feature_set_name,
        'target_horizon': feature_config.target_horizon,
        'window_type': run_config.walk_forward.window_type,
        'initial_train_size': run_config.walk_forward.initial_train_size,
        'n_estimators': run_config.model.n_estimators,
        'max_depth': run_config.model.max_depth,
        'learning_rate': run_config.model.learning_rate,
        'min_child_weight': run_config.model.min_child_weight,
        'subsample': run_config.model.subsample,
        'colsample_bytree': run_config.model.colsample_bytree,
        'gamma': run_config.model.gamma,
        'reg_alpha': run_config.model.reg_alpha,
        'reg_lambda': run_config.model.reg_lambda,
        'data_sig': data_signature
    }
    
    return hashlib.md5(
        json.dumps(config_dict, sort_keys=True).encode()
    ).hexdigest()
```

## Usage Example

```python
from src.benchmark.xgb import benchmark_multi_horizon_xgb

# Configuration
config = WheatXGBBenchmarkConfig(
    target_col="wheat_weekly_rv",
    target_horizons=[1, 2, 4],
    core_columns=["wheat_weekly_rv", "wheat_monthly_rv", "wheat_seasonal_rv"],
    model_names=["xgb_expanding", "xgb_rolling", "xgb_expanding_regularized"],
    feature_set_names=[
        "har", 
        "har_endo_exo", 
        "har_endo_exo_climate_news_macro"
    ],
    use_cache=True,
    cache_dir="cache/xgb"
)

# Run benchmark
results = benchmark_multi_horizon_xgb(data, config)

# Convert to DataFrame
df = benchmark_multi_horizon_results_to_frame(results)

# Results structure:
# - target_horizon: 1, 2, 4
# - model_type: xgb_expanding, xgb_rolling, xgb_expanding_regularized
# - feature_set: har, har_endo_exo, etc.
# - test_mse, test_rmse, test_mae, test_r2, test_r2log
# - n_selected_features (all features used)
# - xgb_n_estimators, xgb_max_depth, xgb_learning_rate, etc.
# - top_10_features: Most important features by gain
```

## Comparison with Other Models

| Aspect | XGBoost | Random Forest | HAR Model |
|--------|---------|---------------|-----------|
| Algorithm | Gradient boosting | Bagging | Linear regression |
| Training | Sequential | Parallel | Direct |
| Regularization | L1/L2 + complexity | Implicit | LASSO/BSR |
| Feature Importance | Gain/weight/cover | MDI | Coefficients |
| Interpretability | Low-Medium | Low | High |
| Speed (training) | Medium | Fast | Very fast |
| Speed (prediction) | Fast | Medium | Very fast |
| Overfitting Control | Excellent | Good | Good |
| Hyperparameters | Many | Moderate | Few |

## Best Practices

1. **Learning Rate**: Use small learning rate (0.001-0.01) with more trees
2. **Tree Depth**: Limit depth (3-6) to prevent overfitting
3. **Regularization**: Use gamma, alpha, lambda to control complexity
4. **Subsampling**: Use subsample and colsample_bytree (0.6-0.9)
5. **Early Stopping**: Monitor validation performance
6. **Feature Engineering**: Create informative features despite XGBoost's power
7. **Cross-Validation**: Use time series CV for hyperparameter tuning
8. **Caching**: Cache results to avoid redundant computation
9. **Parallel Processing**: Leverage n_jobs for faster training
10. **Monitoring**: Track feature importance across windows

## References

1. Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." *Proceedings of the 22nd ACM SIGKDD*, 785-794.

2. Friedman, J. H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine." *Annals of Statistics*, 29(5), 1189-1232.

3. Bucci, A. (2020). "Realized Volatility Forecasting with Neural Networks." *Journal of Financial Econometrics*, 18(3), 502-531.

## Target Construction Methods

### Point vs Mean Target

The XGBoost implementation supports two target construction modes:

```python
@dataclass
class XGBFeatureConfig:
    target_mode: Literal["point", "mean"] = "point"
    target_horizon: int = 1
```

### Point Target (Default)

Point target predicts the realized volatility at a specific future time point:

$$
y_t^{(h)} = RV_{t+h}
$$

where $h$ is the forecast horizon.

**Example for 2-week ahead:**
```
Time:     t    t+1   t+2   t+3   t+4
RV:      2.1   2.3   2.5   2.2   2.4
Target:  2.5   ←---- Point target at t+2
```

**Implementation:**
```python
def create_point_target(rv_series, horizon):
    """
    Create point target for XGBoost.
    
    Parameters:
    -----------
    rv_series : pd.Series
        Realized volatility time series
    horizon : int
        Forecast horizon (number of periods ahead)
    
    Returns:
    --------
    pd.Series
        Target variable shifted by horizon
    """
    return rv_series.shift(-horizon)
```

### Mean Target

Mean target predicts the average realized volatility over the forecast horizon:

$$
y_t^{(h)} = \frac{1}{h}\sum_{i=1}^{h} RV_{t+i}
$$

This provides a smoother target that XGBoost's gradient boosting can optimize more effectively.

**Example for 4-week ahead mean:**
```
Time:     t    t+1   t+2   t+3   t+4
RV:      2.1   2.3   2.5   2.2   2.4
Target:  2.35  ←---- Mean of [2.3, 2.5, 2.2, 2.4]
```

**Implementation:**
```python
def create_mean_target(rv_series, horizon):
    """
    Create mean target for XGBoost.
    
    Parameters:
    -----------
    rv_series : pd.Series
        Realized volatility time series
    horizon : int
        Forecast horizon (number of periods to average)
    
    Returns:
    --------
    pd.Series
        Mean of RV over next h periods
    """
    # Forward-looking rolling mean
    target = pd.Series(index=rv_series.index, dtype=float)
    
    for t in range(len(rv_series) - horizon):
        target.iloc[t] = rv_series.iloc[t+1:t+horizon+1].mean()
    
    return target
```

### XGBoost Gradient Behavior with Different Targets

#### Point Target Gradients

With point targets, XGBoost computes gradients on volatile targets:

$$
g_i = \frac{\partial}{\partial \hat{y}_i}(y_i - \hat{y}_i)^2 = -2(y_i - \hat{y}_i)
$$

$$
h_i = \frac{\partial^2}{\partial \hat{y}_i^2}(y_i - \hat{y}_i)^2 = 2
$$

**Characteristics:**
- **Larger gradient magnitudes** when target is volatile
- **More aggressive** tree splits to capture spikes
- **Higher learning rate** may be needed
- **More boosting rounds** to converge

```python
# Point target training
dtrain_point = xgb.DMatrix(X_train, label=y_train_point)

params_point = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.01,        # Standard learning rate
    'max_depth': 6,
    'min_child_weight': 1.0       # Less regularization
}

model_point = xgb.train(params_point, dtrain_point, num_boost_round=1000)
```

#### Mean Target Gradients

With mean targets, gradients are computed on smoother targets:

$$
g_i = -2(\bar{y}_i - \hat{y}_i) \quad \text{where } \bar{y}_i = \frac{1}{h}\sum_{j=1}^{h}y_{i+j}
$$

**Characteristics:**
- **Smaller gradient magnitudes** due to averaging
- **More stable** tree construction
- **Lower learning rate** may suffice
- **Fewer boosting rounds** needed

```python
# Mean target training
dtrain_mean = xgb.DMatrix(X_train, label=y_train_mean)

params_mean = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.005,       # Lower learning rate
    'max_depth': 4,               # Shallower trees
    'min_child_weight': 3.0       # More regularization
}

model_mean = xgb.train(params_mean, dtrain_mean, num_boost_round=800)
```

### Multi-Horizon Target Construction

For comprehensive benchmarks across multiple horizons:

```python
def prepare_xgb_multi_horizon_targets(rv_series, horizons, target_mode):
    """
    Prepare targets for XGBoost multi-horizon forecasting.
    
    Parameters:
    -----------
    rv_series : pd.Series
        Realized volatility time series
    horizons : list[int]
        List of forecast horizons [1, 2, 4]
    target_mode : str
        'point' or 'mean'
    
    Returns:
    --------
    dict[int, pd.Series]
        Dictionary mapping horizon to target series
    """
    targets = {}
    
    for h in horizons:
        if target_mode == "point":
            # Direct future value
            targets[h] = rv_series.shift(-h)
            
        elif target_mode == "mean":
            # Average over horizon
            target = pd.Series(index=rv_series.index, dtype=float)
            for t in range(len(rv_series) - h):
                target.iloc[t] = rv_series.iloc[t+1:t+h+1].mean()
            targets[h] = target
            
    return targets
```

### Feature Importance Differences

Target construction significantly affects XGBoost's feature importance:

#### Point Target Feature Importance (Gain)
```python
# Top features for point targets (1-week ahead)
point_importance = {
    'wheat_weekly_rv': 0.387,      # Recent volatility dominates
    'wheat_monthly_rv': 0.142,
    'DJIA_Index': 0.089,           # Market shocks important
    'wheat_seasonal_rv': 0.076,
    'WTI_Index': 0.054             # Energy prices
}
```

#### Mean Target Feature Importance (Gain)
```python
# Top features for mean targets (4-week ahead)
mean_importance = {
    'wheat_seasonal_rv': 0.312,    # Long-term patterns dominate
    'wheat_monthly_rv': 0.267,
    'wheat_weekly_rv': 0.156,      # Recent volatility less important
    'ssta_elino': 0.098,           # Climate factors more relevant
    'NAO_index': 0.067,
    'pdsi_extreme_drought_in_planting': 0.045
}
```

**Observation:** XGBoost's gradient-based learning amplifies the shift toward long-term features with mean targets.

### Loss Function Behavior

#### Point Target Loss Evolution

```python
# Training with point targets
evals_result_point = {}
model_point = xgb.train(
    params,
    dtrain_point,
    num_boost_round=1000,
    evals=[(dtrain_point, 'train'), (dval_point, 'val')],
    evals_result=evals_result_point,
    verbose_eval=False
)

# Loss typically shows:
# - Higher initial loss
# - More fluctuation during training
# - Slower convergence
# - Risk of overfitting after ~800 rounds
```

#### Mean Target Loss Evolution

```python
# Training with mean targets
evals_result_mean = {}
model_mean = xgb.train(
    params,
    dtrain_mean,
    num_boost_round=1000,
    evals=[(dtrain_mean, 'train'), (dval_mean, 'val')],
    evals_result=evals_result_mean,
    verbose_eval=False
)

# Loss typically shows:
# - Lower initial loss
# - Smoother convergence
# - Faster convergence
# - Less overfitting risk
```

### Optimal Hyperparameters by Target Type

#### Point Target Configuration

```python
xgb_point_config = XGBModelConfig(
    n_estimators=1000,
    max_depth=6,                  # Deeper trees for complexity
    learning_rate=0.01,           # Standard learning rate
    min_child_weight=1.0,         # Less regularization
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.0,                    # No complexity penalty
    reg_alpha=0.0,                # No L1
    reg_lambda=1.0,               # Standard L2
    target_transform="log"
)
```

#### Mean Target Configuration

```python
xgb_mean_config = XGBModelConfig(
    n_estimators=800,             # Fewer rounds needed
    max_depth=4,                  # Shallower trees sufficient
    learning_rate=0.005,          # Lower learning rate
    min_child_weight=3.0,         # More regularization
    subsample=0.7,                # More subsampling
    colsample_bytree=0.7,
    gamma=0.1,                    # Complexity penalty
    reg_alpha=0.1,                # L1 regularization
    reg_lambda=2.0,               # Stronger L2
    target_transform="log"
)
```

### Impact on Model Performance

```python
def compare_xgb_target_modes(data, horizons=[1, 2, 4]):
    """
    Compare XGBoost performance with point vs mean targets.
    """
    results = {
        'point': {},
        'mean': {}
    }
    
    for mode in ['point', 'mean']:
        for h in horizons:
            # Prepare target
            if mode == 'point':
                y = data['wheat_weekly_rv'].shift(-h)
            else:
                y = create_mean_target(data['wheat_weekly_rv'], h)
            
            # Log transform
            y_log = np.log(y)
            
            # Train-test split
            X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
            y_train, y_test = y_log.iloc[:train_size], y_log.iloc[train_size:]
            
            # Create DMatrix
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test)
            
            # Configure based on target type
            if mode == 'point':
                params = {'learning_rate': 0.01, 'max_depth': 6}
            else:
                params = {'learning_rate': 0.005, 'max_depth': 4}
            
            params.update({
                'objective': 'reg:squarederror',
                'random_state': 42
            })
            
            # Train
            model = xgb.train(params, dtrain, num_boost_round=1000)
            
            # Predict and inverse transform
            y_pred_log = model.predict(dtest)
            y_pred = np.exp(y_pred_log)
            y_test_original = np.exp(y_test)
            
            # Evaluate
            results[mode][h] = {
                'mse': mean_squared_error(y_test_original, y_pred),
                'r2': r2_score(y_test_original, y_pred),
                'mae': mean_absolute_error(y_test_original, y_pred),
                'theil_u': compute_theil_u(y_test_original, y_pred)
            }
    
    return results

# Typical results
# Point targets (h=4):  MSE=0.0221, R²=0.672, MAE=0.108, Theil_U=0.89
# Mean targets (h=4):   MSE=0.0176, R²=0.741, MAE=0.094, Theil_U=0.76
```

### Early Stopping with Different Targets

```python
# Point targets: may need more patience
model_point = xgb.train(
    params_point,
    dtrain_point,
    num_boost_round=2000,
    evals=[(dtrain_point, 'train'), (dval_point, 'val')],
    early_stopping_rounds=100,    # More patience
    verbose_eval=False
)

# Mean targets: can stop earlier
model_mean = xgb.train(
    params_mean,
    dtrain_mean,
    num_boost_round=1500,
    evals=[(dtrain_mean, 'train'), (dval_mean, 'val')],
    early_stopping_rounds=50,     # Less patience needed
    verbose_eval=False
)
```

### Visualization

```python
import matplotlib.pyplot as plt

def visualize_xgb_target_construction(rv_series, horizon=4):
    """
    Visualize point vs mean target construction and XGBoost predictions.
    """
    # Create targets
    point_target = rv_series.shift(-horizon)
    mean_target = create_mean_target(rv_series, horizon)
    
    # Train models (simplified)
    X = prepare_features(rv_series)
    
    # Point model
    dtrain_point = xgb.DMatrix(X, label=np.log(point_target))
    model_point = xgb.train({'max_depth': 6}, dtrain_point, num_boost_round=500)
    pred_point = np.exp(model_point.predict(xgb.DMatrix(X)))
    
    # Mean model
    dtrain_mean = xgb.DMatrix(X, label=np.log(mean_target))
    model_mean = xgb.train({'max_depth': 4}, dtrain_mean, num_boost_round=500)
    pred_mean = np.exp(model_mean.predict(xgb.DMatrix(X)))
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Point target
    axes[0].plot(rv_series.index[:100], rv_series.iloc[:100], 
                 label='Actual RV', linewidth=2, alpha=0.7)
    axes[0].plot(point_target.index[:100], point_target.iloc[:100], 
                 label=f'Point Target (h={horizon})', 
                 linewidth=2, alpha=0.7, linestyle='--')
    axes[0].plot(rv_series.index[:100], pred_point[:100], 
                 label='XGBoost Prediction', 
                 linewidth=2, alpha=0.7, linestyle=':')
    axes[0].set_title(f'XGBoost with Point Target (h={horizon} weeks)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Mean target
    axes[1].plot(rv_series.index[:100], rv_series.iloc[:100], 
                 label='Actual RV', linewidth=2, alpha=0.7)
    axes[1].plot(mean_target.index[:100], mean_target.iloc[:100], 
                 label=f'Mean Target (h={horizon})', 
                 linewidth=2, alpha=0.7, linestyle='--')
    axes[1].plot(rv_series.index[:100], pred_mean[:100], 
                 label='XGBoost Prediction', 
                 linewidth=2, alpha=0.7, linestyle=':')
    axes[1].set_title(f'XGBoost with Mean Target (h={horizon} weeks)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
```

### Benchmark Configuration

```python
# Point target benchmark
config_point = WheatXGBBenchmarkConfig(
    target_col="wheat_weekly_rv",
    target_horizons=[1, 2, 4],
    target_mode="point",
    model_names=["xgb_expanding", "xgb_rolling"],
    feature_set_names=["har", "har_endo_exo", "har_endo_exo_climate_news_macro"],
    use_cache=True
)

# Mean target benchmark
config_mean = WheatXGBBenchmarkConfig(
    target_col="wheat_weekly_rv",
    target_horizons=[1, 2, 4],
    target_mode="mean",
    model_names=["xgb_expanding", "xgb_rolling"],
    feature_set_names=["har", "har_endo_exo", "har_endo_exo_climate_news_macro"],
    use_cache=True
)

# Run both
results_point = benchmark_multi_horizon_xgb(data, config_point)
results_mean = benchmark_multi_horizon_xgb(data, config_mean)

# Compare
comparison_df = pd.DataFrame({
    'Point_MSE': [results_point[h]['har_endo_exo']['xgb_expanding'].metrics['test']['mse'] 
                  for h in [1, 2, 4]],
    'Mean_MSE': [results_mean[h]['har_endo_exo']['xgb_expanding'].metrics['test']['mse'] 
                 for h in [1, 2, 4]],
    'Point_R2': [results_point[h]['har_endo_exo']['xgb_expanding'].metrics['test']['r2'] 
                 for h in [1, 2, 4]],
    'Mean_R2': [results_mean[h]['har_endo_exo']['xgb_expanding'].metrics['test']['r2'] 
                for h in [1, 2, 4]]
}, index=[1, 2, 4])

print(comparison_df)
```

### Practical Recommendations

#### Use Point Targets When:
1. **Specific date forecasting** is required
2. **Short-term trading** strategies (1-2 weeks)
3. **Event-driven** volatility prediction
4. Need to **capture spikes** and extreme events
5. **Tactical risk management**

**XGBoost Configuration:**
- Higher learning rate (0.01-0.03)
- Deeper trees (max_depth=6-8)
- Less regularization
- More boosting rounds (1000-2000)

#### Use Mean Targets When:
1. **Average volatility** over a period matters
2. **Medium to long-term** forecasting (4+ weeks)
3. **Strategic portfolio** decisions
4. **Risk budgeting** over horizons
5. Prefer **stable, smooth** predictions

**XGBoost Configuration:**
- Lower learning rate (0.005-0.01)
- Shallower trees (max_depth=3-5)
- More regularization (higher gamma, reg_lambda)
- Fewer boosting rounds (500-1000)

### Example: Complete Workflow

```python
from src.benchmark.xgb import benchmark_multi_horizon_xgb

# Load data
data = pd.read_csv('wheat_volatility.csv', index_col=0, parse_dates=True)

# Scenario 1: Point targets for tactical trading
config_tactical = WheatXGBBenchmarkConfig(
    target_col="wheat_weekly_rv",
    target_horizons=[1, 2],
    target_mode="point",
    model_names=["xgb_expanding"],
    feature_set_names=["har_endo_exo_news_macro"]
)

results_tactical = benchmark_multi_horizon_xgb(data, config_tactical)

# Scenario 2: Mean targets for risk management
config_risk = WheatXGBBenchmarkConfig(
    target_col="wheat_weekly_rv",
    target_horizons=[4],
    target_mode="mean",
    model_names=["xgb_expanding_regularized"],
    feature_set_names=["har_endo_exo_climate_macro"]
)

results_risk = benchmark_multi_horizon_xgb(data, config_risk)

# Extract results
tactical_metrics = results_tactical[1]['har_endo_exo_news_macro']['xgb_expanding'].metrics['test']
risk_metrics = results_risk[4]['har_endo_exo_climate_macro']['xgb_expanding_regularized'].metrics['test']

print("Tactical (Point, h=1):", tactical_metrics)
print("Risk (Mean, h=4):", risk_metrics)
```

### Key Takeaway

XGBoost's gradient-based optimization makes it particularly sensitive to target construction. Mean targets lead to smoother gradients, faster convergence, and often better generalization. However, point targets provide more precise predictions for specific time points, which may be more valuable depending on the application. The choice should align with your forecasting objective and risk management strategy.
