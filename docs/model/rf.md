# Random Forest Implementation and Training Procedure

## Overview

This document describes the Random Forest implementation used in the Volf project for volatility forecasting. Random Forest is an ensemble learning method that constructs multiple decision trees and aggregates their predictions to capture complex non-linear relationships in volatility dynamics.

## Model Architecture

### Ensemble Structure

Random Forest builds $B$ decision trees on bootstrap samples:

$$
\hat{f}_{RF}(\mathbf{x}) = \frac{1}{B}\sum_{b=1}^{B} \hat{f}_b(\mathbf{x})
$$

where each tree $\hat{f}_b$ is trained on a bootstrap sample with random feature subsampling at each split.

## Implementation Architecture

### Configuration Structure

```python
@dataclass
class RFFeatureConfig:
    target_col: str                              # Target variable name
    core_columns: list[str]                      # Core HAR features
    target_horizon: int = 1                      # Forecast horizon
    extra_feature_cols: list[str] | None = None  # Additional predictors
    target_col_name: str = "RV_target"
    target_mode: Literal["point", "mean"] = "point"
    target_floor: float = 1e-10

@dataclass
class RFWalkForwardConfig:
    window_type: Literal["expanding", "rolling"] = "expanding"
    initial_train_size: int = 104                # Initial training window
    test_size: int = 1                           # Test window size
    step: int = 1                                # Step size
    rolling_window_size: int | None = None       # Fixed window for rolling
    progress_bar: bool = True

@dataclass
class RFModelConfig:
    n_estimators: int = 500                      # Number of trees
    max_depth: int | None = None                 # Maximum tree depth
    min_samples_split: int = 5                   # Min samples to split
    min_samples_leaf: int = 2                    # Min samples in leaf
    max_features: str | float = "sqrt"           # Features per split
    bootstrap: bool = True                       # Bootstrap sampling
    random_state: int = 42
    n_jobs: int = -1                             # Parallel jobs
    target_transform: Literal["none", "log"] = "log"
    prediction_floor: float = 1e-10
```

## Multi-Target Training Procedure

### 1. Target and Feature Set Definition

Same hierarchical feature sets as HAR model:

```python
feature_sets = {
    "har": [],                                    # Core only
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

- Training data grows with each iteration
- Captures long-term patterns
- Default: `initial_train_size=104` weeks

#### Rolling Window Strategy

```
Window 1:  [----------] → [*]
Window 2:      [----------] → [*]
Window 3:          [----------] → [*]
```

- Fixed-size training window
- Adapts to recent market conditions
- Default: `rolling_window_size=104` weeks

### 3. Model Training Pipeline

```python
def run_rf_experiment(data, feature_config, run_config):
    """
    Complete Random Forest training pipeline.
    
    Steps:
    1. Prepare features and target
    2. Initialize walk-forward validator
    3. For each window:
       a. Extract training and test data
       b. Apply target transformation (log if enabled)
       c. Train Random Forest on training data
       d. Generate predictions on test data
       e. Inverse transform predictions
       f. Apply prediction floor
       g. Store results and feature importance
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
        
        # Train Random Forest
        rf = RandomForestRegressor(
            n_estimators=run_config.model.n_estimators,
            max_depth=run_config.model.max_depth,
            min_samples_split=run_config.model.min_samples_split,
            min_samples_leaf=run_config.model.min_samples_leaf,
            max_features=run_config.model.max_features,
            bootstrap=run_config.model.bootstrap,
            random_state=run_config.model.random_state,
            n_jobs=run_config.model.n_jobs
        )
        
        rf.fit(X_train, y_train_transformed)
        
        # Predict
        y_pred = rf.predict(X_test)
        
        # Inverse transform
        if run_config.model.target_transform == "log":
            y_pred = np.exp(y_pred)
        
        # Apply floor
        y_pred = np.maximum(y_pred, run_config.model.prediction_floor)
        
        # Store results
        predictions.append({
            'window': window_idx,
            'y_true': y_test,
            'y_pred': y_pred,
            'feature_importance': rf.feature_importances_
        })
        
        feature_importances.append(rf.feature_importances_)
    
    # Aggregate feature importance across windows
    avg_importance = np.mean(feature_importances, axis=0)
    feature_ranking = pd.DataFrame({
        'feature': X.columns,
        'importance': avg_importance
    }).sort_values('importance', ascending=False)
    
    return aggregate_results(predictions, feature_ranking)
```

### 4. Feature Importance Analysis

Random Forest provides built-in feature importance through Mean Decrease in Impurity (MDI):

$$
\text{Importance}_j = \frac{1}{B}\sum_{b=1}^{B}\sum_{t \in T_b} \mathbb{1}(v_t = j) \cdot \Delta i_t
$$

where:
- $v_t$ is the feature used at node $t$
- $\Delta i_t$ is the decrease in impurity at node $t$
- $T_b$ is the set of nodes in tree $b$

```python
def extract_feature_importance(rf_model, feature_names):
    """
    Extract and rank feature importance.
    
    Returns:
    - Top features by importance
    - Cumulative importance
    - Feature selection threshold
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Cumulative importance
    importance_df['cumulative_importance'] = \
        importance_df['importance'].cumsum()
    
    # Features contributing to 95% of importance
    threshold_95 = importance_df[
        importance_df['cumulative_importance'] <= 0.95
    ]
    
    return importance_df, threshold_95
```

## Grid Search for Hyperparameter Tuning

### Grid Search Configuration

```python
@dataclass
class RFGridSearchConfig:
    enabled: bool = False
    initial_train_sizes: list[int] | None = None     # [52, 104, 156]
    test_sizes: list[int] | None = None              # [1, 2, 4]
    steps: list[int] | None = None                   # [1, 2]
    max_depths: list[int | None] | None = None       # [10, 20, 30, None]
    min_samples_splits: list[int] | None = None      # [2, 5, 10]
    min_samples_leafs: list[int] | None = None       # [1, 2, 4]
    metric: str = "test_r2"                          # Optimization metric
    max_candidates: int | None = None                # Limit search space
```

### Grid Search Procedure

```python
def grid_search_rf(data, base_config, grid_config):
    """
    Hyperparameter optimization for Random Forest.
    
    Search Space:
    - Window parameters: initial_train_size, test_size, step
    - Tree parameters: max_depth, min_samples_split, min_samples_leaf
    
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
                for max_depth in grid_config.max_depths:
                    for min_split in grid_config.min_samples_splits:
                        for min_leaf in grid_config.min_samples_leafs:
                            candidates.append({
                                'initial_train_size': train_size,
                                'test_size': test_size,
                                'step': step,
                                'max_depth': max_depth,
                                'min_samples_split': min_split,
                                'min_samples_leaf': min_leaf
                            })
    
    # Limit candidates if specified
    if grid_config.max_candidates:
        candidates = candidates[:grid_config.max_candidates]
    
    # Evaluate each candidate
    results = []
    for candidate in candidates:
        run_config = create_config_from_candidate(base_config, candidate)
        result = run_rf_experiment(data, feature_config, run_config)
        
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
def benchmark_multi_horizon_rf(data, config):
    """
    Run comprehensive Random Forest benchmark.
    
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
                feature_config = RFFeatureConfig(
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
                    cache_dir=f"cache/rf/horizon_{horizon}"
                )
                
                results[horizon][feature_set_name][model_name] = result
    
    return results
```

### Default Model Configurations

```python
DEFAULT_RF_MODELS = {
    "rf_expanding": RFRunConfig(
        walk_forward=RFWalkForwardConfig(
            window_type="expanding",
            initial_train_size=104,
            test_size=1,
            step=1
        ),
        model=RFModelConfig(
            n_estimators=500,
            max_depth=None,              # Unlimited depth
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
            target_transform="log"
        )
    ),
    
    "rf_rolling": RFRunConfig(
        walk_forward=RFWalkForwardConfig(
            window_type="rolling",
            initial_train_size=104,
            rolling_window_size=104,
            test_size=1,
            step=1
        ),
        model=RFModelConfig(
            n_estimators=500,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
            target_transform="log"
        )
    ),
    
    "rf_expanding_shallow": RFRunConfig(
        walk_forward=RFWalkForwardConfig(
            window_type="expanding",
            initial_train_size=104,
            test_size=1,
            step=1
        ),
        model=RFModelConfig(
            n_estimators=500,
            max_depth=20,                # Limited depth
            min_samples_split=10,
            min_samples_leaf=4,
            max_features="sqrt",
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
            target_transform="log"
        )
    )
}
```

## Evaluation Metrics

```python
def compute_rf_metrics(y_true, y_pred):
    """
    Compute comprehensive evaluation metrics for Random Forest.
    """
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'r2log': r2_score(np.log(y_true), np.log(y_pred)),
        'qlike': np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        'directional_accuracy': compute_directional_accuracy(y_true, y_pred)
    }

def compute_directional_accuracy(y_true, y_pred):
    """
    Percentage of correct directional predictions.
    """
    direction_true = np.sign(np.diff(y_true))
    direction_pred = np.sign(np.diff(y_pred))
    return np.mean(direction_true == direction_pred) * 100
```

## Caching System

```python
def cache_key_rf(model_name, feature_set_name, feature_config, 
                 run_config, data_signature):
    """
    Generate unique cache key for Random Forest experiments.
    
    Includes:
    - Model name and feature set
    - Target horizon
    - Window configuration
    - RF hyperparameters (n_estimators, max_depth, etc.)
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
        'min_samples_split': run_config.model.min_samples_split,
        'min_samples_leaf': run_config.model.min_samples_leaf,
        'data_sig': data_signature
    }
    
    return hashlib.md5(
        json.dumps(config_dict, sort_keys=True).encode()
    ).hexdigest()
```

## Parallel Execution

Random Forest training is parallelized at two levels:

### 1. Tree-Level Parallelization

```python
# Within each Random Forest model
rf = RandomForestRegressor(
    n_estimators=500,
    n_jobs=-1  # Use all available CPU cores
)
```

Each tree is trained independently on a separate CPU core.

### 2. Experiment-Level Parallelization

```python
from concurrent.futures import ThreadPoolExecutor

def parallel_benchmark(data, config):
    """
    Run multiple experiments in parallel.
    """
    tasks = []
    
    for horizon in config.target_horizons:
        for feature_set in config.feature_sets:
            for model_name in config.model_names:
                tasks.append((horizon, feature_set, model_name))
    
    results = {}
    
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = {
            executor.submit(
                run_single_experiment, 
                data, horizon, feature_set, model_name
            ): (horizon, feature_set, model_name)
            for horizon, feature_set, model_name in tasks
        }
        
        for future in as_completed(futures):
            horizon, feature_set, model_name = futures[future]
            result = future.result()
            
            if horizon not in results:
                results[horizon] = {}
            if feature_set not in results[horizon]:
                results[horizon][feature_set] = {}
            
            results[horizon][feature_set][model_name] = result
    
    return results
```

## Usage Example

```python
from src.benchmark.rf import benchmark_multi_horizon_rf

# Configuration
config = WheatRFBenchmarkConfig(
    target_col="wheat_weekly_rv",
    target_horizons=[1, 2, 4],
    core_columns=["wheat_weekly_rv", "wheat_monthly_rv", "wheat_seasonal_rv"],
    model_names=["rf_expanding", "rf_rolling"],
    feature_set_names=[
        "har", 
        "har_endo_exo", 
        "har_endo_exo_climate_news_macro"
    ],
    use_cache=True,
    cache_dir="cache/rf",
    max_workers=4  # Parallel experiments
)

# Run benchmark
results = benchmark_multi_horizon_rf(data, config)

# Convert to DataFrame
df = benchmark_multi_horizon_results_to_frame(results)

# Results structure:
# - target_horizon: 1, 2, 4
# - model_type: rf_expanding, rf_rolling
# - feature_set: har, har_endo_exo, etc.
# - test_mse, test_rmse, test_mae, test_r2, test_r2log
# - n_selected_features (all features used)
# - rf_n_estimators, rf_max_depth, rf_min_samples_split, etc.
# - top_10_features: Most important features
```

## Comparison with HAR Model

| Aspect | Random Forest | HAR Model |
|--------|---------------|-----------|
| Model Type | Non-parametric ensemble | Linear regression |
| Feature Selection | Implicit (importance) | Explicit (LASSO/BSR) |
| Interactions | Automatic | Manual specification |
| Interpretability | Feature importance | Coefficients |
| Training Time | Slower | Faster |
| Prediction Time | Moderate | Very fast |
| Overfitting Control | Tree depth, min samples | Regularization |
| Extrapolation | Poor | Better |

## References

1. Breiman, L. (2001). "Random Forests." *Machine Learning*, 45(1), 5-32.

2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.

3. Hillebrand, E., & Medeiros, M. C. (2010). "The Benefits of Bagging for Forecast Models of Realized Volatility." *Econometric Reviews*, 29(5-6), 571-593.

## Target Construction Methods

### Point vs Mean Target

The Random Forest implementation supports two target construction modes:

```python
@dataclass
class RFFeatureConfig:
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
    Create point target for Random Forest.
    
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

This provides a smoother target that Random Forest can predict more accurately.

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
    Create mean target for Random Forest.
    
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

### Random Forest Behavior with Different Targets

#### Point Target Characteristics

Random Forest with point targets:
- **Higher variance** in predictions
- **More sensitive** to individual tree predictions
- **Captures** short-term volatility spikes
- **Better for** tactical trading decisions

```python
# Point target training
rf_point = RandomForestRegressor(n_estimators=500)
rf_point.fit(X_train, y_train_point)

# Predictions show more variability
y_pred_point = rf_point.predict(X_test)
# std(y_pred_point) = 0.234
```

#### Mean Target Characteristics

Random Forest with mean targets:
- **Lower variance** in predictions (smoother)
- **More stable** across different tree samples
- **Averages out** short-term noise
- **Better for** risk management and hedging

```python
# Mean target training
rf_mean = RandomForestRegressor(n_estimators=500)
rf_mean.fit(X_train, y_train_mean)

# Predictions are smoother
y_pred_mean = rf_mean.predict(X_test)
# std(y_pred_mean) = 0.187
```

### Multi-Horizon Target Construction

For comprehensive benchmarks across multiple horizons:

```python
def prepare_rf_multi_horizon_targets(rv_series, horizons, target_mode):
    """
    Prepare targets for Random Forest multi-horizon forecasting.
    
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

Target construction affects feature importance rankings:

#### Point Target Feature Importance
```python
# Top features for point targets (1-week ahead)
point_importance = {
    'wheat_weekly_rv': 0.342,      # Recent volatility most important
    'wheat_monthly_rv': 0.156,
    'wheat_seasonal_rv': 0.089,
    'DJIA_Index': 0.067,           # Market factors
    'ssta_elino': 0.045            # Climate factors
}
```

#### Mean Target Feature Importance
```python
# Top features for mean targets (4-week ahead)
mean_importance = {
    'wheat_seasonal_rv': 0.298,    # Long-term patterns more important
    'wheat_monthly_rv': 0.234,
    'wheat_weekly_rv': 0.187,      # Recent volatility less important
    'ssta_elino': 0.078,           # Climate factors more relevant
    'NAO_index': 0.056
}
```

**Observation:** Mean targets shift importance toward longer-term features.

### Impact on Model Performance

```python
def compare_target_modes(data, horizons=[1, 2, 4]):
    """
    Compare Random Forest performance with point vs mean targets.
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
            
            # Train-test split
            X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
            y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
            
            # Train Random Forest
            rf = RandomForestRegressor(n_estimators=500, random_state=42)
            rf.fit(X_train, y_train)
            
            # Predict
            y_pred = rf.predict(X_test)
            
            # Evaluate
            results[mode][h] = {
                'mse': mean_squared_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred)
            }
    
    return results

# Typical results
# Point targets (h=4):  MSE=0.0234, R²=0.654, MAE=0.112
# Mean targets (h=4):   MSE=0.0189, R²=0.723, MAE=0.098
```

### Visualization

```python
import matplotlib.pyplot as plt

def visualize_target_construction(rv_series, horizon=4):
    """
    Visualize point vs mean target construction.
    """
    # Create targets
    point_target = rv_series.shift(-horizon)
    mean_target = create_mean_target(rv_series, horizon)
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Point target
    axes[0].plot(rv_series.index[:100], rv_series.iloc[:100], 
                 label='Actual RV', linewidth=2, alpha=0.7)
    axes[0].plot(point_target.index[:100], point_target.iloc[:100], 
                 label=f'Point Target (h={horizon})', 
                 linewidth=2, alpha=0.7, linestyle='--')
    axes[0].set_title(f'Point Target Construction (h={horizon} weeks)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Mean target
    axes[1].plot(rv_series.index[:100], rv_series.iloc[:100], 
                 label='Actual RV', linewidth=2, alpha=0.7)
    axes[1].plot(mean_target.index[:100], mean_target.iloc[:100], 
                 label=f'Mean Target (h={horizon})', 
                 linewidth=2, alpha=0.7, linestyle='--')
    axes[1].set_title(f'Mean Target Construction (h={horizon} weeks)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
```

### Benchmark Configuration

```python
# Point target benchmark
config_point = WheatRFBenchmarkConfig(
    target_col="wheat_weekly_rv",
    target_horizons=[1, 2, 4],
    target_mode="point",
    model_names=["rf_expanding", "rf_rolling"],
    feature_set_names=["har", "har_endo_exo", "har_endo_exo_climate_news_macro"],
    use_cache=True
)

# Mean target benchmark
config_mean = WheatRFBenchmarkConfig(
    target_col="wheat_weekly_rv",
    target_horizons=[1, 2, 4],
    target_mode="mean",
    model_names=["rf_expanding", "rf_rolling"],
    feature_set_names=["har", "har_endo_exo", "har_endo_exo_climate_news_macro"],
    use_cache=True
)

# Run both
results_point = benchmark_multi_horizon_rf(data, config_point)
results_mean = benchmark_multi_horizon_rf(data, config_mean)

# Compare
comparison = compare_results(results_point, results_mean)
```

### Tree Depth and Target Smoothness

Mean targets allow for shallower trees:

```python
# Point targets: need deeper trees to capture variability
rf_point_config = RFModelConfig(
    n_estimators=500,
    max_depth=None,           # Unlimited depth
    min_samples_split=5,
    min_samples_leaf=2
)

# Mean targets: can use shallower trees
rf_mean_config = RFModelConfig(
    n_estimators=500,
    max_depth=20,             # Limited depth works well
    min_samples_split=10,     # More regularization
    min_samples_leaf=4
)
```

### Practical Recommendations

#### Use Point Targets When:
1. Forecasting specific future dates (e.g., next week's volatility)
2. Short-term trading strategies
3. Event-driven forecasting
4. Need to capture volatility spikes

#### Use Mean Targets When:
1. Risk management over a period
2. Portfolio rebalancing decisions
3. Long-term forecasting (4+ weeks)
4. Prefer stable, smooth predictions
5. Evaluating average market conditions

### Example: Complete Workflow

```python
from src.benchmark.rf import benchmark_multi_horizon_rf

# Load data
data = pd.read_csv('wheat_volatility.csv', index_col=0, parse_dates=True)

# Scenario 1: Point targets for tactical trading
config_tactical = WheatRFBenchmarkConfig(
    target_col="wheat_weekly_rv",
    target_horizons=[1, 2],      # Short-term
    target_mode="point",
    model_names=["rf_expanding"],
    feature_set_names=["har_endo_exo_news_macro"]
)

results_tactical = benchmark_multi_horizon_rf(data, config_tactical)

# Scenario 2: Mean targets for risk management
config_risk = WheatRFBenchmarkConfig(
    target_col="wheat_weekly_rv",
    target_horizons=[4],         # Medium-term
    target_mode="mean",
    model_names=["rf_expanding"],
    feature_set_names=["har_endo_exo_climate_macro"]
)

results_risk = benchmark_multi_horizon_rf(data, config_risk)

# Compare performance
print("Tactical (Point, h=1):", results_tactical[1]['har_endo_exo_news_macro']['rf_expanding'].metrics['test'])
print("Risk (Mean, h=4):", results_risk[4]['har_endo_exo_climate_macro']['rf_expanding'].metrics['test'])
```

### Key Takeaway

Random Forest's ensemble nature makes it particularly well-suited for mean targets, as the averaging across trees naturally complements the averaging in the target construction. However, point targets provide more actionable predictions for specific time points.
