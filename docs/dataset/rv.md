
# Realized Volatility and Climate Features

## Overview

This dataset is designed to forecast **weekly realized volatility (RV) of wheat prices** using a combination of:

* **Market-based features** (derived from price returns of agricultural commodities such as wheat, corn, and soybeans)
* **Climate-based features** (capturing environmental conditions affecting agricultural production)

The goal is to model how **market dynamics and climate variability jointly influence volatility** in agricultural commodities.

---

## Feature Naming Convention

All features follow a consistent structure:

```
<commodity> + <frequency> + <feature_name>
```

### Examples

* `wheat_weekly_RV`
* `corn_monthly_RVG`
* `soybean_seasonal_RSK`

### Components

* **Commodity**: wheat, corn, soybeans
* **Frequency**:

  * `weekly`: short-term dynamics
  * `monthly`: medium-term trends
  * `seasonal`: long-term/harvest cycle effects
* **Feature name**: statistical measure (defined below)

---

## Realized Volatility Features

Let ( r_i ) denote intraperiod log-returns (e.g., daily returns within a week), and ( M ) be the number of observations in that period.

---

### 1. Realized Volatility (RV)

$$
RV = \sqrt{\sum_{i=1}^{M} r_i^2}
$$

**Interpretation:**

* Measures total price variability within a period
* Equivalent to the **magnitude of market uncertainty**

**Why it matters:**

* Core target variable for volatility forecasting
* Captures both upward and downward movements

---

### 2. Positive Realized Volatility (RVG)

$$
RVG = \sqrt{\sum_{r_i > 0} r_i^2}
$$

**Interpretation:**

* Volatility contributed only by **positive returns (price increases)**

**Use case:**

* Helps distinguish **bullish volatility regimes**

---

### 3. Negative Realized Volatility (RVB)

$$
RVB = \sqrt{\sum_{r_i < 0} r_i^2}
$$

**Interpretation:**

* Volatility from **negative returns (price drops)**

**Use case:**

* Important for modeling **downside risk** and stress periods

---

### 4. Realized Skewness (RSK)

$$
RSK = \frac{\sqrt{M},\sum_{i=1}^{M} r_i^3}{RV^{1.5}}
\qquad \text{if } RV > 0 \text{ and } M > 1
$$

**Interpretation:**

* Measures **asymmetry** of returns distribution

  * Positive → large upward moves dominate
  * Negative → large downward moves dominate

**Why it matters:**

* Early signal of **market imbalance or directional risk**

---

### 5. Realized Kurtosis (RKU)

$$
RKU = \frac{M,\sum_{i=1}^{M} r_i^4}{RV^2}
\qquad \text{if } RV > 0 \text{ and } M > 1
$$

**Interpretation:**

* Measures **tail risk** (frequency of extreme events)

**Why it matters:**

* High values indicate **jumps, shocks, or extreme volatility bursts**

---

### 6. Bipower Variation (BV) and Jumps

$$
BV = \frac{1}{\mu_1^{2}}
\sum_{i=1}^{M-1} |r_i|;|r_{i+1}|
$$

$$
JUMPS = \max(RV - BV,; 0)
$$

**Interpretation:**

* **BV** estimates the “continuous” part of volatility
* **JUMPS** isolates **discontinuous shocks** (e.g., news, weather events)

**Why it matters:**

* Separates **normal market fluctuations** from **sudden shocks**

---

### 7. Tail Risk Measures (TRu, TRd)

#### Upper Tail Volatility

$$
TRu = \sum_{r_i > q_{95}} r_i^{2}
$$

#### Lower Tail Volatility

$$
TRd = \sum_{r_i < q_{5}} r_i^{2}
$$

**Interpretation:**

* Focuses only on **extreme returns**

  * ( q_{95} ): 95th percentile
  * ( q_{5} ): 5th percentile

**Why it matters:**

* Captures **rare but impactful events**
* Useful for **risk management and stress modeling**

---

### 8. Leverage Effect (LEV)

$$
LEV = \mathrm{corr}(r,; |r|)
$$

**Interpretation:**

* Measures relationship between **returns and volatility**

  * Negative correlation → volatility rises after price drops

**Why it matters:**

* Common in financial markets
* Indicates **asymmetric volatility response**

---

