# Climate Dataset Documentation

## Overview

This document describes only the climate datasets that are actually consumed by the current unified preprocessing pipeline in `src/dataset/climate/preprocessing.py`.

The active pipeline reads four climate inputs:

- `data/climate/noaa_weekly.csv`
- `data/climate/spi_weekly_multiscale.csv`
- `data/climate/co2_weekly_mlo.csv`
- `data/climate/palmer/`

These inputs are transformed into crop-specific climate features and merged into:

- `data/ag/corn.csv`
- `data/ag/soybean.csv`
- `data/ag/wheat.csv`

The pipeline runs per crop and uses crop calendars plus production-by-state weights to turn state-level climate signals into national crop-level features.

---

## 1. Inputs Actually Used

### 1.1 NOAA weekly weather

Source file:
- `data/climate/noaa_weekly.csv`

Used columns:
- `state`
- `date`
- `TMAX`
- `TMIN`
- `AWND`

Notes:
- `PRCP` exists in the file but is not used directly by `src/dataset/climate/preprocessing.py`.
- The pipeline parses `date` as a timestamp and derives ISO `week_of_year` from it.

### 1.2 SPI weekly multiscale precipitation index

Source file:
- `data/climate/spi_weekly_multiscale.csv`

Used columns:
- `state`
- `week_date`
- `SPI_7d`
- `SPI_1m`
- `SPI_3m`

Notes:
- The file may contain additional SPI horizons such as `SPI_2m`, `SPI_6m`, and `SPI_12m`, but the current pipeline only uses `SPI_7d`, `SPI_1m`, and `SPI_3m`.
- `week_date` is copied into a working `date` column before feature generation.

### 1.3 Weekly CO2 series

Source file:
- `data/climate/co2_weekly_mlo.csv`

Used columns:
- `date`
- the first non-date value column, currently `co2_molfrac_ppm`

Notes:
- The loader intentionally picks the first column that is not `date`, empty, or `Unnamed: 0`.
- CO2 is treated as a national series, not a state-level dataset.

### 1.4 Palmer drought index files

Source directory:
- `data/climate/palmer/`

Directory layout used by the loader:
- one directory per state, such as `data/climate/palmer/01_Alabama`
- one CSV per climate division inside each state directory, such as `0101.csv`

Raw file format expected by the loader:
- first line is a text header and is skipped
- remaining lines are `YYYYMMDD,pdsi`

Notes:
- All station or division files inside a state directory are read.
- State-level weekly PDSI is formed by averaging all divisions for the same state and date.

---

## 2. How These Inputs Are Gathered

### 2.1 NOAA weekly weather gathering

The current preprocessing pipeline consumes `data/climate/noaa_weekly.csv`, which is produced upstream by the NOAA fetcher in `src/dataset/climate/noaa.py` and the CLI wrapper in `scripts/dataset/climate/noaa.py`.

Gathering process:
1. The script queries NOAA's `GHCND` dataset by state FIPS code.
2. It requests only four daily data types: `PRCP`, `TMAX`, `TMIN`, and `AWND`.
3. Requests are paginated with `limit=1000` and increasing `offset`.
4. Responses are cached locally in `.cache/noaa/*.parquet`.
5. Multiple NOAA tokens can be supplied with `NOAA_TOKENS`; the client rotates tokens when a daily rate limit is hit.
6. Daily rows are converted from NOAA units by dividing `value` by `10`.
7. Daily state-level values are pivoted into wide format.
8. Daily data is resampled to weekly frequency with `W-MON`, labeled on the left edge of the interval.

Weekly aggregation rules:
- `PRCP`: weekly sum
- `TMAX`: weekly mean
- `TMIN`: weekly mean
- `AWND`: weekly mean

Output produced upstream:
- `data/climate/noaa_daily.csv` or refreshed variants of that file
- `data/climate/noaa_weekly.csv`

### 2.2 SPI weekly gathering

The current preprocessing pipeline consumes `data/climate/spi_weekly_multiscale.csv`, which is produced from `data/climate/noaa_daily.csv` by `scripts/dataset/climate/noaa_spi.py`.

Gathering and derivation process:
1. Load daily NOAA precipitation by state from `data/climate/noaa_daily.csv`.
2. Sort observations by `state` and `date`.
3. Build rolling precipitation totals for six daily windows:
   - `7d = 7`
   - `1m = 30`
   - `2m = 60`
   - `3m = 90`
   - `6m = 180`
   - `12m = 365`
4. For each state and time scale, fit a Gamma distribution to non-zero rolling totals.
5. Convert cumulative probabilities to standard normal values with `norm.ppf` to obtain SPI.
6. Convert daily rows to weekly rows by assigning each date to its week start and keeping the last observation in each state-week.

Important implementation details:
- Gamma fitting is skipped when fewer than 10 non-zero samples are available.
- CDF values are clipped to avoid infinite values in `norm.ppf`.
- Although multiple horizons are written to the SPI file, the downstream climate pipeline uses only `SPI_7d`, `SPI_1m`, and `SPI_3m`.

Output produced upstream:
- `data/climate/spi_weekly_multiscale.csv`

### 2.3 CO2 weekly gathering

The current preprocessing pipeline reads CO₂ data from a file sourced from the NOAA Global Monitoring Laboratory dataset available at:
[https://gml.noaa.gov/data/dataset.php?item=all-co2-flask](https://gml.noaa.gov/data/dataset.php?item=all-co2-flask)

Within this repository, the preprocessing code assumes the dataset has already been downloaded and stored locally (e.g., as `data/climate/co2_weekly_mlo.csv`) and does not handle fetching or regeneration.

The pipeline relies on the following file contract:

* it must contain a `date` column
* it must contain one usable numeric CO₂ value column
* the series is treated as weekly and representative of a single location (e.g., Mauna Loa observations)

Downstream preprocessing uses this CO₂ time series as a univariate weekly climate signal.

### 2.4 Palmer / PDSI gathering

The current preprocessing pipeline reads weekly Palmer drought files from `data/climate/palmer/` directly.

Within this repository, the preprocessing code assumes the Palmer files already exist and does not fetch them. The only gathering behavior visible from the code is the expected on-disk format:
- state directories must be named like `<code>_<state_name>`
- each state directory contains climate-division CSV files
- each CSV contains one header line followed by `date,pdsi` rows

The loader converts the raw division files into a state-level weekly PDSI series by averaging all division files within a state.

---

## 3. How The Active Pipeline Preprocesses Them

The main orchestration happens in `src/dataset/climate/preprocessing.py`.

For each crop in `corn`, `soybean`, and `wheat`, the pipeline:
1. builds state-level NOAA features
2. builds state-level SPI features
3. builds state-level PDSI features
4. merges those state-level features by `date` and `state`
5. aggregates state features to one national crop-weighted time series
6. builds crop-specific CO2 features
7. merges climate features into `data/ag/v6.csv`
8. writes the final crop file to `data/ag/<crop>.csv`

### 3.1 Crop season masking

Many features are only retained during crop-relevant weeks using `src/dataset/climate/crop_seasonal.py`.

Crop calendars:
- corn planting: weeks `10-22`
- corn harvesting: weeks `36-48`
- soybean planting: weeks `14-26`
- soybean harvesting: weeks `36-48`
- wheat planting: weeks `14-22` and `36-48`
- wheat harvesting: weeks `20-28` and `32-38`

If a week is outside the crop's planting or harvesting window, the corresponding seasonal feature is set to `0.0`.

### 3.2 Production-weighted national aggregation

State-level climate features are aggregated to a national crop-level series using `src/dataset/util/production_by_state.py`.

Weight construction:
1. Load `data/production_by_state/<crop>.csv`.
2. Find year columns matching `CropProductionByState_YYYY`.
3. Convert the table to long format.
4. Drop `United States`, missing values, and non-positive production.
5. Compute each state's average production across available years.
6. Normalize averages so state weights sum to `1`.

Aggregation rule:
- for each date and each climate feature, compute the weighted average across states using `state_weight`
- if total weight is zero, fall back to the unweighted mean for that date

---

## 4. Dataset-Specific Preprocessing

### 4.1 NOAA feature preprocessing

NOAA processing is state-by-state.

Steps:
1. Read `data/climate/noaa_weekly.csv` with parsed dates.
2. Derive `week_of_year`.
3. Process only `TMAX`, `TMIN`, and `AWND`.
4. For each state, evaluate each variable month-by-month.
5. Optionally detrend the monthly series when the Augmented Dickey-Fuller test indicates non-stationarity.
6. Standardize the monthly series with z-scores.
7. Convert z-scores into quantile-based extreme bands.
8. Keep values only during planting or harvesting weeks for the selected crop.

Trend handling:
- `run_adf_test` runs `adfuller(..., autolag="AIC")`
- if the p-value is at least `0.05`, the code treats the monthly slice as trending and applies linear detrending with `numpy.polyfit`
- quantile selection is done on the detrended series
- stored values are mapped back to the original scale when a trend was removed

#### Mathematical formulation

For a given climate variable $X$ (e.g., TMAX, TMIN, AWND) observed at state $s$ and time $t$:

**Step 1: Monthly grouping**

Group observations by calendar month $m \in \{1, 2, \ldots, 12\}$:

$$
X_{s,m} = \{X_{s,t} : \text{month}(t) = m\}
$$

**Step 2: Stationarity test**

Apply the Augmented Dickey-Fuller test to $X_{s,m}$. The null hypothesis is that the series has a unit root (non-stationary).

If $p \geq 0.05$, reject stationarity and proceed to detrending.

**Step 3: Linear detrending (conditional)**

If the series is non-stationary, fit a linear trend:

$$
X_{s,m}(i) = \alpha + \beta \cdot i + \epsilon_i
$$

where $i$ is the observation index within the monthly group, and $(\alpha, \beta)$ are estimated via ordinary least squares.

The detrended series is:

$$
\tilde{X}_{s,m}(i) = X_{s,m}(i) - (\alpha + \beta \cdot i)
$$

**Step 4: Standardization**

Compute the z-score for each observation within the monthly group using the detrended series (if detrending was applied) or the original series (if stationary):

$$
Z_{s,m}(i) = \frac{\tilde{X}_{s,m}(i) - \mu_{s,m}}{\sigma_{s,m}}
$$

where:
- $\mu_{s,m}$ is the mean
- $\sigma_{s,m}$ is the standard deviation
- $n$ is the number of observations in month $m$ for state $s$

**Step 5: Quantile-based extreme detection**

Compute quantile thresholds $q_p$ from the empirical distribution of $Z_{s,m}$:

$$
q_p = \inf \{z : P(Z_{s,m} \leq z) \geq p\}
$$

Define extreme bands based on variable-specific quantile rules (see below).

**Step 6: Value selection with trend restoration**

For each extreme band $B$, first identify which observations fall within the band based on their z-scores:

$$
\text{InBand}_{s,m,B}(i) = 
\begin{cases}
1 & \text{if } Z_{s,m}(i) \in B \\
0 & \text{otherwise}
\end{cases}
$$

Then, if detrending was applied, restore the trend before storing the value:

$$
V_{s,m,B}(i) = 
\begin{cases}
\tilde{X}_{s,m}(i) + (\alpha + \beta \cdot i) & \text{if } \text{InBand}_{s,m,B}(i) = 1 \text{ and detrending was applied} \\
X_{s,m}(i) & \text{if } \text{InBand}_{s,m,B}(i) = 1 \text{ and no detrending} \\
0 & \text{otherwise}
\end{cases}
$$

In other words:
- Quantile detection is performed on the detrended z-scores
- If an observation falls within an extreme band, the stored value is the original scale value with the trend restored
- This ensures that the final features reflect the actual magnitude of the climate variable, not the detrended residuals

**Step 7: Seasonal masking**

For crop $c$ with planting weeks $P_c$ and harvesting weeks $H_c$, the final feature is:

$$
F_{s,t,B}^{\text{planting}} = 
\begin{cases}
V_{s,m,B}(i) & \text{if } \text{week}(t) \in P_c \\
0 & \text{otherwise}
\end{cases}
$$

$$
F_{s,t,B}^{\text{harvesting}} = 
\begin{cases}
V_{s,m,B}(i) & \text{if } \text{week}(t) \in H_c \\
0 & \text{otherwise}
\end{cases}
$$

Quantile rules by variable:
- `TMAX`
  - `hot`: $q_{0.75} \leq Z_{s,m} < q_{0.90}$
  - `very_hot`: $Z_{s,m} \geq q_{0.90}$
- `TMIN`
  - `cold`: $Z_{s,m} < q_{0.10}$
  - `very_cold`: $q_{0.10} \leq Z_{s,m} < q_{0.25}$
- `AWND`
  - `moderate_high_wind`: $q_{0.80} \leq Z_{s,m} < q_{0.90}$
  - `extreme_high_wind`: $Z_{s,m} \geq q_{0.90}$

Feature naming pattern:
- `tmax_hot_in_planting`
- `tmax_very_hot_in_harvesting`
- `tmin_cold_in_planting`
- `awnd_extreme_high_wind_in_harvesting`

### 4.2 SPI feature preprocessing

SPI processing is also state-by-state, but it does not use monthly quantiles or detrending.

Steps:
1. Read `data/climate/spi_weekly_multiscale.csv` with parsed `week_date`.
2. Copy `week_date` into `date`.
3. Derive `week_of_year`.
4. Process only `SPI_7d`, `SPI_1m`, and `SPI_3m`.
5. Apply fixed threshold bands directly to the SPI value.
6. Keep values only during planting or harvesting weeks for the selected crop.

#### Mathematical formulation

The Standardized Precipitation Index (SPI) is computed upstream in `scripts/dataset/climate/noaa_spi.py` using the following process:

**Step 1: Rolling precipitation accumulation**

For a given time scale $k$ (e.g., 7 days, 30 days, 90 days), compute the rolling sum:

$$
P_k(t) = \sum_{i=0}^{k-1} P(t - i)
$$

where $P(t)$ is the daily precipitation at time $t$.

**Step 2: Gamma distribution fitting**

Fit a Gamma distribution to the non-zero values of $P_k$:

$$
f(x; \alpha, \beta) = \frac{1}{\beta^\alpha \Gamma(\alpha)} x^{\alpha - 1} e^{-x/\beta}, \quad x > 0
$$

where:
- $\alpha$ is the shape parameter
- $\beta$ is the scale parameter
- $\Gamma(\alpha) = \int_0^\infty t^{\alpha-1} e^{-t} dt$ is the gamma function

Parameters are estimated via maximum likelihood estimation (MLE).

**Step 3: Cumulative probability**

Compute the cumulative distribution function (CDF):

$$
F(x) = \int_0^x f(t; \alpha, \beta) dt
$$

**Step 4: Transformation to standard normal**

Convert the CDF to a standard normal variable:

$$
\text{SPI}_k(t) = \Phi^{-1}(F(P_k(t)))
$$

where $\Phi^{-1}$ is the inverse of the standard normal CDF.

**Step 5: Threshold-based extreme detection**

In the preprocessing pipeline, fixed thresholds are applied to the SPI values to identify extreme wet and dry conditions.

Threshold rules:
- very wet: $1.5 \leq \text{SPI}_k \leq 2.0$
- extreme wet: $\text{SPI}_k > 2.0$
- very dry: $-2.0 \leq \text{SPI}_k \leq -1.5$
- extreme dry: $\text{SPI}_k < -2.0$

**Step 6: Value selection and seasonal masking**

For each threshold band $B$ and crop $c$:

$$
F_{s,t,k,B}^{\text{planting}} = 
\begin{cases}
\text{SPI}_k(t) & \text{if } \text{SPI}_k(t) \in B \text{ and } \text{week}(t) \in P_c \\
0 & \text{otherwise}
\end{cases}
$$

$$
F_{s,t,k,B}^{\text{harvesting}} = 
\begin{cases}
\text{SPI}_k(t) & \text{if } \text{SPI}_k(t) \in B \text{ and } \text{week}(t) \in H_c \\
0 & \text{otherwise}
\end{cases}
$$

Feature naming pattern:
- `spi_7d_very_wet_in_planting`
- `spi_1m_extreme_dry_in_harvesting`
- `spi_3m_extreme_wet_in_planting`

### 4.3 PDSI feature preprocessing

PDSI processing starts from the Palmer directory and builds state-level weekly values first.

Steps:
1. Read every division CSV under each state directory in `data/climate/palmer/`.
2. Skip the first header line in each file.
3. Parse `YYYYMMDD` into `date` and parse the index value as numeric `pdsi`.
4. Convert each date to the Monday of the same week for consistent joins.
5. Average all division values for the same `state` and `date`.
6. Derive `week_of_year`.
7. Apply fixed drought and wetness thresholds.
8. Keep values only during planting or harvesting weeks for the selected crop.

#### Mathematical formulation

The Palmer Drought Severity Index (PDSI) is read from pre-computed files. The preprocessing pipeline applies the following transformations:

**Step 1: State-level aggregation**

For state $s$ with $n_s$ climate divisions, compute the state-level PDSI:

$$
\text{PDSI}_s(t) = \frac{1}{n_s} \sum_{d=1}^{n_s} \text{PDSI}_{s,d}(t)
$$

where $\text{PDSI}_{s,d}(t)$ is the PDSI value for division $d$ in state $s$ at time $t$.

**Step 2: Threshold-based extreme detection**

Fixed thresholds are applied to identify drought and wetness extremes:

Threshold rules:
- very wet: $3.0 \leq \text{PDSI}_s \leq 4.0$
- extreme wet: $\text{PDSI}_s > 4.0$
- extreme drought: $-4.0 \leq \text{PDSI}_s \leq -3.0$
- severe drought: $\text{PDSI}_s < -4.0$

**Step 3: Value selection and seasonal masking**

For each threshold band $B$ and crop $c$:

$$
F_{s,t,B}^{\text{planting}} = 
\begin{cases}
\text{PDSI}_s(t) & \text{if } \text{PDSI}_s(t) \in B \text{ and } \text{week}(t) \in P_c \\
0 & \text{otherwise}
\end{cases}
$$

$$
F_{s,t,B}^{\text{harvesting}} = 
\begin{cases}
\text{PDSI}_s(t) & \text{if } \text{PDSI}_s(t) \in B \text{ and } \text{week}(t) \in H_c \\
0 & \text{otherwise}
\end{cases}
$$

Feature naming pattern:
- `pdsi_very_wet_in_planting`
- `pdsi_extreme_wet_in_harvesting`
- `pdsi_extreme_drought_in_planting`
- `pdsi_severe_drought_in_harvesting`

### 4.4 CO2 feature preprocessing

CO2 processing is national rather than state-by-state.

Steps:
1. Read `data/climate/co2_weekly_mlo.csv` with parsed dates.
2. Select the first valid numeric value column.
3. Derive `week_of_year`.
4. Process the series month-by-month using the same monthly extreme detector used for NOAA.
5. Run ADF-based trend detection and linear detrending when needed.
6. Standardize the monthly series.
7. Mark only the top `5%` tail as extreme.
8. Keep values only during planting or harvesting weeks for the selected crop.

#### Mathematical formulation

CO2 preprocessing follows the same monthly quantile approach as NOAA variables, but with a single extreme band.

**Step 1: Monthly grouping**

Group CO2 observations by calendar month $m$:

$$
\text{CO2}_m = \{\text{CO2}(t) : \text{month}(t) = m\}
$$

**Step 2: Stationarity test and detrending**

Apply the Augmented Dickey-Fuller test. If $p \geq 0.05$, fit a linear trend:

$$
\text{CO2}_m(i) = \alpha + \beta \cdot i + \epsilon_i
$$

The detrended series is:

$$
\tilde{\text{CO2}}_m(i) = \text{CO2}_m(i) - (\alpha + \beta \cdot i)
$$

**Step 3: Standardization**

Compute the z-score:

$$
Z_m(i) = \frac{\tilde{\text{CO2}}_m(i) - \mu_m}{\sigma_m}
$$

**Step 4: Extreme detection**

Compute the 95th percentile threshold:

$$
q_{0.95} = \inf \{z : P(Z_m \leq z) \geq 0.95\}
$$

Identify extreme observations based on z-scores:

$$
\text{IsExtreme}_m(i) = 
\begin{cases}
1 & \text{if } Z_m(i) \geq q_{0.95} \\
0 & \text{otherwise}
\end{cases}
$$

**Step 5: Value selection with trend restoration**

If detrending was applied, restore the trend before storing the value:

$$
V_m^{\text{extreme}}(i) = 
\begin{cases}
\tilde{\text{CO2}}_m(i) + (\alpha + \beta \cdot i) & \text{if } \text{IsExtreme}_m(i) = 1 \text{ and detrending was applied} \\
\text{CO2}_m(i) & \text{if } \text{IsExtreme}_m(i) = 1 \text{ and no detrending} \\
0 & \text{otherwise}
\end{cases}
$$

In other words:
- Extreme detection is performed on the detrended z-scores
- If an observation is extreme, the stored value is the original scale CO2 value with the trend restored
- This ensures that the final CO2 features reflect the actual atmospheric CO2 concentration, not the detrended residuals

**Step 6: Seasonal masking**

For crop $c$:

$$
F_{t}^{\text{planting}} = 
\begin{cases}
V_m^{\text{extreme}}(i) & \text{if } \text{week}(t) \in P_c \\
0 & \text{otherwise}
\end{cases}
$$

$$
F_{t}^{\text{harvesting}} = 
\begin{cases}
V_m^{\text{extreme}}(i) & \text{if } \text{week}(t) \in H_c \\
0 & \text{otherwise}
\end{cases}
$$

Threshold rule:
- `co2_extreme`: $Z_m \geq q_{0.95}$

Feature naming pattern:
- `co2_extreme_in_planting`
- `co2_extreme_in_harvesting`

---

## 5. Merge And Output Behavior

After state-level features are created, they are aggregated to a national crop-weighted series.

### 5.1 Production-weighted aggregation

For each crop $c$, state $s$, and feature $F$:

**Step 1: Load production weights**

Compute the average production for each state across available years:

$$
\bar{Y}_{c,s} = \frac{1}{T} \sum_{y=1}^{T} Y_{c,s,y}
$$

where $Y_{c,s,y}$ is the production of crop $c$ in state $s$ in year $y$.

**Step 2: Normalize weights**

Compute the state weight:

$$
w_{c,s} = \frac{\bar{Y}_{c,s}}{\sum_{s'} \bar{Y}_{c,s'}}
$$

such that $\sum_s w_{c,s} = 1$.

**Step 3: Weighted aggregation**

For each date $t$ and feature $F$, compute the national value:

$$
F_{c,t}^{\text{national}} = \sum_{s} w_{c,s} \cdot F_{s,t}
$$

If the total weight is zero (no production data), fall back to the unweighted mean:

$$
F_{c,t}^{\text{national}} = \frac{1}{n_s} \sum_{s} F_{s,t}
$$

### 5.2 Final merge

After NOAA, SPI, and PDSI features are created at the state level:
- they are outer-joined on `date` and `state`
- missing feature values are filled with `0.0`
- the merged table is aggregated to a national series using crop production weights

Then:
- national weighted state features are outer-joined with the national CO2 features on `date`
- the result is merged into `data/ag/v6.csv` on `date`
- newly added climate columns are filled with `0.0` where missing
- one output file is written per crop

Outputs:
- `data/ag/corn.csv`
- `data/ag/soybean.csv`
- `data/ag/wheat.csv`

---

## 6. Climate Oscillation Indices

The repository includes two major climate oscillation indices that capture large-scale atmospheric circulation patterns. While these indices are not currently consumed by the main preprocessing pipeline in `src/dataset/climate/preprocessing.py`, they are available for analysis and may be integrated into future feature sets.

### 6.1 North Atlantic Oscillation (NAO) Index

#### What is NAO?

The North Atlantic Os ffcillation (NAO) is a large-scale atmospheric circulation pattern that describes the variability in the pressure difference between the Icelandic Low and the Azores High in the North Atlantic region.

**Physical interpretation:**
- **Positive NAO phase**: Strong pressure gradient between the Icelandic Low and Azores High, leading to stronger westerly winds, warmer and wetter winters in Europe, and colder and drier conditions in the Mediterranean and Middle East.
- **Negative NAO phase**: Weak pressure gradient, leading to weaker westerly winds, colder winters in northern Europe, and wetter conditions in the Mediterranean.

#### How NAO data is gathered

Source file:
- `data/climate/nao.csv` (monthly values)
- `data/climate/NAO_index.csv` (weekly interpolated values)

Gathering process:
1. Monthly NAO index values are obtained from external sources (e.g., NOAA Climate Prediction Center, NCAR).
2. The raw monthly data is stored in `data/climate/nao.csv` with columns for `Year` and each month (`Jan`, `Feb`, ..., `Dec`).
3. The script `scripts/dataset/climate/nao.py` processes the monthly data:
   - Melts the wide-format table into long format with `Year`, `Month`, and `NAO_index`
   - Creates daily timestamps by assigning each monthly value to the first day of the month
   - Interpolates daily values using linear time-based interpolation
   - Resamples to weekly frequency using `W-MON` (week ending on Monday)
   - Computes the weekly mean of interpolated daily values
4. The weekly NAO index is saved to `data/climate/NAO_index.csv`.

Data format:
- `Date`: weekly timestamp in `YYYY-MM-DD` format
- `NAO_index`: normalized NAO value (dimensionless)

### 6.2 Southern Oscillation Index (SOI)

#### What is SOI?

The Southern Oscillation Index (SOI) is a standardized measure of the atmospheric pressure difference between Tahiti (French Polynesia) and Darwin (Australia). It is a key indicator of the El Niño-Southern Oscillation (ENSO) phenomenon.

**Physical interpretation:**
- **Negative SOI**: Lower pressure at Tahiti relative to Darwin, associated with El Niño conditions (warmer sea surface temperatures in the eastern Pacific, reduced rainfall in Australia and Indonesia, increased rainfall in the Americas).
- **Positive SOI**: Higher pressure at Tahiti relative to Darwin, associated with La Niña conditions (cooler sea surface temperatures in the eastern Pacific, increased rainfall in Australia and Indonesia, drier conditions in the Americas).
- **Neutral SOI**: Near-zero values indicate neutral ENSO conditions.

#### How SOI data is gathered

Source file:
- `data/climate/soi.csv` (monthly values)
- `data/climate/SOI_index.csv` (weekly interpolated values)

Gathering process:
1. Monthly SOI values are obtained from external sources (e.g., NOAA Climate Prediction Center, Australian Bureau of Meteorology).
2. The raw monthly data is stored in `data/climate/soi.csv` with columns for `YEAR` and each month (`JAN`, `FEB`, ..., `DEC`).
3. The script `scripts/dataset/climate/soi.py` processes the monthly data:
   - Melts the wide-format table into long format with `YEAR`, `MONTH`, and `SOI_index`
   - Creates daily timestamps by assigning each monthly value to the first day of the month
   - Interpolates daily values using linear time-based interpolation
   - Resamples to weekly frequency using `W-MON` (week ending on Monday)
   - Computes the weekly mean of interpolated daily values
4. The weekly SOI index is saved to `data/climate/SOI_index.csv`.

Data format:
- `Date`: weekly timestamp in `YYYY-MM-DD` format
- `SOI_index`: normalized SOI value (dimensionless, typically ranging from -30 to +30)

---

## 7. Code References

Main pipeline:
- `src/dataset/climate/preprocessing.py`

NOAA gathering:
- `src/dataset/climate/noaa.py`
- `scripts/dataset/climate/noaa.py`

SPI derivation:
- `scripts/dataset/climate/noaa_spi.py`

Palmer / PDSI loading:
- `src/dataset/climate/pdsi.py`

Crop seasonal masking:
- `src/dataset/climate/crop_seasonal.py`

Production weighting:
- `src/dataset/util/production_by_state.py`

Climate oscillation indices:
- `scripts/dataset/climate/nao.py`
- `scripts/dataset/climate/soi.py`
