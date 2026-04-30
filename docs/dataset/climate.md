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

The current preprocessing pipeline reads `data/climate/co2_weekly_mlo.csv` directly.

Within this repository, the preprocessing code assumes the file already exists and does not fetch or regenerate it. The only gathering behavior visible from the pipeline is the file contract:
- it must contain a `date` column
- it must contain one usable numeric CO2 value column
- the series is treated as weekly and national

Downstream preprocessing uses the CO2 series as a univariate weekly climate signal.

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

Quantile rules by variable:
- `TMAX`
  - `hot`: `75th` to `90th` percentile of monthly z-score
  - `very_hot`: above `90th` percentile
- `TMIN`
  - `cold`: below `10th` percentile
  - `very_cold`: `10th` to `25th` percentile
- `AWND`
  - `moderate_high_wind`: `80th` to `90th` percentile
  - `extreme_high_wind`: above `90th` percentile

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

Threshold rules:
- very wet: `1.5 <= SPI <= 2.0`
- extreme wet: `SPI > 2.0`
- very dry: `-2.0 <= SPI <= -1.5`
- extreme dry: `SPI < -2.0`

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

Threshold rules:
- very wet: `3.0 <= PDSI <= 4.0`
- extreme wet: `PDSI > 4.0`
- extreme drought: `-4.0 <= PDSI <= -3.0`
- severe drought: `PDSI < -4.0`

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

Threshold rule:
- `co2_extreme`: z-score above the `95th` percentile within the same calendar month

Feature naming pattern:
- `co2_extreme_in_planting`
- `co2_extreme_in_harvesting`

---

## 5. Merge And Output Behavior

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

## 6. What This Document Intentionally Excludes

This document does not describe climate datasets that are present in the repository but are not consumed by `src/dataset/climate/preprocessing.py`, including:
- PRISM download helpers in `src/dataset/climate/prism.py`
- NAO, SOI, temperature anomaly, and Google Trends climate files
- legacy or alternative preprocessing flows that are not part of the active merged pipeline

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
