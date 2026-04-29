 # Climate Data Documentation
 
 ## Overview
 
 This document provides a comprehensive description of climate data collection, processing, and feature engineering used in the Volf financial AI system. The climate data serves as a critical component for understanding environmental factors that may influence financial markets.
 
 ---
 
 ## 1. Data Collection from GHCN-D Dataset
 
 ### 1.1 Data Source
 
 Climate data is collected from the **Global Historical Climatology Network Daily (GHCN-D)** dataset, maintained by the National Oceanic and Atmospheric Administration (NOAA). GHCN-D is an integrated database of daily climate summaries from land surface stations across the globe.
 
 **Dataset Details:**
 - **Provider:** NOAA National Centers for Environmental Information (NCEI)
 - **Coverage:** Global, with over 100,000 stations
 - **Temporal Range:** 1763 to present (varies by station)
 - **Update Frequency:** Daily
 - **Access Method:** FTP servers, API endpoints, or bulk downloads
 
 ### 1.2 Primary Climate Variables
 
 The following core variables are extracted from GHCN-D:
 
 #### 1.2.1 PRCP - Precipitation
 - **Description:** Total daily precipitation (rain and/or melted snow)
 - **Units:** Tenths of millimeters (converted to millimeters)
 - **Measurement:** 24-hour accumulation
 - **Quality Flags:** Checked for measurement quality and source reliability
 
 #### 1.2.2 TMAX - Maximum Temperature
 - **Description:** Daily maximum temperature
 - **Units:** Tenths of degrees Celsius (converted to degrees Celsius)
 - **Measurement:** Highest temperature recorded during the 24-hour period
 - **Quality Control:** Outlier detection and consistency checks applied
 
 #### 1.2.3 TMIN - Minimum Temperature
 - **Description:** Daily minimum temperature
 - **Units:** Tenths of degrees Celsius (converted to degrees Celsius)
 - **Measurement:** Lowest temperature recorded during the 24-hour period
 - **Quality Control:** Outlier detection and consistency checks applied
 
 #### 1.2.4 AWND - Average Wind Speed
 - **Description:** Daily average wind speed
 - **Units:** Tenths of meters per second (converted to meters per second)
 - **Measurement:** Average of wind speed measurements throughout the day
 - **Note:** Not available at all stations; requires interpolation or station selection
 
 #### 1.2.5 SNOW - Snowfall
 - **Description:** Daily snowfall
 - **Units:** Millimeters
 - **Measurement:** 24-hour accumulation of snow
 - **Seasonal Relevance:** Particularly important for winter months
 
 #### 1.2.6 SNWD - Snow Depth
 - **Description:** Depth of snow on the ground
 - **Units:** Millimeters
 - **Measurement:** Depth at time of observation
 - **Application:** Useful for understanding accumulated snow conditions
 
 ### 1.3 Data Acquisition Process
 
 ```
 1. Station Selection
    ↓
 2. API Query / FTP Download
    ↓
 3. Data Parsing (CSV/Fixed-width format)
    ↓
 4. Unit Conversion
    ↓
 5. Quality Flag Filtering
    ↓
 6. Missing Data Handling
    ↓
 7. Temporal Alignment
    ↓
 8. Storage in Database
 ```
 
 **Station Selection Criteria:**
 - Geographic relevance to financial markets (e.g., major financial centers)
 - Data completeness (>90% coverage for target period)
 - Measurement quality (high-quality flags)
 - Temporal continuity (minimal gaps)
 
 **Data Quality Filters:**
 - Remove records with quality flags indicating errors
 - Filter extreme outliers using statistical methods (e.g., 5-sigma rule)
 - Validate temporal consistency (e.g., TMIN < TMAX)
 
 ---
 
 ## 2. Standardized Precipitation Index (SPI) Calculation
 
 ### 2.1 Overview
 
 The **Standardized Precipitation Index (SPI)** is a widely used drought indicator that quantifies precipitation deficits or surpluses over multiple time scales. SPI is calculated by fitting a probability distribution to precipitation data and transforming it to a standard normal distribution.
 
 ### 2.2 Mathematical Formulation
 
 #### 2.2.1 Data Aggregation
 
 For a given time scale $k$ (e.g., 1, 3, 6, 12, or 24 months), aggregate precipitation data:
 
 $$
 P_k(i) = \sum_{j=0}^{k-1} P(i-j)
 $$
 
 where:
 - $P_k(i)$ is the accumulated precipitation for time scale $k$ ending at time $i$
 - $P(i)$ is the precipitation at time $i$
 - $k$ is the time scale in months
 
 #### 2.2.2 Probability Distribution Fitting
 
 The **Gamma distribution** is typically used to model precipitation data, as it can handle the skewed nature of precipitation (including zero values):
 
 $$
 g(x) = \frac{1}{\beta^\alpha \Gamma(\alpha)} x^{\alpha-1} e^{-x/\beta}
 $$
 
 where:
 - $x > 0$ is the precipitation amount
 - $\alpha > 0$ is the shape parameter
 - $\beta > 0$ is the scale parameter
 - $\Gamma(\alpha)$ is the gamma function: $\Gamma(\alpha) = \int_0^\infty t^{\alpha-1} e^{-t} dt$
 
 #### 2.2.3 Parameter Estimation
 
 Parameters $\alpha$ and $\beta$ are estimated using **Maximum Likelihood Estimation (MLE)**:
 
 $$
 \hat{\alpha} = \frac{1}{4A}\left(1 + \sqrt{1 + \frac{4A}{3}}\right)
 $$
 
 $$
 \hat{\beta} = \frac{\bar{x}}{\hat{\alpha}}
 $$
 
 where:
 
 $$
 A = \ln(\bar{x}) - \frac{\sum \ln(x)}{n}
 $$
 
 - $\bar{x}$ is the mean precipitation
 - $n$ is the number of observations
 
 #### 2.2.4 Cumulative Probability
 
 The cumulative probability for a given precipitation value $x$ is:
 
 $$
 G(x) = \int_0^x g(t) dt = \frac{1}{\beta^\alpha \Gamma(\alpha)} \int_0^x t^{\alpha-1} e^{-t/\beta} dt
 $$
 
 This can be expressed using the incomplete gamma function:
 
 $$
 G(x) = \frac{1}{\Gamma(\alpha)} \int_0^{x/\beta} t^{\alpha-1} e^{-t} dt = \frac{\gamma(\alpha, x/\beta)}{\Gamma(\alpha)}
 $$
 
 where $\gamma(\alpha, x/\beta)$ is the incomplete gamma function.
 
 #### 2.2.5 Handling Zero Precipitation
 
 Since precipitation can be zero, the cumulative probability is adjusted:
 
 $$
 H(x) = q + (1-q) \cdot G(x)
 $$
 
 where:
 - $q$ is the probability of zero precipitation: $q = \frac{m}{n}$
 - $m$ is the number of zero precipitation observations
 - $n$ is the total number of observations
 
 #### 2.2.6 Transformation to Standard Normal Distribution
 
 The SPI is obtained by transforming the cumulative probability $H(x)$ to a standard normal distribution using the inverse normal cumulative distribution function:
 
 $$
 \text{SPI} = \Phi^{-1}(H(x))
 $$
 
 where $\Phi^{-1}$ is the inverse of the standard normal cumulative distribution function.
 
 For computational efficiency, the **Abramowitz and Stegun approximation** is commonly used:
 
 For $0 < H(x) \leq 0.5$:
 
 $$
 \text{SPI} = -\left(t - \frac{c_0 + c_1 t + c_2 t^2}{1 + d_1 t + d_2 t^2 + d_3 t^3}\right)
 $$
 
 For $0.5 < H(x) < 1$:
 
 $$
 \text{SPI} = +\left(t - \frac{c_0 + c_1 t + c_2 t^2}{1 + d_1 t + d_2 t^2 + d_3 t^3}\right)
 $$
 
 where:
 
 $$
 t = \sqrt{\ln\left(\frac{1}{(H(x))^2}\right)} \quad \text{for } H(x) \leq 0.5
 $$
 
 $$
 t = \sqrt{\ln\left(\frac{1}{(1-H(x))^2}\right)} \quad \text{for } H(x) > 0.5
 $$
 
 Constants:
 - $c_0 = 2.515517$
 - $c_1 = 0.802853$
 - $c_2 = 0.010328$
 - $d_1 = 1.432788$
 - $d_2 = 0.189269$
 - $d_3 = 0.001308$
 
 ### 2.3 SPI Interpretation
 
 | SPI Value | Category | Probability |
 |-----------|----------|-------------|
 | ≥ 2.0 | Extremely wet | ~2.3% |
 | 1.5 to 1.99 | Very wet | ~4.4% |
 | 1.0 to 1.49 | Moderately wet | ~9.2% |
 | -0.99 to 0.99 | Near normal | ~68.2% |
 | -1.0 to -1.49 | Moderately dry | ~9.2% |
 | -1.5 to -1.99 | Severely dry | ~4.4% |
 | ≤ -2.0 | Extremely dry | ~2.3% |
 
 ### 2.4 Multiple Time Scales
 
 SPI is calculated for multiple time scales to capture different drought/wetness phenomena:
 
 - **SPI-1:** 1-month scale (short-term soil moisture, agricultural impacts)
 - **SPI-3:** 3-month scale (seasonal precipitation patterns)
 - **SPI-6:** 6-month scale (medium-term trends, reservoir levels)
 - **SPI-12:** 12-month scale (long-term hydrological drought)
 - **SPI-24:** 24-month scale (multi-year drought cycles)
 
 ---
 
 ## 3. Additional Climate Features
 
 ### 3.1 Derived Temperature Features
 
 #### 3.1.1 Daily Temperature Range (DTR)
 
 $$
 \text{DTR} = T_{\max} - T_{\min}
 $$
 
 **Significance:** Indicates diurnal temperature variation, which affects energy demand and agricultural productivity.
 
 #### 3.1.2 Mean Temperature (TAVG)
 
 $$
 T_{\text{avg}} = \frac{T_{\max} + T_{\min}}{2}
 $$
 
 **Significance:** General temperature indicator for daily conditions.
 
 #### 3.1.3 Growing Degree Days (GDD)
 
 $$
 \text{GDD} = \max\left(0, T_{\text{avg}} - T_{\text{base}}\right)
 $$
 
 where $T_{\text{base}}$ is typically 10°C for most crops.
 
 **Significance:** Measures heat accumulation for crop development.
 
 #### 3.1.4 Heating Degree Days (HDD)
 
 $$
 \text{HDD} = \max\left(0, T_{\text{base}} - T_{\text{avg}}\right)
 $$
 
 where $T_{\text{base}}$ is typically 18°C (65°F).
 
 **Significance:** Estimates energy demand for heating.
 
 #### 3.1.5 Cooling Degree Days (CDD)
 
 $$
 \text{CDD} = \max\left(0, T_{\text{avg}} - T_{\text{base}}\right)
 $$
 
 where $T_{\text{base}}$ is typically 18°C (65°F).
 
 **Significance:** Estimates energy demand for cooling.
 
 ### 3.2 Temperature Anomalies
 
 #### 3.2.1 Daily Temperature Anomaly
 
 $$
 \text{Anomaly}_T(d) = T(d) - \bar{T}_{\text{climatology}}(d)
 $$
 
 where $\bar{T}_{\text{climatology}}(d)$ is the long-term average temperature for day-of-year $d$ (typically 30-year baseline).
 
 #### 3.2.2 Standardized Temperature Anomaly
 
 $$
 \text{Z-score}_T(d) = \frac{T(d) - \bar{T}_{\text{climatology}}(d)}{\sigma_{\text{climatology}}(d)}
 $$
 
 where $\sigma_{\text{climatology}}(d)$ is the standard deviation for day-of-year $d$.
 
 ### 3.3 Precipitation Features
 
 #### 3.3.1 Cumulative Precipitation
 
 $$
 P_{\text{cum}}(t, k) = \sum_{i=t-k+1}^{t} P(i)
 $$
 
 where $k$ is the accumulation period (e.g., 7, 30, 90 days).
 
 #### 3.3.2 Precipitation Intensity
 
 $$
 I_P = \frac{P_{\text{total}}}{N_{\text{wet days}}}
 $$
 
 where $N_{\text{wet days}}$ is the number of days with precipitation > threshold (typically 1mm).
 
 #### 3.3.3 Dry Spell Duration
 
 Consecutive days with precipitation < threshold (typically 1mm).
 
 #### 3.3.4 Wet Spell Duration
 
 Consecutive days with precipitation ≥ threshold.
 
 ### 3.4 Wind Features
 
 #### 3.4.1 Wind Power Density
 
 $$
 \text{WPD} = \frac{1}{2} \rho v^3
 $$
 
 where:
 - $\rho$ is air density (approximately 1.225 kg/m³ at sea level)
 - $v$ is wind speed (m/s)
 
 **Significance:** Indicates potential wind energy generation.
 
 #### 3.4.2 Wind Chill Index (for cold conditions)
 
 $$
 \text{WCI} = 13.12 + 0.6215 T - 11.37 v^{0.16} + 0.3965 T v^{0.16}
 $$
 
 where:
 - $T$ is air temperature (°C)
 - $v$ is wind speed (km/h)
 
 ### 3.5 Extreme Event Indicators
 
 #### 3.5.1 Extreme Heat Days
 
 Binary indicator: $\mathbb{1}(T_{\max} > P_{90})$
 
 where $P_{90}$ is the 90th percentile of historical maximum temperatures.
 
 #### 3.5.2 Extreme Cold Days
 
 Binary indicator: $\mathbb{1}(T_{\min} < P_{10})$
 
 where $P_{10}$ is the 10th percentile of historical minimum temperatures.
 
 #### 3.5.3 Heavy Precipitation Days
 
 Binary indicator: $\mathbb{1}(P > P_{95})$
 
 where $P_{95}$ is the 95th percentile of historical precipitation.
 
 #### 3.5.4 Frost Days
 
 Binary indicator: $\mathbb{1}(T_{\min} < 0°C)$
 
 ### 3.6 Composite Climate Indices
 
 #### 3.6.1 Palmer Drought Severity Index (PDSI)
 
 A complex water balance model that considers:
 - Precipitation
 - Temperature
 - Soil moisture capacity
 - Evapotranspiration
 
 $$
 \text{PDSI}_t = 0.897 \cdot \text{PDSI}_{t-1} + \frac{Z_t}{3}
 $$
 
 where $Z_t$ is the moisture anomaly index (calculated from water balance).
 
 #### 3.6.2 Evapotranspiration (ET)
 
 Estimated using the **Penman-Monteith equation** or simplified **Hargreaves equation**:
 
 **Hargreaves Equation:**
 
 $$
 \text{ET}_0 = 0.0023 \cdot R_a \cdot (T_{\text{avg}} + 17.8) \cdot \sqrt{T_{\max} - T_{\min}}
 $$
 
 where:
 - $\text{ET}_0$ is reference evapotranspiration (mm/day)
 - $R_a$ is extraterrestrial radiation (MJ/m²/day, calculated from latitude and day of year)
 - Temperatures in °C
 
 ### 3.7 Seasonal and Temporal Features
 
 #### 3.7.1 Moving Averages
 
 $$
 \text{MA}_k(t) = \frac{1}{k} \sum_{i=t-k+1}^{t} X(i)
 $$
 
 Applied to temperature, precipitation, and wind speed with windows $k \in \{7, 14, 30, 90\}$ days.
 
 #### 3.7.2 Exponential Moving Averages
 
 $$
 \text{EMA}_t = \alpha \cdot X_t + (1-\alpha) \cdot \text{EMA}_{t-1}
 $$
 
 where $\alpha = \frac{2}{k+1}$ is the smoothing factor.
 
 #### 3.7.3 Rate of Change
 
 $$
 \text{ROC}_k(t) = \frac{X(t) - X(t-k)}{k}
 $$
 
 Measures the rate of change over $k$ days.
 
 #### 3.7.4 Volatility
 
 $$
 \sigma_k(t) = \sqrt{\frac{1}{k-1} \sum_{i=t-k+1}^{t} (X(i) - \bar{X}_k(t))^2}
 $$
 
 Rolling standard deviation over $k$ days.
 
 ---
 
 ## 4. Data Processing Pipeline
 
 ### 4.1 Workflow
 
 ```
 Raw GHCN-D Data
        ↓
 [Quality Control & Filtering]
        ↓
 [Unit Conversion & Standardization]
        ↓
 [Missing Data Imputation]
        ↓
 [Feature Engineering]
        ├─→ [SPI Calculation]
        ├─→ [Temperature Derivatives]
        ├─→ [Precipitation Features]
        ├─→ [Wind Features]
        └─→ [Extreme Event Detection]
        ↓
 [Temporal Aggregation]
        ↓
 [Normalization & Scaling]
        ↓
 Feature Store / Database
 ```
 
 ### 4.2 Missing Data Handling
 
 **Strategies:**
 1. **Linear Interpolation:** For short gaps (< 3 days)
 2. **Climatological Mean:** Replace with long-term average for that day-of-year
 3. **Spatial Interpolation:** Use nearby stations (inverse distance weighting)
 4. **Forward/Backward Fill:** For non-critical features
 5. **Model-based Imputation:** Use machine learning models trained on complete data
 
 ### 4.3 Temporal Alignment
 
 All climate features are aligned to a common temporal grid:
 - **Frequency:** Daily
 - **Time Zone:** UTC
 - **Aggregation:** When multiple observations exist, use mean or sum as appropriate
 
 ### 4.4 Feature Scaling
 
 Different scaling methods applied based on feature characteristics:
 
 **Standardization (Z-score):**
 $$
 X_{\text{scaled}} = \frac{X - \mu}{\sigma}
 $$
 
 **Min-Max Normalization:**
 $$
 X_{\text{scaled}} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}
 $$
 
 **Robust Scaling:**
 $$
 X_{\text{scaled}} = \frac{X - \text{median}(X)}{\text{IQR}(X)}
 $$
 
 ---
 
 ## 5. Integration with Financial Models
 
 ### 5.1 Climate-Finance Linkages
 
 Climate features are integrated into financial models through several channels:
 
 1. **Direct Impact:** Energy sector stocks affected by temperature extremes
 2. **Agricultural Commodities:** Crop prices influenced by precipitation and temperature
 3. **Insurance Sector:** Extreme weather events affect claims and premiums
 4. **Macroeconomic Indicators:** Climate anomalies impact GDP growth
 5. **Risk Assessment:** Climate volatility as a risk factor
 
 ### 5.2 Feature Selection
 
 Climate features are selected based on:
 - **Correlation Analysis:** With target financial variables
 - **Mutual Information:** Non-linear dependencies
 - **Granger Causality:** Temporal predictive power
 - **Domain Knowledge:** Known climate-finance relationships
 
 ### 5.3 Lag Structure
 
 Climate impacts on financial markets often have time lags:
 - **Immediate (0-7 days):** Energy demand, weather derivatives
 - **Short-term (1-4 weeks):** Agricultural futures, retail sales
 - **Medium-term (1-3 months):** Crop yields, insurance claims
 - **Long-term (3-12 months):** Macroeconomic indicators, infrastructure investments
 
 ---
 
 ## 6. Data Quality and Validation
 
 ### 6.1 Quality Metrics
 
 - **Completeness:** Percentage of non-missing values
 - **Consistency:** Logical relationships (e.g., TMIN < TMAX)
 - **Accuracy:** Comparison with alternative data sources
 - **Timeliness:** Data availability lag
 
 ### 6.2 Validation Procedures
 
 1. **Cross-validation with satellite data** (e.g., MODIS, ERA5)
 2. **Comparison with regional climate models**
 3. **Statistical outlier detection** (Tukey's fences, Z-scores)
 4. **Temporal consistency checks** (sudden jumps, trends)
 5. **Spatial consistency checks** (comparison with nearby stations)
 
 ### 6.3 Uncertainty Quantification
 
 - **Measurement Uncertainty:** Instrument precision
 - **Interpolation Uncertainty:** Spatial/temporal gaps
 - **Model Uncertainty:** SPI and derived features
 - **Propagation:** How uncertainty affects downstream models
 
 ---
 
 ## 7. References and Resources
 
 ### 7.1 Data Sources
 
 - **NOAA GHCN-D:** https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily
 - **GHCN-D Documentation:** https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt
 
 ### 7.2 Key Publications
 
 - McKee, T. B., Doesken, N. J., & Kleist, J. (1993). "The relationship of drought frequency and duration to time scales." *Proceedings of the 8th Conference on Applied Climatology*.
 - Edwards, D. C., & McKee, T. B. (1997). "Characteristics of 20th century drought in the United States at multiple time scales." *Climatology Report Number 97-2*, Colorado State University.
 - Vicente-Serrano, S. M., et al. (2010). "A multiscalar drought index sensitive to global warming: the standardized precipitation evapotranspiration index." *Journal of Climate*, 23(7), 1696-1718.
 
 ### 7.3 Software Libraries
 
 - **climate-indices (Python):** SPI, SPEI, PDSI calculations
 - **xclim (Python):** Climate indices and indicators
 - **ClimateIndices.jl (Julia):** High-performance climate index computation
 
 ---
 
 ## 8. Appendix: Example Calculations
 
 ### 8.1 SPI-3 Calculation Example
 
 **Given:** Monthly precipitation data (mm) for a location:
 
 | Month | Precipitation |
 |-------|---------------|
 | Jan | 45.2 |
 | Feb | 38.7 |
 | Mar | 52.3 |
 | Apr | 61.8 |
 | May | 73.5 |
 | Jun | 82.1 |
 
 **Step 1:** Calculate 3-month accumulated precipitation for June:
 $$
 P_3(\text{Jun}) = 73.5 + 82.1 + 61.8 = 217.4 \text{ mm}
 $$
 
 **Step 2:** Fit gamma distribution to historical 3-month accumulations for June (using 30+ years of data).
 
 **Step 3:** Calculate cumulative probability $H(217.4)$ using fitted parameters.
 
 **Step 4:** Transform to standard normal: $\text{SPI-3} = \Phi^{-1}(H(217.4))$
 
 ### 8.2 Growing Degree Days Example
 
 **Given:** Daily temperatures for a week in growing season:
 
 | Day | TMAX (°C) | TMIN (°C) | TAVG (°C) | GDD |
 |-----|-----------|-----------|-----------|-----|
 | 1 | 28.5 | 15.2 | 21.85 | 11.85 |
 | 2 | 30.1 | 16.8 | 23.45 | 13.45 |
 | 3 | 27.3 | 14.5 | 20.90 | 10.90 |
 | 4 | 25.8 | 13.1 | 19.45 | 9.45 |
 | 5 | 29.2 | 15.9 | 22.55 | 12.55 |
 | 6 | 31.5 | 17.6 | 24.55 | 14.55 |
 | 7 | 28.9 | 16.2 | 22.55 | 12.55 |
 
 **Cumulative GDD for week:** $11.85 + 13.45 + 10.90 + 9.45 + 12.55 + 14.55 + 12.55 = 85.30$
 
 ---
 
 **Document Version:** 1.0  
 **Last Updated:** 2025  
 **Maintained by:** Volf Financial AI Team
