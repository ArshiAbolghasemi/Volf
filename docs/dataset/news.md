# News Features

## Overview

This dataset incorporates **macroeconomic uncertainty and sentiment indices** to enhance forecasting of **weekly realized volatility (RV) of wheat prices**.

It combines:

* **News-based sentiment features** (capturing economic tone and narratives from media)
* **Policy uncertainty measures** (reflecting ambiguity in government economic decisions)

The goal is to model how **macroeconomic sentiment and policy uncertainty influence commodity market volatility**, complementing market-based and climate-based features.

---

## FRSBF News Sentiment Index

The **Daily News Sentiment Index** is a high-frequency measure of economic sentiment derived from textual analysis of news articles.

### Description

* Built from economics-related articles across major U.S. newspapers
* Uses lexical analysis (general + custom news-specific dictionaries)
* Aggregated into a daily time series and smoothed using a **geometrically weighted average**
* Typically aggregated to **weekly frequency** for modeling

### Interpretation

* Higher values → more **positive economic sentiment**
* Lower values → more **negative sentiment**

### Why it matters

* Captures **market expectations and narratives**
* Provides early signals of:

  * Economic shifts
  * Financial stress
  * Policy reactions

### Download

* [https://www.frbsf.org/research-and-insights/data-and-indicators/daily-news-sentiment-index/](https://www.frbsf.org/research-and-insights/data-and-indicators/daily-news-sentiment-index/)

---

## Economic Policy Uncertainty (EPU) Index

The **Economic Policy Uncertainty Index** measures uncertainty related to economic policy using news coverage and economic indicators.

### Description

* Based on frequency of policy-related uncertainty terms in newspapers
* Incorporates additional signals such as:

  * Policy changes (e.g., tax provisions)
  * Forecast disagreement
* Available as a standardized time series via FRED

### Interpretation

* Higher values → greater **policy uncertainty**
* Lower values → more **stable policy environment**

### Why it matters

* Influences:

  * Investment and trading behavior
  * Risk premiums
  * Commodity demand expectations

### Download

* [https://fred.stlouisfed.org/series/USEPUINDXD](https://fred.stlouisfed.org/series/USEPUINDXD)

---

## Text Climate Anomaly (`Text_Climate_Anomaly`)

`Text_Climate_Anomaly` is a **text-attention proxy for climate narratives** used as an exogenous news feature in volatility modeling.

### How it is gathered in this project

In this codebase, the feature is built from **Google Trends daily interest** for the keyword:

- `climate change` (geo: `US`)

The pipeline:

1. Pulls daily search interest in monthly chunks
2. Concatenates all chunks into one daily series
3. Standardizes the series as a z-score
4. Resamples to weekly frequency (`W-MON`) using weekly mean
5. Outputs `date` and `Text_Climate_Anomaly`

Implementation source:

- `src/dataset/google_trend/climate_change.py`
- `src/dataset/google_trend/climate_change.py`

---
