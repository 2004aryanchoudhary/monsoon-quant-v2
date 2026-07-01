# Monsoon Quant V2: Alternative Data & Cross-Sectional Alpha

**Author:** Aryan Choudhary  
**Domain:** Quantitative Finance / Alternative Data / Agricultural Equities  

## Overview
Monsoon Quant V2 is a systematic, cross-sectional trading model that predicts the 21-day forward relative outperformance of Indian agricultural equities (KSCL, PIIND, UPL, DHANUKA) against the NIFTY 50 index. 

Instead of relying purely on noisy daily price action, the model derives alpha from a **National Crop Stress Index**—a geographical basket of historical precipitation and temperature data aggregated across critical Indian agricultural hubs (Rajasthan, Maharashtra, Andhra Pradesh). 

## Core Architecture
* **Data Engineering:** Automated ingestion of multi-asset financial panel data (yfinance) and historical meteorological data (Open-Meteo Archive API).
* **Feature Engineering:** 21-day rolling cumulative environmental shocks (Precipitation Surprise, Heat Averages) joined with localized 21-day volatility and momentum structures.
* **Predictive Engine:** An XGBoost Classifier heavily regularized to prevent overfitting on daily market noise, predicting a binary 1-month forward outperformance target.
* **Execution Strategy:** A vectorized cross-sectional backtester that dynamically allocates capital every 21 days to the single asset with the highest probabilistic edge.

## Out-of-Sample Performance Metrics (415-Day Test Window)
* **Strategy Win Rate (Accuracy):** 55.31%
* **Strategy Cumulative Return:** +28.58%
* **Benchmark (NIFTY 50) Return:** -5.51%
* **Equal-Weighted Sector Return:** -2.32%
* **Signal Drivers:** Model feature importance rankings mathematically proved that 21-day cumulative national rainfall is the #2 highest driver of predictive edge, superseding the broader market index return.

## Tech Stack
* **Environment:** `uv` (Lightning-fast Python package manager)
* **Pipeline:** Pandas, NumPy, Requests
* **Machine Learning:** XGBoost, Scikit-Learn

## How to Run the Pipeline
Ensure `uv` is installed, then execute the data dominoes in order:
1. `uv run weather_loader.py` (Ingests the National Agricultural Basket)
2. `uv run feature_engineer.py` (Constructs the Multi-Asset Panel Matrix)
3. `uv run train_classifier.py` (Trains the XGBoost engine)
4. `uv run backtest_strategy.py` (Executes the cross-sectional portfolio evaluation)
