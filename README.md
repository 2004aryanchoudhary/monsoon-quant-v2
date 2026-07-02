# ⛈️ Monsoon Quant V2: Agrochemical Alpha Engine

An end-to-end multi-asset quantitative trading pipeline leveraging alternative climate data to forecast Indian agrochemical equity performance.

## 📌 Project Overview
This system shifts away from standard technical analysis by utilizing meteorological anomalies (cumulative localized rainfall and temperature averages) as primary alpha signals. The architecture executes chronological out-of-sample inferences using an XGBoost classifier, protected by a strict mathematical capital preservation filter.

## 🚀 Core Features
* **Alternative Data Pipeline:** Automated ingestion and normalization of domestic weather basket data alongside historical equity panels.
* **Chronological Data Partitioning:** Strict time-series splitting to guarantee zero lookahead bias and prevent model data leakage.
* **Capital Preservation Protocol:** Implemented a calibrated 60% probability threshold filter. The portfolio dynamically migrates to 100% cash during low-conviction market regimes, minimizing drawdowns.
* **Production Dashboard:** Live, interactive Streamlit web application backed by a persistent SQLite database for real-time strategy visualization and performance tracking.

## 📊 Out-of-Sample Performance (Late 2024 - Mid 2026)
* **Benchmark (NIFTY 50):** -1.53%
* **Passive Sector Basket:** -6.72%
* **Monsoon Alpha Strategy:** **+9.28%**

## 🛠️ Tech Stack
* **Data Engineering:** Pandas, NumPy, yfinance
* **Machine Learning:** XGBoost, Scikit-Learn
* **Infrastructure:** SQLite, Streamlit