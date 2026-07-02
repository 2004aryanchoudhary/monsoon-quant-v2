# ⛈️ Monsoon Quant V2: Agrochemical Alpha Engine

An end-to-end multi-asset quantitative trading pipeline leveraging alternative climate data to forecast Indian agrochemical equity performance, featuring live paper-trading execution.

## 📌 Project Overview
This system shifts away from standard technical analysis by utilizing meteorological anomalies (cumulative localized rainfall and temperature averages) as primary alpha signals. The architecture executes chronological out-of-sample inferences using an XGBoost classifier, protected by a strict mathematical capital preservation filter and an automated order management system.

## 🚀 Core Architecture
* **Alternative Data Pipeline:** Automated ingestion and normalization of domestic weather basket data alongside historical equity panels.
* **Algorithmic Transparency (SHAP):** Live inference engine utilizing SHAP (SHapley Additive exPlanations) values to mathematically prove feature attribution and eliminate "black box" model risk.
* **Risk Management & Position Sizing:** Implemented a calibrated 60% probability threshold filter. The portfolio dynamically migrates to 100% cash during low-conviction market regimes, minimizing drawdowns.
* **Autonomous Execution System:** A Python daemon that triggers daily at 3:15 PM IST, fetching live market depth and executing risk-adjusted paper trades (max 5% capital risk per trade) into a persistent SQLite ledger.
* **Production Dashboard:** Live, interactive Streamlit web application featuring a cross-sectional backtest visualizer, a What-If scenario engine, and an autonomous execution ledger.

## 📊 Out-of-Sample Performance (Late 2024 - Mid 2026)
* **Benchmark (NIFTY 50):** -1.53%
* **Passive Sector Basket:** -6.72%
* **Monsoon Alpha Strategy:** **+9.28%**

## 🛠️ Tech Stack
* **Quantitative Research:** Pandas, NumPy, yfinance
* **Machine Learning:** XGBoost, Scikit-Learn, SHAP
* **Execution & Infrastructure:** SQLite, Python Schedule, Streamlit, Altair