# ⚡ Hourly Energy Consumption Forecasting

Time series forecasting of hourly electricity load using classical statistical modeling (SARIMA) and gradient-boosted trees (XGBoost), benchmarked against each other on 14 years of real utility data.

## 📌 Overview

This project forecasts daily energy demand (MW) for American Electric Power (AEP) using the [Hourly Energy Consumption dataset](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) from Kaggle. It compares a traditional time series approach (SARIMA) against a machine learning approach (XGBoost with engineered lag/rolling features) to see which handles real-world seasonality and demand spikes better.

**Result: XGBoost outperformed SARIMA by ~59% on MAE**, cutting forecast error from a 10.2% MAPE down to 4.08%.

## 📊 Dataset

- **Source:** AEP hourly load data (PJM Interconnection region)
- **Size:** 121,273 hourly readings
- **Date range:** October 2004 – August 2018 (~14 years)
- **Target:** Energy consumption in megawatts (MW)
- **Missing values:** None

## 🔍 Approach

1. **EDA** — visualized full 14-year series, and average consumption by hour of day, day of week, and month to surface daily/weekly/seasonal demand cycles
2. **Stationarity check** — Augmented Dickey-Fuller test confirmed the series is stationary
3. **Decomposition** — resampled to daily frequency and decomposed into trend, weekly seasonality, and residual components
4. **Feature engineering** — built lag features (1, 7, 14 days), rolling means (7-day, 30-day), plus calendar features (day of week, month, day of year)
5. **Modeling** — trained and compared two forecasters on a chronological train/test split:
   - **SARIMA** `(1,0,1)(1,1,1,7)` on the raw daily series
   - **XGBoost** on the engineered feature set
6. **Evaluation** — scored both on held-out test data using MAE, RMSE, and MAPE

## 🧠 Models & Results

| Model   | MAE     | RMSE    | MAPE   |
|---------|---------|---------|--------|
| SARIMA  | 1465.84 | 1773.60 | 10.20% |
| XGBoost | 597.62  | 775.40  | 4.08%  |

- **Train period:** 2004-10-31 → 2015-11-02 (4,020 days)
- **Test period:** 2015-11-03 → 2018-08-03 (1,005 days)

XGBoost's ability to leverage lagged demand and rolling averages let it capture short-term momentum and weekly patterns that a single seasonal-ARIMA specification struggled to keep up with over a long, evolving series.

## 🛠️ Tech Stack

- **Python** — pandas, NumPy
- **Statistical modeling** — statsmodels (SARIMAX, seasonal_decompose, ADF test)
- **Machine learning** — XGBoost, scikit-learn (metrics, train/test split)
- **Visualization** — Matplotlib

## 📁 Repository Structure

```
├── AEP_hourly.csv                     # Raw hourly energy consumption data
├── energy_forecasting.ipynb           # Full analysis: EDA → modeling → evaluation
├── energy_forecasting_presentation.html  # Presentation summary of findings
└── README.md
```

## 🚀 Running It

```bash
pip install pandas numpy matplotlib statsmodels scikit-learn xgboost
jupyter notebook energy_forecasting.ipynb
```

## 🔮 Possible Extensions

- Add exogenous features (temperature, holidays) to both models
- Try LSTM/Prophet for comparison
- Forecast at hourly rather than daily resolution
- Hyperparameter tuning via grid/Bayesian search for XGBoost

## 👤 Author

**Akshit Gajera** — MSc Data Science
