# Energy Demand Forecasting with XGBoost

A time series forecasting project using XGBoost to predict hourly energy demand for the PJM Interconnection grid, covering 15 years of data from 2002 to 2018.

---

## Project Overview

This project forecasts electricity demand (measured in megawatts) using gradient boosting regression. Rather than treating the problem as a pure time series task, the approach converts the datetime index into structured calendar features, allowing XGBoost to learn temporal patterns without requiring sequential modelling techniques.

The pipeline covers data loading, exploratory visualisation, feature engineering, model training with early stopping, feature importance analysis, and a post-prediction error audit to identify where the model underperforms and why.

---

## Dataset

**File:** `PJME_hourly.csv`

| Column | Description |
|--------|-------------|
| Datetime | Hourly timestamp (2002-12-31 to 2018-01-02) |
| PJME_MW | Energy consumption in megawatts |

**Scale:** 145,366 rows of hourly observations
**Source:** PJM Interconnection — a regional transmission organisation covering the US Mid-Atlantic and Midwest

---

## Notebook Structure

**File:** `Time_series.ipynb`

### Step 1 — Data Loading and Inspection
Load the CSV, set the Datetime column as the index, and convert it to a proper datetime type. Initial plots reveal clear seasonal structure across years, months, and hours of the day.

### Step 2 — Train/Test Split
Split the data at 1 January 2015:
- Training set: 2002 to 2014 (~105,000 rows)
- Test set: 2015 to 2018 (~40,000 rows)

A temporal split is used deliberately — shuffling would leak future information into training.

### Step 3 — Feature Engineering
A `create_features()` function extracts six calendar features from the datetime index:

```python
def create_features(df):
    df = df.copy()
    df['hour']      = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter']   = df.index.quarter
    df['month']     = df.index.month
    df['year']      = df.index.year
    df['dayofyear'] = df.index.dayofyear
    return df
```

### Step 4 — Visualising Feature/Target Relationships
Boxplots of energy demand by hour and by month confirm the patterns the model needs to capture:
- Demand peaks mid-afternoon and drops overnight
- Summer and winter months show consistently higher demand than spring and autumn

### Step 5 — Model Training
XGBoost regressor trained with 1,000 estimators, early stopping (50 rounds), and a learning rate of 0.01. Both train and test sets are passed as evaluation sets to monitor for overfitting during training.

```python
reg = xgb.XGBRegressor(
    n_estimators=1000,
    early_stopping_rounds=50,
    learning_rate=0.01
)
reg.fit(x_train, y_train,
        eval_set=[(x_train, y_train), (x_test, y_test)],
        verbose=100)
```

### Step 6 — Feature Importance
Horizontal bar chart of feature importances. `hour` and `year` rank as the most predictive features, reflecting both the strong intra-day demand cycle and the long-term upward trend in grid demand.

### Step 7 — Forecast on Test Set
Predictions are merged back onto the full dataset and plotted against actuals. A zoomed week-level view (April 2018) shows the model tracking the hourly demand cycle closely.

### Step 8 — Error Analysis
Mean absolute error is computed per day and sorted to surface the best and worst predictions:

**Worst prediction days:**
| Date | Mean Absolute Error (MW) |
|------|--------------------------|
| 2016-08-13 | 16,512 |
| 2016-08-14 | 16,438 |
| 2016-08-12 | 12,419 |

The three worst days all fall in mid-August 2016 — a period of extreme heat across the PJM region. These are demand spikes driven by external conditions (temperature) that the calendar features alone cannot capture. Adding weather data as an exogenous feature would likely close this gap.

**Best prediction days (MAE below 420 MW):** mid-autumn and spring dates with mild, stable demand patterns.

---

## Results

| Metric | Value |
|--------|-------|
| RMSE on test set | **3,958.84 MW** |
| Test period | Jan 2015 to Jan 2018 |
| Mean demand (test) | ~33,000 MW (approx.) |

---

## Key Takeaways

- Calendar features alone are sufficient to capture the bulk of demand variability — hour, year, and month drive the majority of predictive power.
- The model's largest errors are concentrated in extreme weather events, not random noise. This is a meaningful finding: it isolates exactly what the model is missing (temperature/weather data) rather than suggesting a general modelling failure.
- Error analysis by date is more informative than headline RMSE for understanding where a time series model breaks down.

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
xgboost
scikit-learn
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn xgboost scikit-learn
```

---

## File Structure

```
├── Time_series.ipynb       # Main Jupyter Notebook
├── PJME_hourly.csv         # Raw hourly energy data
└── README.md               # This file
```

---

*Author: Dwiti Bhavsar — Senior Data Analyst & Data Scientist*
