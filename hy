import pandas as pd
import numpy as np
import pyodbc
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from dateutil.relativedelta import relativedelta

# Connect to SQL Server


# Load June 2025 actual data
query = "SELECT MetricID, ReportingLevel, ReportingView, ReportingDate, MetricValue FROM metrics_june25_data_for_comp"
df_actual = pd.read_sql(query, conn)
df_actual["ReportingDate"] = pd.to_datetime(df_actual["ReportingDate"])
actual_dict = df_actual.set_index(["MetricID", "ReportingLevel", "ReportingView"])["MetricValue"].to_dict()

# Load full time series data
query_hist = "SELECT MetricID, ReportingLevel, ReportingView, ReportingDate, MetricValue FROM your_historical_table"
df_hist = pd.read_sql(query_hist, conn)
df_hist["ReportingDate"] = pd.to_datetime(df_hist["ReportingDate"])

# Define models
model_list = ["Theta", "ARIMA", "Holt", "Prophet", "SMA", "RandomForest", "LinearRegression", "SVR", "LightGBM", "XGBoost"]
model_mae = {model: [] for model in model_list}

# Forecast for each metric group
for (metric_id, level, view), group in df_hist.groupby(["MetricID", "ReportingLevel", "ReportingView"]):
    group = group.sort_values("ReportingDate").set_index("ReportingDate")
    
    if group.index[-1] != pd.Timestamp("2025-05-01"):
        continue

    train = group["MetricValue"]
    df = group.copy()
    steps = 1

    for model_choice in model_list:
        try:
            if model_choice == "Theta":
                model = ThetaModel(train, period=1)
                forecast = model.fit().forecast(steps=steps)

            elif model_choice == "ARIMA":
                model = ARIMA(train, order=(2, 2, 2))
                forecast = model.fit().forecast(steps=steps)

            elif model_choice == "Holt":
                model = ExponentialSmoothing(train, trend="add")
                forecast = model.fit().forecast(steps=steps)

            elif model_choice == "Prophet":
                df_prophet = train.reset_index().rename(columns={"ReportingDate": "ds", "MetricValue": "y"})
                prophet = Prophet()
                prophet.fit(df_prophet)
                future = prophet.make_future_dataframe(periods=steps, freq="MS")
                forecast = prophet.predict(future)["yhat"].iloc[-steps:].values

            elif model_choice == "SMA":
                sma_value = train.rolling(window=3).mean().iloc[-1]
                forecast = [sma_value] * steps

            elif model_choice in ["RandomForest", "LinearRegression", "SVR", "LightGBM", "XGBoost"]:
                df_reset = df.reset_index()
                df_reset["month"] = df_reset["ReportingDate"].dt.month
                df_reset["year"] = df_reset["ReportingDate"].dt.year
                df_reset["t"] = np.arange(len(df_reset))
                X = df_reset[["month", "year", "t"]].values
                y = df_reset["MetricValue"].values

                last_t = X[-1, 2]
                future_t = np.arange(last_t + 1, last_t + 1 + steps)
                future_months = [(df.index[-1] + relativedelta(months=i + 1)).month for i in range(steps)]
                future_years = [(df.index[-1] + relativedelta(months=i + 1)).year for i in range(steps)]
                X_future = np.column_stack([future_months, future_years, future_t])

                model_map = {
                    "RandomForest": RandomForestRegressor(),
                    "LinearRegression": LinearRegression(),
                    "SVR": SVR(),
                    "LightGBM": LGBMRegressor(),
                    "XGBoost": XGBRegressor()
                }
                model = model_map[model_choice]
                model.fit(X, y)
                forecast = model.predict(X_future)

            actual = actual_dict.get((metric_id, level, view), None)
            if actual is not None:
                mae = mean_absolute_error([actual], forecast)
                model_mae[model_choice].append(mae)

        except Exception as e:
            print(f"Error for {model_choice} on {metric_id}, {level}, {view}: {e}")

# Final MAE report
final_mae = {model: np.mean(errors) if errors else None for model, errors in model_mae.items()}
print("Model MAE Comparison for June 2025:\n")
for model, mae in final_mae.items():
    print(f"{model}: {mae:.2f}" if mae is not None else f"{model}: No predictions")

PROJECT NAME: Forecasting and Model Evaluation System

DESCRIPTION:
------------
This project predicts metric values using 10 different time series and machine learning models 
and compares them using MAE (Mean Absolute Error). 
It uses SQL Server as a data source and generates SHAP value visualizations using Streamlit.

REQUIREMENTS:
-------------
- Python 3.8 or above
- SQL Server (ODBC connection)
- Internet connection (for installing packages)

SETUP INSTRUCTIONS:
-------------------

1. Clone or Download the Project
--------------------------------
- Clone the repo using Git:
    git clone <your-repo-url>
  OR
- Download and extract the zip file.

2. Create Virtual Environment (Optional but Recommended)
--------------------------------------------------------
- Windows:
    python -m venv venv
    venv\Scripts\activate

- Linux/Mac:
    python3 -m venv venv
    source venv/bin/activate

3. Install Required Python Packages
-----------------------------------
- Run the following command in the project root folder:
    pip install -r requirements.txt

4. Set Up Database Connection
-----------------------------
- Make sure you have SQL Server installed and running.
- This app uses Windows Authentication. Update your connection if needed.

Connection string used in the code:
conn = pyodbc.connect(
'DRIVER={ODBC Driver 17 for SQL Server};'
'SERVER=GBRMSD500003053\FUS_SQLVIRT_Dev;'
'DATABASE=FUSION_MI_DEV;'
'Trusted_Connection=yes;'
)

markdown
Copy
Edit

5. Run the Streamlit App
------------------------
- From the terminal, navigate to the project folder and run:
    streamlit run app.py

6. Expected Output
------------------
- The app allows you to select a metric.
- It predicts the value for 1st June 2025 using the past 5 months.
- It shows the predictions from 10 models and the MAE for each.
- SHAP waterfall plots visualize feature contributions.

TROUBLESHOOTING:
----------------
- If you face a connection error, verify your SQL Server and ODBC configuration.
- If SHAP plots don’t display, ensure `matplotlib` and `shap` are installed.
