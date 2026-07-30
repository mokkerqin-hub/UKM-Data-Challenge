import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

# --- SUPPRESS WARNINGS ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="No supported index is available")

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Water Consumption Forecast", layout="wide")

st.title("💧 Malaysia Water Consumption Forecasting Dashboard")


# --- DATA LOADING ---
@st.cache_data
def load_data():
    df = pd.read_csv("water_consumption.csv")
    df['date'] = pd.to_datetime(df['date'], format='%Y', exact=False)
    df['year'] = df['date'].dt.year
    return df

df = load_data()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filter Settings")

selected_state = st.sidebar.selectbox("Select State", df['state'].unique())
selected_sector = st.sidebar.selectbox("Select Sector", df['sector'].unique())

test_size = 5
forecast_years = st.sidebar.slider("Forecast Horizon (Years)", 1, 10, 5)

# --- FILTER DATA ---
full_series = df[
    (df['state'] == selected_state) &
    (df['sector'] == selected_sector)
].sort_values('year')

train_df = full_series.iloc[:-test_size]
test_df = full_series.iloc[-test_size:]

# --- HELPER FUNCTIONS ---
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def get_ml_forecast(model_type, train_data, test_len, forecast_len):

    temp_df = train_data.copy()

    temp_df['lag_1'] = temp_df['value'].shift(1)
    temp_df['lag_2'] = temp_df['value'].shift(2)
    temp_df['rolling_mean_3'] = temp_df['value'].rolling(window=3).mean()

    temp_df = temp_df.dropna()

    features = ['year', 'lag_1', 'lag_2', 'rolling_mean_3']

    X = temp_df[features]
    y = temp_df['value']

    if model_type == 'XGB':
        model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
    else:
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )

    model.fit(X, y)

    results = []

    last_row = temp_df.iloc[-1].copy()

    for _ in range(test_len + forecast_len):

        feat = pd.DataFrame([[

            last_row['year'] + 1,
            last_row['value'],
            last_row['lag_1'],
            np.mean([last_row['value'], last_row['lag_1'], last_row['lag_2']])

        ]], columns=features)

        pred = model.predict(feat)[0]

        results.append(pred)

        last_row['year'] += 1
        last_row['lag_2'] = last_row['lag_1']
        last_row['lag_1'] = last_row['value']
        last_row['value'] = pred

    return results[:test_len], results[test_len:]


# --- MODEL CALCULATIONS ---

# 1️⃣ ARIMA
with st.spinner('Optimizing ARIMA...'):
    arima_model = pm.auto_arima(train_df['value'], seasonal=False, stepwise=True)
    arima_test = arima_model.predict(n_periods=test_size)
    arima_fore = arima_model.predict(n_periods=test_size + forecast_years)[test_size:]

# Convert train series
train_series = train_df['value'].values

# 2️⃣ Simple Exponential Smoothing (SES)
ses_model = SimpleExpSmoothing(train_series).fit()
ses_test = ses_model.forecast(test_size)
ses_fore = ses_model.forecast(test_size + forecast_years)[test_size:]

# 3️⃣ Double Exponential Smoothing (DES)
des_model = ExponentialSmoothing(train_series, trend='add').fit()
des_test = des_model.forecast(test_size)
des_fore = des_model.forecast(test_size + forecast_years)[test_size:]

# 4️⃣ Machine Learning Models
xgb_test, xgb_fore = get_ml_forecast('XGB', train_df, test_size, forecast_years)
rf_test, rf_fore = get_ml_forecast('RF', train_df, test_size, forecast_years)

# --- PLOTTING ---
col1, col2 = st.columns([2, 1])

with col1:

    st.subheader("Visual Comparison")

    fig, ax = plt.subplots(figsize=(8, 4))

    future_years = np.arange(
        test_df['year'].max() + 1,
        test_df['year'].max() + 1 + forecast_years
    )

    ax.plot(train_df['year'], train_df['value'], color='black', label='Actual (Train)')
    ax.plot(test_df['year'], test_df['value'], color='black', linestyle=':', label='Actual (Test)')

    plot_data = [

        (test_df['year'], arima_test, future_years, arima_fore, 'blue', 'ARIMA'),
        (test_df['year'], ses_test, future_years, ses_fore, 'purple', 'SES'),
        (test_df['year'], des_test, future_years, des_fore, 'red', 'DES'),
        (test_df['year'], xgb_test, future_years, xgb_fore, 'green', 'XGBoost'),
        (test_df['year'], rf_test, future_years, rf_fore, 'orange', 'Random Forest')

    ]

    for t_yr, t_pred, f_yr, f_pred, color, label in plot_data:
        ax.plot(t_yr, t_pred, color=color, linestyle='--', alpha=0.5)
        ax.plot(f_yr, f_pred, color=color, label=f'{label} Forecast')

    ax.legend(fontsize='small', loc='upper left')

    st.pyplot(fig)

# --- FORECAST TABLE ---
with col2:

    st.subheader("Predicted Values")

    pred_df = pd.DataFrame({

        "Year": future_years,
        "ARIMA": arima_fore,
        "SES": ses_fore,
        "DES": des_fore,
        "XGBoost": xgb_fore,
        "Random Forest": rf_fore

    }).set_index("Year")

    st.dataframe(pred_df.style.format("{:.2f}"))

# --- PERFORMANCE RANKING ---

st.subheader("Model Performance Evaluation")

def get_metrics(actual, predicted):

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = calculate_mape(actual, predicted)

    return mae, rmse, mape


actual_vals = test_df['value'].values

metrics_dict = {

    "ARIMA": get_metrics(actual_vals, arima_test),
    "SES": get_metrics(actual_vals, ses_test),
    "DES": get_metrics(actual_vals, des_test),
    "XGBoost": get_metrics(actual_vals, xgb_test),
    "Random Forest": get_metrics(actual_vals, rf_test)

}

perf_df = pd.DataFrame(
    metrics_dict,
    index=["MAE", "RMSE", "MAPE (%)"]
).T

# --- RANKING LOGIC ---

perf_df['Rank_MAE'] = perf_df['MAE'].rank(ascending=True)
perf_df['Rank_RMSE'] = perf_df['RMSE'].rank(ascending=True)
perf_df['Rank_MAPE'] = perf_df['MAPE (%)'].rank(ascending=True)

perf_df['Average_Rank'] = (
    perf_df['Rank_MAE'] +
    perf_df['Rank_RMSE'] +
    perf_df['Rank_MAPE']
) / 3

perf_df['Ranking'] = perf_df['Average_Rank'].rank(ascending=True).astype(int)

perf_df = perf_df.sort_values("Ranking")

st.table(
    perf_df[["MAE", "RMSE", "MAPE (%)", "Ranking"]].style.format({
        "MAE": "{:.2f}",
        "RMSE": "{:.2f}",
        "MAPE (%)": "{:.2f}%"
    })
)