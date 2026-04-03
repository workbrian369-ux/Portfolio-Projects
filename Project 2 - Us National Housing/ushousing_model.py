""""
Streamlit Dashboard: Us national housing market analysis & prediction
Business Goal: Predict Median Sales price of houses sold using economic & house supply/demand indicators (monthly us data 1990-2025)
The Dashboard provides:
    1. Ridge Regression model evaluation and feature importance.
    2. Time series forecasting (ARIMA/Prophet) for future months.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

# Time series libraries
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# Optional: auto Arima(requires pmdarima)
try:
    from pmdarima import auto_arima # type: ignore
    AUTO_ARIMA_AVAILABLE = True
except ImportError:
    AUTO_ARIMA_AVAILABLE = False

# 1. Data loading (cached for perfomance)
@st.cache_data
def load_data():
    """Loads and Preprocess the housing dataset."""
    df = pd.read_csv("unified_monthly_data_interpolated_1990_20250101.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    # Sort by date and set as index for time series models
    df = df.sort_values('Date').reset_index(drop=True)
    return df

@st.cache_data
def get_time_series_data(df):
    """Extract target series with date as index for forecasting."""
    ts = df[['Date', 'MedianSalesPriceofHousesSold']].copy()
    ts.set_index('Date', inplace=True)
    return ts

# 2. Ridge regression model
def train_ridge_model(df, target='MedianSalesPriceofHousesSold'):
    """Train ridge regression on all features and return model + metric."""
    # Drop date if present columns
    X = df.drop(columns=[target, 'Date'], errors='ignore')

    # keep only numeric columns
    X = X.select_dtypes(include=[np.number])
    y = df[target]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Ridge with default alpha = 1.0\
    bat_model = Ridge(alpha=1.0)
    bat_model.fit(X_train, y_train)

    # Predictions
    y_pred = bat_model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Feature importance
    coef = pd.Series(bat_model.coef_, index=X.columns).sort_values(ascending=False)

    return bat_model, mae, r2, coef, y_test, y_pred

# Time series foreecasting Functions
def forecast_arima(series, steps, seasonal=True):
    """Forecast using ARIMA (AUTO_ARIMA if available, else manual order.)"""
    if AUTO_ARIMA_AVAILABLE:
        # Use auto_arima to select best(p,d,q) and seasonal components.
        wayne_model = auto_arima( # type: ignore
            series,
            start_p=0, max_p=5,
            start_d=0, max_d=2,
            start_q=0, max_q=5,
            seasonal=seasonal, m=12,
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )
        forecast = wayne_model.prediction(n_periods=steps)
        return forecast, wayne_model
    else:
        st.warning("pmdarima not installed. using ARIMA(1, 1, 1) results may be suboptimal.")
        wayne_model = ARIMA(series, order=(1, 1, 1))
        fitted = wayne_model.fit()
        forecast = fitted.forecast(steps=steps)
        return forecast, fitted
    
def forecast_prophet(series_df, steps):
    """Forecast using Prophet (automatic seasonality)"""
    # Prophet expects columns 'ds' and 'y'
    df_prophet =series_df.reset_index().rename(columns={'Date': 'ds', 'MedianSalesPriceofHousesSold': 'y'})
    bruce_model = Prophet(daily_seasonality=False, yearly_seasonality=True) # pyright: ignore[reportArgumentType]
    bruce_model.fit(df_prophet)

    # Model future dataframe
    future = bruce_model.make_future_dataframe(periods=steps, freq='MS')
    forecast = bruce_model.predict(future)

    return forecast, bruce_model 

# 4. Streamlit UI
st.set_page_config(page_title="BATMAN Housing price predictor", layout="wide")
st.title("Us National Housing Market Dashboard by Batman!")
st.markdown("Predict **Median Sales Price of Houses Sold** using economic indicators or time series models.")

# Navigation sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Ridge Regression Model", "Time Series Forecast"])

df = load_data()
st.sidebar.write(f"Data Shape: {df.shape}")
st.sidebar.write(f"Data range: {df['Date'].min().date()} to {df['Date'].max().date()}")

# PAGE 1: Ridge Regression Model.
if page == "Ridge Regression Model":
    st.header("Ridge Regression (All features)")
    st.markdown("This Model Uses all Available economic and housing indicators to predict the median sales price.")

    with st.spinner("Training Ridge Model..."):
        bat_model, mae, r2, coef, y_test, y_pred = train_ridge_model(df)

    col1, col2 = st.columns(2)
    col1.metric("Mean Absolute Error", f"${mae:,.2f}")
    col2.metric("R2 score", f"{r2:.4f}")

    # Actual vs Predicted Plot
    st.subheader("Actual vs Predicted (Test Set)")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel("Actual Price ($)")
    ax.set_ylabel("Predicted Price ($)")
    ax.set_title("Ridge Regression Perfomance")
    st.pyplot(fig)

    # Feature importance (Top 10)
    st.subheader("Feature importance (Coefficients)")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    coef.head(10).plot(kind='barh', ax=ax2, color='teal')
    ax2.set_xlabel("Coefficient Value")
    ax2.set_title("Top 10 Most influential Features")
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)

    # Show full coefficient table.
    with st.expander("See all coefficients"):
        st.dataframe(coef)

# PAGE 2: TIME SERIES FORECAST
else:
    st.header("Time Series Forecast For Future Months.")
    st.markdown("Predict Future median Sales price using either **ARIMA** or **Prophet**.")

    # Prepare time series data
    ts = get_time_series_data(df)

    # Sidebar controls for forecasting.
    st.sidebar.subheader("Forecast Settings")
    model_type = st.sidebar.selectbox("Model", ["Prophet", "ARIMA"])
    forecast_horizon = st.sidebar.slider("Forecast Horizon (months)", 1, 24, 12)
    
    # run forecast when button is clicked.
    if st.sidebar.button("Run Forecast"):

        # Initialize variables
        forecast_vals = None
        # Plot historical Data
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(ts.index, ts['MedianSalesPriceofHousesSold'], label='Historical', color='blue')

        if model_type == "Prophet":
            st.info("Fitting Prophet Model (This may take a few seconds)...")
            forecast, _= forecast_prophet(ts, forecast_horizon)
            # Extract forecast values and dates
            future = forecast.tail(forecast_horizon)
            forecast_dates = future['ds']
            forecast_vals = future['yhat']
            ax.plot(forecast_dates, forecast_vals, 'ro-', label='Prophet Forecast')
            forecast_table = future.rename(columns={'ds': "Date", 'yhat': 'Predicted Price'})
        else: # ARIMA
            st.info("Batman Fitting ARIMA Model....")
            series = ts['MedianSalesPriceofHousesSold']
            forecast_vals, _ = forecast_arima(series, forecast_horizon)
            # Generate future dates (monthly frequency)
            last_date = ts.index[-1]
            forecast_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1),
                                         periods=forecast_horizon, freq='MS')
            ax.plot(forecast_dates, forecast_vals, 'go-', label='ARIMA Forecast')
            # Build forecast table
            forecast_table = pd.DataFrame({
                'Date': forecast_dates,
                'Predicted Price': forecast_vals
            })
        
        # Customise Plot
        ax.set_title(f"{model_type} Forecast - Next {forecast_horizon} Months")
        ax.set_xlabel("Date")
        ax.set_ylabel("Median Sales Price ($)")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

        # Display forecast table
        st.subheader("Forecast Values")
        st.dataframe(forecast_table)

        # Option to download forecast as csv.
        csv = forecast_table.to_csv(index=False)
        st.download_button(
            label="Download Forecast as CSV",
            data=csv,
            file_name=f"{model_type}_Darknight_forecast_{forecast_horizon}months.csv",
            mime="text/csv"
        )
    else:
        st.info("Select forecast settings in the sidebar and click 'Run Forecast'.")

# Footer
st.sidebar.markdown("---gotham finest---")
st.sidebar.write("**Data source:** Us housing & Economic indicators (1990-2025)")
