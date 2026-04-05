"""
Streamlit Dashboard: US National Housing Market Analysis & Prediction
Business Goal: Predict Median Sales price of houses sold using economic & house supply/demand indicators (monthly US data 1990-2025)
The Dashboard provides:
    1. Ridge Regression model evaluation and feature importance.
    2. Time series forecasting (ARIMA/Prophet) for future months.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
import pickle # NEW: Imported pickle to allow saving the trained model

# Time series libraries
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# Optional: Auto-ARIMA (requires pmdarima)
try:
    from pmdarima import auto_arima
    AUTO_ARIMA_AVAILABLE = True
except ImportError:
    AUTO_ARIMA_AVAILABLE = False


# 1. DATA LOADING & PREPARATION

@st.cache_data
def load_data():
    """
    Loads and preprocesses the housing dataset.
    Using @st.cache_data ensures Streamlit only loads the data once, 
    making the app run much faster on subsequent reloads.
    """
    df = pd.read_csv("unified_monthly_data_interpolated_1990_20250101.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date and reset index so time series models read it sequentially
    df = df.sort_values('Date').reset_index(drop=True)
    return df

@st.cache_data
def get_time_series_data(df):
    """
    Extracts just the Date and Target variable for time series forecasting.
    Sets the Date as the index, which is standard practice for ARIMA models.
    """
    ts = df[['Date', 'MedianSalesPriceofHousesSold']].copy()
    ts.set_index('Date', inplace=True)
    return ts


# 2. MACHINE LEARNING: RIDGE REGRESSION


def train_ridge_model(df, target='MedianSalesPriceofHousesSold'):
    """
    Trains a Ridge Regression model to predict the housing price based on all other features.
    It handles isolating numeric columns, splitting data for testing, and evaluating performance.
    """
    # Isolate predictive features (X) and target variable (y)
    X = df.drop(columns=[target, 'Date'], errors='ignore')
    X = X.select_dtypes(include=[np.number]) # Keep only numerical data
    y = df[target]
    
    # Split data into 80% training data and 20% testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize and train the Ridge model
    bat_model = Ridge(alpha=1.0)
    bat_model.fit(X_train, y_train)
    
    # Make predictions on our test set
    y_pred = bat_model.predict(X_test)
    
    # Calculate accuracy metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Extract how much weight the model gave to each feature
    coef = pd.Series(bat_model.coef_, index=X.columns).sort_values(ascending=False)

    return bat_model, mae, r2, coef, y_test, y_pred


# 3. TIME SERIES FORECASTING


def forecast_arima(series, steps, seasonal=True):
    """
    Forecasts future prices using ARIMA. If auto_arima is installed, it uses it to 
    automatically find the best parameters. Otherwise, it defaults to standard ARIMA.
    """
    if AUTO_ARIMA_AVAILABLE:
        # Let auto_arima figure out the best statistical parameters for the data
        wayne_model = auto_arima( 
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
        forecast = wayne_model.predict(n_periods=steps)
        return forecast, wayne_model
    else:
        st.warning("pmdarima not installed. Using standard ARIMA(1, 1, 1). Results may be suboptimal.")
        wayne_model = ARIMA(series, order=(1, 1, 1))
        fitted = wayne_model.fit()
        forecast = fitted.forecast(steps=steps)
        return forecast, fitted
    
def forecast_prophet(series_df, steps):
    """
    Forecasts using Facebook's Prophet model. Prophet expects data in a very specific
    format with columns named 'ds' (datestamp) and 'y' (target value).
    """
    # Rename columns to meet Prophet's strict formatting requirements
    df_prophet = series_df.reset_index().rename(columns={'Date': 'ds', 'MedianSalesPriceofHousesSold': 'y'})

    # Double check data types
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')
    
    # Remove any missing data points
    df_prophet = df_prophet.dropna(subset=['y', 'ds'])

    if len(df_prophet) < 2:
        raise ValueError("Not enough valid data points for Prophet forecasting")
    
    # Initialize and train the Prophet model
    bruce_model = Prophet(daily_seasonality=False, yearly_seasonality=True) 
    bruce_model.fit(df_prophet)

    # Generate dates for the future
    future = bruce_model.make_future_dataframe(periods=steps, freq='MS') # 'MS' means Month Start
    forecast = bruce_model.predict(future)

    return forecast, bruce_model 


# 4. STREAMLIT USER INTERFACE


# Configure the look and feel of the web page
st.set_page_config(page_title="BATMAN Housing Price Predictor", layout="wide")
st.title("US National Housing Market Dashboard by Batman!")
st.markdown("Predict **Median Sales Price of Houses Sold** using economic indicators or time series models.")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Ridge Regression Model", "Time Series Forecast"])

# Load data to show stats in the sidebar
try:
    df = load_data()
    st.sidebar.write(f"Data Shape: {df.shape}")
    st.sidebar.write(f"Data Range: {df['Date'].min().date()} to {df['Date'].max().date()}")
except Exception as e:
    st.error(f"Error loading data. Please ensure 'unified_monthly_data_interpolated_1990_20250101.csv' is in the same directory. Details: {e}")
    st.stop()


# --- PAGE 1: RIDGE REGRESSION MODEL ---
if page == "Ridge Regression Model":
    st.header("Ridge Regression (All Features)")
    st.markdown("This model uses all available economic and housing indicators to predict the median sales price.")

    # Show a loading spinner while the model trains
    with st.spinner("Training Ridge Model..."):
        bat_model, mae, r2, coef, y_test, y_pred = train_ridge_model(df)

    # Display key metrics
    col1, col2 = st.columns(2)
    col1.metric("Mean Absolute Error", f"${mae:,.2f}")
    col2.metric("R2 Score", f"{r2:.4f}")

    # Plot 1: Actual vs Predicted Price
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=y_test, y=y_pred,
        mode='markers',
        marker=dict(color='royalblue', size=6, opacity=0.7, line=dict(width=1, color='darkblue')),
        name='Predictions'
    ))
    
    # Add a reference line for perfect predictions
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    fig1.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name='Perfect Prediction'
    ))
    
    fig1.update_layout(
        title="Actual vs Predicted Price (Test Set)",
        xaxis_title="Actual Price ($)",
        yaxis_title="Predicted Price ($)",
        width=800, height=600,
        hovermode='closest'
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Plot 2: Feature Importance Bar Chart
    top_coef = coef.head(10).sort_values() # Sort ascending to plot the highest values at the top
    fig2 = go.Figure(go.Bar(
        x=top_coef.values,
        y=top_coef.index,
        orientation='h',
        marker_color='teal',
        text=top_coef.round(2).values,
        textposition='outside'
    ))
    fig2.update_layout(
        title="Top 10 Most Influential Features (Ridge Coefficients)",
        xaxis_title="Coefficient Value (Impact on predicted price)",
        yaxis_title="Feature",
        height=500,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Allow the user to see every single coefficient if they want
    with st.expander("See all coefficients"):
        st.dataframe(coef)

    # NEW: --- Download Model Section ---
    st.markdown("---") # Add a visual divider
    st.subheader("Export Trained Model")
    st.markdown("You can download this trained Ridge Regression model to use in other applications without retraining.")
    
    # Convert the Python model object into a downloadable byte stream using pickle
    model_bytes = pickle.dumps(bat_model)
    
    # Create the download button
    st.download_button(
        label="Download Trained Ridge Model (.pkl)",
        data=model_bytes,
        file_name="batman_ridge_model.pkl",
        mime="application/octet-stream"
    )


# --- PAGE 2: TIME SERIES FORECAST ---
else:
    st.header("Time Series Forecast For Future Months")
    st.markdown("Predict future median sales price using either **ARIMA** or **Prophet**.")

    # Prepare data specifically for the time series models
    ts = get_time_series_data(df)

    # UI controls
    st.sidebar.subheader("Forecast Settings")
    model_type = st.sidebar.selectbox("Model", ["Prophet", "ARIMA"])
    forecast_horizon = st.sidebar.slider("Forecast Horizon (months)", 1, 24, 12)
    
    if st.sidebar.button("Run Forecast"):
        # Setup base variables for our charts and tables
        forecast_vals = None
        forecast_table = None
        forecast_dates = None

        # Grab historical data points for the visualization
        hist_dates = ts.index
        hist_prices = ts['MedianSalesPriceofHousesSold']

        # Path 1: Prophet
        if model_type == "Prophet":
            st.info("Fitting Prophet Model (This may take a few seconds)...")
            forecast, _ = forecast_prophet(ts, forecast_horizon)
            
            # Extract just the newly predicted dates and values
            future = forecast.tail(forecast_horizon)
            forecast_dates = future['ds']
            forecast_vals = future['yhat']
            forecast_table = future[['ds', 'yhat']].rename(columns={'ds': "Date", 'yhat': 'Predicted Price'})

        # Path 2: ARIMA
        else:
            st.info("Batman Fitting ARIMA Model...")
            series = ts['MedianSalesPriceofHousesSold']
            forecast_vals, _ = forecast_arima(series, forecast_horizon)
            
            # Generate future monthly dates
            last_date = ts.index[-1]
            forecast_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1),
                                         periods=forecast_horizon, freq='MS')
            
            # Create a clean DataFrame for the results table
            forecast_table = pd.DataFrame({
                'Date': forecast_dates,
                'Predicted Price': forecast_vals
            })
        
        # Plotly Line Chart for historical + forecasted data
        fig = go.Figure()
        
        # Plot History
        fig.add_trace(go.Scatter(
            x=hist_dates, y=hist_prices,
            mode='lines',
            name='Historical',
            line=dict(color='blue', width=2)
        ))
        
        # Plot Forecast
        if model_type == "Prophet":
            fig.add_trace(go.Scatter(
                x=forecast_dates, y=forecast_vals,
                mode='lines+markers',
                name='Prophet Forecast',
                line=dict(color='red', width=2, dash='dot'),
                marker=dict(size=6, symbol='circle')
            ))
        else:
            fig.add_trace(go.Scatter(
                x=forecast_dates, y=forecast_vals,
                mode='lines+markers',
                name='ARIMA Forecast',
                line=dict(color='green', width=2, dash='dot'),
                marker=dict(size=6, symbol='circle')
            ))
            
        fig.update_layout(
            title=f"{model_type} Forecast - Next {forecast_horizon} Months",
            xaxis_title="Date",
            yaxis_title="Median Sales Price ($)",
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display the tabular results
        st.subheader("Forecast Values")
        st.dataframe(forecast_table)

        # Allow the user to download their forecast
        csv = forecast_table.to_csv(index=False)
        st.download_button(
            label="Download Forecast as CSV",
            data=csv,
            file_name=f"{model_type}_Darknight_forecast_{forecast_horizon}months.csv",
            mime="text/csv"
        )
    else:
        st.info("Select forecast settings in the sidebar and click 'Run Forecast'.")

# --- Footer ---
st.sidebar.markdown("--- Gotham's Finest ---")
st.sidebar.write("**Data source:** US Housing & Economic Indicators (1990-2025)")
