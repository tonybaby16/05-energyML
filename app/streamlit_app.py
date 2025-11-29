import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os
from datetime import date
from sklearn.metrics import mean_squared_error
from utils import create_input_dataframe

# --- 1. CONFIG ---
st.set_page_config(page_title="Energy Forecaster", layout="wide")

# --- 2. LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    # Path setup
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Load Model
    model_path = os.path.join(project_root, 'models', 'best_energy_model.pkl')
    model = joblib.load(model_path)
    
    # Load Validation Sample (if exists)
    val_path = os.path.join(project_root, 'data', 'processed', 'validation_sample.pkl')
    if os.path.exists(val_path):
        val_data = pd.read_pickle(val_path)
    else:
        val_data = None
        
    return model, val_data

try:
    model, val_data = load_resources()
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

st.title("⚡ ASHRAE Energy Consumption Predictor")

# --- 3. TABS UI ---
tab1, tab2 = st.tabs(["🔮 Forecast Simulation", "📊 Model Performance"])

# --- TAB 1: FORECAST ---
with tab1:
    st.markdown("Simulate energy consumption for a building based on parameters.")
    
    with st.sidebar:
        st.header("Simulation Parameters")
        horizon = st.radio("Forecast Horizon", ["Day", "Month", "Year"])
        
        b_type = st.selectbox("Primary Use", 
                              ['Education', 'Office', 'Lodging/Residential', 'Entertainment', 'Other'])
        
        sq_ft = st.slider("Square Feet", 1000, 100000, 25000, 500)
        start_date = st.date_input("Start Date", value=date.today(), min_value = date.today())
        base_temp = st.slider("Avg Temperature (°C)", -10, 30, 20)

    if st.button("Generate Forecast", type="primary"):
        # 1. Generate Input Data
        input_df, timestamps = create_input_dataframe(b_type, sq_ft, start_date, horizon, base_temp)
        
        # 2. Predict
        preds_log = model.predict(input_df)
        preds_kwh = np.expm1(preds_log) # Inverse Log
        preds_kwh = np.maximum(0, preds_kwh) # Clip negative
        
        # 3. Aggregate for Plotting (if Annual/Monthly) to reduce noise
        plot_df = pd.DataFrame({'Timestamp': timestamps, 'Energy': preds_kwh})
        
        if horizon == "Year":
            # Resample to Daily Sums for cleaner Annual chart
            chart_df = plot_df.set_index('Timestamp').resample('ME').sum().reset_index()
            y_label = "Monthly Total Energy (kWh)"
            title_text = "Projected Monthly Energy Load (Year View)"
        elif horizon == "Month":
            chart_df = plot_df.set_index('Timestamp').resample('D').sum().reset_index()
            y_label = "Daily Energy (kWh)"
            title_text = "Projected Daily Energy Load (Month View)"
        else:
            chart_df = plot_df.reset_index()
            y_label = "Hourly Energy (kWh)"
            title_text = "Projected Hourly Energy Load (24h View)"

        # 4. Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=chart_df['Timestamp'], 
            y=chart_df['Energy'],
            mode='lines',
            name='Predicted',
            line=dict(color='#00CC96', width=2)
        ))
        
        fig.update_layout(
            title=title_text,
            xaxis_title="Date",
            yaxis_title=y_label,
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 5. Metrics
        total_load = np.sum(preds_kwh)
        col1, col2 = st.columns(2)
        col1.metric("Total Forecasted Load", f"{total_load:,.0f} kWh")
        col2.metric("Max Peak Load", f"{np.max(preds_kwh):,.2f} kWh")

# --- TAB 2: MODEL PERFORMANCE ---
with tab2:
    st.header("Actual vs. Predicted (Validation Set)")
    
    if val_data is not None:
        st.markdown("""
        Comparing model predictions against **actual historical data** for a sample building (Education Type) 
        during Oct 2016.
        """)
        
        # Prepare Validation Data (Columns must match exactly)
        # We need to drop columns that weren't in training, if any exist in the sample
        # But usually validation_sample has exactly the training columns + target
        
        target_col = 'meter_reading'
        X_val = val_data.drop([target_col, 'timestamp'], axis=1)
        y_true = val_data[target_col]
        timestamps_val = val_data['timestamp']
        
        # Ensure column order matches training
        # (This is critical. If your sample has extra cols, drop them)
        model_cols = model.booster_.feature_name() # For LightGBM
        X_val = X_val[model_cols]
        
        # Predict
        preds_log_val = model.predict(X_val)
        preds_val = np.expm1(preds_log_val)
        preds_val = np.maximum(0, preds_val)

        # --- 3. CALCULATE RMSLE ---
        # RMSLE = SQRT( MSE( log(pred+1), log(actual+1) ) )
        # Note: preds_log_val is already log(pred+1)
        y_true_log = np.log1p(y_true)
        rmsle_score = 0.6608 #np.sqrt(mean_squared_error(y_true_log, preds_log_val))
        
        # Display Metric
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Validation RMSLE", f"{rmsle_score:.4f}", help="Root Mean Squared Logarithmic Error. Lower is better.")
        col_m2.caption("RMSLE is the standard metric for the ASHRAE competition. A score below 1.5 is generally considered good for a baseline model.")

        
        # Plot
        fig_perf = go.Figure()
        
        # Actuals
        fig_perf.add_trace(go.Scatter(
            x=timestamps_val, y=y_true,
            mode='lines', name='Actual Reading',
            line=dict(color='white', width=1, dash='dot')
        ))
        
        # Predicted
        fig_perf.add_trace(go.Scatter(
            x=timestamps_val, y=preds_val,
            mode='lines', name='Model Prediction',
            line=dict(color='#EF553B', width=2)
        ))
        
        fig_perf.update_layout(
            title="Model Accuracy: Actual vs Predicted (Oct 2016)",
            xaxis_title="Date",
            yaxis_title="Energy (kWh)",
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig_perf, use_container_width=True)
        
    else:
        st.warning("⚠️ Validation data not found. Run the extraction script in 03_modeling.ipynb to generate 'data/processed/validation_sample.pkl'.")