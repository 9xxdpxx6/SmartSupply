# file: streamlit_app/app.py
import streamlit as st
import pandas as pd
import requests
import io
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os


# Set up the page configuration
st.set_page_config(
    page_title="Sales Forecasting App",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Sales Forecasting App")

# Initialize session state
if 'uploaded_file_path' not in st.session_state:
    st.session_state.uploaded_file_path = None
if 'preprocessed_shop_csv' not in st.session_state:
    st.session_state.preprocessed_shop_csv = None
if 'trained_model_path' not in st.session_state:
    st.session_state.trained_model_path = None
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None
if 'forecast_csv_path' not in st.session_state:
    st.session_state.forecast_csv_path = None

# FastAPI backend URL
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

# File uploader
uploaded_file = st.file_uploader("Upload your sales CSV file", type=["csv"])

if uploaded_file is not None:
    # Display raw data preview
    bytes_data = uploaded_file.getvalue()
    df_raw = pd.read_csv(io.StringIO(bytes_data.decode("utf-8")))
    
    st.subheader("Raw Data Preview")
    st.dataframe(df_raw.head(10))
    
    st.subheader("Detected Columns")
    st.write(f"Columns: {', '.join(df_raw.columns.tolist())}")
    
    # Upload file to backend
    if st.button("Upload to Backend"):
        try:
            # Create a file-like object for the POST request
            files = {"file": (uploaded_file.name, bytes_data, "text/csv")}
            response = requests.post(f"{FASTAPI_URL}/upload", files=files)
            
            if response.status_code == 200:
                result = response.json()
                st.session_state.uploaded_file_path = result["file_path"]
                st.success(f"File uploaded successfully: {result['file_path']}")
            else:
                st.error(f"Upload failed: {response.text}")
        except Exception as e:
            st.error(f"Error uploading file: {str(e)}")

# Preprocess button
if st.session_state.uploaded_file_path:
    if st.button("Preprocess Data"):
        try:
            payload = {"file_path": st.session_state.uploaded_file_path}
            response = requests.post(f"{FASTAPI_URL}/preprocess", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                st.session_state.preprocessed_shop_csv = result["shop_csv"]
                st.success("Data preprocessed successfully!")
                
                # Show stats
                st.subheader("Preprocessing Stats")
                st.json(result["stats"])
            else:
                st.error(f"Preprocessing failed: {response.text}")
        except Exception as e:
            st.error(f"Error preprocessing data: {str(e)}")

# Train button
if st.session_state.preprocessed_shop_csv:
    if st.button("Train Model"):
        try:
            # Create model output path
            model_path = st.session_state.preprocessed_shop_csv.replace("_shop.csv", "_model.pkl")
            payload = {
                "shop_csv": st.session_state.preprocessed_shop_csv,
                "model_out": model_path
            }
            
            response = requests.post(f"{FASTAPI_URL}/train", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                st.session_state.trained_model_path = result["model_path"]
                st.success("Model trained successfully!")
                
                # Show metrics
                st.subheader("Training Metrics")
                st.json(result["backtest_metrics"])
            else:
                st.error(f"Training failed: {response.text}")
        except Exception as e:
            st.error(f"Error training model: {str(e)}")

# Predict section
if st.session_state.trained_model_path:
    st.subheader("Generate Forecast")
    horizon = st.number_input("Forecast Horizon (days)", min_value=1, value=30, step=1)
    
    if st.button("Generate Forecast"):
        try:
            payload = {
                "model_path": st.session_state.trained_model_path,
                "horizon": int(horizon)
            }
            
            response = requests.post(f"{FASTAPI_URL}/predict", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                st.session_state.forecast_data = result["forecast"]
                
                # Load data as DataFrame
                df_forecast = pd.DataFrame(result["forecast"])
                
                # Store the CSV path for download
                st.session_state.forecast_csv_path = "data/processed/forecast_shop.csv"
                
                st.success("Forecast generated successfully!")
            else:
                st.error(f"Prediction failed: {response.text}")
        except Exception as e:
            st.error(f"Error generating forecast: {str(e)}")

# Display forecast
if st.session_state.forecast_data:
    df_forecast = pd.DataFrame(st.session_state.forecast_data)
    
    st.subheader("Forecast Visualization")
    
    # Create plotly chart
    fig = go.Figure()
    
    # Add forecast line
    fig.add_trace(go.Scatter(
        x=df_forecast['ds'],
        y=df_forecast['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='red')
    ))
    
    # Add confidence intervals if available
    if 'yhat_lower' in df_forecast.columns and 'yhat_upper' in df_forecast.columns:
        fig.add_trace(go.Scatter(
            x=df_forecast['ds'],
            y=df_forecast['yhat_upper'],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=df_forecast['ds'],
            y=df_forecast['yhat_lower'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.2)',
            name='Confidence Interval',
            showlegend=True
        ))
    
    fig.update_layout(
        title="Sales Forecast",
        xaxis_title="Date",
        yaxis_title="Sales",
        hovermode='x unified',
        width=800,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display forecast table
    st.subheader("Forecast Table (First 50 Rows)")
    st.dataframe(df_forecast.head(50))

# Add to session state to store PDF content
if 'pdf_content' not in st.session_state:
    st.session_state.pdf_content = None

# Download PDF button
if st.session_state.forecast_csv_path:
    if st.button("Download PDF Report"):
        try:
            # Generate PDF report via FastAPI
            params = {"path": st.session_state.forecast_csv_path}
            response = requests.get(f"{FASTAPI_URL}/forecast/download", params=params)
            
            if response.status_code == 200:
                st.session_state.pdf_content = response.content
                st.success("PDF report generated!")
            else:
                st.error(f"PDF generation failed: {response.text}")
        except Exception as e:
            st.error(f"Error generating PDF: {str(e)}")

    if st.session_state.pdf_content:
        st.download_button(
            label="Click here to download PDF",
            data=st.session_state.pdf_content,
            file_name="forecast_report.pdf",
            mime="application/pdf",
            key='download_pdf'  # Unique key prevents re-render issues
        )

# Health check
try:
    health_response = requests.get(f"{FASTAPI_URL}/health")
    if health_response.status_code == 200:
        st.sidebar.success("✅ Backend API is running")
    else:
        st.sidebar.error("❌ Backend API is not responding")
except:
    st.sidebar.error("❌ Unable to connect to backend API")