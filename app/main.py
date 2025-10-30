# file: app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import logging
from app.preprocessing import parse_and_process
from app.train import train_prophet
from app.predict import predict_prophet, save_forecast_csv
from app.utils import export_report_pdf
import pandas as pd


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Sales Forecasting API", description="API for sales forecasting using Prophet")

# Add CORS middleware to allow localhost origins (for Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501", "http://localhost:*", "http://127.0.0.1:*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class PreprocessRequest(BaseModel):
    file_path: str


class TrainRequest(BaseModel):
    shop_csv: str
    model_out: str


class PredictRequest(BaseModel):
    model_path: str
    horizon: int


class PreprocessResponse(BaseModel):
    shop_csv: str
    category_csv: str
    stats: dict


class TrainResponse(BaseModel):
    model_path: str
    backtest_metrics: dict
    data_info: dict


class PredictResponse(BaseModel):
    forecast: list


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a CSV file to the server."""
    try:
        # Create directory if it doesn't exist
        os.makedirs("data/raw", exist_ok=True)
        
        # Save the file to the data/raw directory
        file_path = os.path.join("data/raw", file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"File uploaded: {file_path}")
        return {"file_path": file_path}
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


@app.post("/preprocess", response_model=PreprocessResponse)
async def preprocess_data(request: PreprocessRequest):
    """Preprocess the uploaded CSV file."""
    try:
        # Check if the file exists
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Create output directories if they don't exist
        os.makedirs("data/processed", exist_ok=True)
        
        # Define output file paths
        base_name = os.path.splitext(os.path.basename(request.file_path))[0]
        out_shop_csv = os.path.join("data/processed", f"{base_name}_shop.csv")
        out_category_csv = os.path.join("data/processed", f"{base_name}_category.csv")
        
        # Call the preprocessing function
        result = parse_and_process(request.file_path, out_shop_csv, out_category_csv)
        
        logger.info(f"Preprocessing completed: {request.file_path}")
        return PreprocessResponse(
            shop_csv=result["shop_csv"],
            category_csv=result["category_csv"],
            stats=result["stats"]
        )
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during preprocessing: {str(e)}")


@app.post("/train", response_model=TrainResponse)
async def train_model(request: TrainRequest):
    """Train a Prophet model on shop CSV data."""
    try:
        # Check if the CSV file exists
        if not os.path.exists(request.shop_csv):
            raise HTTPException(status_code=404, detail="Shop CSV file not found")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(request.model_out), exist_ok=True)
        
        # Train the model
        result = train_prophet(request.shop_csv, request.model_out, include_regressors=False)
        
        logger.info(f"Model training completed: {request.model_out}")
        return TrainResponse(
            model_path=result["model_path"],
            backtest_metrics=result["backtest_metrics"],
            data_info=result["data_info"]
        )
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during model training: {str(e)}")


@app.post("/predict", response_model=PredictResponse)
async def predict_data(request: PredictRequest):
    """Generate forecast using a trained model."""
    try:
        # Check if the model file exists
        if not os.path.exists(request.model_path):
            raise HTTPException(status_code=404, detail="Model file not found")
        
        # Generate forecast
        df_forecast = predict_prophet(request.model_path, request.horizon)
        
        # Save forecast to CSV
        os.makedirs("data/processed", exist_ok=True)
        forecast_csv_path = "data/processed/forecast_shop.csv"
        save_forecast_csv(df_forecast, forecast_csv_path)
        
        # Convert DataFrame to list of dictionaries for JSON response
        forecast_list = df_forecast.to_dict(orient="records")
        
        logger.info(f"Forecasting completed, results saved to: {forecast_csv_path}")
        return PredictResponse(forecast=forecast_list)
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")


from fastapi.responses import FileResponse
import tempfile
import uuid


@app.get("/forecast/download")
async def download_forecast_pdf(path: str = Query(..., description="Path to the forecast CSV file")):
    """Download a forecast report as PDF."""
    try:
        # Validate that the path is for a CSV file
        if not path.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Path must point to a CSV file")
        
        # Read the forecast CSV
        df_forecast = pd.read_csv(path)
        
        # Try to read the history data (using the shop CSV if available)
        shop_csv_path = path.replace("_category.csv", "_shop.csv").replace("forecast_shop.csv", "shop.csv")
        if os.path.exists(shop_csv_path):
            df_history = pd.read_csv(shop_csv_path)
        else:
            # If history is not available, create an empty history DataFrame
            df_history = pd.DataFrame(columns=['ds', 'y'])
        
        # Create a simple metrics dictionary for the report
        metrics = {
            'mae': 'N/A',
            'rmse': 'N/A',
            'mape': 'N/A'
        }
        
        # Generate a PDF path using a temporary file
        pdf_path = path.replace('.csv', '_report.pdf')
        
        # Generate the PDF report
        export_report_pdf(pdf_path, df_history, df_forecast, metrics)
        
        logger.info(f"PDF report generated: {pdf_path}")
        
        # Return the PDF file as a download
        return FileResponse(
            path=pdf_path,
            media_type='application/pdf',
            filename=os.path.basename(pdf_path)
        )
        
    except Exception as e:
        logger.error(f"Error during PDF generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during PDF generation: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Sales Forecasting API is running"}


# If running as a script, start the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)