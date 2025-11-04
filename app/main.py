# file: app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Literal
import os
import logging
from app.preprocessing import parse_and_process
from app.train import train_prophet
from app.predict import predict_prophet, save_forecast_csv
from app.evaluation import rolling_cross_validation_prophet
from app.utils import export_report_pdf
from app.diagnostics import diagnose_model, save_diagnostics
from app.category_forecast import distribute_shop_forecast_to_categories, get_category_forecast_by_name
import pandas as pd
import pandas.errors


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
    force_weekly: bool = False


class TrainRequest(BaseModel):
    shop_csv: str
    model_out: str
    include_regressors: bool = False
    log_transform: bool = False
    interval_width: float = Field(default=0.95, ge=0.5, le=0.99)
    holdout_frac: float = Field(default=0.2, ge=0.1, le=0.5)
    changepoint_prior_scale: float = Field(default=0.01, ge=0.001, le=0.5)
    seasonality_prior_scale: float = Field(default=10.0, ge=0.01, le=100.0)
    seasonality_mode: Literal["additive", "multiplicative"] = "additive"
    auto_tune: bool = False
    skip_holdout: bool = False  # Если True, использует ВСЕ данные для обучения (без теста)
    filter_column: Optional[Literal["category", "product_id"]] = None  # Фильтр по категории или товару
    filter_value: Optional[str] = None  # Значение для фильтрации (название категории или ID товара)


class PredictRequest(BaseModel):
    model_path: str
    horizon: int = Field(..., ge=1, le=365)
    log_transform: bool = False
    future_regressor_strategy: Literal["ffill", "median"] = "ffill"
    last_known_regressors_csv: Optional[str] = None
    smooth_transition: bool = False
    smooth_days: int = Field(default=14, ge=0, le=30)
    smooth_alpha: float = Field(default=0.6, ge=0.0, le=1.0)
    max_change_pct: float = Field(default=0.015, ge=0.001, le=0.1)


class EvaluateRequest(BaseModel):
    shop_csv: str
    initial_days: int = Field(default=180, ge=30)
    horizon_days: int = Field(default=30, ge=1)
    period_days: int = Field(default=30, ge=1)
    include_regressors: bool = False
    log_transform: bool = False


class PreprocessResponse(BaseModel):
    shop_csv: str
    category_csv: str
    product_csv: Optional[str] = None
    stats: dict
    aggregation_suggestion: dict  # freq: 'D' or 'W', reason: str


class TrainResponse(BaseModel):
    model_path: str
    metrics_path: str
    metrics: dict
    train_range: dict
    test_range: dict
    n_train: int
    n_test: int


class CategoryForecastRequest(BaseModel):
    shop_forecast_csv: str
    category_csv: str
    horizon_days: int = Field(default=90, ge=1, le=365)
    category_name: Optional[str] = None  # Если указано, возвращает только эту категорию


class CategoryForecastResponse(BaseModel):
    forecast: list
    forecast_csv_path: str
    n_predictions: int
    n_categories: Optional[int] = None  # Количество категорий
    method: str = "distributed_from_shop"


class PredictResponse(BaseModel):
    forecast: list
    forecast_csv_path: str
    n_predictions: int


class EvaluateResponse(BaseModel):
    metrics: dict
    cv_predictions_csv: str
    n_cv_steps: int
    summary: dict


class DiagnoseRequest(BaseModel):
    shop_csv: str
    model_path: str
    include_regressors: bool = False


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a CSV file to the server."""
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV file")
        
        # Create directory if it doesn't exist
        os.makedirs("data/raw", exist_ok=True)
        
        # Save the file to the data/raw directory
        file_path = os.path.join("data/raw", file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"File uploaded successfully: {file_path}")
        return {
            "status": "success",
            "file_path": file_path,
            "message": f"File '{file.filename}' uploaded successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


@app.post("/preprocess", response_model=PreprocessResponse)
async def preprocess_data(request: PreprocessRequest):
    """Preprocess the uploaded CSV file."""
    try:
        # Normalize input file path for cross-platform compatibility
        normalized_input_path = os.path.normpath(request.file_path)
        
        # Check if the file exists
        if not os.path.exists(normalized_input_path):
            raise HTTPException(status_code=404, detail=f"File not found: {normalized_input_path}")
        
        # Create output directories if they don't exist
        os.makedirs("data/processed", exist_ok=True)
        
        # Define output file paths
        base_name = os.path.splitext(os.path.basename(normalized_input_path))[0]
        out_shop_csv = os.path.join("data/processed", f"{base_name}_shop.csv")
        out_category_csv = os.path.join("data/processed", f"{base_name}_category.csv")
        out_product_csv = os.path.join("data/processed", f"{base_name}_product.csv")
        
        # Normalize output paths
        out_shop_csv = os.path.normpath(out_shop_csv)
        out_category_csv = os.path.normpath(out_category_csv)
        out_product_csv = os.path.normpath(out_product_csv)
        
        # Call the preprocessing function
        result = parse_and_process(
            normalized_input_path, 
            out_shop_csv, 
            out_category_csv,
            out_product_csv=out_product_csv,
            force_weekly=request.force_weekly
        )
        
        # Extract aggregation suggestion from stats
        aggregation_suggestion = {
            "freq": result["stats"].get("freq_used", "D"),
            "reason": result["stats"].get("freq_reason", "Default daily aggregation")
        }
        
        logger.info(f"Preprocessing completed: {normalized_input_path}")
        logger.info(f"Aggregation: {aggregation_suggestion['freq']} - {aggregation_suggestion['reason']}")
        
        # Normalize paths for JSON response (use forward slashes for cross-platform compatibility)
        shop_csv_normalized = result["shop_csv"].replace("\\", "/")
        category_csv_normalized = result["category_csv"].replace("\\", "/")
        product_csv_normalized = result.get("product_csv", "").replace("\\", "/") if result.get("product_csv") else None
        
        # Ensure stats are fully JSON-serializable
        import json
        stats_serialized = json.loads(json.dumps(result["stats"], default=str))
        
        return PreprocessResponse(
            shop_csv=shop_csv_normalized,
            category_csv=category_csv_normalized,
            product_csv=product_csv_normalized,
            stats=stats_serialized,
            aggregation_suggestion=aggregation_suggestion
        )
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")
    except ValueError as e:
        logger.error(f"Validation error during preprocessing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except pandas.errors.EmptyDataError as e:
        logger.error(f"Empty CSV file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"CSV file is empty or invalid: {str(e)}")
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Unexpected error during preprocessing: {str(e)}", exc_info=True)
        logger.error(f"Full traceback:\n{error_traceback}")
        error_detail = f"Internal error: {str(e)}. Please check the CSV file format and required columns. Check server logs for details."
        raise HTTPException(status_code=500, detail=error_detail)


@app.post("/train", response_model=TrainResponse)
async def train_model(request: TrainRequest):
    """Train a Prophet model on shop CSV data."""
    try:
        # Check if the CSV file exists
        if not os.path.exists(request.shop_csv):
            raise HTTPException(status_code=404, detail=f"Shop CSV file not found: {request.shop_csv}")
        
        # Validate model output path
        if not request.model_out.endswith('.pkl'):
            raise HTTPException(status_code=400, detail="Model output path must end with .pkl")
        
        # Create directory if it doesn't exist
        model_dir = os.path.dirname(request.model_out)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        
        # Train the model
        logger.info(f"Training model with parameters: include_regressors={request.include_regressors}, "
                   f"log_transform={request.log_transform}, interval_width={request.interval_width}, "
                   f"holdout_frac={request.holdout_frac}, changepoint_prior_scale={request.changepoint_prior_scale}, "
                   f"seasonality_prior_scale={request.seasonality_prior_scale}, seasonality_mode={request.seasonality_mode}, "
                   f"auto_tune={request.auto_tune}")
        
        result = train_prophet(
            shop_csv_path=request.shop_csv,
            model_out_path=request.model_out,
            include_regressors=request.include_regressors,
            log_transform=request.log_transform,
            interval_width=request.interval_width,
            holdout_frac=request.holdout_frac,
            changepoint_prior_scale=request.changepoint_prior_scale,
            seasonality_prior_scale=request.seasonality_prior_scale,
            seasonality_mode=request.seasonality_mode,
            auto_tune=request.auto_tune,
            skip_holdout=getattr(request, 'skip_holdout', False),
            filter_column=request.filter_column,
            filter_value=request.filter_value
        )
        
        metrics_path = request.model_out.replace('.pkl', '_metrics.json')
        
        logger.info(f"Model training completed: {request.model_out}")
        
        # Безопасное логирование метрик (могут быть None при skip_holdout=True)
        mae_val = result['metrics'].get('mae')
        rmse_val = result['metrics'].get('rmse')
        mape_val = result['metrics'].get('mape')
        
        if mae_val is not None and rmse_val is not None and mape_val is not None:
            logger.info(f"Metrics: MAE={mae_val:.2f}, RMSE={rmse_val:.2f}, MAPE={mape_val:.2f}%")
        else:
            logger.info(f"Metrics: N/A (skip_holdout=True, no test set)")
        
        return TrainResponse(
            model_path=result["model_path"],
            metrics_path=metrics_path,
            metrics=result["metrics"],
            train_range=result["train_range"],
            test_range=result["test_range"],
            n_train=result["n_train"],
            n_test=result["n_test"]
        )
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(f"Validation error during training: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during model training: {str(e)}")


@app.post("/predict", response_model=PredictResponse)
async def predict_data(request: PredictRequest):
    """Generate forecast using a trained model."""
    try:
        # Check if the model file exists
        if not os.path.exists(request.model_path):
            raise HTTPException(status_code=404, detail=f"Model file not found: {request.model_path}")
        
        # Validate horizon
        if not (1 <= request.horizon <= 365):
            raise HTTPException(status_code=400, detail="Horizon must be between 1 and 365 days")
        
        # Validate regressor strategy
        if request.future_regressor_strategy not in ["ffill", "median"]:
            raise HTTPException(status_code=400, detail="future_regressor_strategy must be 'ffill' or 'median'")
        
        # Map strategy name to function parameter
        regressor_fill_method = "forward" if request.future_regressor_strategy == "ffill" else "median"
        
        # Generate forecast
        logger.info(f"Generating forecast: horizon={request.horizon}, log_transform={request.log_transform}, "
                   f"regressor_strategy={request.future_regressor_strategy}")
        
        df_forecast = predict_prophet(
            model_path=request.model_path,
            horizon_days=request.horizon,
            last_known_regressors_csv=request.last_known_regressors_csv,
            log_transform=request.log_transform,
            regressor_fill_method=regressor_fill_method,
            smooth_transition=request.smooth_transition,
            smooth_days=request.smooth_days,
            smooth_alpha=request.smooth_alpha,
            max_change_pct=request.max_change_pct
        )
        
        # Forecast is already saved to default location by predict_prophet
        forecast_csv_path = "data/processed/forecast_shop.csv"
        
        # Convert DataFrame to list of dictionaries for JSON response
        forecast_list = df_forecast.to_dict(orient="records")
        
        logger.info(f"Forecasting completed: {len(forecast_list)} predictions saved to {forecast_csv_path}")
        
        return PredictResponse(
            forecast=forecast_list,
            forecast_csv_path=forecast_csv_path,
            n_predictions=len(forecast_list)
        )
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(f"Validation error during prediction: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_model(request: EvaluateRequest):
    """Perform rolling cross-validation for model evaluation."""
    try:
        # Check if the CSV file exists
        if not os.path.exists(request.shop_csv):
            raise HTTPException(status_code=404, detail=f"Shop CSV file not found: {request.shop_csv}")
        
        # Load data
        logger.info(f"Loading data from: {request.shop_csv}")
        df = pd.read_csv(request.shop_csv)
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values('ds').reset_index(drop=True)
        
        # Perform cross-validation
        logger.info(f"Starting CV with: initial={request.initial_days}, horizon={request.horizon_days}, "
                   f"period={request.period_days}, include_regressors={request.include_regressors}, "
                   f"log_transform={request.log_transform}")
        
        result = rolling_cross_validation_prophet(
            shop_df=df,
            initial_days=request.initial_days,
            horizon_days=request.horizon_days,
            period_days=request.period_days,
            include_regressors=request.include_regressors,
            log_transform=request.log_transform
        )
        
        # Save predictions CSV
        os.makedirs("data/processed", exist_ok=True)
        cv_predictions_csv = "data/processed/cv_predictions.csv"
        result['predictions_df'].to_csv(cv_predictions_csv, index=False)
        
        # Prepare summary
        summary = {
            "mae_mean": result['metrics']['mae']['mean'],
            "mae_std": result['metrics']['mae']['std'],
            "rmse_mean": result['metrics']['rmse']['mean'],
            "rmse_std": result['metrics']['rmse']['std'],
            "mape_mean": result['metrics']['mape']['mean'],
            "mape_std": result['metrics']['mape']['std']
        }
        
        logger.info(f"CV completed: {result['metrics']['n_cv_steps']} steps")
        logger.info(f"Summary: MAE={summary['mae_mean']:.2f}±{summary['mae_std']:.2f}, "
                   f"RMSE={summary['rmse_mean']:.2f}±{summary['rmse_std']:.2f}, "
                   f"MAPE={summary['mape_mean']:.2f}%±{summary['mape_std']:.2f}%")
        
        return EvaluateResponse(
            metrics=result['metrics'],
            cv_predictions_csv=cv_predictions_csv,
            n_cv_steps=result['metrics']['n_cv_steps'],
            summary=summary
        )
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(f"Validation error during evaluation: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during evaluation: {str(e)}")


from fastapi.responses import FileResponse


@app.get("/forecast/download")
async def download_forecast_pdf(path: str = Query(..., description="Path to the forecast CSV file")):
    """Download a forecast report as PDF."""
    try:
        # Validate that the path is for a CSV file
        if not path.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Path must point to a CSV file")
        
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Forecast CSV file not found: {path}")
        
        # Read the forecast CSV
        df_forecast = pd.read_csv(path)
        
        # Validate required columns
        required_cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
        missing_cols = [col for col in required_cols if col not in df_forecast.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Forecast CSV missing required columns: {', '.join(missing_cols)}")
        
        # Try to read the history data (using the shop CSV if available)
        # Try multiple possible paths for the shop CSV
        base_dir = os.path.dirname(path)
        possible_paths = [
            path.replace("_category.csv", "_shop.csv").replace("forecast_shop.csv", "sales_data_shop.csv"),
            os.path.join(base_dir, "sales_data_shop.csv"),
            path.replace("forecast_shop.csv", "sales_data_shop.csv"),
            path.replace("forecast_category.csv", "sales_data_category.csv").replace("_category.csv", "_shop.csv"),
        ]
        
        df_history = pd.DataFrame(columns=['ds', 'y'])
        shop_csv_path = None
        
        for possible_path in possible_paths:
            if os.path.exists(possible_path):
                try:
                    df_history_temp = pd.read_csv(possible_path)
                    # Validate that it has required columns
                    if 'ds' in df_history_temp.columns and 'y' in df_history_temp.columns:
                        df_history = df_history_temp
                        shop_csv_path = possible_path
                        logger.info(f"Found history CSV at: {shop_csv_path} ({len(df_history)} rows)")
                        break
                    else:
                        logger.warning(f"History CSV at {possible_path} missing required columns (ds, y)")
                except Exception as e:
                    logger.warning(f"Error reading history CSV at {possible_path}: {str(e)}")
                    continue
        
        if shop_csv_path is None:
            # If history is not available, create an empty history DataFrame
            logger.warning(f"History CSV not found in any of the attempted paths, using empty history")
        
        # Try to load metrics from JSON if available
        # Metrics are saved next to the model, try standard location first
        possible_metrics_paths = [
            "models/prophet_model_metrics.json",  # Standard location
            path.replace('.csv', '_metrics.json').replace('forecast', 'model'),
            os.path.join(os.path.dirname(path), "prophet_model_metrics.json"),
            os.path.join("models", os.path.basename(path).replace("forecast_shop.csv", "prophet_model_metrics.json")),
        ]
        
        metrics = {
            'mae': 'N/A',
            'rmse': 'N/A',
            'mape': 'N/A',
            'log_transform': False,
            'interval_width': 0.95
        }
        
        metrics_found = False
        for metrics_json_path in possible_metrics_paths:
            if os.path.exists(metrics_json_path):
                try:
                    import json
                    with open(metrics_json_path, 'r') as f:
                        loaded_metrics = json.load(f)
                        metrics.update(loaded_metrics)
                    logger.info(f"Loaded metrics from: {metrics_json_path}")
                    metrics_found = True
                    break
                except Exception as e:
                    logger.warning(f"Error reading metrics from {metrics_json_path}: {str(e)}")
                    continue
        
        if not metrics_found:
            logger.warning("Metrics JSON not found, using default values (N/A)")
        
        # Generate a PDF path
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
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during PDF generation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during PDF generation: {str(e)}")


@app.post("/diagnose")
async def diagnose_model_endpoint(request: DiagnoseRequest):
    """Perform model diagnostics."""
    try:
        # Check if files exist
        if not os.path.exists(request.shop_csv):
            raise HTTPException(status_code=404, detail=f"Shop CSV file not found: {request.shop_csv}")
        if not os.path.exists(request.model_path):
            raise HTTPException(status_code=404, detail=f"Model file not found: {request.model_path}")
        
        # Load data
        logger.info(f"Loading data from: {request.shop_csv}")
        df_history = pd.read_csv(request.shop_csv)
        df_history['ds'] = pd.to_datetime(df_history['ds'])
        df_history = df_history.sort_values('ds').reset_index(drop=True)
        
        # Load model and generate forecast for diagnostics
        import joblib
        
        # Generate forecast on historical data for comparison
        model = joblib.load(request.model_path)
        
        # Check if model requires regressors
        requires_regressors = len(model.extra_regressors) > 0 if hasattr(model, 'extra_regressors') else False
        
        # Auto-detect if regressors are needed
        if not request.include_regressors and requires_regressors:
            logger.info("Model requires regressors but not provided, auto-detecting from model")
            request.include_regressors = True
        
        # Create future dataframe for all historical dates
        future = model.make_future_dataframe(periods=0, freq='D')
        
        if requires_regressors or request.include_regressors:
            if 'avg_price' in df_history.columns and 'avg_discount' in df_history.columns:
                regressors = df_history[['ds', 'avg_price', 'avg_discount']]
                future = future.merge(regressors, on='ds', how='left')
                for col in ['avg_price', 'avg_discount']:
                    future[col] = future[col].ffill().fillna(regressors[col].median())
            else:
                logger.warning("Model requires regressors but they are not in data")
                if requires_regressors:
                    raise ValueError("Model was trained with regressors but they are missing in historical data")
        
        forecast = model.predict(future)
        df_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        
        # Perform diagnostics
        logger.info("Performing model diagnostics...")
        diagnostics = diagnose_model(
            df_history=df_history,
            df_forecast=df_forecast,
            model_name=os.path.basename(request.model_path),
            include_regressors=request.include_regressors
        )
        
        # Save diagnostics
        os.makedirs("analysis", exist_ok=True)
        diagnostics_path = "analysis/model_diagnostics.json"
        save_diagnostics(diagnostics, diagnostics_path)
        
        logger.info(f"Diagnostics completed and saved to {diagnostics_path}")
        
        return diagnostics
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error during diagnostics: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during diagnostics: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Sales Forecasting API is running"}


@app.post("/predict_category_distributed", response_model=CategoryForecastResponse)
async def predict_category_distributed(request: CategoryForecastRequest):
    """
    Альтернативный подход для прогнозирования категорий:
    распределяет shop-level прогноз по категориям пропорционально их историческим долям.
    
    Это решение для случаев, когда Prophet плохо работает с разреженными данными категорий.
    Использует проверенный shop-level прогноз и распределяет его пропорционально.
    """
    try:
        # Проверяем наличие файлов
        if not os.path.exists(request.shop_forecast_csv):
            raise HTTPException(status_code=404, detail=f"Shop forecast CSV not found: {request.shop_forecast_csv}")
        if not os.path.exists(request.category_csv):
            raise HTTPException(status_code=404, detail=f"Category CSV not found: {request.category_csv}")
        
        logger.info("=" * 80)
        logger.info("ИСПОЛЬЗУЕМ АЛЬТЕРНАТИВНЫЙ ПОДХОД: Распределение shop-level прогноза по категориям")
        logger.info(f"Shop forecast: {request.shop_forecast_csv}")
        logger.info(f"Category history: {request.category_csv}")
        logger.info("=" * 80)
        
        # Распределяем прогноз
        output_csv = "data/processed/forecast_category_distributed.csv"
        distributed_forecast = distribute_shop_forecast_to_categories(
            shop_forecast_csv=request.shop_forecast_csv,
            category_csv=request.category_csv,
            horizon_days=request.horizon_days,
            output_csv=output_csv
        )
        
        # Если указана конкретная категория, фильтруем
        if request.category_name:
            filtered_forecast = distributed_forecast[distributed_forecast['category'] == request.category_name]
            if len(filtered_forecast) == 0:
                available = distributed_forecast['category'].unique().tolist()
                raise HTTPException(
                    status_code=404,
                    detail=f"Category '{request.category_name}' not found. Available: {available}"
                )
            distributed_forecast = filtered_forecast
        
        # Конвертируем в список словарей, включая категорию
        forecast_list = distributed_forecast[['category', 'ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict('records')
        
        # Конвертируем даты в строки
        for item in forecast_list:
            if isinstance(item['ds'], pd.Timestamp):
                item['ds'] = item['ds'].isoformat()
        
        logger.info(f"Category forecast generated: {len(forecast_list)} predictions for {distributed_forecast['category'].nunique()} categories")
        
        # Подсчитываем количество категорий
        n_categories = distributed_forecast['category'].nunique()
        
        return CategoryForecastResponse(
            forecast=forecast_list,
            forecast_csv_path=output_csv,
            n_predictions=len(forecast_list),
            n_categories=n_categories,  # Добавляем количество категорий
            method="distributed_from_shop"
        )
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error during category forecast distribution: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during category forecast distribution: {str(e)}")


@app.get("/categories")
async def get_categories(
    category_csv: Optional[str] = Query(None, description="Path to preprocessed category CSV file"),
    raw_csv: Optional[str] = Query(None, description="Path to raw CSV file with actual transactions")
):
    """
    Get list of available categories with record counts.
    If raw_csv is provided, counts real transactions from raw file.
    Otherwise, uses preprocessed category_csv file.
    """
    try:
        # Priority: use raw_csv if provided (for real transaction counts)
        if raw_csv:
            if not os.path.exists(raw_csv):
                raise HTTPException(status_code=404, detail=f"Raw CSV file not found: {raw_csv}")
            
            logger.info(f"Loading categories from raw CSV: {raw_csv}")
            df = pd.read_csv(raw_csv, low_memory=False)
            
            # Check for category column (handle both old and new formats)
            if 'category' not in df.columns and 'Product_Category' in df.columns:
                df['category'] = df['Product_Category']
            elif 'category' not in df.columns:
                raise HTTPException(status_code=400, detail="Raw CSV file must contain 'category' or 'Product_Category' column")
            
            # Count REAL transactions per category (not time points)
            category_counts = df['category'].value_counts().reset_index()
            category_counts.columns = ['name', 'count']
            category_counts = category_counts.sort_values('name')
            
            logger.info(f"Found {len(category_counts)} categories with real transaction counts from raw file")
        else:
            # Fallback to preprocessed file
            if not category_csv:
                raise HTTPException(status_code=400, detail="Either 'category_csv' or 'raw_csv' must be provided")
            
            if not os.path.exists(category_csv):
                raise HTTPException(status_code=404, detail=f"Category CSV file not found: {category_csv}")
            
            logger.info(f"Loading categories from preprocessed CSV: {category_csv}")
            df = pd.read_csv(category_csv)
            if 'category' not in df.columns:
                raise HTTPException(status_code=400, detail="CSV file must contain 'category' column")
            
            # Calculate counts for each category (these are time points, not transactions)
            category_counts = df['category'].value_counts().reset_index()
            category_counts.columns = ['name', 'count']
            category_counts = category_counts.sort_values('name')
            
            logger.info(f"Found {len(category_counts)} categories from preprocessed file (time point counts)")
        
        # Convert to list of dicts
        categories_with_counts = [
            {"name": row['name'], "count": int(row['count'])} 
            for _, row in category_counts.iterrows()
        ]
        
        return {
            "categories": categories_with_counts,
            "count": len(categories_with_counts),
            "from_raw": raw_csv is not None
        }
    except Exception as e:
        logger.error(f"Error getting categories: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting categories: {str(e)}")


@app.get("/products")
async def get_products(
    product_csv: Optional[str] = Query(None, description="Path to preprocessed product CSV file"),
    raw_csv: Optional[str] = Query(None, description="Path to raw CSV file with actual transactions"),
    limit: int = Query(default=500, ge=1, le=10000, description="Maximum number of products to return")
):
    """
    Get list of available products with record counts.
    If raw_csv is provided, counts real transactions from raw file.
    Otherwise, uses preprocessed product_csv file.
    """
    try:
        # Priority: use raw_csv if provided (for real transaction counts)
        if raw_csv:
            if not os.path.exists(raw_csv):
                raise HTTPException(status_code=404, detail=f"Raw CSV file not found: {raw_csv}")
            
            logger.info(f"Loading products from raw CSV: {raw_csv}...")
            df = pd.read_csv(raw_csv, low_memory=False)
            
            # Check for product_id column (handle both old and new formats)
            if 'product_id' not in df.columns:
                if 'item_id' in df.columns:
                    df['product_id'] = df['item_id'].astype(str)
                elif 'Product_ID' in df.columns:
                    df['product_id'] = df['Product_ID'].astype(str)
                elif 'sku' in df.columns:
                    df['product_id'] = df['sku'].astype(str)
                else:
                    raise HTTPException(status_code=400, detail="Raw CSV file must contain 'product_id', 'item_id', 'Product_ID', or 'sku' column")
            
            logger.info(f"Found {len(df)} total transactions, {df['product_id'].nunique()} unique products")
            
            # Calculate counts for each product_id (REAL transactions, not time points)
            product_counts = df['product_id'].value_counts().reset_index()
            product_counts.columns = ['name', 'count']
            # Sort by count descending, then by name for products with same count
            product_counts = product_counts.sort_values(['count', 'name'], ascending=[False, True])
            
            logger.info(f"Calculated real transaction counts for {len(product_counts)} products")
        else:
            # Fallback to preprocessed file
            if not product_csv:
                raise HTTPException(status_code=400, detail="Either 'product_csv' or 'raw_csv' must be provided")
            
            if not os.path.exists(product_csv):
                raise HTTPException(status_code=404, detail=f"Product CSV file not found: {product_csv}")
            
            logger.info(f"Loading products from preprocessed CSV: {product_csv}...")
            df = pd.read_csv(product_csv)
            
            if 'product_id' not in df.columns:
                raise HTTPException(status_code=400, detail="CSV file must contain 'product_id' column")
            
            logger.info(f"Found {len(df)} total records (time points), {df['product_id'].nunique()} unique products")
            
            # Calculate counts for each product_id (these are time points, not transactions)
            product_counts = df['product_id'].value_counts().reset_index()
            product_counts.columns = ['name', 'count']
            # Sort by count descending, then by name for products with same count
            product_counts = product_counts.sort_values(['count', 'name'], ascending=[False, True])
        
        # Limit results if too many products
        total_products = len(product_counts)
        if total_products > limit:
            logger.info(f"Limiting results to top {limit} products (out of {total_products} total)")
            product_counts = product_counts.head(limit)
        
        # Convert to list of dicts
        products_with_counts = [
            {"name": str(row['name']), "count": int(row['count'])} 
            for _, row in product_counts.iterrows()
        ]
        
        logger.info(f"Returning {len(products_with_counts)} products")
        
        return {
            "products": products_with_counts,
            "count": len(products_with_counts),
            "total_available": total_products,
            "limited": total_products > limit,
            "from_raw": raw_csv is not None
        }
    except Exception as e:
        logger.error(f"Error getting products: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting products: {str(e)}")


# If running as a script, start the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
