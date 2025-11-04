# file: README.md
# SmartSupply - Sales Forecasting Application

SmartSupply is a comprehensive sales forecasting application that leverages Prophet for time series forecasting. The application provides both a FastAPI backend for processing and a Streamlit frontend for interactive data analysis and visualization.

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### FastAPI Backend
To start the FastAPI server:
```bash
uvicorn app.main:app --reload --port 8888
```
The API will be available at `http://localhost:8888`.

API documentation is available at:
- Swagger UI: `http://localhost:8888/docs`
- ReDoc: `http://localhost:8888/redoc`

### Streamlit Frontend
To start the Streamlit application:
```bash
streamlit run streamlit_app/app.py
```
The UI will be available at `http://localhost:8501`.

**Режимы работы:**
- **Simple Mode (Простой режим)**: Автоматизированный процесс для нетехнических пользователей
  - Оптимальные настройки по умолчанию (без регрессоров, с log-transform, с auto-tune)
  - Распределенный прогноз по всем категориям
  - Отдельные графики для каждой категории
- **Advanced Mode (Продвинутый режим)**: Полный контроль над всеми параметрами
  - Все настройки настраиваемы вручную
  - Доступны все разделы: обучение, диагностика, кросс-валидация, прогноз

## Example Workflow

1. **Prepare your data**: Ensure your CSV file has the following columns:
   - `Sale_Date`: Date of the sale (YYYY-MM-DD format)
   - `Product_ID`: Unique identifier for the product
   - `Product_Category`: Category of the product
   - `Unit_Price`: Price of the unit (numeric)
   - `Discount`: Discount applied (0.0-1.0 or percentage)
   - `Quantity_Sold`: Quantity sold (numeric, this is the target variable)

2. **Run preprocessing**:
   ```bash
   python scripts/cli_tools.py preprocess \
       --input data/raw/sales.csv \
       --out_shop data/processed/shop.csv \
       --out_cat data/processed/category.csv
   ```
   
   The preprocessing will:
   - Validate data (minimum 100 rows, check for missing values, duplicates)
   - Normalize column names to canonical format
   - Aggregate data to shop-level (daily/weekly) and category-level
   - Generate regressors: `avg_price`, `avg_discount`, `day_of_week`, `is_weekend`
   - Return statistics and aggregation frequency recommendation

3. **Train the model**:
   ```bash
   python scripts/cli_tools.py train \
       --shop_csv data/processed/shop.csv \
       --model_out models/prophet_model.pkl \
       --include_regressors \
       --log_transform \
       --interval_width 0.95 \
       --holdout_frac 0.2
   ```
   
   Outputs:
   - Model file: `models/prophet_model.pkl`
   - Metrics JSON: `models/prophet_model_metrics.json`
   - Training metrics displayed in console

4. **Evaluate with cross-validation** (recommended):
   ```bash
   python scripts/cli_tools.py evaluate \
       --shop_csv data/processed/shop.csv \
       --initial_days 180 \
       --horizon_days 30 \
       --period_days 30 \
       --include_regressors \
       --log_transform \
       --out_csv data/processed/cv_predictions.csv \
       --out_plot data/processed/cv_plot.png
   ```
   
   Outputs:
   - CV predictions CSV: `data/processed/cv_predictions.csv`
   - CV visualization plot: `data/processed/cv_plot.png` (if specified)
   - Aggregated metrics (mean ± std) for MAE, RMSE, MAPE

5. **Generate predictions**:
   ```bash
   python scripts/cli_tools.py predict \
       --model_path models/prophet_model.pkl \
       --horizon 30 \
       --out_csv data/processed/forecast.csv \
       --log_transform \
       --regressor_strategy ffill \
       --regressors_csv data/processed/shop.csv
   ```
   
   Outputs:
   - Forecast CSV: `data/processed/forecast.csv` (or `data/processed/forecast_shop.csv` by default)
   - Contains columns: `ds`, `yhat`, `yhat_lower`, `yhat_upper`

6. **Download PDF report**:
   ```bash
   curl "http://localhost:8888/forecast/download?path=data/processed/forecast_shop.csv" \
       -o forecast_report.pdf
   ```
   
   Or use the Streamlit UI: Click "Download PDF Report" button after generating forecast.

   Outputs:
   - PDF report: `forecast_report.pdf`
   - Metrics JSON: `forecast_report_metrics.json` (saved alongside PDF)
   - Contains: summary statistics, fact vs forecast plot, residuals plot, forecast table

## How to Tune Parameters

### Log Transform (`log_transform`)

**When to use:**
- Use when the variance of sales increases with magnitude (heteroscedasticity)
- Useful for data with exponential growth patterns
- Recommended when sales values span several orders of magnitude

**How it works:**
- Applies `log1p(y)` transformation before training: `y_transformed = log(1 + y)`
- Applies inverse `expm1()` to predictions: `y_pred = exp(yhat) - 1`
- Helps stabilize variance and handle skewed distributions

**Example:**
```bash
# Check if log transform is needed
# If CV metrics improve with log_transform=True, use it
python scripts/cli_tools.py train --shop_csv data/processed/shop.csv \
    --model_out models/model_with_log.pkl --log_transform
```

### Regressors (`include_regressors`)

**When to use:**
- Use when price and discount have significant impact on sales
- Reduces forecast uncertainty if external factors are predictive
- Recommended when you have reliable price/discount data for future periods

**How it works:**
- Adds `avg_price` and `avg_discount` as external regressors to Prophet model
- For future dates, uses forward-fill (last known values) or median strategy
- Requires `last_known_regressors_csv` in predict step if model uses regressors

**Example:**
```bash
# Train with regressors
python scripts/cli_tools.py train --shop_csv data/processed/shop.csv \
    --model_out models/model_with_regressors.pkl --include_regressors

# Predict with regressors (must provide regressors CSV)
python scripts/cli_tools.py predict --model_path models/model_with_regressors.pkl \
    --horizon 30 --regressors_csv data/processed/shop.csv --regressor_strategy ffill
```

**Regressor Strategy:**
- `ffill`: Forward-fill using last known values (default, good for stable prices)
- `median`: Use median of historical values (good when prices fluctuate)

### Interval Width (`interval_width`)

**When to adjust:**
- **Increase** (e.g., 0.99) for more conservative forecasts (wider confidence intervals)
- **Decrease** (e.g., 0.80) for tighter intervals (less uncertainty, but may miss outliers)
- Default 0.95 (95% confidence) is usually a good balance

**How it works:**
- Controls the width of uncertainty intervals (`yhat_lower`, `yhat_upper`)
- Affects Prophet's internal uncertainty estimation
- Higher values = wider intervals = more conservative predictions

**Example:**
```bash
# More conservative (99% confidence)
python scripts/cli_tools.py train --shop_csv data/processed/shop.csv \
    --model_out models/model_conservative.pkl --interval_width 0.99

# Less conservative (80% confidence)
python scripts/cli_tools.py train --shop_csv data/processed/shop.csv \
    --model_out models/model_tight.pkl --interval_width 0.80
```

### Holdout Fraction (`holdout_frac`)

**When to adjust:**
- **Larger** (e.g., 0.3): More data for testing, less for training (use with large datasets)
- **Smaller** (e.g., 0.1): More data for training, less for testing (use with small datasets)
- Default 0.2 (20% for testing) works well for most cases
- Minimum recommended: 0.1, Maximum: 0.5

**How it works:**
- Splits data chronologically: first (1 - holdout_frac) for training, last holdout_frac for testing
- If test set < 7 samples, automatically uses time-series cross-validation instead
- Used to calculate validation metrics (MAE, RMSE, MAPE)

**Example:**
```bash
# With large dataset (1000+ days), use larger holdout
python scripts/cli_tools.py train --shop_csv data/processed/shop.csv \
    --model_out models/model.pkl --holdout_frac 0.3

# With small dataset (<300 days), use smaller holdout
python scripts/cli_tools.py train --shop_csv data/processed/shop.csv \
    --model_out models/model.pkl --holdout_frac 0.15
```

### Rolling Cross-Validation

**When to use:**
- Always recommended before deploying a model
- Provides more robust evaluation than single train/test split
- Shows how model performance varies across different time periods
- Helps identify if model degrades over time

**How to run:**
```bash
python scripts/cli_tools.py evaluate \
    --shop_csv data/processed/shop.csv \
    --initial_days 180 \
    --horizon_days 30 \
    --period_days 30 \
    --include_regressors \
    --log_transform
```

**Parameters:**
- `initial_days`: Minimum training period (recommended: 90-180 days)
- `horizon_days`: Forecast horizon for each CV step (should match production forecast horizon)
- `period_days`: How many days to slide window between steps (smaller = more steps, longer runtime)

**How to interpret:**
- **Mean metrics**: Average performance across all CV steps
- **Std metrics**: Standard deviation (lower = more stable performance)
- **Min/Max range**: Shows best and worst case scenarios
- Look for: low mean, low std (consistent performance), reasonable min-max range

**Example output:**
```
CV Steps: 5
MAE (mean ± std): 12.34 ± 2.15
RMSE (mean ± std): 15.67 ± 3.21
MAPE (mean ± std): 8.5% ± 1.2%
```

Good metrics: Low values, small standard deviations, reasonable ranges.

## API Endpoints

- `POST /upload`: Upload sales CSV file
  - Returns: `{file_path: str}`

- `POST /preprocess`: Preprocess uploaded data
  - Parameters: `file_path`, `force_weekly` (optional)
  - Returns: `{shop_csv, category_csv, stats, aggregation_suggestion}`

- `POST /train`: Train the forecasting model
  - Parameters: `shop_csv`, `model_out`, `include_regressors`, `log_transform`, `interval_width`, `holdout_frac`
  - Returns: `{model_path, metrics_path, metrics, train_range, test_range, n_train, n_test}`

- `POST /evaluate`: Run cross-validation evaluation
  - Parameters: `shop_csv`, `initial_days`, `horizon_days`, `period_days`, `include_regressors`, `log_transform`
  - Returns: `{metrics, cv_predictions_csv, n_cv_steps, summary}`

- `POST /predict`: Generate forecast
  - Parameters: `model_path`, `horizon`, `log_transform`, `future_regressor_strategy`, `last_known_regressors_csv`
  - Returns: `{forecast, forecast_csv_path, n_predictions}`

- `GET /forecast/download`: Download forecast as PDF
  - Parameters: `path` (query parameter: path to forecast CSV)
  - Returns: PDF file

- `GET /health`: Health check
  - Returns: `{status: "healthy", message: "..."}`

## Output Files

### Model Files
- `*.pkl`: Trained Prophet model (save with joblib)
- `*_metrics.json`: Training metrics and configuration
  - Contains: MAE, RMSE, MAPE, log_transform flag, interval_width, holdout_frac

### Data Files
- `*_shop.csv`: Shop-level aggregated data
  - Columns: `ds`, `y`, `avg_price`, `avg_discount`, `day_of_week`, `is_weekend`
- `*_category.csv`: Category-level aggregated data
  - Columns: `category`, `ds`, `y`

### Forecast Files
- `forecast_shop.csv`: Generated forecasts
  - Columns: `ds`, `yhat`, `yhat_lower`, `yhat_upper`

### Evaluation Files
- `cv_predictions.csv`: Cross-validation predictions
  - Columns: `ds`, `actual`, `predicted`, `cv_step`

### Reports
- `*_report.pdf`: PDF report with:
  - Page 1: Summary statistics and metrics table
  - Page 2: Fact vs forecast plot
  - Page 3: Residuals plot
  - Page 4: Forecast table preview
- `*_metrics.json`: Metrics saved alongside PDF

## Development Checklist

### Initial Setup
- [ ] Create virtual environment and install dependencies
- [ ] Verify FastAPI server starts: `uvicorn app.main:app --reload --port 8888`
- [ ] Verify Streamlit app starts: `streamlit run streamlit_app/app.py`

### Data Preparation
- [ ] Prepare CSV with required columns: `Sale_Date`, `Product_ID`, `Product_Category`, `Unit_Price`, `Discount`, `Quantity_Sold`
- [ ] Check data quality: minimum 100 rows, no critical NaN values
- [ ] Run preprocessing and verify outputs
- [ ] Check aggregation frequency recommendation (D or W)

### Model Training
- [ ] Decide on parameters based on data characteristics:
  - [ ] Use log_transform if variance increases with magnitude
  - [ ] Use include_regressors if price/discount data is available and reliable
  - [ ] Set interval_width based on risk tolerance (default 0.95)
  - [ ] Set holdout_frac based on dataset size (default 0.2)
- [ ] Train model and verify metrics are reasonable
- [ ] Check training metrics JSON file is created

### Model Evaluation
- [ ] Run cross-validation with appropriate parameters
- [ ] Review CV metrics: low mean and std are good
- [ ] Check CV predictions CSV for any obvious issues
- [ ] Visualize CV results (plot should show predictions following actuals)

### Prediction Generation
- [ ] Generate predictions with correct parameters
- [ ] Verify forecast CSV contains expected number of rows
- [ ] Check forecast values are reasonable (not negative, not extreme)
- [ ] If using regressors, verify regressor strategy is appropriate

### Report Generation
- [ ] Download PDF report
- [ ] Verify PDF contains all expected pages
- [ ] Check graphs are readable and properly formatted
- [ ] Verify metrics match training metrics

### Production Readiness
- [ ] Run smoke test: `bash tests/smoke_test.sh`
- [ ] Verify all endpoints respond correctly
- [ ] Test with real production data (if available)
- [ ] Document any production-specific configuration

## Project Structure

- `app/`: Contains the core application modules
  - `main.py`: FastAPI application with all endpoints
  - `preprocessing.py`: Data parsing, validation, and aggregation
  - `train.py`: Model training with Prophet
  - `predict.py`: Forecast generation
  - `evaluation.py`: Cross-validation utilities
  - `utils.py`: Metrics, PDF export, helper functions
- `streamlit_app/`: Contains the Streamlit UI
  - `app.py`: Main Streamlit application
- `scripts/`: Contains CLI tools
  - `cli_tools.py`: Command-line interface for all operations
- `tests/`: Contains test scripts
  - `smoke_test.sh`: End-to-end API test
- `data/`: Directory for input/output data files
  - `raw/`: Uploaded CSV files
  - `processed/`: Preprocessed and forecast CSV files
- `models/`: Directory for saved models

## Troubleshooting

### Common Issues

**Issue**: "Insufficient data for training"
- **Solution**: Ensure dataset has at least 100 rows and 30+ unique dates

**Issue**: "Model requires regressors but CSV not provided"
- **Solution**: Pass `--regressors_csv` or `last_known_regressors_csv` parameter when predicting

**Issue**: "Test set too small"
- **Solution**: Automatically handled - uses cross-validation instead. Consider reducing `holdout_frac`

**Issue**: "Missing required columns"
- **Solution**: Verify CSV has columns: `Sale_Date`, `Product_ID`, `Product_Category`, `Unit_Price`, `Discount`, `Quantity_Sold`

**Issue**: "Bad metrics (NaN or very large values)"
- **Solution**: Check data quality, try with/without log_transform, verify regressors if used

## License

[Your License Here]
