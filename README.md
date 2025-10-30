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
uvicorn app.main:app --reload
```
The API will be available at `http://localhost:8000`.

### Streamlit Frontend
To start the Streamlit application:
```bash
streamlit run streamlit_app/app.py
```
The UI will be available at `http://localhost:8000` (typically `http://localhost:8501`).

## Example Workflow

1. **Prepare your data**: Ensure your CSV file has the following columns:
   - `Sale_Date`: Date of the sale
   - `Product_ID`: Unique identifier for the product
   - `Product_Category`: Category of the product
   - `Unit_Price`: Price of the unit
   - `Discount`: Discount applied
   - `Quantity_Sold`: Quantity sold

2. **Run preprocessing**:
   ```bash
   python scripts/cli_tools.py preprocess data/sales.csv data/processed/shop.csv data/processed/category.csv
   ```

3. **Train the model**:
   ```bash
   python scripts/cli_tools.py train data/processed/shop.csv models/prophet_model.pkl
   ```

4. **Generate predictions**:
   ```bash
   python scripts/cli_tools.py predict models/prophet_model.pkl 30 data/forecast.csv
   ```

5. **Or run the full pipeline**:
   ```bash
   python scripts/cli_tools.py full-pipeline data/sales.csv models/prophet_model.pkl 30 data/forecast.csv
   ```

## API Endpoints

- `POST /upload`: Upload sales CSV file
- `POST /preprocess`: Preprocess uploaded data
- `POST /train`: Train the forecasting model
- `POST /predict`: Generate forecast
- `GET /forecast/download`: Download forecast as PDF
- `GET /health`: Health check

## Project Structure

- `app/`: Contains the core application modules
- `streamlit_app/`: Contains the Streamlit UI
- `scripts/`: Contains CLI tools
- `tests/`: Contains test scripts
- `data/`: Directory for input/output data files
- `models/`: Directory for saved models