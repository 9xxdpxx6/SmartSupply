# file: tests/smoke_test.sh
#!/bin/bash
#
# Smoke test for the Sales Forecasting API
# 
# This script performs a comprehensive end-to-end test of the API:
# 1. Uploads a sample CSV file
# 2. Preprocesses the data
# 3. Trains a model with advanced parameters (regressors, log-transform)
# 4. Evaluates the model with cross-validation
# 5. Generates predictions with regressor strategy
# 6. Downloads a PDF report
#
# Prerequisites:
# - The FastAPI server must be running at http://localhost:8888
# - Python 3 with requests library (or curl with jq for JSON parsing)
# - A sample CSV file named 'sample_sales.csv' must exist in the current directory
#   (or the script will create one)
# 
# To run the test:
# 1. Start the FastAPI server: uvicorn app.main:app --host 0.0.0.0 --port 8888
# 2. Run this script: bash tests/smoke_test.sh
#
# What to check:
# - Each step should return HTTP 200 status
# - Files should be created in data/raw, data/processed, models/
# - Metrics should be reasonable (not NaN or extremely large values)
# - PDF should be generated successfully
#

set -e  # Exit immediately if a command exits with a non-zero status

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Sales Forecasting API - Smoke Test"
echo "=========================================="
echo ""

# Check if server is running
echo -e "${YELLOW}Step 0: Checking if server is running at http://localhost:8888...${NC}"
if ! curl -s --connect-timeout 5 http://localhost:8888/health > /dev/null; then
    echo -e "${RED}Error: Server is not running at http://localhost:8888${NC}"
    echo "Please start the FastAPI server before running this test:"
    echo "  uvicorn app.main:app --host 0.0.0.0 --port 8888"
    exit 1
fi
echo -e "${GREEN}✓ Server is running${NC}"
echo ""

# Create a sample CSV file if it doesn't exist
if [ ! -f "sample_sales.csv" ]; then
    echo -e "${YELLOW}Creating sample CSV file...${NC}"
    cat > sample_sales.csv << EOF
Sale_Date,Product_ID,Product_Category,Unit_Price,Discount,Quantity_Sold
2023-01-01,P001,Electronics,100.0,5.0,10
2023-01-02,P002,Clothing,50.0,2.0,15
2023-01-03,P003,Electronics,120.0,10.0,8
2023-01-04,P004,Books,20.0,1.0,20
2023-01-05,P005,Clothing,40.0,3.0,12
2023-01-06,P006,Electronics,110.0,8.0,9
2023-01-07,P007,Books,25.0,2.0,18
2023-01-08,P008,Clothing,45.0,4.0,14
2023-01-09,P009,Electronics,105.0,6.0,11
2023-01-10,P010,Books,22.0,1.5,22
EOF
    # Add more dates to ensure we have enough data for training
    for i in {11..200}; do
        date=$(date -d "2023-01-01 +${i} days" +%Y-%m-%d 2>/dev/null || date -v+${i}d -j -f "%Y-%m-%d" "2023-01-01" +%Y-%m-%d 2>/dev/null || echo "2023-01-$(printf "%02d" $i)")
        echo "${date},P$(printf "%03d" $((i % 50 + 1))),Electronics,$((90 + i % 30)).0,$((3 + i % 10)).0,$((5 + i % 20))"
    done >> sample_sales.csv
    echo -e "${GREEN}✓ Sample CSV file created (sample_sales.csv)${NC}"
fi

# Step 1: Upload the sample CSV file
echo -e "${YELLOW}Step 1: Uploading sample CSV file...${NC}"
UPLOAD_RESPONSE=$(curl -s -X POST "http://localhost:8888/upload" \
    -H "accept: application/json" \
    -H "Content-Type: multipart/form-data" \
    -F "file=@sample_sales.csv")

# Check if upload was successful
if echo "$UPLOAD_RESPONSE" | grep -q "file_path"; then
    UPLOAD_FILE_PATH=$(echo $UPLOAD_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin)['file_path'])" 2>/dev/null || echo "data/raw/sample_sales.csv")
    echo -e "${GREEN}✓ File uploaded successfully: $UPLOAD_FILE_PATH${NC}"
else
    echo -e "${RED}Error: Failed to upload file${NC}"
    echo "Response: $UPLOAD_RESPONSE"
    exit 1
fi
echo ""

# Step 2: Preprocess the uploaded data
echo -e "${YELLOW}Step 2: Preprocessing data...${NC}"
PREPROCESS_PAYLOAD=$(cat <<EOF
{
    "file_path": "$UPLOAD_FILE_PATH",
    "force_weekly": false
}
EOF
)
PREPROCESS_RESPONSE=$(curl -s -X POST "http://localhost:8888/preprocess" \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d "$PREPROCESS_PAYLOAD")

# Extract paths from response
SHOP_CSV_PATH=$(echo $PREPROCESS_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin)['shop_csv'])" 2>/dev/null || echo "")
CATEGORY_CSV_PATH=$(echo $PREPROCESS_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin)['category_csv'])" 2>/dev/null || echo "")

if [ -z "$SHOP_CSV_PATH" ] || [ -z "$CATEGORY_CSV_PATH" ]; then
    echo -e "${RED}Error: Failed to preprocess data${NC}"
    echo "Response: $PREPROCESS_RESPONSE"
    exit 1
fi
echo -e "${GREEN}✓ Data preprocessed successfully!${NC}"
echo "  Shop CSV: $SHOP_CSV_PATH"
echo "  Category CSV: $CATEGORY_CSV_PATH"

# Extract and display stats
AGGREGATION_SUGGESTION=$(echo $PREPROCESS_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin).get('aggregation_suggestion', {}).get('freq', 'N/A'))" 2>/dev/null || echo "N/A")
echo "  Aggregation suggestion: $AGGREGATION_SUGGESTION"
echo ""

# Step 3: Train the model with advanced parameters
echo -e "${YELLOW}Step 3: Training model with advanced parameters (regressors, log-transform)...${NC}"
MODEL_OUT_PATH="models/test_model.pkl"
TRAIN_PAYLOAD=$(cat <<EOF
{
    "shop_csv": "$SHOP_CSV_PATH",
    "model_out": "$MODEL_OUT_PATH",
    "include_regressors": true,
    "log_transform": true,
    "interval_width": 0.9,
    "holdout_frac": 0.2
}
EOF
)
TRAIN_RESPONSE=$(curl -s -X POST "http://localhost:8888/train" \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d "$TRAIN_PAYLOAD")

MODEL_PATH=$(echo $TRAIN_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin)['model_path'])" 2>/dev/null || echo "")

if [ -z "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Failed to train model${NC}"
    echo "Response: $TRAIN_RESPONSE"
    exit 1
fi

# Extract and display metrics
MAE=$(echo $TRAIN_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin)['metrics']['mae'])" 2>/dev/null || echo "N/A")
RMSE=$(echo $TRAIN_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin)['metrics']['rmse'])" 2>/dev/null || echo "N/A")
MAPE=$(echo $TRAIN_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin)['metrics']['mape'])" 2>/dev/null || echo "N/A")

echo -e "${GREEN}✓ Model trained successfully: $MODEL_PATH${NC}"
echo "  Metrics:"
echo "    MAE:  $MAE"
echo "    RMSE: $RMSE"
echo "    MAPE: $MAPE%"
echo ""

# Step 4: Evaluate model with cross-validation
echo -e "${YELLOW}Step 4: Running cross-validation evaluation...${NC}"
EVALUATE_PAYLOAD=$(cat <<EOF
{
    "shop_csv": "$SHOP_CSV_PATH",
    "initial_days": 180,
    "horizon_days": 30,
    "period_days": 30,
    "include_regressors": true,
    "log_transform": true
}
EOF
)
EVALUATE_RESPONSE=$(curl -s -X POST "http://localhost:8888/evaluate" \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d "$EVALUATE_PAYLOAD")

# Extract CV results
CV_MAE_MEAN=$(echo $EVALUATE_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin)['summary']['mae_mean'])" 2>/dev/null || echo "N/A")
CV_RMSE_MEAN=$(echo $EVALUATE_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin)['summary']['rmse_mean'])" 2>/dev/null || echo "N/A")
CV_N_STEPS=$(echo $EVALUATE_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin)['n_cv_steps'])" 2>/dev/null || echo "N/A")
CV_PREDICTIONS_CSV=$(echo $EVALUATE_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin)['cv_predictions_csv'])" 2>/dev/null || echo "")

if [ -z "$CV_PREDICTIONS_CSV" ]; then
    echo -e "${RED}Error: Failed to run cross-validation${NC}"
    echo "Response: $EVALUATE_RESPONSE"
    exit 1
fi

echo -e "${GREEN}✓ Cross-validation completed successfully!${NC}"
echo "  CV Steps: $CV_N_STEPS"
echo "  CV MAE (mean): $CV_MAE_MEAN"
echo "  CV RMSE (mean): $CV_RMSE_MEAN"
echo "  CV Predictions CSV: $CV_PREDICTIONS_CSV"
echo ""

# Step 5: Generate predictions
echo -e "${YELLOW}Step 5: Generating predictions with regressor strategy...${NC}"
PREDICT_PAYLOAD=$(cat <<EOF
{
    "model_path": "$MODEL_PATH",
    "horizon": 30,
    "log_transform": true,
    "future_regressor_strategy": "ffill",
    "last_known_regressors_csv": "$SHOP_CSV_PATH"
}
EOF
)
PREDICT_RESPONSE=$(curl -s -X POST "http://localhost:8888/predict" \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d "$PREDICT_PAYLOAD")

FORECAST_CSV_PATH=$(echo $PREDICT_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin)['forecast_csv_path'])" 2>/dev/null || echo "data/processed/forecast_shop.csv")
N_PREDICTIONS=$(echo $PREDICT_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin)['n_predictions'])" 2>/dev/null || echo "N/A")

if [ "$N_PREDICTIONS" = "N/A" ] || [ -z "$FORECAST_CSV_PATH" ]; then
    echo -e "${RED}Error: Failed to generate predictions${NC}"
    echo "Response: $PREDICT_RESPONSE"
    exit 1
fi

echo -e "${GREEN}✓ Predictions generated successfully!${NC}"
echo "  Forecast CSV: $FORECAST_CSV_PATH"
echo "  Number of predictions: $N_PREDICTIONS"
echo ""

# Step 6: Download PDF report
echo -e "${YELLOW}Step 6: Downloading PDF report...${NC}"
PDF_OUTPUT="forecast_report.pdf"
PDF_STATUS=$(curl -s -w "%{http_code}" -o "$PDF_OUTPUT" \
    "http://localhost:8888/forecast/download?path=$FORECAST_CSV_PATH" \
    -H "accept: application/pdf")

if [ "$PDF_STATUS" = "200" ] && [ -f "$PDF_OUTPUT" ]; then
    PDF_SIZE=$(stat -f%z "$PDF_OUTPUT" 2>/dev/null || stat -c%s "$PDF_OUTPUT" 2>/dev/null || echo "unknown")
    echo -e "${GREEN}✓ PDF report downloaded successfully!${NC}"
    echo "  File: $PDF_OUTPUT"
    echo "  Size: $PDF_SIZE bytes"
else
    echo -e "${RED}Error: Failed to download PDF report (HTTP $PDF_STATUS)${NC}"
    exit 1
fi
echo ""

# Summary
echo "=========================================="
echo -e "${GREEN}✅ Smoke test completed successfully!${NC}"
echo "=========================================="
echo ""
echo "Summary of generated files:"
echo "  📄 Uploaded file: $UPLOAD_FILE_PATH"
echo "  📊 Shop CSV: $SHOP_CSV_PATH"
echo "  📊 Category CSV: $CATEGORY_CSV_PATH"
echo "  🤖 Model: $MODEL_PATH"
echo "  📈 CV Predictions: $CV_PREDICTIONS_CSV"
echo "  🔮 Forecast: $FORECAST_CSV_PATH"
echo "  📄 PDF Report: $PDF_OUTPUT"
echo ""
echo "Next steps to verify:"
echo "  1. Check that all files exist and have reasonable sizes"
echo "  2. Verify model metrics are reasonable (MAE, RMSE, MAPE not NaN or extremely large)"
echo "  3. Open the PDF report and verify it contains graphs and tables"
echo "  4. Check CV predictions CSV has actual vs predicted values"
echo ""
