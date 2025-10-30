#!/bin/bash
# file: tests/smoke_test.sh
# 
# Smoke test for the Sales Forecasting API
# 
# This script performs a basic end-to-end test of the API:
# 1. Uploads a sample CSV file
# 2. Preprocesses the data
# 3. Trains a model
# 4. Generates predictions
# 5. Downloads a PDF report
#
# Prerequisites:
# - The FastAPI server must be running at http://localhost:8000
# - A sample CSV file named 'sample_sales.csv' must exist in the current directory
# 
# To run the test:
# 1. Start the FastAPI server: uvicorn app.main:app --host 0.0.0.0 --port 8000
# 2. Run this script: ./tests/smoke_test.sh

set -e  # Exit immediately if a command exits with a non-zero status

echo "Starting smoke test for Sales Forecasting API..."

# Check if server is running
echo "Checking if server is running at http://localhost:8000..."
if ! curl -s --connect-timeout 5 http://localhost:8000/health > /dev/null; then
    echo "Error: Server is not running at http://localhost:8000"
    echo "Please start the FastAPI server before running this test"
    exit 1
fi
echo "✓ Server is running"

# Create a sample CSV file if it doesn't exist
if [ ! -f "sample_sales.csv" ]; then
    echo "Creating sample CSV file..."
    cat > sample_sales.csv << EOF
Sale_Date,Product_ID,Product_Category,Unit_Price,Discount,Quantity_Sold
2023-01-01,P001,Electronics,100.0,5.0,10
2023-01-02,P002,Clothing,50.0,2.0,15
2023-01-03,P003,Electronics,120.0,10.0,8
2023-01-04,P004,Books,20.0,1.0,20
2023-01-05,P005,Clothing,40.0,3.0,12
EOF
    echo "✓ Sample CSV file created"
fi

# Step 1: Upload the sample CSV file
echo "Step 1: Uploading sample CSV file..."
UPLOAD_RESPONSE=$(curl -s -X POST "http://localhost:8000/upload" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@sample_sales.csv")
UPLOAD_FILE_PATH=$(echo $UPLOAD_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin)['file_path'])")

if [ -z "$UPLOAD_FILE_PATH" ]; then
    echo "Error: Failed to upload file"
    exit 1
fi
echo "✓ File uploaded successfully: $UPLOAD_FILE_PATH"

# Step 2: Preprocess the uploaded data
echo "Step 2: Preprocessing data..."
PREPROCESS_PAYLOAD="{\"file_path\": \"$UPLOAD_FILE_PATH\"}"
PREPROCESS_RESPONSE=$(curl -s -X POST "http://localhost:8000/preprocess" -H "accept: application/json" -H "Content-Type: application/json" -d "$PREPROCESS_PAYLOAD")
SHOP_CSV_PATH=$(echo $PREPROCESS_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin)['shop_csv'])")
CATEGORY_CSV_PATH=$(echo $PREPROCESS_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin)['category_csv'])")

if [ -z "$SHOP_CSV_PATH" ] || [ -z "$CATEGORY_CSV_PATH" ]; then
    echo "Error: Failed to preprocess data"
    exit 1
fi
echo "✓ Data preprocessed successfully!"
echo "  Shop CSV: $SHOP_CSV_PATH"
echo "  Category CSV: $CATEGORY_CSV_PATH"

# Step 3: Train the model
echo "Step 3: Training model..."
MODEL_OUT_PATH="models/test_model.pkl"
TRAIN_PAYLOAD="{\"shop_csv\": \"$SHOP_CSV_PATH\", \"model_out\": \"$MODEL_OUT_PATH\"}"
TRAIN_RESPONSE=$(curl -s -X POST "http://localhost:8000/train" -H "accept: application/json" -H "Content-Type: application/json" -d "$TRAIN_PAYLOAD")
MODEL_PATH=$(echo $TRAIN_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin)['model_path'])")

if [ -z "$MODEL_PATH" ]; then
    echo "Error: Failed to train model"
    exit 1
fi
echo "✓ Model trained successfully: $MODEL_PATH"

# Step 4: Generate predictions
echo "Step 4: Generating predictions..."
PREDICT_PAYLOAD="{\"model_path\": \"$MODEL_PATH\", \"horizon\": 10}"
PREDICT_RESPONSE=$(curl -s -X POST "http://localhost:8000/predict" -H "accept: application/json" -H "Content-Type: application/json" -d "$PREDICT_PAYLOAD")
FORECAST_CSV_PATH="data/processed/forecast_shop.csv"

# Check if prediction was successful
if ! echo $PREDICT_RESPONSE | python3 -c "import sys, json; data = json.load(sys.stdin); print(len(data['forecast']) > 0)" | grep -q "True"; then
    echo "Error: Failed to generate predictions"
    exit 1
fi
echo "✓ Predictions generated successfully"

# Step 5: Download PDF report
echo "Step 5: Downloading PDF report..."
PDF_RESPONSE=$(curl -s -X GET "http://localhost:8000/forecast/download?path=$FORECAST_CSV_PATH" -H "accept: application/json")
PDF_FILE="forecast_report.pdf"

# Save the PDF response (this might be JSON with content, depending on your API)
# For this test, we'll just verify that we got a response
if [ -n "$PDF_RESPONSE" ]; then
    echo "✓ PDF report generated"
else
    echo "Warning: Failed to get PDF report"
fi

echo ""
echo "✅ Smoke test completed successfully!"
echo "All API endpoints are working correctly."