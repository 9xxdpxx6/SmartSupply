# file: scripts/cli_tools.py
"""
CLI Tools for Sales Forecasting

This script provides command-line tools for preprocessing data, training models,
and generating predictions using the local modules.

Usage examples:
    # Preprocess a CSV file
    python cli_tools.py preprocess data/sales.csv data/processed/shop.csv data/processed/category.csv

    # Train a model
    python cli_tools.py train data/processed/shop.csv models/prophet_model.pkl

    # Generate predictions
    python cli_tools.py predict models/prophet_model.pkl 30 data/forecast.csv

    # Full pipeline
    python cli_tools.py full-pipeline data/sales.csv models/prophet_model.pkl 30 data/forecast.csv
"""

import argparse
import os
from app.preprocessing import parse_and_process
from app.train import train_prophet
from app.predict import predict_prophet, save_forecast_csv


def main():
    parser = argparse.ArgumentParser(description="CLI Tools for Sales Forecasting")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Preprocess subcommand
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess raw CSV file")
    preprocess_parser.add_argument("input_csv", help="Path to input CSV file")
    preprocess_parser.add_argument("out_shop_csv", help="Output path for shop-level CSV")
    preprocess_parser.add_argument("out_category_csv", help="Output path for category-level CSV")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train Prophet model")
    train_parser.add_argument("shop_csv", help="Path to shop-level CSV file")
    train_parser.add_argument("model_out", help="Output path for trained model")

    # Predict subcommand
    predict_parser = subparsers.add_parser("predict", help="Generate predictions")
    predict_parser.add_argument("model_path", help="Path to trained model file")
    predict_parser.add_argument("horizon", type=int, help="Number of days to forecast")
    predict_parser.add_argument("output_csv", help="Output path for forecast CSV")

    # Full pipeline subcommand
    full_parser = subparsers.add_parser("full-pipeline", help="Run the complete pipeline")
    full_parser.add_argument("input_csv", help="Path to input CSV file")
    full_parser.add_argument("model_out", help="Output path for trained model")
    full_parser.add_argument("horizon", type=int, help="Number of days to forecast")
    full_parser.add_argument("output_csv", help="Output path for forecast CSV")

    args = parser.parse_args()

    if args.command == "preprocess":
        print(f"Preprocessing {args.input_csv}...")
        result = parse_and_process(args.input_csv, args.out_shop_csv, args.out_category_csv)
        print(f"Preprocessing completed!")
        print(f"Shop CSV: {result['shop_csv']}")
        print(f"Category CSV: {result['category_csv']}")
        print(f"Stats: {result['stats']}")

    elif args.command == "train":
        print(f"Training model with {args.shop_csv}...")
        result = train_prophet(args.shop_csv, args.model_out, include_regressors=False)
        print(f"Training completed!")
        print(f"Model saved to: {result['model_path']}")
        print(f"Backtest metrics: {result['backtest_metrics']}")

    elif args.command == "predict":
        print(f"Generating predictions with model {args.model_path} for {args.horizon} days...")
        df_forecast = predict_prophet(args.model_path, args.horizon)
        save_forecast_csv(df_forecast, args.output_csv)
        print(f"Predictions saved to: {args.output_csv}")
        print(f"Forecast shape: {df_forecast.shape}")

    elif args.command == "full-pipeline":
        print("Running full pipeline...")
        
        # Create output directories
        os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        
        # Generate output paths for intermediate files
        base_name = os.path.splitext(os.path.basename(args.input_csv))[0]
        shop_csv_path = os.path.join(os.path.dirname(args.model_out), f"{base_name}_shop.csv")
        category_csv_path = os.path.join(os.path.dirname(args.model_out), f"{base_name}_category.csv")
        
        # Preprocess
        print(f"Step 1: Preprocessing {args.input_csv}...")
        preprocess_result = parse_and_process(args.input_csv, shop_csv_path, category_csv_path)
        print("Preprocessing completed!")
        
        # Train
        print(f"Step 2: Training model with {shop_csv_path}...")
        train_result = train_prophet(shop_csv_path, args.model_out, include_regressors=False)
        print("Training completed!")
        
        # Predict
        print(f"Step 3: Generating predictions for {args.horizon} days...")
        df_forecast = predict_prophet(args.model_out, args.horizon)
        save_forecast_csv(df_forecast, args.output_csv)
        print(f"Predictions saved to: {args.output_csv}")
        
        print("Full pipeline completed successfully!")
        print(f"Model saved to: {args.model_out}")
        print(f"Forecast saved to: {args.output_csv}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()