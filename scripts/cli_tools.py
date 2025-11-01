# file: scripts/cli_tools.py
"""
CLI Tools for Sales Forecasting

This script provides command-line tools for preprocessing data, training models,
evaluating models with cross-validation, and generating predictions.

Usage examples:
    # Preprocess a CSV file
    python scripts/cli_tools.py preprocess --input data/raw/sales.csv --out_shop data/processed/shop.csv --out_cat data/processed/category.csv
    
    # Preprocess with weekly aggregation
    python scripts/cli_tools.py preprocess --input data/raw/sales.csv --out_shop data/processed/shop.csv --out_cat data/processed/category.csv --force_weekly
    
    # Train a model
    python scripts/cli_tools.py train --shop_csv data/processed/shop.csv --model_out models/prophet_model.pkl
    
    # Train with regressors and log-transform
    python scripts/cli_tools.py train --shop_csv data/processed/shop.csv --model_out models/prophet_model.pkl --include_regressors --log_transform --interval_width 0.9 --holdout_frac 0.15
    
    # Generate predictions
    python scripts/cli_tools.py predict --model_path models/prophet_model.pkl --horizon 30 --out_csv data/processed/forecast.csv
    
    # Generate predictions with log-transform
    python scripts/cli_tools.py predict --model_path models/prophet_model.pkl --horizon 30 --out_csv data/processed/forecast.csv --log_transform --regressor_strategy median
    
    # Run cross-validation evaluation
    python scripts/cli_tools.py evaluate --shop_csv data/processed/shop.csv --initial_days 180 --horizon_days 30 --period_days 30
    
    # Full pipeline
    python scripts/cli_tools.py full-pipeline --input data/raw/sales.csv --model_out models/prophet_model.pkl --horizon 30
"""

import argparse
import os
import sys
from app.preprocessing import parse_and_process
from app.train import train_prophet
from app.predict import predict_prophet, save_forecast_csv
from app.evaluation import rolling_cross_validation_prophet, plot_cv_results
import pandas as pd


def print_summary(title: str, info: dict):
    """Print a formatted summary."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)
    for key, value in info.items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="CLI Tools for Sales Forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Preprocess subcommand
    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Preprocess raw CSV file",
        description="Parse and preprocess raw sales data CSV file with validation and aggregation"
    )
    preprocess_parser.add_argument("--input", required=True, help="Path to input CSV file")
    preprocess_parser.add_argument("--out_shop", required=True, help="Output path for shop-level CSV")
    preprocess_parser.add_argument("--out_cat", required=True, help="Output path for category-level CSV")
    preprocess_parser.add_argument(
        "--force_weekly",
        action="store_true",
        help="Force weekly aggregation regardless of data density"
    )

    # Train subcommand
    train_parser = subparsers.add_parser(
        "train",
        help="Train Prophet model",
        description="Train a Prophet model on shop-level sales data"
    )
    train_parser.add_argument("--shop_csv", required=True, help="Path to shop-level CSV file")
    train_parser.add_argument("--model_out", required=True, help="Output path for trained model (.pkl)")
    train_parser.add_argument(
        "--include_regressors",
        action="store_true",
        help="Include avg_price and avg_discount as regressors"
    )
    train_parser.add_argument(
        "--log_transform",
        action="store_true",
        help="Apply log1p transformation to target variable"
    )
    train_parser.add_argument(
        "--interval_width",
        type=float,
        default=0.95,
        help="Confidence interval width (default: 0.95)"
    )
    train_parser.add_argument(
        "--holdout_frac",
        type=float,
        default=0.2,
        help="Fraction of data for testing (default: 0.2)"
    )
    train_parser.add_argument(
        "--changepoint_prior_scale",
        type=float,
        default=0.05,
        help="Flexibility of automatic changepoint detection (default: 0.05, higher = more flexible)"
    )
    train_parser.add_argument(
        "--seasonality_prior_scale",
        type=float,
        default=10.0,
        help="Strength of seasonality components (default: 10.0, higher = stronger seasonality)"
    )
    train_parser.add_argument(
        "--seasonality_mode",
        choices=["additive", "multiplicative"],
        default="additive",
        help="Seasonality mode: additive or multiplicative (default: additive)"
    )

    # Predict subcommand
    predict_parser = subparsers.add_parser(
        "predict",
        help="Generate predictions",
        description="Generate forecast using a trained Prophet model"
    )
    predict_parser.add_argument("--model_path", required=True, help="Path to trained model file (.pkl)")
    predict_parser.add_argument(
        "--horizon",
        type=int,
        required=True,
        help="Number of days to forecast (1-365)"
    )
    predict_parser.add_argument("--out_csv", required=True, help="Output path for forecast CSV")
    predict_parser.add_argument(
        "--log_transform",
        action="store_true",
        help="Apply inverse log1p (expm1) transformation to predictions"
    )
    predict_parser.add_argument(
        "--regressor_strategy",
        choices=["ffill", "median"],
        default="ffill",
        help="Strategy for filling regressors on future dates (default: ffill)"
    )
    predict_parser.add_argument(
        "--regressors_csv",
        type=str,
        default=None,
        help="Path to CSV with regressors (optional, required if model uses regressors)"
    )

    # Evaluate subcommand
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Run cross-validation evaluation",
        description="Perform rolling cross-validation to evaluate model performance"
    )
    evaluate_parser.add_argument("--shop_csv", required=True, help="Path to shop-level CSV file")
    evaluate_parser.add_argument(
        "--initial_days",
        type=int,
        default=180,
        help="Number of days for initial training period (default: 180)"
    )
    evaluate_parser.add_argument(
        "--horizon_days",
        type=int,
        default=30,
        help="Number of days to forecast ahead (default: 30)"
    )
    evaluate_parser.add_argument(
        "--period_days",
        type=int,
        default=30,
        help="Number of days to slide the window forward (default: 30)"
    )
    evaluate_parser.add_argument(
        "--include_regressors",
        action="store_true",
        help="Include avg_price and avg_discount as regressors"
    )
    evaluate_parser.add_argument(
        "--log_transform",
        action="store_true",
        help="Apply log1p transformation to target variable"
    )
    evaluate_parser.add_argument(
        "--out_csv",
        type=str,
        default="data/processed/cv_predictions.csv",
        help="Output path for CV predictions CSV (default: data/processed/cv_predictions.csv)"
    )
    evaluate_parser.add_argument(
        "--out_plot",
        type=str,
        default=None,
        help="Output path for CV visualization plot (optional)"
    )

    # Full pipeline subcommand
    full_parser = subparsers.add_parser(
        "full-pipeline",
        help="Run the complete pipeline",
        description="Run the complete pipeline: preprocess -> train -> predict"
    )
    full_parser.add_argument("--input", required=True, help="Path to input CSV file")
    full_parser.add_argument("--model_out", required=True, help="Output path for trained model")
    full_parser.add_argument("--horizon", type=int, default=30, help="Number of days to forecast (default: 30)")
    full_parser.add_argument("--forecast_csv", default="data/processed/forecast_shop.csv",
                            help="Output path for forecast CSV (default: data/processed/forecast_shop.csv)")
    full_parser.add_argument("--force_weekly", action="store_true", help="Force weekly aggregation")
    full_parser.add_argument("--include_regressors", action="store_true", help="Include regressors")
    full_parser.add_argument("--log_transform", action="store_true", help="Apply log transform")

    args = parser.parse_args()

    if args.command == "preprocess":
        print(f"Preprocessing {args.input}...")
        result = parse_and_process(
            args.input,
            args.out_shop,
            args.out_cat,
            force_weekly=args.force_weekly
        )
        
        print_summary("Preprocessing Completed", {
            "Shop CSV": result['shop_csv'],
            "Category CSV": result['category_csv'],
            "Rows (raw)": result['stats']['n_rows_raw'],
            "Rows (clean)": result['stats']['n_rows_clean'],
            "Unique dates": result['stats']['n_unique_dates'],
            "Frequency": result['stats']['freq_used'],
            "Frequency reason": result['stats']['freq_reason'],
            "Date range": f"{result['stats']['date_min']} to {result['stats']['date_max']}",
            "Duplicates removed": result['stats']['duplicates_removed']
        })

    elif args.command == "train":
        print(f"Training model with {args.shop_csv}...")
        result = train_prophet(
            shop_csv_path=args.shop_csv,
            model_out_path=args.model_out,
            include_regressors=args.include_regressors,
            log_transform=args.log_transform,
            interval_width=args.interval_width,
            holdout_frac=args.holdout_frac,
            changepoint_prior_scale=args.changepoint_prior_scale,
            seasonality_prior_scale=args.seasonality_prior_scale,
            seasonality_mode=args.seasonality_mode
        )
        
        metrics_path = args.model_out.replace('.pkl', '_metrics.json')
        
        print_summary("Training Completed", {
            "Model path": result['model_path'],
            "Metrics path": metrics_path,
            "Training samples": result['n_train'],
            "Test samples": result['n_test'],
            "Train period": f"{result['train_range']['start']} to {result['train_range']['end']}",
            "Test period": f"{result['test_range']['start']} to {result['test_range']['end']}",
            "MAE": f"{result['metrics']['mae']:.4f}",
            "RMSE": f"{result['metrics']['rmse']:.4f}",
            "MAPE": f"{result['metrics']['mape']:.2f}%",
            "Log transform": result['metrics']['log_transform'],
            "Interval width": result['metrics']['interval_width']
        })

    elif args.command == "predict":
        print(f"Generating predictions with model {args.model_path} for {args.horizon} days...")
        df_forecast = predict_prophet(
            model_path=args.model_path,
            horizon_days=args.horizon,
            last_known_regressors_csv=args.regressors_csv,
            log_transform=args.log_transform,
            regressor_fill_method=args.regressor_strategy
        )
        
        # Save to specified output path
        save_forecast_csv(df_forecast, args.out_csv)
        
        print_summary("Prediction Completed", {
            "Forecast CSV": args.out_csv,
            "Forecast shape": df_forecast.shape,
            "Forecast period": f"{df_forecast['ds'].min()} to {df_forecast['ds'].max()}",
            "Mean forecast": f"{df_forecast['yhat'].mean():.2f}",
            "Min forecast": f"{df_forecast['yhat'].min():.2f}",
            "Max forecast": f"{df_forecast['yhat'].max():.2f}",
            "Log transform applied": args.log_transform,
            "Regressor strategy": args.regressor_strategy
        })

    elif args.command == "evaluate":
        print(f"Running cross-validation evaluation with {args.shop_csv}...")
        df = pd.read_csv(args.shop_csv)
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values('ds').reset_index(drop=True)
        
        result = rolling_cross_validation_prophet(
            shop_df=df,
            initial_days=args.initial_days,
            horizon_days=args.horizon_days,
            period_days=args.period_days,
            include_regressors=args.include_regressors,
            log_transform=args.log_transform
        )
        
        # Save predictions
        result['predictions_df'].to_csv(args.out_csv, index=False)
        
        # Plot if requested
        if args.out_plot:
            plot_cv_results(result['predictions_df'], args.out_plot)
        
        metrics = result['metrics']
        
        print_summary("Cross-Validation Completed", {
            "CV predictions CSV": args.out_csv,
            "CV plot": args.out_plot if args.out_plot else "Not generated",
            "Number of CV steps": metrics['n_cv_steps'],
            "MAE (mean ± std)": f"{metrics['mae']['mean']:.2f} ± {metrics['mae']['std']:.2f}",
            "RMSE (mean ± std)": f"{metrics['rmse']['mean']:.2f} ± {metrics['rmse']['std']:.2f}",
            "MAPE (mean ± std)": f"{metrics['mape']['mean']:.2f}% ± {metrics['mape']['std']:.2f}%",
            "MAE range": f"{metrics['mae']['min']:.2f} - {metrics['mae']['max']:.2f}",
            "RMSE range": f"{metrics['rmse']['min']:.2f} - {metrics['rmse']['max']:.2f}",
            "MAPE range": f"{metrics['mape']['min']:.2f}% - {metrics['mape']['max']:.2f}%",
            "Log transform": metrics['log_transform'],
            "Include regressors": metrics['include_regressors']
        })

    elif args.command == "full-pipeline":
        print("Running full pipeline...")
        
        # Create output directories
        os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
        os.makedirs(os.path.dirname(args.forecast_csv), exist_ok=True)
        
        # Generate output paths for intermediate files
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        shop_csv_path = os.path.join(os.path.dirname(args.model_out), f"{base_name}_shop.csv")
        category_csv_path = os.path.join(os.path.dirname(args.model_out), f"{base_name}_category.csv")
        
        # Step 1: Preprocess
        print(f"Step 1: Preprocessing {args.input}...")
        preprocess_result = parse_and_process(
            args.input,
            shop_csv_path,
            category_csv_path,
            force_weekly=args.force_weekly
        )
        print(f"✓ Preprocessing completed: {preprocess_result['shop_csv']}")
        
        # Step 2: Train
        print(f"Step 2: Training model with {shop_csv_path}...")
        train_result = train_prophet(
            shop_csv_path=shop_csv_path,
            model_out_path=args.model_out,
            include_regressors=args.include_regressors,
            log_transform=args.log_transform,
            interval_width=0.95,
            holdout_frac=0.2,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            seasonality_mode='additive'
        )
        print(f"✓ Training completed: {train_result['model_path']}")
        print(f"  Metrics: MAE={train_result['metrics']['mae']:.2f}, "
              f"RMSE={train_result['metrics']['rmse']:.2f}, "
              f"MAPE={train_result['metrics']['mape']:.2f}%")
        
        # Step 3: Predict
        print(f"Step 3: Generating predictions for {args.horizon} days...")
        df_forecast = predict_prophet(
            model_path=args.model_out,
            horizon_days=args.horizon,
            log_transform=args.log_transform,
            regressor_fill_method='forward'
        )
        save_forecast_csv(df_forecast, args.forecast_csv)
        print(f"✓ Predictions saved to: {args.forecast_csv}")
        
        print_summary("Full Pipeline Completed", {
            "Input file": args.input,
            "Shop CSV": preprocess_result['shop_csv'],
            "Category CSV": preprocess_result['category_csv'],
            "Model": train_result['model_path'],
            "Metrics JSON": args.model_out.replace('.pkl', '_metrics.json'),
            "Forecast CSV": args.forecast_csv,
            "Forecast shape": df_forecast.shape
        })

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
