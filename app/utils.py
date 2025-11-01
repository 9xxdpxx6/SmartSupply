# file: app/utils.py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import os


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        
    Returns:
        MAE value
    """
    return mean_absolute_error(actual, predicted)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error.
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        
    Returns:
        RMSE value
    """
    return np.sqrt(mean_squared_error(actual, predicted))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        
    Returns:
        MAPE value as percentage
    """
    # Handle potential division by zero
    mask = actual != 0
    if mask.any():
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    else:
        return 0.0


def save_metrics_json(metrics: dict, path: str) -> None:
    """
    Save metrics dictionary to JSON file.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        path: Path where the JSON file will be saved
    """
    # Ensure output directory exists
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    # Save to JSON
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)


def export_report_pdf(
    output_pdf_path: str, 
    df_history: pd.DataFrame, 
    df_forecast: pd.DataFrame, 
    metrics: dict,
    n_preview: int = 30
) -> None:
    """
    Export forecast report as a multi-page PDF.
    
    Args:
        output_pdf_path: Path to save the PDF report
        df_history: Historical data with columns 'ds' and 'y'
        df_forecast: Forecast data with columns 'ds', 'yhat', 'yhat_lower', 'yhat_upper'
                    May include both historical predictions and future forecast
        metrics: Dictionary containing evaluation metrics (mae, rmse, mape, log_transform, etc.)
        n_preview: Number of forecast rows to show in the preview table (default: 30)
    """
    # Ensure output directory exists
    out_dir = os.path.dirname(output_pdf_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    # Extract log_transform info if available
    log_transform_applied = metrics.get('log_transform', False)
    
    # Prepare data
    df_history = df_history.copy()
    if not df_history.empty:
        df_history['ds'] = pd.to_datetime(df_history['ds'])
        df_history = df_history.sort_values('ds')
        df_history = df_history.dropna(subset=['ds'])  # Remove rows with NaT dates
    
    df_forecast = df_forecast.copy()
    df_forecast['ds'] = pd.to_datetime(df_forecast['ds'])
    df_forecast = df_forecast.sort_values('ds')
    df_forecast = df_forecast.dropna(subset=['ds'])  # Remove rows with NaT dates
    
    # Separate historical predictions and future forecast
    if not df_history.empty:
        hist_end = df_history['ds'].max()
        if pd.isna(hist_end):
            hist_end = None
    else:
        hist_end = None
    
    if hist_end is not None and not pd.isna(hist_end):
        df_forecast_historical = df_forecast[df_forecast['ds'] <= hist_end].copy()
        df_forecast_future = df_forecast[df_forecast['ds'] > hist_end].copy()
    else:
        # No history, treat all forecast as future
        df_forecast_historical = pd.DataFrame()
        df_forecast_future = df_forecast.copy()
    
    # Calculate statistics
    if not df_history.empty:
        hist_start = df_history['ds'].min()
        hist_end_date = df_history['ds'].max()
        if pd.isna(hist_start) or pd.isna(hist_end_date):
            hist_start = None
            hist_end_date = None
            total_days = 0
        else:
            total_days = (hist_end_date - hist_start).days + 1
        total_sales = df_history['y'].sum() if 'y' in df_history.columns else 0
        avg_per_day = total_sales / total_days if total_days > 0 else 0
    else:
        hist_start = None
        hist_end_date = None
        total_days = 0
        total_sales = 0
        avg_per_day = 0
    
    with PdfPages(output_pdf_path) as pdf:
        # Page 1: Summary text with metrics table
        fig1, ax1 = plt.subplots(figsize=(8.5, 11))
        ax1.axis('off')
        
        # Create summary text
        summary_lines = [
            "Sales Forecasting Report",
            "",
            "=" * 60,
            "",
            "Historical Data Period:",
        ]
        
        if hist_start is not None and hist_end_date is not None and not pd.isna(hist_start) and not pd.isna(hist_end_date):
            summary_lines.extend([
                f"  Start: {hist_start.strftime('%Y-%m-%d')}",
                f"  End:   {hist_end_date.strftime('%Y-%m-%d')}",
                f"  Total Days: {total_days}",
            ])
        else:
            summary_lines.append("  No historical data available")
        
        summary_lines.extend([
            "",
            "Sales Statistics:",
            f"  Total Sales:    {total_sales:,.2f}",
            f"  Average/Day:     {avg_per_day:,.2f}",
            "",
            "Forecast Horizon:",
            f"  Future Period: {len(df_forecast_future)} days",
        ])
        
        # Add metrics table
        summary_lines.extend([
            "",
            "Performance Metrics:",
            "-" * 60,
        ])
        
        # Format metrics
        mae_val = metrics.get('mae', 'N/A')
        rmse_val = metrics.get('rmse', 'N/A')
        mape_val = metrics.get('mape', 'N/A')
        
        if isinstance(mae_val, (int, float)):
            summary_lines.append(f"  MAE:  {mae_val:,.2f}")
        else:
            summary_lines.append(f"  MAE:  {mae_val}")
            
        if isinstance(rmse_val, (int, float)):
            summary_lines.append(f"  RMSE: {rmse_val:,.2f}")
        else:
            summary_lines.append(f"  RMSE: {rmse_val}")
            
        if isinstance(mape_val, (int, float)):
            summary_lines.append(f"  MAPE: {mape_val:.2f}%")
        else:
            summary_lines.append(f"  MAPE: {mape_val}")
        
        # Add model details
        summary_lines.extend([
            "",
            "Model Configuration:",
            "-" * 60,
        ])
        
        if log_transform_applied:
            summary_lines.append("  Log Transformation: Applied (log1p/expm1)")
        else:
            summary_lines.append("  Log Transformation: Not applied")
        
        interval_width = metrics.get('interval_width', 0.95)
        summary_lines.append(f"  Confidence Interval: {interval_width * 100:.0f}%")
        
        summary_text = "\n".join(summary_lines)
        
        ax1.text(0.1, 0.95, summary_text, transform=ax1.transAxes, fontsize=11, 
                 verticalalignment='top', fontfamily='monospace')
        
        pdf.savefig(fig1, bbox_inches='tight')
        plt.close(fig1)
        
        # Page 2: Line plot of actual vs forecast
        fig2, ax2 = plt.subplots(figsize=(12, 7))
        
        # Plot historical data
        if not df_history.empty:
            ax2.plot(df_history['ds'], df_history['y'], 
                    label='Historical Sales', color='blue', linewidth=2, alpha=0.7)
        
        # Plot historical predictions if available
        if not df_forecast_historical.empty:
            ax2.plot(df_forecast_historical['ds'], df_forecast_historical['yhat'], 
                    label='Historical Predictions', color='green', linewidth=1.5, linestyle='--', alpha=0.8)
        
        # Plot future forecast
        if not df_forecast_future.empty:
            ax2.plot(df_forecast_future['ds'], df_forecast_future['yhat'], 
                    label='Future Forecast', color='red', linewidth=2)
            
            # Plot confidence intervals for future forecast
            if 'yhat_lower' in df_forecast_future.columns and 'yhat_upper' in df_forecast_future.columns:
                ax2.fill_between(df_forecast_future['ds'], 
                                df_forecast_future['yhat_lower'], 
                                df_forecast_future['yhat_upper'], 
                                color='red', alpha=0.2, label=f'{interval_width*100:.0f}% Confidence Interval')
        
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Sales', fontsize=12, fontweight='bold')
        
        title = 'Historical Sales vs Forecast'
        if log_transform_applied:
            title += ' (Log Transform Applied)'
        ax2.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis to show dates nicely
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        pdf.savefig(fig2, bbox_inches='tight')
        plt.close(fig2)
        
        # Page 3: Residuals plot
        fig3, ax3 = plt.subplots(figsize=(12, 7))
        
        if not df_forecast_historical.empty:
            # Align historical predictions with actual values
            merged = df_forecast_historical[['ds', 'yhat']].merge(
                df_history[['ds', 'y']],
                on='ds',
                how='inner'
            )
            
            if not merged.empty:
                residuals = merged['y'] - merged['yhat']
                
                # Plot residuals over time
                ax3.plot(merged['ds'], residuals, 'o-', color='purple', 
                        markersize=4, linewidth=1, alpha=0.6, label='Residuals')
                
                # Add zero line
                ax3.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
                
                # Add mean residual line
                mean_residual = residuals.mean()
                ax3.axhline(y=mean_residual, color='red', linestyle='--', 
                           linewidth=1.5, alpha=0.7, label=f'Mean Residual: {mean_residual:.2f}')
                
                ax3.set_xlabel('Date', fontsize=12, fontweight='bold')
                ax3.set_ylabel('Residual (Actual - Predicted)', fontsize=12, fontweight='bold')
                ax3.set_title('Residuals Plot\n(Historical Predictions)', fontsize=14, fontweight='bold', pad=15)
                
                # Add explanation text
                explanation = (
                    "Residuals = Actual Sales - Predicted Sales\n"
                    "Positive values indicate underestimation, negative values indicate overestimation.\n"
                    "Ideally, residuals should be randomly distributed around zero."
                )
                ax3.text(0.02, 0.98, explanation, transform=ax3.transAxes, 
                        fontsize=9, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                ax3.legend(loc='best', fontsize=10)
                ax3.grid(True, alpha=0.3)
                
                # Format x-axis
                ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
                plt.xticks(rotation=45, ha='right')
            else:
                ax3.text(0.5, 0.5, 'No historical predictions available\nfor residuals calculation', 
                        transform=ax3.transAxes, ha='center', va='center', fontsize=12)
                ax3.set_title('Residuals Plot', fontsize=14, fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No historical predictions available\nfor residuals calculation', 
                    transform=ax3.transAxes, ha='center', va='center', fontsize=12)
            ax3.set_title('Residuals Plot', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        pdf.savefig(fig3, bbox_inches='tight')
        plt.close(fig3)
        
        # Page 4: Table with first n_preview forecast rows
        fig4, ax4 = plt.subplots(figsize=(12, 10))
        ax4.axis('off')
        
        # Get first n_preview forecast rows (prefer future forecast, but can include historical)
        if not df_forecast_future.empty:
            df_table = df_forecast_future.head(n_preview).copy()
        else:
            df_table = df_forecast.head(n_preview).copy()
        
        # Format date column
        df_table_display = df_table.copy()
        # Handle NaT values in date column
        df_table_display['ds'] = pd.to_datetime(df_table_display['ds'])
        df_table_display['ds'] = df_table_display['ds'].apply(
            lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else 'N/A'
        )
        
        # Round numeric columns
        for col in df_table_display.select_dtypes(include=[np.number]).columns:
            df_table_display[col] = df_table_display[col].round(2)
        
        # Create table
        table = ax4.table(cellText=df_table_display.values,
                         colLabels=df_table_display.columns,
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)
        
        # Alternate row colors for better readability
        for i in range(len(df_table_display)+1):
            for j in range(len(df_table_display.columns)):
                if i == 0:  # Header row
                    table[(i, j)].set_facecolor('#2E7D32')
                    table[(i, j)].set_text_props(weight='bold', color='white')
                elif i % 2 == 0:  # Even rows
                    table[(i, j)].set_facecolor('#f5f5f5')
                else:  # Odd rows
                    table[(i, j)].set_facecolor('white')
        
        title_text = f'Forecast Preview (First {len(df_table_display)} Rows)'
        ax4.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
        
        pdf.savefig(fig4, bbox_inches='tight')
        plt.close(fig4)
    
    # Save metrics JSON file alongside PDF
    metrics_json_path = output_pdf_path.replace('.pdf', '_metrics.json')
    save_metrics_json(metrics, metrics_json_path)


def read_csv_path(csv_path: str) -> pd.DataFrame:
    """
    Helper function to read a CSV file with proper error handling.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        DataFrame with the CSV data
    """
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    except Exception as e:
        raise Exception(f"Error reading CSV file {csv_path}: {str(e)}")
