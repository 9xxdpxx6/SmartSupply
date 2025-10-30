# file: app/utils.py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


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
        MAPE value
    """
    # Handle potential division by zero
    mask = actual != 0
    if mask.any():
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    else:
        return 0.0


def export_report_pdf(output_pdf_path: str, df_history: pd.DataFrame, df_forecast: pd.DataFrame, metrics: dict) -> None:
    """
    Export forecast report as a multi-page PDF.
    
    Args:
        output_pdf_path: Path to save the PDF report
        df_history: Historical data with columns 'ds' and 'y'
        df_forecast: Forecast data with columns 'ds', 'yhat', 'yhat_lower', 'yhat_upper'
        metrics: Dictionary containing evaluation metrics
    """
    with PdfPages(output_pdf_path) as pdf:
        # Page 1: Summary text
        fig1, ax1 = plt.subplots(figsize=(8.5, 11))
        ax1.axis('off')
        
        # Extract summary information
        hist_start = df_history['ds'].min()
        hist_end = df_history['ds'].max()
        total_sales = df_history['y'].sum()
        
        # Create summary text
        summary_text = f"""
Forecast Report

Historical Data Range: {hist_start.strftime('%Y-%m-%d')} to {hist_end.strftime('%Y-%m-%d')}
Total Historical Sales: {total_sales:,.2f}

Performance Metrics:
- MAE: {metrics.get('mae', 'N/A')}
- RMSE: {metrics.get('rmse', 'N/A')}
- MAPE: {metrics.get('mape', 'N/A')}%

Model Details:
- Forecast Horizon: {len(df_forecast) - len(df_history)} days
- Confidence Interval: 95%
        """
        
        ax1.text(0.1, 0.9, summary_text, transform=ax1.transAxes, fontsize=12, 
                 verticalalignment='top', fontfamily='monospace')
        
        pdf.savefig(fig1, bbox_inches='tight')
        plt.close(fig1)
        
        # Page 2: Line plot of actual vs forecast
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        # Plot historical data
        ax2.plot(df_history['ds'], df_history['y'], label='Historical', color='blue')
        
        # Plot forecast data
        ax2.plot(df_forecast['ds'], df_forecast['yhat'], label='Forecast', color='red')
        
        # Plot confidence intervals if available
        if 'yhat_lower' in df_forecast.columns and 'yhat_upper' in df_forecast.columns:
            ax2.fill_between(df_forecast['ds'], 
                            df_forecast['yhat_lower'], 
                            df_forecast['yhat_upper'], 
                            color='red', alpha=0.2, label='Confidence Interval')
        
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Sales')
        ax2.set_title('Historical vs Forecast')
        ax2.legend()
        ax2.grid(True)
        
        # Format x-axis to show dates nicely
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        pdf.savefig(fig2, bbox_inches='tight')
        plt.close(fig2)
        
        # Page 3: Table with first 30 forecast rows
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        ax3.axis('off')
        
        # Get first 30 forecast rows
        df_table = df_forecast.head(30).copy()
        df_table['ds'] = df_table['ds'].dt.strftime('%Y-%m-%d')
        
        # Round numeric columns
        for col in df_table.select_dtypes(include=[np.number]).columns:
            df_table[col] = df_table[col].round(2)
        
        # Create table
        table = ax3.table(cellText=df_table.values,
                         colLabels=df_table.columns,
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Alternate row colors for better readability
        for i in range(len(df_table)+1):
            for j in range(len(df_table.columns)):
                if i == 0:  # Header row
                    table[(i, j)].set_facecolor('#4CAF50')
                    table[(i, j)].set_text_props(weight='bold', color='white')
                elif i % 2 == 0:  # Even rows
                    table[(i, j)].set_facecolor('#f5f5f5')
        
        ax3.set_title('First 30 Forecast Rows', fontsize=14, pad=20)
        
        pdf.savefig(fig3, bbox_inches='tight')
        plt.close(fig3)


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