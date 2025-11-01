# file: app/preprocessing.py
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _improve_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Улучшает качество данных:
    - Заменяет нулевые значения на медиану соседних дней
    - Обрабатывает выбросы (winsorization)
    - Применяет скользящее среднее для стабилизации
    
    Args:
        df: DataFrame с колонками 'ds' и 'y'
        
    Returns:
        Улучшенный DataFrame
    """
    df = df.copy()
    df = df.sort_values('ds').reset_index(drop=True)
    
    # 1. Обработка нулевых значений
    zero_mask = df['y'] == 0
    zero_count = zero_mask.sum()
    if zero_count > 0:
        logger.info(f"Replacing {zero_count} zero values with median of neighboring days...")
        for idx in df[zero_mask].index:
            window_start = max(0, idx - 3)
            window_end = min(len(df), idx + 4)
            window = df.iloc[window_start:window_end]
            window_nonzero = window[window['y'] > 0]['y']
            if len(window_nonzero) > 0:
                replacement = window_nonzero.median()
                df.loc[idx, 'y'] = replacement
            else:
                # Если все окно нули, используем общую медиану
                df.loc[idx, 'y'] = df[df['y'] > 0]['y'].median()
    
    # 2. Winsorization (обработка выбросов)
    Q1 = df['y'].quantile(0.05)
    Q3 = df['y'].quantile(0.95)
    outliers = (df['y'] < Q1) | (df['y'] > Q3)
    outlier_count = outliers.sum()
    if outlier_count > 0:
        logger.info(f"Winsorizing {outlier_count} outliers (clipping to 5th and 95th percentiles)...")
        df.loc[df['y'] < Q1, 'y'] = Q1
        df.loc[df['y'] > Q3, 'y'] = Q3
    
    # 3. Скользящее среднее для стабилизации (7 дней)
    logger.info("Applying 7-day moving average for data stabilization...")
    df['y'] = df['y'].rolling(window=7, center=True, min_periods=1).mean()
    
    return df


def suggest_aggregation(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Suggest aggregation frequency based on data density.
    
    Args:
        df: DataFrame with 'ds' column containing datetime values
        
    Returns:
        Tuple of (frequency: 'D' or 'W', reason: str)
    """
    if df.empty or 'ds' not in df.columns:
        return 'D', "No data available"
    
    df_copy = df.copy()
    df_copy['ds'] = pd.to_datetime(df_copy['ds'])
    
    n_unique_dates = df_copy['ds'].nunique()
    date_range_days = (df_copy['ds'].max() - df_copy['ds'].min()).days + 1
    
    if date_range_days == 0:
        return 'D', "Single date in dataset"
    
    avg_records_per_day = len(df_copy) / date_range_days if date_range_days > 0 else 0
    
    if n_unique_dates < 90:
        return 'W', f"Too few unique dates ({n_unique_dates} < 90), recommend weekly aggregation"
    elif avg_records_per_day < 1.0:
        return 'W', f"Data too sparse (avg {avg_records_per_day:.2f} records/day < 1.0), recommend weekly aggregation"
    else:
        return 'D', f"Data density acceptable ({avg_records_per_day:.2f} records/day, {n_unique_dates} unique dates)"


def parse_and_process(
    raw_csv_path: str, 
    out_shop_csv: str, 
    out_category_csv: str, 
    force_weekly: bool = False
) -> Dict[str, Any]:
    """
    Parse and process raw sales data CSV file with validation and regressor computation.
    
    Args:
        raw_csv_path: Path to the input CSV file with sales data
        out_shop_csv: Path to save shop-level aggregated data
        out_category_csv: Path to save category-level aggregated data
        force_weekly: If True, force weekly aggregation regardless of data density
        
    Returns:
        Dictionary containing:
        - shop_csv: path to the shop-level CSV
        - category_csv: path to the category-level CSV
        - stats: statistics about the processed data
    """
    logger.info(f"Starting preprocessing for file: {raw_csv_path}")
    
    # Normalize path for cross-platform compatibility
    raw_csv_path = os.path.normpath(raw_csv_path)
    
    # Read the CSV file
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"File not found: {raw_csv_path}")
    
    try:
        # Try reading with different encodings
        try:
            df = pd.read_csv(raw_csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            logger.warning("UTF-8 encoding failed, trying latin-1")
            try:
                df = pd.read_csv(raw_csv_path, encoding='latin-1')
            except Exception as e:
                logger.error(f"Failed to read CSV with both UTF-8 and latin-1 encoding: {str(e)}")
                raise ValueError(f"Unable to read CSV file. Encoding error: {str(e)}")
        
        n_rows_raw = len(df)
        logger.info(f"Loaded {n_rows_raw} rows from CSV")
        
        if n_rows_raw == 0:
            raise ValueError("CSV file is empty")
            
    except pd.errors.EmptyDataError:
        raise ValueError("CSV file is empty or invalid")
    except Exception as e:
        logger.error(f"Error reading CSV file: {str(e)}", exc_info=True)
        raise ValueError(f"Error reading CSV file: {str(e)}")
    
    # Validation: minimum 100 rows
    if n_rows_raw < 100:
        warning_msg = f"Warning: Dataset has only {n_rows_raw} rows (minimum recommended: 100)"
        logger.warning(warning_msg)
        stats_warning = warning_msg
    else:
        stats_warning = None
    
    # Detect format: old format (Sale_Date) or new format (order_date)
    has_old_format = 'Sale_Date' in df.columns
    has_new_format = 'order_date' in df.columns
    
    if has_old_format:
        # Old format: Sale_Date, Product_ID, Product_Category, Unit_Price, Discount, Quantity_Sold
        required_cols = ['Sale_Date', 'Product_ID', 'Product_Category', 'Unit_Price', 'Discount', 'Quantity_Sold']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
        format_type = 'old'
    elif has_new_format:
        # New format: order_date, item_id/sku, category, price, discount_amount, qty_ordered
        required_cols = ['order_date', 'category', 'price', 'qty_ordered']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
        format_type = 'new'
    else:
        raise ValueError("Unknown format: neither 'Sale_Date' nor 'order_date' column found. "
                        "Expected columns: Sale_Date/order_date, Product_ID/item_id, Product_Category/category, "
                        "Unit_Price/price, Discount/discount_amount, Quantity_Sold/qty_ordered")
    
    # Validate crucial NaN values (depending on format)
    if format_type == 'old':
        date_col = 'Sale_Date'
        qty_col = 'Quantity_Sold'
    else:
        date_col = 'order_date'
        qty_col = 'qty_ordered'
    
    crucial_nan_ds = df[date_col].isna().sum()
    crucial_nan_y = df[qty_col].isna().sum()
    
    if crucial_nan_ds > 0:
        logger.warning(f"Found {crucial_nan_ds} NaN values in {date_col}, removing these rows")
        df = df.dropna(subset=[date_col])
    
    if crucial_nan_y > 0:
        logger.warning(f"Found {crucial_nan_y} NaN values in {qty_col}, will fill with 0")
    
    # For new format: filter only valid orders (complete or received)
    if format_type == 'new' and 'status' in df.columns:
        valid_statuses = ['complete', 'received']  # Only count completed/received orders
        before_filter = len(df)
        df = df[df['status'].isin(valid_statuses)].copy()
        filtered = before_filter - len(df)
        if filtered > 0:
            logger.info(f"Filtered out {filtered} rows with invalid status (keeping only 'complete' and 'received' orders)")
    
    # Check and remove duplicates
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        logger.info(f"Found {n_duplicates} duplicate rows, removing them")
        df = df.drop_duplicates()
    
    n_rows_clean = len(df)
    logger.info(f"After cleaning: {n_rows_clean} rows (removed {n_rows_raw - n_rows_clean} rows)")
    
    # Normalize column names (depending on format)
    if format_type == 'old':
        column_mapping = {
            'Sale_Date': 'ds',
            'Product_ID': 'product_id',
            'Product_Category': 'category',
            'Unit_Price': 'price',
            'Discount': 'discount',
            'Quantity_Sold': 'y'
        }
    else:
        # New format mapping
        column_mapping = {
            'order_date': 'ds',
            'category': 'category',  # Already correct
            'price': 'price',  # Already correct
            'qty_ordered': 'y',
        }
        # Map product identifier (item_id or sku)
        if 'item_id' in df.columns:
            column_mapping['item_id'] = 'product_id'
        elif 'sku' in df.columns:
            column_mapping['sku'] = 'product_id'
        
        # Map discount (discount_amount or Discount_Percent)
        if 'discount_amount' in df.columns:
            column_mapping['discount_amount'] = 'discount'
        elif 'Discount_Percent' in df.columns:
            # Convert percent to amount if needed
            if 'price' in df.columns and 'qty_ordered' in df.columns:
                df['discount_amount'] = df['price'] * df['qty_ordered'] * (df['Discount_Percent'] / 100.0)
                column_mapping['discount_amount'] = 'discount'
        
        # If discount column not found, create zero discount
        if 'discount' not in column_mapping.values():
            df['discount'] = 0.0
            logger.info("No discount column found, setting discount to 0")
    
    df = df.rename(columns=column_mapping)
    
    # Coerce types
    logger.info("Converting data types...")
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    
    # Check for invalid dates after conversion
    invalid_dates = df['ds'].isna().sum()
    if invalid_dates > 0:
        logger.warning(f"Found {invalid_dates} invalid dates after conversion, removing these rows")
        df = df.dropna(subset=['ds'])
    
    df['y'] = pd.to_numeric(df['y'], errors='coerce').fillna(0)
    df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
    df['discount'] = pd.to_numeric(df['discount'], errors='coerce').fillna(0)
    
    # Fill other potentially missing values
    df = df.fillna({
        'product_id': 'unknown',
        'category': 'unknown'
    })
    
    # Validate date range
    if df.empty:
        raise ValueError("No data remaining after cleaning. Please check your CSV file.")
    
    if df['ds'].isna().all():
        raise ValueError("No valid dates found after data cleaning. Please check Sale_Date column format.")
    
    date_min = df['ds'].min()
    date_max = df['ds'].max()
    n_unique_dates = df['ds'].nunique()
    
    if pd.isna(date_min) or pd.isna(date_max):
        raise ValueError("Invalid date range after conversion. Please check Sale_Date values.")
    
    logger.info(f"Date range: {date_min} to {date_max} ({n_unique_dates} unique dates)")
    
    # Determine aggregation frequency
    suggested_freq, freq_reason = suggest_aggregation(df)
    if force_weekly:
        freq_used = 'W'
        logger.info(f"Forcing weekly aggregation (force_weekly=True)")
    else:
        freq_used = suggested_freq
        logger.info(f"Suggested frequency: {freq_used} - {freq_reason}")
    
    # Aggregate to shop-level with regressors
    logger.info("Aggregating shop-level data...")
    
    try:
        if freq_used == 'W':
            # Weekly aggregation: start week on Monday
            df['ds_week'] = df['ds'].dt.to_period('W-MON').dt.start_time
            shop_level = df.groupby('ds_week').agg({
                'y': 'sum',
                'price': 'mean',
                'discount': 'mean'
            }).reset_index()
            shop_level.columns = ['ds', 'y', 'avg_price', 'avg_discount']
        else:
            # Daily aggregation
            shop_level = df.groupby('ds').agg({
                'y': 'sum',
                'price': 'mean',
                'discount': 'mean'
            }).reset_index()
            shop_level.columns = ['ds', 'y', 'avg_price', 'avg_discount']
    except Exception as e:
        logger.error(f"Error during shop-level aggregation: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to aggregate shop-level data: {str(e)}")
    
    # Add day_of_week and is_weekend
    shop_level['day_of_week'] = shop_level['ds'].dt.dayofweek.astype(int)
    shop_level['is_weekend'] = (shop_level['day_of_week'] >= 5).astype(int)
    
    # Ensure continuous date range
    if shop_level.empty:
        raise ValueError("No data available for shop-level aggregation after processing")
    
    shop_date_min = shop_level['ds'].min()
    shop_date_max = shop_level['ds'].max()
    
    if pd.isna(shop_date_min) or pd.isna(shop_date_max):
        raise ValueError("Invalid dates in shop-level aggregation")
    
    if freq_used == 'W':
        date_range = pd.date_range(
            start=shop_date_min,
            end=shop_date_max,
            freq='W-MON'
        )
    else:
        date_range = pd.date_range(
            start=shop_date_min,
            end=shop_date_max,
            freq='D'
        )
    
    # Reindex with forward-fill for regressors
    try:
        shop_level = shop_level.set_index('ds')
        shop_level_full = pd.DataFrame(index=date_range)
        shop_level_full = shop_level_full.join(shop_level, how='left')
        
        # Fill y with 0 for missing dates
        shop_level_full['y'] = shop_level_full['y'].fillna(0)
        
        # Forward-fill regressors, then fill remaining NaN with median
        for col in ['avg_price', 'avg_discount']:
            if shop_level_full[col].notna().any():
                shop_level_full[col] = shop_level_full[col].ffill()
                # Fill any remaining NaN with median
                if shop_level_full[col].isna().any():
                    median_val = shop_level_full[col].median()
                    shop_level_full[col] = shop_level_full[col].fillna(median_val if not pd.isna(median_val) else 0)
            else:
                shop_level_full[col] = 0
        
        # Recalculate day_of_week and is_weekend for filled dates
        shop_level_full['day_of_week'] = shop_level_full.index.dayofweek.astype(int)
        shop_level_full['is_weekend'] = (shop_level_full['day_of_week'] >= 5).astype(int)
        
        shop_level = shop_level_full.reset_index()
        shop_level.columns = ['ds', 'y', 'avg_price', 'avg_discount', 'day_of_week', 'is_weekend']
    except Exception as e:
        logger.error(f"Error during date range filling: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to create continuous date range: {str(e)}")
    
    # Apply data quality fixes: handle zeros, outliers, and smoothing
    logger.info("Applying data quality improvements...")
    shop_level = _improve_data_quality(shop_level)
    
    # Ensure output directory exists and normalize paths
    shop_dir = os.path.dirname(out_shop_csv)
    if shop_dir:
        os.makedirs(shop_dir, exist_ok=True)
    
    # Normalize path for cross-platform compatibility
    out_shop_csv = os.path.normpath(out_shop_csv)
    
    try:
        shop_level.to_csv(out_shop_csv, index=False)
    except Exception as e:
        logger.error(f"Error saving shop CSV to {out_shop_csv}: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to save shop-level CSV: {str(e)}")
    logger.info(f"Shop-level data saved to: {out_shop_csv} ({len(shop_level)} rows)")
    
    # Aggregate to category-level
    logger.info("Aggregating category-level data...")
    
    try:
        if freq_used == 'W':
            df['ds_week'] = df['ds'].dt.to_period('W-MON').dt.start_time
            category_level = df.groupby(['category', 'ds_week']).agg({'y': 'sum'}).reset_index()
            category_level.columns = ['category', 'ds', 'y']
        else:
            category_level = df.groupby(['category', 'ds']).agg({'y': 'sum'}).reset_index()
    except Exception as e:
        logger.error(f"Error during category-level aggregation: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to aggregate category-level data: {str(e)}")
    
    # Ensure continuous date range for each category
    if not category_level.empty:
        try:
            all_categories = category_level['category'].unique()
            
            cat_date_min = category_level['ds'].min()
            cat_date_max = category_level['ds'].max()
            
            if pd.isna(cat_date_min) or pd.isna(cat_date_max):
                raise ValueError("Invalid dates in category-level data")
            
            if freq_used == 'W':
                all_dates = pd.date_range(
                    start=cat_date_min,
                    end=cat_date_max,
                    freq='W-MON'
                )
            else:
                all_dates = pd.date_range(
                    start=cat_date_min,
                    end=cat_date_max,
                    freq='D'
                )
            
            # Create complete index
            full_index = pd.MultiIndex.from_product(
                [all_categories, all_dates],
                names=['category', 'ds']
            )
            
            # Reindex and fill missing values with 0
            category_level = category_level.set_index(['category', 'ds']).reindex(
                full_index, fill_value=0
            ).reset_index()
        except Exception as e:
            logger.error(f"Error during category-level date range filling: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to create continuous date range for categories: {str(e)}")
    
    # Ensure output directory exists and normalize paths
    category_dir = os.path.dirname(out_category_csv)
    if category_dir:
        os.makedirs(category_dir, exist_ok=True)
    
    # Normalize path for cross-platform compatibility
    out_category_csv = os.path.normpath(out_category_csv)
    
    try:
        category_level.to_csv(out_category_csv, index=False)
    except Exception as e:
        logger.error(f"Error saving category CSV to {out_category_csv}: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to save category-level CSV: {str(e)}")
    logger.info(f"Category-level data saved to: {out_category_csv} ({len(category_level)} rows)")
    
    # Prepare stats (ensure all values are JSON-serializable)
    stats = {
        'n_rows_raw': int(n_rows_raw),
        'n_rows_clean': int(n_rows_clean),
        'n_unique_dates': int(n_unique_dates),
        'freq_used': freq_used,
        'freq_reason': freq_reason,
        'date_min': date_min.isoformat() if not pd.isna(date_min) else None,
        'date_max': date_max.isoformat() if not pd.isna(date_max) else None,
        'duplicates_removed': int(n_duplicates),
        'unique_categories': int(df['category'].nunique()),
        'unique_products': int(df['product_id'].nunique()),
        'shop_data_rows': int(len(shop_level)),
        'category_data_rows': int(len(category_level)),
        'warning': stats_warning
    }
    
    logger.info("Preprocessing completed successfully")
    logger.info(f"Stats: {stats}")
    
    return {
        'shop_csv': out_shop_csv,
        'category_csv': out_category_csv,
        'stats': stats
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse and process sales data")
    parser.add_argument("input_csv", help="Input CSV file path")
    parser.add_argument("output_shop_csv", help="Output shop-level CSV file path")
    parser.add_argument("output_category_csv", help="Output category-level CSV file path")
    parser.add_argument("--force-weekly", action="store_true", 
                       help="Force weekly aggregation regardless of data density")
    
    args = parser.parse_args()
    
    result = parse_and_process(
        args.input_csv,
        args.output_shop_csv,
        args.output_category_csv,
        force_weekly=args.force_weekly
    )
    
    print("Processing completed successfully!")
    print(f"Shop-level data saved to: {result['shop_csv']}")
    print(f"Category-level data saved to: {result['category_csv']}")
    print("\nStats:")
    for key, value in result['stats'].items():
        print(f"  {key}: {value}")
