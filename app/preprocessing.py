# file: app/preprocessing.py
import pandas as pd
from datetime import datetime
from typing import Dict, Any


def parse_and_process(
    raw_csv_path: str, out_shop_csv: str, out_category_csv: str
) -> Dict[str, Any]:
    """
    Parse and process raw sales data CSV file.

    Args:
        raw_csv_path: Path to the input CSV file with sales data
        out_shop_csv: Path to save shop-level aggregated data
        out_category_csv: Path to save category-level aggregated data

    Returns:
        Dictionary containing:
        - shop_csv: path to the shop-level CSV
        - category_csv: path to the category-level CSV
        - stats: statistics about the processed data
    """
    # Read the CSV file
    df = pd.read_csv(raw_csv_path)
    
    # Normalize column names
    column_mapping = {
        'Sale_Date': 'ds',
        'Product_ID': 'product_id',
        'Product_Category': 'category',
        'Unit_Price': 'price',
        'Discount': 'discount',
        'Quantity_Sold': 'y'
    }
    df = df.rename(columns=column_mapping)
    
    # Ensure required columns exist
    required_cols = ['ds', 'product_id', 'category', 'price', 'discount', 'y']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Coerce types
    df['ds'] = pd.to_datetime(df['ds'])
    df['y'] = pd.to_numeric(df['y'], errors='coerce').fillna(0)
    df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
    df['discount'] = pd.to_numeric(df['discount'], errors='coerce').fillna(0)
    
    # Fill other potentially missing values with zeros where logical
    df = df.fillna({
        'product_id': 'unknown',
        'category': 'unknown'
    })
    
    # Aggregate to shop-level: sum(quantity) per ds
    shop_level = df.groupby('ds').agg({'y': 'sum'}).reset_index()
    shop_level.columns = ['ds', 'y']
    
    # Ensure continuous daily range for shop-level data
    if not shop_level.empty:
        date_range_shop = pd.date_range(
            start=shop_level['ds'].min(),
            end=shop_level['ds'].max(),
            freq='D'
        )
        shop_level = shop_level.set_index('ds').reindex(date_range_shop, fill_value=0).reset_index()
        shop_level.columns = ['ds', 'y']
    
    # Save shop-level data
    shop_level.to_csv(out_shop_csv, index=False)
    
    # Aggregate to category-level: sum per (category, ds)
    category_level = df.groupby(['category', 'ds']).agg({'y': 'sum'}).reset_index()
    category_level.columns = ['category', 'ds', 'y']
    
    # Ensure continuous daily range for each category
    if not category_level.empty:
        all_categories = category_level['category'].unique()
        all_dates = pd.date_range(
            start=category_level['ds'].min(),
            end=category_level['ds'].max(),
            freq='D'
        )
        
        # Create a complete index of all category-date combinations
        full_index = pd.MultiIndex.from_product(
            [all_categories, all_dates],
            names=['category', 'ds']
        )
        
        # Reindex the category data to fill missing dates
        category_level = category_level.set_index(['category', 'ds']).reindex(
            full_index, fill_value=0
        ).reset_index()
    
    # Save category-level data
    category_level.to_csv(out_category_csv, index=False)
    
    # Prepare stats
    stats = {
        'total_records': len(df),
        'date_range': {
            'start': df['ds'].min().isoformat() if not df.empty else None,
            'end': df['ds'].max().isoformat() if not df.empty else None
        },
        'unique_categories': df['category'].nunique(),
        'unique_products': df['product_id'].nunique(),
        'shop_data_rows': len(shop_level),
        'category_data_rows': len(category_level)
    }
    
    return {
        'shop_csv': out_shop_csv,
        'category_csv': out_category_csv,
        'stats': stats
    }


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse and process sales data")
    parser.add_argument("input_csv", help="Input CSV file path")
    parser.add_argument("output_shop_csv", help="Output shop-level CSV file path")
    parser.add_argument("output_category_csv", help="Output category-level CSV file path")
    
    args = parser.parse_args()
    
    result = parse_and_process(
        args.input_csv,
        args.output_shop_csv,
        args.output_category_csv
    )
    
    print("Processing completed successfully!")
    print(f"Shop-level data saved to: {result['shop_csv']}")
    print(f"Category-level data saved to: {result['category_csv']}")
    print(f"Stats: {result['stats']}")