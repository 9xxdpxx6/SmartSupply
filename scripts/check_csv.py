# file: scripts/check_csv.py
"""
Quick CSV validation script to check if a CSV file is ready for preprocessing.

Usage:
    python scripts/check_csv.py path/to/file.csv
"""

import sys
import pandas as pd
import os


def check_csv(csv_path: str):
    """Check if CSV file meets preprocessing requirements."""
    print(f"Checking CSV file: {csv_path}")
    print("=" * 60)
    
    # Check file exists
    if not os.path.exists(csv_path):
        print(f"❌ ERROR: File not found: {csv_path}")
        return False
    
    print(f"✓ File exists")
    
    # Check file size
    file_size = os.path.getsize(csv_path)
    print(f"✓ File size: {file_size} bytes")
    
    if file_size == 0:
        print(f"❌ ERROR: File is empty")
        return False
    
    # Try to read CSV
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
        print(f"✓ Successfully read CSV with UTF-8 encoding")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_path, encoding='latin-1')
            print(f"✓ Successfully read CSV with latin-1 encoding")
        except Exception as e:
            print(f"❌ ERROR: Could not read CSV file: {str(e)}")
            return False
    except Exception as e:
        print(f"❌ ERROR: Could not read CSV file: {str(e)}")
        return False
    
    print(f"✓ Total rows: {len(df)}")
    
    if len(df) == 0:
        print(f"❌ ERROR: CSV file has no data rows")
        return False
    
    # Check required columns
    required_cols = ['Sale_Date', 'Product_ID', 'Product_Category', 'Unit_Price', 'Discount', 'Quantity_Sold']
    print(f"\nChecking required columns:")
    print(f"  Required: {', '.join(required_cols)}")
    print(f"  Found:   {', '.join(df.columns.tolist())}")
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ ERROR: Missing required columns: {', '.join(missing_cols)}")
        return False
    
    print(f"✓ All required columns present")
    
    # Check for critical NaN values
    print(f"\nChecking data quality:")
    nan_ds = df['Sale_Date'].isna().sum()
    nan_y = df['Quantity_Sold'].isna().sum()
    
    print(f"  NaN in Sale_Date: {nan_ds}")
    print(f"  NaN in Quantity_Sold: {nan_y}")
    
    if nan_ds == len(df):
        print(f"❌ ERROR: All Sale_Date values are NaN")
        return False
    
    # Try to parse dates
    try:
        dates = pd.to_datetime(df['Sale_Date'], errors='coerce')
        valid_dates = dates.notna().sum()
        print(f"  Valid dates: {valid_dates} / {len(df)}")
        
        if valid_dates == 0:
            print(f"❌ ERROR: No valid dates found in Sale_Date column")
            print(f"  Sample values: {df['Sale_Date'].head().tolist()}")
            return False
        
        if valid_dates < len(df) * 0.5:
            print(f"⚠ WARNING: Less than 50% of dates are valid")
        
        date_min = dates.min()
        date_max = dates.max()
        print(f"  Date range: {date_min} to {date_max}")
        print(f"  Unique dates: {dates.nunique()}")
        
    except Exception as e:
        print(f"❌ ERROR: Could not parse dates: {str(e)}")
        return False
    
    # Check numeric columns
    try:
        qty = pd.to_numeric(df['Quantity_Sold'], errors='coerce')
        valid_qty = qty.notna().sum()
        print(f"  Valid Quantity_Sold: {valid_qty} / {len(df)}")
        
        if valid_qty == 0:
            print(f"❌ ERROR: No valid Quantity_Sold values")
            return False
    except Exception as e:
        print(f"⚠ WARNING: Could not validate Quantity_Sold: {str(e)}")
    
    print(f"\n✓ CSV file appears to be valid for preprocessing")
    print("=" * 60)
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/check_csv.py <csv_file_path>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    success = check_csv(csv_path)
    sys.exit(0 if success else 1)

