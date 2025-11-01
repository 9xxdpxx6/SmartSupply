import pandas as pd
df = pd.read_csv('data/processed/sales_data_shop.csv')
print(f'Rows: {len(df)}')
print(f'Zeros: {(df["y"] == 0).sum()}')
print(f'Mean: {df["y"].mean():.2f}')
print(f'First 10 values:')
print(df[['ds', 'y']].head(10))

