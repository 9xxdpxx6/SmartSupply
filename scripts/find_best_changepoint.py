"""Поиск оптимального changepoint_prior_scale для уменьшения переоценки"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.train import train_prophet

print("Поиск оптимального changepoint_prior_scale...")
print("=" * 60)

# Тестируем разные значения changepoint
changepoint_values = [0.003, 0.005, 0.007, 0.01, 0.015]

results = []
for chp in changepoint_values:
    print(f"\nТест changepoint_prior_scale = {chp}")
    result = train_prophet(
        shop_csv_path="data/processed/sales_data_shop.csv",
        model_out_path=f"models/test_chp_{chp}.pkl",
        include_regressors=False,
        log_transform=False,
        interval_width=0.95,
        holdout_frac=0.2,
        changepoint_prior_scale=chp,
        seasonality_prior_scale=10.0,
        seasonality_mode="additive"
    )
    
    mape = result['metrics']['mape']
    mae = result['metrics']['mae']
    
    print(f"  MAPE: {mape:.2f}%, MAE: {mae:.2f}")
    
    results.append({
        'changepoint': chp,
        'mape': mape,
        'mae': mae,
        'rmse': result['metrics']['rmse']
    })

# Сортируем по MAPE
results.sort(key=lambda x: x['mape'])

print(f"\n{'=' * 60}")
print("РЕЗУЛЬТАТЫ:")
print(f"{'=' * 60}")
print(f"{'Changepoint':<15} {'MAPE':<10} {'MAE':<10}")
print("-" * 35)

for r in results:
    best_mark = " <-- ЛУЧШИЙ" if r == results[0] else ""
    print(f"{r['changepoint']:<15.3f} {r['mape']:<10.2f} {r['mae']:<10.2f}{best_mark}")

print(f"\nРекомендуется: changepoint_prior_scale = {results[0]['changepoint']}")


