"""Быстрый тест оптимальной конфигурации"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.train import train_prophet

print("Тестирование оптимальной конфигурации:")
print("=" * 60)

result = train_prophet(
    shop_csv_path="data/processed/sales_data_shop.csv",
    model_out_path="models/optimal_test.pkl",
    include_regressors=False,
    log_transform=False,  # БЕЗ log-transform!
    interval_width=0.95,
    holdout_frac=0.2,
    changepoint_prior_scale=0.01,  # Консервативный
    seasonality_prior_scale=10.0,
    seasonality_mode="additive"
)

print(f"\nРезультаты:")
print(f"MAE:  {result['metrics']['mae']:.2f}")
print(f"RMSE: {result['metrics']['rmse']:.2f}")
print(f"MAPE: {result['metrics']['mape']:.2f}%")

