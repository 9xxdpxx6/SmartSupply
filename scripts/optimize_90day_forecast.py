"""Оптимизация параметров для 90-дневного прогноза"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.train import train_prophet
from app.predict import predict_prophet
import pandas as pd
import numpy as np

print("=" * 80)
print("ОПТИМИЗАЦИЯ ДЛЯ 90-ДНЕВНОГО ПРОГНОЗА")
print("=" * 80)

# Загружаем данные для проверки
df_history = pd.read_csv('data/processed/sales_data_shop.csv')
df_history['ds'] = pd.to_datetime(df_history['ds'])
n_total = len(df_history)
n_train = int(n_total * 0.8)
df_test = df_history.iloc[n_train:]

# Тестируем разные комбинации
configs = [
    {
        "name": "Вариант 1: changepoint=0.005, seasonality=8 (текущий)",
        "changepoint": 0.005,
        "seasonality": 8.0
    },
    {
        "name": "Вариант 2: changepoint=0.005, seasonality=6 (меньше сезонности)",
        "changepoint": 0.005,
        "seasonality": 6.0
    },
    {
        "name": "Вариант 3: changepoint=0.007, seasonality=7",
        "changepoint": 0.007,
        "seasonality": 7.0
    },
    {
        "name": "Вариант 4: changepoint=0.005, seasonality=5 (минимальная сезонность)",
        "changepoint": 0.005,
        "seasonality": 5.0
    },
]

results = []

for config in configs:
    print(f"\n{config['name']}")
    print("-" * 60)
    
    # Обучаем модель
    result = train_prophet(
        shop_csv_path="data/processed/sales_data_shop.csv",
        model_out_path=f"models/test_90day_{config['changepoint']}_{config['seasonality']}.pkl",
        include_regressors=False,
        log_transform=False,
        interval_width=0.95,
        holdout_frac=0.2,
        changepoint_prior_scale=config['changepoint'],
        seasonality_prior_scale=config['seasonality'],
        seasonality_mode="additive"
    )
    
    # Генерируем прогноз на 90 дней
    forecast = predict_prophet(
        model_path=f"models/test_90day_{config['changepoint']}_{config['seasonality']}.pkl",
        horizon_days=90,
        log_transform=False
    )
    
    # Сравниваем с реальными данными (на доступных)
    df_merged = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].merge(
        df_test[['ds', 'y']],
        on='ds',
        how='inner'
    )
    
    if len(df_merged) > 0:
        errors = df_merged['yhat'] - df_merged['y']
        mae = np.abs(errors).mean()
        rmse = np.sqrt((errors**2).mean())
        mask = df_merged['y'] != 0
        mape = np.mean(np.abs((df_merged.loc[mask, 'y'] - df_merged.loc[mask, 'yhat']) / df_merged.loc[mask, 'y'])) * 100
        
        # Coverage
        in_interval = ((df_merged['y'] >= df_merged['yhat_lower']) & 
                       (df_merged['y'] <= df_merged['yhat_upper'])).sum()
        coverage = (in_interval / len(df_merged)) * 100
        
        # Систематическая ошибка
        mean_error = errors.mean()
        
        print(f"  MAPE: {mape:.2f}%")
        print(f"  MAE: {mae:.2f}")
        print(f"  Coverage CI: {coverage:.1f}%")
        print(f"  Систематическая ошибка: {mean_error:.2f}")
        
        results.append({
            'name': config['name'],
            'changepoint': config['changepoint'],
            'seasonality': config['seasonality'],
            'mape': mape,
            'mae': mae,
            'rmse': rmse,
            'coverage': coverage,
            'mean_error': mean_error
        })
    else:
        print("  Нет данных для сравнения")

# Сортируем по MAPE
results.sort(key=lambda x: x['mape'])

print(f"\n{'=' * 80}")
print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
print(f"{'=' * 80}")
print(f"{'Конфигурация':<50} {'MAPE':<8} {'Coverage':<10} {'Ошибка':<10}")
print("-" * 80)

for r in results:
    best = " <-- ЛУЧШИЙ" if r == results[0] else ""
    print(f"{r['name']:<50} {r['mape']:<8.2f} {r['coverage']:<10.1f} {r['mean_error']:<10.2f}{best}")

print(f"\nРЕКОМЕНДАЦИЯ:")
best = results[0]
print(f"  changepoint_prior_scale = {best['changepoint']}")
print(f"  seasonality_prior_scale = {best['seasonality']}")
print(f"  Ожидаемый MAPE: {best['mape']:.2f}%")
print(f"  Ожидаемое покрытие CI: {best['coverage']:.1f}%")

# Сохраняем результаты
with open('models/90day_optimization_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

