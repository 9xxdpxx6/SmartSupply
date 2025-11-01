"""Поиск решения проблемы завышения в начале прогноза"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.train import train_prophet
from app.predict import predict_prophet

print("=" * 80)
print("ПОИСК РЕШЕНИЯ ПРОБЛЕМЫ ЗАВЫШЕНИЯ В НАЧАЛЕ ПРОГНОЗА")
print("=" * 80)

# Загружаем данные для анализа
df_history = pd.read_csv('data/processed/sales_data_shop.csv')
df_history['ds'] = pd.to_datetime(df_history['ds'])
n_total = len(df_history)
n_train = int(n_total * 0.8)
df_test = df_history.iloc[n_train:]

# Фокусируемся на первых 14 днях (2 недели) - где проблема
df_test_first14 = df_test.head(14)

configs = [
    {
        "name": "Вариант 1: seasonality=5 (меньше сезонности)",
        "changepoint": 0.005,
        "seasonality": 5.0
    },
    {
        "name": "Вариант 2: seasonality=4 (еще меньше)",
        "changepoint": 0.005,
        "seasonality": 4.0
    },
    {
        "name": "Вариант 3: seasonality=6 + changepoint=0.004 (более консервативный)",
        "changepoint": 0.004,
        "seasonality": 6.0
    },
    {
        "name": "Вариант 4: seasonality=7 + changepoint=0.006 (более гибкий тренд)",
        "changepoint": 0.006,
        "seasonality": 7.0
    },
    {
        "name": "Вариант 5: seasonality=3 (минимальная сезонность)",
        "changepoint": 0.005,
        "seasonality": 3.0
    },
]

results = []

for config in configs:
    print(f"\n{config['name']}")
    print("-" * 60)
    
    # Обучаем
    result = train_prophet(
        shop_csv_path="data/processed/sales_data_shop.csv",
        model_out_path=f"models/test_fix_{config['changepoint']}_{config['seasonality']}.pkl",
        include_regressors=False,
        log_transform=False,
        interval_width=0.95,
        holdout_frac=0.2,
        changepoint_prior_scale=config['changepoint'],
        seasonality_prior_scale=config['seasonality'],
        seasonality_mode="additive"
    )
    
    # Генерируем прогноз на 50 дней
    forecast = predict_prophet(
        model_path=f"models/test_fix_{config['changepoint']}_{config['seasonality']}.pkl",
        horizon_days=50,
        log_transform=False
    )
    
    # Анализируем первые 14 дней
    df_merged = forecast[['ds', 'yhat']].merge(
        df_test_first14[['ds', 'y']],
        on='ds',
        how='inner'
    )
    
    if len(df_merged) > 0:
        # Ошибка в первые 14 дней
        errors = df_merged['yhat'] - df_merged['y']
        mae_first14 = np.abs(errors).mean()
        mean_error_first14 = errors.mean()
        rmse_first14 = np.sqrt((errors**2).mean())
        
        mask = df_merged['y'] != 0
        mape_first14 = np.mean(np.abs((df_merged.loc[mask, 'y'] - df_merged.loc[mask, 'yhat']) / df_merged.loc[mask, 'y'])) * 100
        
        # Общая ошибка на всем тесте
        df_all = forecast[['ds', 'yhat']].merge(
            df_test[['ds', 'y']],
            on='ds',
            how='inner'
        )
        errors_all = df_all['yhat'] - df_all['y']
        mape_all = np.mean(np.abs((df_all.loc[df_all['y'] != 0, 'y'] - df_all.loc[df_all['y'] != 0, 'yhat']) / df_all.loc[df_all['y'] != 0, 'y'])) * 100
        
        print(f"  Первые 14 дней:")
        print(f"    MAPE: {mape_first14:.2f}%")
        print(f"    MAE: {mae_first14:.2f}")
        print(f"    Средняя ошибка: {mean_error_first14:.2f} ({'ПЕРЕОЦЕНКА' if mean_error_first14 > 0 else 'НЕДООЦЕНКА'})")
        print(f"  Весь тест (74 дня):")
        print(f"    MAPE: {mape_all:.2f}%")
        
        results.append({
            'name': config['name'],
            'changepoint': config['changepoint'],
            'seasonality': config['seasonality'],
            'mape_first14': mape_first14,
            'mae_first14': mae_first14,
            'mean_error_first14': mean_error_first14,
            'mape_all': mape_all
        })
    else:
        print("  Нет данных для сравнения")

# Сортируем по ошибке в первые 14 дней (абсолютное значение)
results.sort(key=lambda x: abs(x['mean_error_first14']))

print(f"\n{'=' * 80}")
print("РЕЗУЛЬТАТЫ (отсортировано по завышению в первые 14 дней):")
print(f"{'=' * 80}")
print(f"{'Конфигурация':<55} {'Ошибка 14д':<12} {'MAPE 14д':<10} {'MAPE весь':<10}")
print("-" * 90)

for r in results:
    best = " <-- ЛУЧШИЙ" if r == results[0] else ""
    error_str = f"{r['mean_error_first14']:+.2f}"
    print(f"{r['name']:<55} {error_str:<12} {r['mape_first14']:<10.2f} {r['mape_all']:<10.2f}{best}")

print(f"\nРЕКОМЕНДАЦИЯ:")
best = results[0]
print(f"  changepoint_prior_scale = {best['changepoint']}")
print(f"  seasonality_prior_scale = {best['seasonality']}")
print(f"  Ожидаемая ошибка в первые 14 дней: {best['mean_error_first14']:.2f}")
print(f"  Ожидаемый MAPE (первые 14 дней): {best['mape_first14']:.2f}%")
print(f"  Ожидаемый MAPE (весь тест): {best['mape_all']:.2f}%")

# Сохраняем
import json
with open('models/fix_overestimation_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

