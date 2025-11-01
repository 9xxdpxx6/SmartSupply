"""Тест сглаживания перехода от истории к прогнозу"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.train import train_prophet
from app.predict import predict_prophet

print("=" * 80)
print("ТЕСТ СГЛАЖИВАНИЯ НАЧАЛА ПРОГНОЗА")
print("=" * 80)

# Загружаем данные
df_history = pd.read_csv('data/processed/sales_data_shop.csv')
df_history['ds'] = pd.to_datetime(df_history['ds'])
n_total = len(df_history)
n_train = int(n_total * 0.8)
df_train = df_history.iloc[:n_train].copy()
df_test = df_history.iloc[n_train:].copy()

# Обучаем модель с текущими параметрами
print("\nОбучаем модель (changepoint=0.005, seasonality=7)...")
result = train_prophet(
    shop_csv_path="data/processed/sales_data_shop.csv",
    model_out_path="models/test_smooth.pkl",
    include_regressors=False,
    log_transform=False,
    interval_width=0.95,
    holdout_frac=0.2,
    changepoint_prior_scale=0.005,
    seasonality_prior_scale=7.0,
    seasonality_mode="additive"
)

# Генерируем прогноз
forecast = predict_prophet(
    model_path="models/test_smooth.pkl",
    horizon_days=50,
    log_transform=False
)

print("\nПрименяем сглаживание...")

# Получаем последние значения истории (последние 7 дней)
last_7_days_history = df_train.tail(7)['y'].values
last_avg = last_7_days_history.mean()
last_median = np.median(last_7_days_history)

# Первые 14 дней прогноза
forecast_first14 = forecast.head(14).copy()

# Стратегия 1: Простое сглаживание - смешиваем прогноз с последним средним
alpha = 0.3  # Вес последнего среднего (30%)
forecast_smooth1 = forecast.copy()
forecast_smooth1.loc[:13, 'yhat'] = (
    forecast_smooth1.loc[:13, 'yhat'] * (1 - alpha) + last_avg * alpha
)
forecast_smooth1.loc[:13, 'yhat_lower'] = (
    forecast_smooth1.loc[:13, 'yhat_lower'] * (1 - alpha) + last_avg * alpha
)
forecast_smooth1.loc[:13, 'yhat_upper'] = (
    forecast_smooth1.loc[:13, 'yhat_upper'] * (1 - alpha) + last_avg * alpha
)

# Стратегия 2: Постепенное затухание перехода (экспоненциальное)
forecast_smooth2 = forecast.copy()
for i in range(14):
    # Экспоненциальное затухание влияния истории
    weight = np.exp(-i / 5.0)  # Затухает за 5 дней
    forecast_smooth2.iloc[i, forecast_smooth2.columns.get_loc('yhat')] = (
        forecast.iloc[i]['yhat'] * (1 - weight) + last_avg * weight
    )
    forecast_smooth2.iloc[i, forecast_smooth2.columns.get_loc('yhat_lower')] = (
        forecast.iloc[i]['yhat_lower'] * (1 - weight) + last_avg * weight
    )
    forecast_smooth2.iloc[i, forecast_smooth2.columns.get_loc('yhat_upper')] = (
        forecast.iloc[i]['yhat_upper'] * (1 - weight) + last_avg * weight
    )

# Стратегия 3: Использовать медиану вместо среднего
forecast_smooth3 = forecast.copy()
for i in range(14):
    weight = np.exp(-i / 7.0)  # Затухает за 7 дней
    forecast_smooth3.iloc[i, forecast_smooth3.columns.get_loc('yhat')] = (
        forecast.iloc[i]['yhat'] * (1 - weight) + last_median * weight
    )
    forecast_smooth3.iloc[i, forecast_smooth3.columns.get_loc('yhat_lower')] = (
        forecast.iloc[i]['yhat_lower'] * (1 - weight) + last_median * weight
    )
    forecast_smooth3.iloc[i, forecast_smooth3.columns.get_loc('yhat_upper')] = (
        forecast.iloc[i]['yhat_upper'] * (1 - weight) + last_median * weight
    )

# Сравниваем все варианты
strategies = [
    ("Без сглаживания", forecast),
    ("Стратегия 1: Смешивание 30%", forecast_smooth1),
    ("Стратегия 2: Эксп. затухание (среднее)", forecast_smooth2),
    ("Стратегия 3: Эксп. затухание (медиана)", forecast_smooth3),
]

print(f"\nПоследние 7 дней истории: {last_7_days_history}")
print(f"Среднее: {last_avg:.2f}, Медиана: {last_median:.2f}")

print(f"\n{'=' * 80}")
print("СРАВНЕНИЕ СТРАТЕГИЙ (первые 14 дней):")
print(f"{'=' * 80}")

for name, fcst in strategies:
    df_merged = fcst.head(14)[['ds', 'yhat']].merge(
        df_test.head(14)[['ds', 'y']],
        on='ds',
        how='inner'
    )
    
    if len(df_merged) > 0:
        errors = df_merged['yhat'] - df_merged['y']
        mae = np.abs(errors).mean()
        mean_error = errors.mean()
        rmse = np.sqrt((errors**2).mean())
        mask = df_merged['y'] != 0
        mape_val = np.mean(np.abs((df_merged.loc[mask, 'y'] - df_merged.loc[mask, 'yhat']) / df_merged.loc[mask, 'y'])) * 100
        
        print(f"\n{name}:")
        print(f"  MAPE: {mape_val:.2f}%")
        print(f"  MAE: {mae:.2f}")
        print(f"  Средняя ошибка: {mean_error:.2f} ({'ПЕРЕОЦЕНКА' if mean_error > 0 else 'НЕДООЦЕНКА'})")
        print(f"  RMSE: {rmse:.2f}")

# Находим лучшую стратегию
best_mape = float('inf')
best_strategy = None
for name, fcst in strategies:
    df_merged = fcst.head(14)[['ds', 'yhat']].merge(
        df_test.head(14)[['ds', 'y']],
        on='ds',
        how='inner'
    )
    if len(df_merged) > 0:
        mask = df_merged['y'] != 0
        mape_val = np.mean(np.abs((df_merged.loc[mask, 'y'] - df_merged.loc[mask, 'yhat']) / df_merged.loc[mask, 'y'])) * 100
        if mape_val < best_mape:
            best_mape = mape_val
            best_strategy = (name, fcst)

if best_strategy:
    print(f"\n{'=' * 80}")
    print(f"ЛУЧШАЯ СТРАТЕГИЯ: {best_strategy[0]} (MAPE: {best_mape:.2f}%)")
    print(f"{'=' * 80}")

