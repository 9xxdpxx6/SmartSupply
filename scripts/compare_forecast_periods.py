"""Сравнение прогноза с реальностью по периодам"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Загружаем данные
df_history = pd.read_csv('data/processed/sales_data_shop.csv')
df_history['ds'] = pd.to_datetime(df_history['ds'])
df_forecast = pd.read_csv('data/processed/forecast_shop.csv')
df_forecast['ds'] = pd.to_datetime(df_forecast['ds'])

n_total = len(df_history)
n_train = int(n_total * 0.8)
df_test = df_history.iloc[n_train:]

df_merged = df_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].merge(
    df_test[['ds', 'y']],
    on='ds',
    how='inner'
)

print("=" * 80)
print("ДЕТАЛЬНОЕ СРАВНЕНИЕ ПРОГНОЗА С РЕАЛЬНОСТЬЮ")
print("=" * 80)

print(f"\nОбщие метрики ({len(df_merged)} дней):")
errors = df_merged['yhat'] - df_merged['y']
mae = np.abs(errors).mean()
mask = df_merged['y'] != 0
mape = np.mean(np.abs((df_merged.loc[mask, 'y'] - df_merged.loc[mask, 'yhat']) / df_merged.loc[mask, 'y'])) * 100
mean_error = errors.mean()

print(f"  MAPE: {mape:.2f}%")
print(f"  MAE: {mae:.2f}")
print(f"  Средняя ошибка: {mean_error:+.2f} ({'ПЕРЕОЦЕНКА' if mean_error > 0 else 'НЕДООЦЕНКА' if mean_error < 0 else 'БАЛАНС'})")

# Анализ по декадам (10 дней)
print(f"\nАнализ по декадам:")
for i in range(0, len(df_merged), 10):
    period = df_merged.iloc[i:i+10]
    if len(period) == 0:
        continue
    
    period_mape = np.mean(np.abs((period.loc[period['y'] != 0, 'y'] - period.loc[period['y'] != 0, 'yhat']) / 
                                 period.loc[period['y'] != 0, 'y'])) * 100
    mean_pred = period['yhat'].mean()
    mean_actual = period['y'].mean()
    mean_error_period = (period['yhat'] - period['y']).mean()
    
    print(f"\n  Декада {i//10 + 1} ({period['ds'].min().date()} - {period['ds'].max().date()}):")
    print(f"    MAPE: {period_mape:.2f}%")
    print(f"    Прогноз: {mean_pred:.2f}, Реальность: {mean_actual:.2f}")
    print(f"    Ошибка: {mean_error_period:+.2f} ({'переоценка' if mean_error_period > 0 else 'недооценка'})")

# Анализ направления тренда
print(f"\nАнализ трендов:")
# Реальные данные - первые 15 дней vs последние 15 дней
if len(df_merged) >= 15:
    first_half = df_merged.head(15)
    last_half = df_merged.tail(15)
    
    actual_trend = last_half['y'].mean() - first_half['y'].mean()
    forecast_trend = last_half['yhat'].mean() - first_half['yhat'].mean()
    
    print(f"  Реальный тренд (первые 15 дней -> последние 15 дней): {actual_trend:+.2f}")
    print(f"  Прогноз тренд (первые 15 дней -> последние 15 дней): {forecast_trend:+.2f}")
    print(f"  Совпадение направления: {'ДА' if (actual_trend > 0) == (forecast_trend > 0) else 'НЕТ'}")

# Точки перегиба
print(f"\nАнализ точек перегиба:")
# Находим дни с максимальным/минимальным значением
actual_min_idx = df_merged['y'].idxmin()
actual_max_idx = df_merged['y'].idxmax()
forecast_min_idx = df_merged['yhat'].idxmin()
forecast_max_idx = df_merged['yhat'].idxmax()

print(f"  Реальный минимум: {df_merged.loc[actual_min_idx, 'ds'].date()} ({df_merged.loc[actual_min_idx, 'y']:.1f})")
print(f"  Прогноз минимум: {df_merged.loc[forecast_min_idx, 'ds'].date()} ({df_merged.loc[forecast_min_idx, 'yhat']:.1f})")
print(f"  Реальный максимум: {df_merged.loc[actual_max_idx, 'ds'].date()} ({df_merged.loc[actual_max_idx, 'y']:.1f})")
print(f"  Прогноз максимум: {df_merged.loc[forecast_max_idx, 'ds'].date()} ({df_merged.loc[forecast_max_idx, 'yhat']:.1f})")

# Детальный просмотр первых 10 дней
print(f"\nДетальный просмотр первых 10 дней:")
print(f"{'Дата':<12} {'Прогноз':<10} {'Реальное':<10} {'Ошибка':<10} {'В CI':<6}")
print("-" * 55)
for i, row in df_merged.head(10).iterrows():
    error = row['yhat'] - row['y']
    in_ci = 'YES' if (row['y'] >= row['yhat_lower'] and row['y'] <= row['yhat_upper']) else 'NO'
    print(f"{row['ds'].date()} {row['yhat']:<10.2f} {row['y']:<10.2f} {error:<10.2f} {in_ci:<6}")

