"""Анализ точности прогноза vs реальные данные"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Загружаем данные
df_history = pd.read_csv('data/processed/sales_data_shop.csv')
df_history['ds'] = pd.to_datetime(df_history['ds'])

df_forecast = pd.read_csv('data/processed/forecast_shop.csv')
df_forecast['ds'] = pd.to_datetime(df_forecast['ds'])

# Определяем периоды (holdout_frac = 0.2, значит последние 20% = тест)
n_total = len(df_history)
n_train = int(n_total * 0.8)
df_train = df_history.iloc[:n_train]
df_test = df_history.iloc[n_train:]

print("=" * 80)
print("АНАЛИЗ ТОЧНОСТИ ПРОГНОЗА")
print("=" * 80)

# Объединяем прогноз с реальными данными тестового периода
df_merged = df_forecast[['ds', 'yhat']].merge(
    df_test[['ds', 'y']],
    on='ds',
    how='inner'
)

if len(df_merged) > 0:
    print(f"\n1. СРАВНЕНИЕ ПРОГНОЗА С РЕАЛЬНЫМИ ДАННЫМИ:")
    print(f"   Период сравнения: {df_merged['ds'].min().date()} до {df_merged['ds'].max().date()}")
    print(f"   Дней для сравнения: {len(df_merged)}")
    
    # Метрики
    errors = df_merged['yhat'] - df_merged['y']
    mae = np.abs(errors).mean()
    rmse = np.sqrt((errors**2).mean())
    
    # MAPE
    mask = df_merged['y'] != 0
    mape = np.mean(np.abs((df_merged.loc[mask, 'y'] - df_merged.loc[mask, 'yhat']) / df_merged.loc[mask, 'y'])) * 100
    
    print(f"\n   Метрики на тестовом периоде:")
    print(f"   MAE:  {mae:.2f}")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   MAPE: {mape:.2f}%")
    
    # Анализ по периодам
    print(f"\n2. АНАЛИЗ ПО ПЕРИОДАМ:")
    first_week = df_merged.head(7)
    last_week = df_merged.tail(7)
    
    print(f"\n   Первая неделя прогноза:")
    print(f"   Средний прогноз: {first_week['yhat'].mean():.2f}")
    print(f"   Средние реальные: {first_week['y'].mean():.2f}")
    print(f"   Разница: {first_week['yhat'].mean() - first_week['y'].mean():.2f} (прогноз {'завышен' if first_week['yhat'].mean() > first_week['y'].mean() else 'занижен'})")
    
    print(f"\n   Последняя неделя прогноза:")
    print(f"   Средний прогноз: {last_week['yhat'].mean():.2f}")
    print(f"   Средние реальные: {last_week['y'].mean():.2f}")
    print(f"   Разница: {last_week['yhat'].mean() - last_week['y'].mean():.2f} (прогноз {'завышен' if last_week['yhat'].mean() > last_week['y'].mean() else 'занижен'})")
    
    # Волатильность
    print(f"\n3. ВОЛАТИЛЬНОСТЬ:")
    print(f"   Стд. откл. прогноза: {df_merged['yhat'].std():.2f}")
    print(f"   Стд. откл. реальных: {df_merged['y'].std():.2f}")
    print(f"   Прогноз {'более' if df_merged['yhat'].std() > df_merged['y'].std() else 'менее'} волатилен чем реальные данные")
    
    # Систематическая ошибка
    mean_error = errors.mean()
    print(f"\n4. СИСТЕМАТИЧЕСКАЯ ОШИБКА:")
    print(f"   Средняя ошибка: {mean_error:.2f}")
    if abs(mean_error) > 10:
        print(f"   {'ПЕРЕОЦЕНКА' if mean_error > 0 else 'НЕДООЦЕНКА'}: модель {'переоценивает' if mean_error > 0 else 'недооценивает'} в среднем на {abs(mean_error):.2f} единиц")
    
    # Первые 5 дней для детального анализа
    print(f"\n5. ПЕРВЫЕ 5 ДНЕЙ ПРОГНОЗА (детально):")
    print(f"{'Дата':<12} {'Прогноз':<10} {'Реальное':<10} {'Ошибка':<10}")
    print("-" * 45)
    for _, row in df_merged.head(5).iterrows():
        error = row['yhat'] - row['y']
        print(f"{row['ds'].date()} {row['yhat']:<10.2f} {row['y']:<10.2f} {error:<10.2f}")
else:
    print("\n⚠️  Нет пересекающихся дат между прогнозом и тестовым периодом!")
    print(f"   Прогноз: {df_forecast['ds'].min().date()} до {df_forecast['ds'].max().date()}")
    print(f"   Тест: {df_test['ds'].min().date()} до {df_test['ds'].max().date()}")

