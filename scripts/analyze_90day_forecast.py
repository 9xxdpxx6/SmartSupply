"""Анализ качества прогноза на 90 дней"""
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

print("=" * 80)
print("АНАЛИЗ ПРОГНОЗА НА 90 ДНЕЙ")
print("=" * 80)

# Определяем периоды (holdout_frac = 0.2)
n_total = len(df_history)
n_train = int(n_total * 0.8)
df_train = df_history.iloc[:n_train]
df_test = df_history.iloc[n_train:]

print(f"\n1. ОБЩАЯ ИНФОРМАЦИЯ:")
print(f"   Размер прогноза: {len(df_forecast)} дней")
print(f"   Период прогноза: {df_forecast['ds'].min().date()} до {df_forecast['ds'].max().date()}")
print(f"   Размер тестового периода: {len(df_test)} дней")
print(f"   Период теста: {df_test['ds'].min().date()} до {df_test['ds'].max().date()}")

# Объединяем прогноз с реальными данными
df_merged = df_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].merge(
    df_test[['ds', 'y']],
    on='ds',
    how='inner'
)

if len(df_merged) > 0:
    print(f"\n2. СРАВНЕНИЕ С РЕАЛЬНЫМИ ДАННЫМИ:")
    print(f"   Дней для сравнения: {len(df_merged)} (из {len(df_test)} доступных)")
    
    # Метрики на доступных данных
    errors = df_merged['yhat'] - df_merged['y']
    mae = np.abs(errors).mean()
    rmse = np.sqrt((errors**2).mean())
    
    mask = df_merged['y'] != 0
    mape = np.mean(np.abs((df_merged.loc[mask, 'y'] - df_merged.loc[mask, 'yhat']) / df_merged.loc[mask, 'y'])) * 100
    
    print(f"\n   Метрики (на первых {len(df_merged)} днях):")
    print(f"   MAE:  {mae:.2f}")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   MAPE: {mape:.2f}%")
    
    # Анализ покрытия confidence interval
    in_interval = ((df_merged['y'] >= df_merged['yhat_lower']) & 
                   (df_merged['y'] <= df_merged['yhat_upper'])).sum()
    coverage = (in_interval / len(df_merged)) * 100
    
    print(f"\n   Покрытие confidence interval: {coverage:.1f}% ({in_interval}/{len(df_merged)} дней)")
    if coverage < 80:
        print(f"   WARNING: Слишком низкое покрытие! Должно быть ~95% для interval_width=0.95")
    
    # Анализ систематической ошибки
    mean_error = errors.mean()
    print(f"\n3. СИСТЕМАТИЧЕСКАЯ ОШИБКА:")
    print(f"   Средняя ошибка: {mean_error:.2f}")
    if abs(mean_error) > 5:
        direction = "ПЕРЕОЦЕНКА" if mean_error > 0 else "НЕДООЦЕНКА"
        print(f"   CRITICAL: {direction}: модель {'переоценивает' if mean_error > 0 else 'недооценивает'} в среднем на {abs(mean_error):.2f} единиц")
    
    # Анализ по неделям
    df_merged['week'] = (df_merged['ds'] - df_merged['ds'].min()).dt.days // 7
    print(f"\n4. АНАЛИЗ ПО НЕДЕЛЯМ:")
    for week in sorted(df_merged['week'].unique()):
        week_data = df_merged[df_merged['week'] == week]
        week_mape = np.mean(np.abs((week_data['y'] - week_data['yhat']) / week_data['y'])) * 100
        mean_pred = week_data['yhat'].mean()
        mean_actual = week_data['y'].mean()
        print(f"\n   Неделя {week + 1} ({week_data['ds'].min().date()} - {week_data['ds'].max().date()}):")
        print(f"      Средний прогноз: {mean_pred:.2f}")
        print(f"      Средние реальные: {mean_actual:.2f}")
        print(f"      MAPE: {week_mape:.2f}%")
    
    # Ширина confidence interval
    df_merged['ci_width'] = df_merged['yhat_upper'] - df_merged['yhat_lower']
    print(f"\n5. НЕОПРЕДЕЛЕННОСТЬ (ширина confidence interval):")
    print(f"   Средняя ширина CI: {df_merged['ci_width'].mean():.2f}")
    print(f"   Минимальная ширина: {df_merged['ci_width'].min():.2f}")
    print(f"   Максимальная ширина: {df_merged['ci_width'].max():.2f}")
    
    if df_merged['ci_width'].mean() > 60:
        print(f"   WARNING: Слишком широкие интервалы! Модель очень не уверена в прогнозе.")
    
    # Первые 10 дней детально
    print(f"\n6. ПЕРВЫЕ 10 ДНЕЙ ПРОГНОЗА (детально):")
    print(f"{'Дата':<12} {'Прогноз':<10} {'Реальное':<10} {'Ошибка':<10} {'В CI?':<8}")
    print("-" * 55)
    for _, row in df_merged.head(10).iterrows():
        error = row['yhat'] - row['y']
        in_ci = 'YES' if (row['y'] >= row['yhat_lower'] and row['y'] <= row['yhat_upper']) else 'NO'
        print(f"{row['ds'].date()} {row['yhat']:<10.2f} {row['y']:<10.2f} {error:<10.2f} {in_ci:<8}")
else:
    print("\nWARNING: Нет пересекающихся дат между прогнозом и тестовым периодом!")

# Анализ полного 90-дневного прогноза
print(f"\n7. АНАЛИЗ ПОЛНОГО 90-ДНЕВНОГО ПРОГНОЗА:")
print(f"   Всего прогнозируемых дней: {len(df_forecast)}")
print(f"   Средний прогноз: {df_forecast['yhat'].mean():.2f}")
print(f"   Медианный прогноз: {df_forecast['yhat'].median():.2f}")
print(f"   Мин/Макс прогноз: {df_forecast['yhat'].min():.2f} / {df_forecast['yhat'].max():.2f}")
print(f"   Волатильность прогноза (std): {df_forecast['yhat'].std():.2f}")

# Проверка на разумность
if df_forecast['yhat'].max() > 150:
    print(f"\n   WARNING: Максимальный прогноз ({df_forecast['yhat'].max():.2f}) очень высокий!")
if df_forecast['yhat'].min() < 0:
    print(f"   WARNING: Есть отрицательные прогнозы!")
if df_forecast['yhat'].std() > df_history['y'].std() * 1.5:
    print(f"   WARNING: Прогноз слишком волатилен по сравнению с историей!")

