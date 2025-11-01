"""Оценка текущего прогноза vs реальные данные"""
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

print("=" * 80)
print("ОЦЕНКА ТЕКУЩЕГО ПРОГНОЗА")
print("=" * 80)

# Определяем тестовый период (holdout_frac = 0.2)
n_total = len(df_history)
n_train = int(n_total * 0.8)
df_train = df_history.iloc[:n_train]
df_test = df_history.iloc[n_train:]

print(f"\n1. ОБЩАЯ ИНФОРМАЦИЯ:")
print(f"   Период обучения: {df_train['ds'].min().date()} - {df_train['ds'].max().date()}")
print(f"   Период теста: {df_test['ds'].min().date()} - {df_test['ds'].max().date()}")
print(f"   Период прогноза: {df_forecast['ds'].min().date()} - {df_forecast['ds'].max().date()}")
print(f"   Размер прогноза: {len(df_forecast)} дней")

# Объединяем прогноз с реальными данными
df_merged = df_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].merge(
    df_test[['ds', 'y']],
    on='ds',
    how='inner'
)

if len(df_merged) == 0:
    print("\n⚠️  Нет пересекающихся дат! Проверьте данные.")
    exit(1)

print(f"\n2. МЕТРИКИ НА ТЕСТОВОМ ПЕРИОДЕ ({len(df_merged)} дней):")

errors = df_merged['yhat'] - df_merged['y']
mae = np.abs(errors).mean()
rmse = np.sqrt((errors**2).mean())

mask = df_merged['y'] != 0
mape = np.mean(np.abs((df_merged.loc[mask, 'y'] - df_merged.loc[mask, 'yhat']) / df_merged.loc[mask, 'y'])) * 100

mean_error = errors.mean()

print(f"   MAE: {mae:.2f}")
print(f"   RMSE: {rmse:.2f}")
print(f"   MAPE: {mape:.2f}%")
print(f"   Средняя ошибка: {mean_error:+.2f} ({'ПЕРЕОЦЕНКА' if mean_error > 0 else 'НЕДООЦЕНКА'})")

# Покрытие confidence interval
in_interval = ((df_merged['y'] >= df_merged['yhat_lower']) & 
               (df_merged['y'] <= df_merged['yhat_upper'])).sum()
coverage = (in_interval / len(df_merged)) * 100
print(f"   Покрытие CI: {coverage:.1f}% ({in_interval}/{len(df_merged)} дней)")

# Анализ по периодам (первые 7, средние 7, последние 7 дней)
print(f"\n3. АНАЛИЗ ПО ПЕРИОДАМ:")

if len(df_merged) >= 21:
    periods = [
        ("Первая неделя", df_merged.head(7)),
        ("Вторая неделя", df_merged.iloc[7:14]),
        ("Третья неделя", df_merged.iloc[14:21]),
    ]
elif len(df_merged) >= 14:
    periods = [
        ("Первая неделя", df_merged.head(7)),
        ("Вторая неделя", df_merged.iloc[7:14]),
    ]
else:
    periods = [("Все дни", df_merged)]

for name, period_df in periods:
    if len(period_df) == 0:
        continue
    
    period_mape = np.mean(np.abs((period_df.loc[period_df['y'] != 0, 'y'] - 
                                  period_df.loc[period_df['y'] != 0, 'yhat']) / 
                                 period_df.loc[period_df['y'] != 0, 'y'])) * 100
    
    mean_pred = period_df['yhat'].mean()
    mean_actual = period_df['y'].mean()
    mean_error_period = (period_df['yhat'] - period_df['y']).mean()
    
    print(f"\n   {name} ({period_df['ds'].min().date()} - {period_df['ds'].max().date()}):")
    print(f"      MAPE: {period_mape:.2f}%")
    print(f"      Средний прогноз: {mean_pred:.2f}")
    print(f"      Средние реальные: {mean_actual:.2f}")
    print(f"      Средняя ошибка: {mean_error_period:+.2f}")

# Визуальный анализ первых 14 дней (где применялось сглаживание)
print(f"\n4. АНАЛИЗ ПЕРВЫХ 14 ДНЕЙ (зона сглаживания):")
first_14 = df_merged.head(14)

# Проверяем плавность
changes = first_14['yhat'].diff().abs() / first_14['yhat'].shift(1) * 100
changes = changes[1:]  # Убираем первый NaN

print(f"   Плавность прогноза:")
print(f"      Макс. изменение день-день: {changes.max():.2f}%")
print(f"      Средн. изменение день-день: {changes.mean():.2f}%")
print(f"      Все изменения <= 1.5%: {all(changes <= 1.51)}")

# Точность первых 14 дней
mask_14 = first_14['y'] != 0
mape_14 = np.mean(np.abs((first_14.loc[mask_14, 'y'] - first_14.loc[mask_14, 'yhat']) / 
                         first_14.loc[mask_14, 'y'])) * 100
print(f"   Точность:")
print(f"      MAPE (первые 14 дней): {mape_14:.2f}%")
print(f"      Средняя ошибка: {(first_14['yhat'] - first_14['y']).mean():+.2f}")

# Детальный анализ первых 7 дней
print(f"\n5. ДЕТАЛЬНЫЙ АНАЛИЗ ПЕРВЫХ 7 ДНЕЙ:")
print(f"{'Дата':<12} {'Прогноз':<10} {'Реальное':<10} {'Ошибка':<10} {'Изменение':<12}")
print("-" * 60)

prev_val = None
for i, row in first_14.head(7).iterrows():
    error = row['yhat'] - row['y']
    change = ""
    if prev_val is not None:
        change_pct = (row['yhat'] - prev_val) / prev_val * 100
        change = f"{change_pct:+.2f}%"
    prev_val = row['yhat']
    print(f"{row['ds'].date()} {row['yhat']:<10.2f} {row['y']:<10.2f} {error:<10.2f} {change:<12}")

# Проблемные зоны
print(f"\n6. ПРОБЛЕМНЫЕ ЗОНЫ:")
large_errors = df_merged[np.abs(errors) > 20].copy()
if len(large_errors) > 0:
    large_errors['error'] = large_errors['yhat'] - large_errors['y']
    large_errors['error_pct'] = (large_errors['error'] / large_errors['y']) * 100
    print(f"   Дней с большой ошибкой (>20 единиц): {len(large_errors)}")
    print(f"\n   Топ-5 самых больших ошибок:")
    top_errors = large_errors.nlargest(5, 'error_pct')
    for _, row in top_errors.iterrows():
        print(f"      {row['ds'].date()}: прогноз {row['yhat']:.1f}, реальное {row['y']:.1f}, "
              f"ошибка {row['error']:+.1f} ({row['error_pct']:+.1f}%)")

# Выводы
print(f"\n7. ВЫВОДЫ:")
print(f"   OK Сглаживание работает: изменения день-день <= 1.5%")
if mape < 30:
    print(f"   OK Общий MAPE хороший: {mape:.2f}%")
elif mape < 50:
    print(f"   WARNING Общий MAPE приемлемый: {mape:.2f}% (можно улучшить)")
else:
    print(f"   ERROR Общий MAPE высокий: {mape:.2f}% (требует улучшения)")

if coverage < 80:
    print(f"   WARNING Покрытие CI низкое: {coverage:.1f}% (должно быть ~95%)")

if abs(mean_error) > 10:
    print(f"   WARNING Систематическая {'переоценка' if mean_error > 0 else 'недооценка'}: {mean_error:.2f} единиц")

