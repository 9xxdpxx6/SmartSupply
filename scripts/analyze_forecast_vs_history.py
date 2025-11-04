"""Анализ соответствия прогноза историческим данным"""
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Загружаем историю и прогноз
df_history = pd.read_csv('data/processed/sales_data_shop.csv')
df_history['ds'] = pd.to_datetime(df_history['ds'])

df_forecast = pd.read_csv('data/processed/forecast_shop.csv')
df_forecast['ds'] = pd.to_datetime(df_forecast['ds'])

print("=" * 80)
print("АНАЛИЗ ПРОГНОЗА VS ИСТОРИЯ")
print("=" * 80)

# Последние дни истории перед прогнозом
last_hist_date = df_history['ds'].max()
last_30_hist = df_history[df_history['ds'] >= (last_hist_date - pd.Timedelta(days=30))].copy()

print(f"\n1. ПОСЛЕДНИЕ 30 ДНЕЙ ИСТОРИИ (до прогноза):")
print(f"   Период: {last_30_hist['ds'].min().date()} до {last_30_hist['ds'].max().date()}")
print(f"   Среднее: {last_30_hist['y'].mean():.2f}")
print(f"   Медиана: {last_30_hist['y'].median():.2f}")
print(f"   Последнее значение: {last_30_hist.iloc[-1]['y']:.2f}")

# Первые дни прогноза
first_forecast_date = df_forecast['ds'].min()
first_7_forecast = df_forecast[df_forecast['ds'] <= (first_forecast_date + pd.Timedelta(days=7))].copy()

print(f"\n2. ПЕРВЫЕ 7 ДНЕЙ ПРОГНОЗА:")
print(f"   Период: {first_7_forecast['ds'].min().date()} до {first_7_forecast['ds'].max().date()}")
print(f"   Средний прогноз: {first_7_forecast['yhat'].mean():.2f}")
print(f"   Первый прогноз: {first_7_forecast.iloc[0]['yhat']:.2f}")
print(f"   Последнее историческое значение: {last_30_hist.iloc[-1]['y']:.2f}")

# Разрыв между историей и прогнозом
gap = first_7_forecast.iloc[0]['yhat'] - last_30_hist.iloc[-1]['y']
print(f"\n3. РАЗРЫВ:")
print(f"   Разница между последним историческим и первым прогнозом: {gap:.2f}")
if abs(gap) > 20:
    print(f"   ВНИМАНИЕ: Большой разрыв! ({abs(gap):.2f} единиц)")

# Проверка тренда в прогнозе
print(f"\n4. ТРЕНД В ПРОГНОЗЕ:")
forecast_start = df_forecast['yhat'].head(10).mean()
forecast_end = df_forecast['yhat'].tail(10).mean()
trend = forecast_end - forecast_start
print(f"   Среднее первых 10 дней: {forecast_start:.2f}")
print(f"   Среднее последних 10 дней: {forecast_end:.2f}")
print(f"   Изменение тренда: {trend:.2f} ({'рост' if trend > 0 else 'падение'})")

# Сравнение с исторической волатильностью
hist_std = last_30_hist['y'].std()
forecast_std = df_forecast['yhat'].std()
print(f"\n5. ВОЛАТИЛЬНОСТЬ:")
print(f"   Стд. откл. истории (последние 30 дней): {hist_std:.2f}")
print(f"   Стд. откл. прогноза (30 дней): {forecast_std:.2f}")
if forecast_std > hist_std * 1.5:
    print(f"   ВНИМАНИЕ: Прогноз более волатилен чем история!")


