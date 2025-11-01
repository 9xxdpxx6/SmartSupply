"""Анализ качества данных и диагностика проблем"""
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# Загружаем данные
df = pd.read_csv('data/processed/sales_data_shop.csv')
df['ds'] = pd.to_datetime(df['ds'])
df = df.sort_values('ds').reset_index(drop=True)

print("=" * 80)
print("ДИАГНОСТИКА ДАННЫХ И МОДЕЛИ")
print("=" * 80)

# 1. Анализ объема данных
print(f"\n1. ОБЪЕМ ДАННЫХ:")
print(f"   Всего записей: {len(df)}")
print(f"   Период: {df['ds'].min().date()} to {df['ds'].max().date()}")
days_total = (df['ds'].max() - df['ds'].min()).days
print(f"   Всего дней: {days_total}")
print(f"   Рекомендация Prophet: минимум 365 дней для yearly seasonality")

if days_total < 365:
    print(f"   ⚠️ ВНИМАНИЕ: Меньше года данных! Prophet может плохо улавливать yearly seasonality")
    print(f"   💡 Рекомендация: использовать weekly_seasonality=True, yearly_seasonality=False")

# 2. Анализ качества данных
print(f"\n2. КАЧЕСТВО ДАННЫХ:")
print(f"   Среднее продаж: {df['y'].mean():.2f}")
print(f"   Стд отклонение: {df['y'].std():.2f}")
print(f"   Коэффициент вариации: {(df['y'].std() / df['y'].mean() * 100):.1f}%")
print(f"   Мин: {df['y'].min():.2f}, Макс: {df['y'].max():.2f}")
print(f"   Медиана: {df['y'].median():.2f}")

# Выбросы
z_scores = np.abs((df['y'] - df['y'].mean()) / df['y'].std())
outliers = (z_scores > 3).sum()
print(f"   Выбросы (> 3*std): {outliers} ({outliers/len(df)*100:.1f}%)")

# Пропуски
missing = df['y'].isna().sum()
zeros = (df['y'] == 0).sum()
print(f"   Пропуски: {missing}")
print(f"   Нули: {zeros} ({zeros/len(df)*100:.1f}%)")

# 3. Анализ тренда
print(f"\n3. ТРЕНД:")
from scipy import stats
dates_num = pd.to_datetime(df['ds']).astype(int) / 1e9
slope, intercept, r_value, p_value, std_err = stats.linregress(dates_num, df['y'].values)
print(f"   Наклон тренда: {slope*86400:.4f} (единиц в день)")
print(f"   R-squared: {r_value**2:.4f}")
print(f"   p-value: {p_value:.6f}")
if p_value < 0.05:
    print(f"   OK: Тренд статистически значим")
else:
    print(f"   WARNING: Тренд не значим (p > 0.05)")

# 4. Сезонность
print(f"\n4. СЕЗОННОСТЬ:")
# Недельная
df['day_of_week'] = df['ds'].dt.dayofweek
weekly_pattern = df.groupby('day_of_week')['y'].mean()
weekly_range = weekly_pattern.max() - weekly_pattern.min()
print(f"   Недельная вариация: {weekly_range:.2f} ({weekly_range/df['y'].mean()*100:.1f}%)")

# Месячная
df['month'] = df['ds'].dt.month
monthly_pattern = df.groupby('month')['y'].mean()
monthly_range = monthly_pattern.max() - monthly_pattern.min()
print(f"   Месячная вариация: {monthly_range:.2f} ({monthly_range/df['y'].mean()*100:.1f}%)")

# 5. Анализ соотношения train/test
print(f"\n5. РАЗДЕЛЕНИЕ ДАННЫХ:")
holdout_frac = 0.2
n_train = int(len(df) * (1 - holdout_frac))
n_test = len(df) - n_train
train_days = (df.iloc[n_train-1]['ds'] - df.iloc[0]['ds']).days
test_days = (df.iloc[-1]['ds'] - df.iloc[n_train]['ds']).days

print(f"   Train: {n_train} записей, {train_days} дней")
print(f"   Test: {n_test} записей, {test_days} дней")
print(f"   Соотношение: {n_train/n_test:.2f}:1")
print(f"   Рекомендация: минимум 3:1 для надежного обучения")

# Для прогноза на 60 дней
print(f"\n6. ПРОГНОЗ НА 60 ДНЕЙ:")
horizon = 60
train_to_horizon_ratio = n_train / horizon
print(f"   Train дней / Horizon: {train_to_horizon_ratio:.1f}:1")
if train_to_horizon_ratio < 4:
    print(f"   ⚠️ ВНИМАНИЕ: Мало данных для прогноза на {horizon} дней!")
    print(f"   💡 Рекомендация: сократить horizon до 30 дней или собрать больше данных")

# 7. Параметры модели
print(f"\n7. ПАРАМЕТРЫ МОДЕЛИ:")
print(f"   Prophet параметры:")
print(f"   - changepoint_prior_scale: 0.05 (высокая гибкость)")
print(f"   - seasonality_prior_scale: 10.0 (сильная сезонность)")
print(f"   - yearly_seasonality: True")
print(f"   - weekly_seasonality: True")
print(f"   Регуляризация: Prophet использует Bayesian подход, но:")
if days_total < 365:
    print(f"   ⚠️ С данными < 1 года yearly_seasonality может переобучаться")

# 8. Рекомендации
print(f"\n8. РЕКОМЕНДАЦИИ:")
issues = []

if days_total < 365:
    issues.append("Мало данных для yearly seasonality")
if n_train / horizon < 4:
    issues.append("Мало данных для долгосрочного прогноза")
if outliers > len(df) * 0.05:
    issues.append("Много выбросов (>5%)")
if df['y'].std() / df['y'].mean() > 0.5:
    issues.append("Высокая волатильность (CV > 50%)")

if issues:
    print(f"   🚨 ПРОБЛЕМЫ:")
    for issue in issues:
        print(f"      - {issue}")
    
    print(f"\n   💡 РЕШЕНИЯ:")
    if days_total < 365:
        print(f"      1. Отключить yearly_seasonality (yearly_seasonality=False)")
        print(f"      2. Использовать только weekly_seasonality")
    if n_train / horizon < 4:
        print(f"      3. Сократить horizon до 30 дней")
        print(f"      4. Собрать больше исторических данных")
    if outliers > len(df) * 0.05:
        print(f"      5. Очистить выбросы в preprocessing")
    if df['y'].std() / df['y'].mean() > 0.5:
        print(f"      6. Использовать log_transform для стабилизации")
        print(f"      7. Увеличить seasonality_prior_scale для более стабильной сезонности")
else:
    print(f"   ✅ Данные выглядят нормально")

print("\n" + "=" * 80)

