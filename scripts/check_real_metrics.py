"""
Проверка реальных метрик для объяснения MAPE
"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Загружаем данные нового датасета
df = pd.read_csv('data/processed/test_shop.csv')
df['ds'] = pd.to_datetime(df['ds'])

print("=" * 80)
print("АНАЛИЗ ДАННЫХ И ОБЪЯСНЕНИЕ ВЫСОКОГО MAPE")
print("=" * 80)

print(f"\n1. ХАРАКТЕРИСТИКИ ДАННЫХ:")
print(f"   Период: {df['ds'].min().date()} to {df['ds'].max().date()}")
print(f"   Всего дней: {len(df)}")
print(f"   Продажи: min={df['y'].min():.0f}, max={df['y'].max():.0f}, mean={df['y'].mean():.0f}")
print(f"   Волатильность (CV): {(df['y'].std() / df['y'].mean() * 100):.1f}%")

# Показываем распределение значений
low_values = (df['y'] < 1000).sum()
medium_values = ((df['y'] >= 1000) & (df['y'] < 2000)).sum()
high_values = (df['y'] >= 2000).sum()

print(f"\n2. РАСПРЕДЕЛЕНИЕ ЗНАЧЕНИЙ:")
print(f"   Низкие (< 1000):     {low_values} дней ({low_values/len(df)*100:.1f}%)")
print(f"   Средние (1000-2000): {medium_values} дней ({medium_values/len(df)*100:.1f}%)")
print(f"   Высокие (>= 2000):   {high_values} дней ({high_values/len(df)*100:.1f}%)")

print(f"\n3. ПОЧЕМУ MAPE МОЖЕТ БЫТЬ ВЫСОКИМ:")
print(f"   Пример: если факт = 600, а прогноз = 800")
print(f"   → Абсолютная ошибка: 200")
print(f"   → MAPE: 200/600 * 100% = 33%")
print(f"   ")
print(f"   Если факт = 4000, а прогноз = 3800")
print(f"   → Абсолютная ошибка: 200 (та же!)")
print(f"   → MAPE: 200/4000 * 100% = 5%")
print(f"   ")
print(f"   Та же ошибка на малом значении = в 6 раз больший MAPE!")

print(f"\n4. ВИЗУАЛЬНОЕ КАЧЕСТВО vs MAPE:")
print(f"   ✓ Если прогноз ЗАХВАТЫВАЕТ ТРЕНД:")
print(f"     - Корреляция факт-прогноз будет высокой (>0.7)")
print(f"     - График будет выглядеть похожим")
print(f"     - Но MAPE может быть высоким из-за:")
print(f"       * Небольших ошибок на малых значениях")
print(f"       * Сдвигов пиков на 1-2 дня")
print(f"       * Систематического смещения на ±10-20%")

print(f"\n5. ЭТО НОРМАЛЬНО ЕСЛИ:")
print(f"   ✓ Coverage (CI) > 80%  → модель калибрована правильно")
print(f"   ✓ Корреляция > 0.7     → тренд захвачен")
print(f"   ✓ MAE разумный         → абсолютные ошибки небольшие")
print(f"   ✓ График визуально похож → модель работает")

print(f"\n6. РЕКОМЕНДАЦИИ:")
print(f"   Для волатильных данных (CV > 50%):")
print(f"   1. Используйте MAE или RMSE вместо MAPE")
print(f"   2. Проверяйте Coverage (CI hit rate)")
print(f"   3. Смотрите на визуальное сходство тренда")
print(f"   4. Рассмотрите log_transform для стабилизации")

print("\n" + "=" * 80)

