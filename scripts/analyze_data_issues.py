"""
Анализ данных для поиска причин плохого качества модели.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def analyze_data(shop_csv_path: str):
    """Анализ данных на проблемы."""
    print("=" * 80)
    print("АНАЛИЗ ДАННЫХ ДЛЯ ПОИСКА ПРОБЛЕМ")
    print("=" * 80)
    
    df = pd.read_csv(shop_csv_path)
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds').reset_index(drop=True)
    
    print(f"\n1. ОБЩАЯ ИНФОРМАЦИЯ:")
    print(f"   Всего записей: {len(df)}")
    print(f"   Период: {df['ds'].min().date()} до {df['ds'].max().date()}")
    print(f"   Дней данных: {(df['ds'].max() - df['ds'].min()).days + 1}")
    
    print(f"\n2. СТАТИСТИКА ПРОДАЖ (y):")
    print(f"   Среднее: {df['y'].mean():.2f}")
    print(f"   Медиана: {df['y'].median():.2f}")
    print(f"   Стд. откл.: {df['y'].std():.2f}")
    print(f"   Мин: {df['y'].min():.2f}")
    print(f"   Макс: {df['y'].max():.2f}")
    print(f"   Коэффициент вариации: {df['y'].std() / df['y'].mean():.2f}")
    
    # Проверка на выбросы
    Q1 = df['y'].quantile(0.25)
    Q3 = df['y'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df['y'] < lower_bound) | (df['y'] > upper_bound)]
    print(f"\n3. ВЫБРОСЫ (метод IQR):")
    print(f"   Нижняя граница: {lower_bound:.2f}")
    print(f"   Верхняя граница: {upper_bound:.2f}")
    print(f"   Количество выбросов: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
    
    if len(outliers) > 0:
        print(f"   Максимальный выброс: {outliers['y'].max():.2f}")
        print(f"   Минимальный выброс: {outliers['y'].min():.2f}")
    
    # Проверка на пропуски
    print(f"\n4. ПРОПУСКИ:")
    missing_dates = pd.date_range(start=df['ds'].min(), end=df['ds'].max(), freq='D')
    missing = missing_dates.difference(df['ds'])
    print(f"   Пропущено дат: {len(missing)} ({len(missing)/len(missing_dates)*100:.1f}%)")
    
    # Проверка на нули
    zero_sales = (df['y'] == 0).sum()
    print(f"\n5. НУЛЕВЫЕ ПРОДАЖИ:")
    print(f"   Дней с нулевыми продажами: {zero_sales} ({zero_sales/len(df)*100:.1f}%)")
    
    # Анализ тренда
    print(f"\n6. АНАЛИЗ ТРЕНДА:")
    df['day_num'] = (df['ds'] - df['ds'].min()).dt.days
    correlation = df['day_num'].corr(df['y'])
    print(f"   Корреляция день-продажи: {correlation:.3f}")
    
    # Сезонность
    print(f"\n7. СЕЗОННОСТЬ:")
    df['weekday'] = df['ds'].dt.dayofweek
    df['month'] = df['ds'].dt.month
    weekday_avg = df.groupby('weekday')['y'].mean()
    print(f"   Средние продажи по дням недели:")
    days = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
    for i, day in enumerate(days):
        print(f"     {day}: {weekday_avg.iloc[i]:.2f}")
    
    month_avg = df.groupby('month')['y'].mean()
    print(f"   Средние продажи по месяцам:")
    months = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
    for month_num in range(1, 13):
        if month_num in month_avg.index:
            print(f"     {months[month_num-1]}: {month_avg[month_num]:.2f}")
    
    # Проверка стационарности (тест на автокорреляцию)
    print(f"\n8. АВТОКОРРЕЛЯЦИЯ:")
    autocorr_1 = df['y'].autocorr(lag=1)
    autocorr_7 = df['y'].autocorr(lag=7)
    autocorr_30 = df['y'].autocorr(lag=30)
    print(f"   Лаг 1 день: {autocorr_1:.3f}")
    print(f"   Лаг 7 дней: {autocorr_7:.3f}")
    print(f"   Лаг 30 дней: {autocorr_30:.3f}")
    
    # Рекомендации
    print(f"\n9. РЕКОМЕНДАЦИИ:")
    recommendations = []
    
    if len(outliers) > len(df) * 0.05:
        recommendations.append("⚠️  Много выбросов (>5%) - нужно обработать")
    
    if len(missing) > len(missing_dates) * 0.1:
        recommendations.append("⚠️  Много пропусков дат (>10%) - нужно заполнить")
    
    if zero_sales > len(df) * 0.1:
        recommendations.append("⚠️  Много нулевых продаж (>10%) - может влиять на модель")
    
    if abs(correlation) < 0.1:
        recommendations.append("⚠️  Слабая корреляция с временем - возможно нет тренда")
    
    if abs(autocorr_1) < 0.3:
        recommendations.append("⚠️  Слабая автокорреляция - данные могут быть нестационарными")
    
    if df['y'].std() / df['y'].mean() > 1.0:
        recommendations.append("✅ Высокая волатильность - log-transform необходим")
    
    if len(recommendations) == 0:
        print("   ✅ Данные выглядят нормально")
    else:
        for rec in recommendations:
            print(f"   {rec}")
    
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--shop_csv", default="data/processed/sales_data_shop.csv")
    args = parser.parse_args()
    
    analyze_data(args.shop_csv)

