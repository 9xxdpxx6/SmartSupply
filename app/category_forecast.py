"""
Альтернативный подход для прогнозирования категорий:
использует shop-level прогноз и распределяет его пропорционально историческим долям категорий.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional
import os

logger = logging.getLogger(__name__)


def distribute_shop_forecast_to_categories(
    shop_forecast_csv: str,
    category_csv: str,
    horizon_days: int,
    output_csv: Optional[str] = None
) -> pd.DataFrame:
    """
    Распределяет shop-level прогноз по категориям пропорционально их историческим долям.
    
    Это альтернативный подход для категорий с разреженными данными, где Prophet плохо работает.
    Использует проверенный shop-level прогноз и распределяет его пропорционально.
    
    Args:
        shop_forecast_csv: Путь к CSV с shop-level прогнозом (ds, yhat, yhat_lower, yhat_upper)
        category_csv: Путь к CSV с историческими данными категорий (category, ds, y)
        horizon_days: Количество дней в прогнозе
        output_csv: Опциональный путь для сохранения результата
        
    Returns:
        DataFrame с колонками: category, ds, yhat, yhat_lower, yhat_upper
    """
    logger.info("=" * 80)
    logger.info("ИСПОЛЬЗУЕМ АЛЬТЕРНАТИВНЫЙ ПОДХОД: Распределение shop-level прогноза по категориям")
    logger.info("=" * 80)
    
    # Загружаем shop-level прогноз
    if not os.path.exists(shop_forecast_csv):
        raise FileNotFoundError(f"Shop forecast CSV not found: {shop_forecast_csv}")
    
    shop_forecast = pd.read_csv(shop_forecast_csv)
    shop_forecast['ds'] = pd.to_datetime(shop_forecast['ds'])
    logger.info(f"Loaded shop-level forecast: {len(shop_forecast)} days")
    
    # Загружаем исторические данные категорий
    if not os.path.exists(category_csv):
        raise FileNotFoundError(f"Category CSV not found: {category_csv}")
    
    category_history = pd.read_csv(category_csv)
    category_history['ds'] = pd.to_datetime(category_history['ds'])
    logger.info(f"Loaded category history: {len(category_history)} rows")
    
    # Используем более короткий период для актуальности (приоритет последним 30 дням)
    category_history_sorted = category_history.sort_values('ds')
    last_date = category_history_sorted['ds'].max()
    
    lookback_periods = [30, 60, 90]
    available_days = (last_date - category_history_sorted['ds'].min()).days
    lookback_days = min([p for p in lookback_periods if available_days >= p], default=available_days)
    
    # Используем более короткий период если данных достаточно
    if available_days >= 60:
        lookback_days = 30  # Приоритет последним 30 дням для актуальности
    elif available_days >= 30:
        lookback_days = 30
    
    cutoff_date = last_date - pd.Timedelta(days=lookback_days)
    recent_history = category_history_sorted[category_history_sorted['ds'] >= cutoff_date].copy()
    logger.info(f"Using last {lookback_days} days for category share calculation (prioritizing recent data)")
    
    # Вычисляем доли категорий с экспоненциальным взвешиванием (более свежие данные важнее)
    category_shares_dict = {}
    
    for category in recent_history['category'].unique():
        cat_history = recent_history[recent_history['category'] == category].copy()
        cat_history = cat_history.sort_values('ds')
        
        # Используем экспоненциальное взвешивание: более свежие дни имеют больший вес
        days_since_start = (cat_history['ds'] - cat_history['ds'].min()).dt.days
        max_days = days_since_start.max() if len(days_since_start) > 0 else 1
        # Экспоненциальные веса: последние дни имеют вес ~1.0, первые ~0.5
        weights = 0.5 + 0.5 * (days_since_start / max_days) if max_days > 0 else pd.Series([1.0] * len(cat_history))
        
        # Взвешенная сумма продаж категории
        weighted_cat_total = (cat_history['y'] * weights).sum()
        
        # Взвешенная сумма всех продаж
        all_history = recent_history.sort_values('ds')
        all_days_since_start = (all_history['ds'] - all_history['ds'].min()).dt.days
        all_max_days = all_days_since_start.max() if len(all_days_since_start) > 0 else 1
        all_weights = 0.5 + 0.5 * (all_days_since_start / all_max_days) if all_max_days > 0 else pd.Series([1.0] * len(all_history))
        weighted_all_total = (all_history['y'] * all_weights).sum()
        
        if weighted_all_total > 0:
            category_shares_dict[category] = weighted_cat_total / weighted_all_total
        else:
            # Fallback: равномерное распределение
            category_shares_dict[category] = 1.0 / len(recent_history['category'].unique())
    
    # Нормализуем доли
    total_share = sum(category_shares_dict.values())
    if total_share > 0:
        category_shares = pd.Series({k: v / total_share for k, v in category_shares_dict.items()})
    else:
        category_shares = pd.Series({cat: 1.0 / len(category_shares_dict) for cat in category_shares_dict.keys()})
    
    logger.info(f"Category shares (exponentially weighted, last {lookback_days} days):")
    for cat, share in category_shares.items():
        logger.info(f"  {cat}: {share:.2%}")
    
    # Распределяем прогноз по категориям
    results = []
    for category in category_shares.index:
        share = category_shares[category]
        
        # Создаем прогноз для категории
        cat_forecast = shop_forecast.copy()
        cat_forecast['category'] = category
        cat_forecast['yhat'] = cat_forecast['yhat'] * share
        cat_forecast['yhat_lower'] = cat_forecast['yhat_lower'] * share
        cat_forecast['yhat_upper'] = cat_forecast['yhat_upper'] * share
        
        results.append(cat_forecast)
    
    # Объединяем все категории
    result_df = pd.concat(results, ignore_index=True)
    result_df = result_df[['category', 'ds', 'yhat', 'yhat_lower', 'yhat_upper']].sort_values(['category', 'ds'])
    
    logger.info(f"Generated forecasts for {len(category_shares)} categories")
    logger.info(f"Total forecast rows: {len(result_df)}")
    
    # Сохраняем если указан путь
    if output_csv:
        os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
        result_df.to_csv(output_csv, index=False)
        logger.info(f"Saved category forecasts to: {output_csv}")
    
    return result_df


def get_category_forecast_by_name(
    category_name: str,
    distributed_forecast_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Извлекает прогноз для конкретной категории из распределенного прогноза.
    
    Args:
        category_name: Название категории
        distributed_forecast_df: DataFrame с распределенными прогнозами (от distribute_shop_forecast_to_categories)
        
    Returns:
        DataFrame с прогнозом для категории (ds, yhat, yhat_lower, yhat_upper)
    """
    cat_forecast = distributed_forecast_df[distributed_forecast_df['category'] == category_name].copy()
    
    if len(cat_forecast) == 0:
        raise ValueError(f"Category '{category_name}' not found in forecast")
    
    return cat_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].reset_index(drop=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Distribute shop-level forecast to categories")
    parser.add_argument("shop_forecast_csv", help="Path to shop-level forecast CSV")
    parser.add_argument("category_csv", help="Path to category history CSV")
    parser.add_argument("--output", type=str, default="data/processed/forecast_category_distributed.csv",
                       help="Output CSV path")
    
    args = parser.parse_args()
    
    result = distribute_shop_forecast_to_categories(
        shop_forecast_csv=args.shop_forecast_csv,
        category_csv=args.category_csv,
        horizon_days=90,
        output_csv=args.output
    )
    
    print("\n" + "="*60)
    print("Category forecast distribution completed!")
    print("="*60)
    print(f"\nOutput: {args.output}")
    print(f"\nPreview:")
    print(result.head(20).to_string(index=False))

