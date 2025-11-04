"""
Скрипт для исправления проблем с данными и переобучения модели.
"""
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.train import train_prophet
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_data_and_retrain(shop_csv_path: str, output_dir: str = "models/fixed"):
    """Исправляет проблемы в данных и переобучает модель."""
    logger.info("=" * 80)
    logger.info("ИСПРАВЛЕНИЕ ДАННЫХ И ПЕРЕОБУЧЕНИЕ")
    logger.info("=" * 80)
    
    # Загружаем данные
    df = pd.read_csv(shop_csv_path)
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds').reset_index(drop=True)
    
    logger.info(f"\n1. АНАЛИЗ ПРОБЛЕМ:")
    zero_count = (df['y'] == 0).sum()
    logger.info(f"   Нулевых значений: {zero_count} ({zero_count/len(df)*100:.1f}%)")
    
    # Исправление 1: Обработка нулевых значений (заменяем на медиану соседних дней)
    logger.info(f"\n2. ИСПРАВЛЕНИЕ 1: Обработка нулевых значений")
    df_fixed = df.copy()
    zero_mask = df_fixed['y'] == 0
    
    if zero_mask.sum() > 0:
        # Для каждого нуля берем медиану соседних 7 дней (3 до, 3 после, не включая нули)
        for idx in df_fixed[zero_mask].index:
            window_start = max(0, idx - 3)
            window_end = min(len(df_fixed), idx + 4)
            window = df_fixed.iloc[window_start:window_end]
            window_nonzero = window[window['y'] > 0]['y']
            if len(window_nonzero) > 0:
                replacement = window_nonzero.median()
                df_fixed.loc[idx, 'y'] = replacement
                logger.info(f"   Заменен 0 в {df_fixed.loc[idx, 'ds'].date()} на {replacement:.2f}")
            else:
                # Если все нули, используем общую медиану
                df_fixed.loc[idx, 'y'] = df_fixed[df_fixed['y'] > 0]['y'].median()
    
    # Исправление 2: Сглаживание выбросов (winsorization)
    logger.info(f"\n3. ИСПРАВЛЕНИЕ 2: Обработка выбросов (winsorization)")
    Q1 = df_fixed['y'].quantile(0.05)
    Q3 = df_fixed['y'].quantile(0.95)
    outliers = (df_fixed['y'] < Q1) | (df_fixed['y'] > Q3)
    outlier_count = outliers.sum()
    
    if outlier_count > 0:
        df_fixed.loc[df_fixed['y'] < Q1, 'y'] = Q1
        df_fixed.loc[df_fixed['y'] > Q3, 'y'] = Q3
        logger.info(f"   Обработано выбросов: {outlier_count}")
    
    # Исправление 3: Скользящее среднее для стабилизации
    logger.info(f"\n4. ИСПРАВЛЕНИЕ 3: Применение скользящего среднего (7 дней)")
    df_fixed['y_smooth'] = df_fixed['y'].rolling(window=7, center=True, min_periods=1).mean()
    df_fixed['y'] = df_fixed['y_smooth']
    df_fixed = df_fixed.drop(columns=['y_smooth'])
    
    # Сохраняем исправленные данные
    fixed_csv_path = shop_csv_path.replace('.csv', '_fixed.csv')
    df_fixed.to_csv(fixed_csv_path, index=False)
    logger.info(f"\n5. Исправленные данные сохранены: {fixed_csv_path}")
    
    # Тестируем разные конфигурации
    logger.info(f"\n6. ТЕСТИРОВАНИЕ РАЗНЫХ КОНФИГУРАЦИЙ:")
    
    configs = [
        {
            "name": "Конфигурация 1: Log-transform + Additive + Низкий changepoint",
            "log_transform": True,
            "changepoint_prior_scale": 0.01,  # Более консервативный
            "seasonality_prior_scale": 15.0,
            "seasonality_mode": "additive"
        },
        {
            "name": "Конфигурация 2: Без log-transform + Multiplicative + Средний changepoint",
            "log_transform": False,
            "changepoint_prior_scale": 0.05,
            "seasonality_prior_scale": 20.0,
            "seasonality_mode": "multiplicative"
        },
        {
            "name": "Конфигурация 3: Log-transform + Additive + Высокий changepoint",
            "log_transform": True,
            "changepoint_prior_scale": 0.20,
            "seasonality_prior_scale": 25.0,
            "seasonality_mode": "additive"
        },
        {
            "name": "Конфигурация 4: Без преобразований + Консервативные параметры",
            "log_transform": False,
            "changepoint_prior_scale": 0.01,
            "seasonality_prior_scale": 10.0,
            "seasonality_mode": "additive"
        },
    ]
    
    os.makedirs(output_dir, exist_ok=True)
    best_mape = float('inf')
    best_config = None
    best_result = None
    
    for i, config in enumerate(configs, 1):
        logger.info(f"\n   Тест {i}/{len(configs)}: {config['name']}")
        model_path = os.path.join(output_dir, f"model_{i}.pkl")
        
        try:
            result = train_prophet(
                shop_csv_path=fixed_csv_path,
                model_out_path=model_path,
                include_regressors=False,
                log_transform=config["log_transform"],
                interval_width=0.95,
                holdout_frac=0.2,
                changepoint_prior_scale=config["changepoint_prior_scale"],
                seasonality_prior_scale=config["seasonality_prior_scale"],
                seasonality_mode=config["seasonality_mode"]
            )
            
            mape = result["metrics"]["mape"]
            logger.info(f"      MAPE: {mape:.2f}%")
            
            if mape < best_mape:
                best_mape = mape
                best_config = config
                best_result = result
                
        except Exception as e:
            logger.error(f"      Ошибка: {str(e)}")
    
    logger.info(f"\n{'=' * 80}")
    logger.info("РЕЗУЛЬТАТЫ:")
    logger.info(f"{'=' * 80}")
    logger.info(f"Лучшая конфигурация: {best_config['name']}")
    logger.info(f"MAPE: {best_mape:.2f}%")
    logger.info(f"MAE: {best_result['metrics']['mae']:.2f}")
    logger.info(f"RMSE: {best_result['metrics']['rmse']:.2f}")
    
    if best_mape < 30:
        logger.info("✅ УЛУЧШЕНИЕ ДОСТИГНУТО!")
    elif best_mape < 50:
        logger.info("⚠️  Есть улучшение, но качество все еще низкое")
    else:
        logger.info("❌ Качество все еще плохое - возможно нужен другой подход")
    
    return fixed_csv_path, best_result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--shop_csv", default="data/processed/sales_data_shop.csv")
    parser.add_argument("--output_dir", default="models/fixed")
    args = parser.parse_args()
    
    fix_data_and_retrain(args.shop_csv, args.output_dir)


