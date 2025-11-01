"""
Скрипт для автоматического тестирования различных конфигураций модели.
Позволяет быстро проверить комбинации параметров без использования GUI.
"""
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.train import train_prophet
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_configurations(shop_csv_path: str, output_dir: str = "models/test_configs"):
    """
    Тестирует различные конфигурации модели и выводит результаты.
    
    Args:
        shop_csv_path: Путь к обработанным данным shop-level CSV
        output_dir: Директория для сохранения тестовых моделей
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Проверяем наличие данных
    if not os.path.exists(shop_csv_path):
        logger.error(f"Файл не найден: {shop_csv_path}")
        return
    
    logger.info(f"Тестирование конфигураций модели на данных: {shop_csv_path}")
    logger.info("=" * 80)
    
    # Конфигурации для тестирования
    configs = [
        {
            "name": "Вариант А (рекомендуется): Log-transform + Additive",
            "include_regressors": False,
            "log_transform": True,
            "interval_width": 0.95,
            "holdout_frac": 0.2,
            "changepoint_prior_scale": 0.10,
            "seasonality_prior_scale": 18.0,
            "seasonality_mode": "additive"
        },
        {
            "name": "Вариант Б: Без log-transform + Multiplicative",
            "include_regressors": False,
            "log_transform": False,
            "interval_width": 0.95,
            "holdout_frac": 0.2,
            "changepoint_prior_scale": 0.15,
            "seasonality_prior_scale": 22.0,
            "seasonality_mode": "multiplicative"
        },
        {
            "name": "Вариант В: Log-transform + Additive (более гибкий)",
            "include_regressors": False,
            "log_transform": True,
            "interval_width": 0.95,
            "holdout_frac": 0.2,
            "changepoint_prior_scale": 0.15,
            "seasonality_prior_scale": 20.0,
            "seasonality_mode": "additive"
        },
        {
            "name": "Вариант Г: Без log-transform + Multiplicative (высокая сезонность)",
            "include_regressors": False,
            "log_transform": False,
            "interval_width": 0.95,
            "holdout_frac": 0.2,
            "changepoint_prior_scale": 0.20,
            "seasonality_prior_scale": 25.0,
            "seasonality_mode": "multiplicative"
        },
        {
            "name": "Вариант Д: Log-transform + Additive + Regressors",
            "include_regressors": True,
            "log_transform": True,
            "interval_width": 0.95,
            "holdout_frac": 0.2,
            "changepoint_prior_scale": 0.10,
            "seasonality_prior_scale": 18.0,
            "seasonality_mode": "additive"
        },
        {
            "name": "Вариант Е: Базовый (дефолтные параметры без изменений)",
            "include_regressors": False,
            "log_transform": False,
            "interval_width": 0.95,
            "holdout_frac": 0.2,
            "changepoint_prior_scale": 0.05,
            "seasonality_prior_scale": 10.0,
            "seasonality_mode": "additive"
        },
    ]
    
    results = []
    
    for i, config in enumerate(configs, 1):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Тест {i}/{len(configs)}: {config['name']}")
        logger.info(f"{'=' * 80}")
        
        model_name = f"test_model_{i}"
        model_path = os.path.join(output_dir, f"{model_name}.pkl")
        
        try:
            # Обучаем модель
            result = train_prophet(
                shop_csv_path=shop_csv_path,
                model_out_path=model_path,
                include_regressors=config["include_regressors"],
                log_transform=config["log_transform"],
                interval_width=config["interval_width"],
                holdout_frac=config["holdout_frac"],
                changepoint_prior_scale=config["changepoint_prior_scale"],
                seasonality_prior_scale=config["seasonality_prior_scale"],
                seasonality_mode=config["seasonality_mode"]
            )
            
            metrics = result["metrics"]
            
            # Сохраняем результат
            test_result = {
                "config_name": config["name"],
                "model_path": model_path,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape": metrics["mape"],
                "config": config,
                "train_samples": result["n_train"],
                "test_samples": result["n_test"]
            }
            
            results.append(test_result)
            
            # Выводим метрики
            logger.info(f"✅ Успешно обучена модель")
            logger.info(f"   MAE:  {metrics['mae']:.2f}")
            logger.info(f"   RMSE: {metrics['rmse']:.2f}")
            logger.info(f"   MAPE: {metrics['mape']:.2f}%")
            
            # Оценка качества
            mape_val = metrics['mape']
            if mape_val > 50:
                logger.info(f"   🚨 КРИТИЧЕСКОЕ качество (MAPE > 50%)")
            elif mape_val > 30:
                logger.info(f"   ⚠️  Плохое качество (MAPE > 30%)")
            elif mape_val > 20:
                logger.info(f"   🟡 Удовлетворительное качество (MAPE > 20%)")
            elif mape_val > 15:
                logger.info(f"   ✅ Хорошее качество (MAPE > 15%)")
            else:
                logger.info(f"   ✅✅ ОТЛИЧНОЕ качество (MAPE ≤ 15%)")
            
        except Exception as e:
            logger.error(f"❌ Ошибка при обучении: {str(e)}")
            results.append({
                "config_name": config["name"],
                "error": str(e)
            })
    
    # Сводная таблица результатов
    logger.info(f"\n{'=' * 80}")
    logger.info("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    logger.info(f"{'=' * 80}")
    
    # Сортируем по MAPE (лучший = первый)
    valid_results = [r for r in results if "mape" in r]
    valid_results.sort(key=lambda x: x["mape"])
    
    logger.info(f"\n{'Конфигурация':<50} {'MAPE':<10} {'MAE':<10} {'RMSE':<10}")
    logger.info("-" * 80)
    
    for result in valid_results:
        mape_val = result["mape"]
        if isinstance(mape_val, (int, float)):
            status = "✅" if mape_val <= 20 else "⚠️" if mape_val <= 30 else "🚨"
            logger.info(f"{status} {result['config_name']:<45} {mape_val:>6.2f}%  {result['mae']:>8.2f}  {result['rmse']:>8.2f}")
        else:
            logger.info(f"❌ {result['config_name']:<45} {'N/A':<10}")
    
    # Рекомендация
    if valid_results:
        best = valid_results[0]
        logger.info(f"\n{'=' * 80}")
        logger.info("🏆 ЛУЧШАЯ КОНФИГУРАЦИЯ:")
        logger.info(f"{'=' * 80}")
        logger.info(f"Название: {best['config_name']}")
        logger.info(f"MAPE: {best['mape']:.2f}%")
        logger.info(f"MAE:  {best['mae']:.2f}")
        logger.info(f"RMSE: {best['rmse']:.2f}")
        logger.info(f"\nПараметры:")
        for key, value in best['config'].items():
            if key != 'name':
                logger.info(f"  {key}: {value}")
        
        # Сохраняем лучший результат в JSON
        best_result_path = os.path.join(output_dir, "best_config.json")
        with open(best_result_path, 'w', encoding='utf-8') as f:
            json.dump(best, f, indent=2, ensure_ascii=False)
        logger.info(f"\n💾 Лучшая конфигурация сохранена в: {best_result_path}")
    
    # Сохраняем все результаты
    all_results_path = os.path.join(output_dir, "all_results.json")
    with open(all_results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 Все результаты сохранены в: {all_results_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Тестирование различных конфигураций модели Prophet")
    parser.add_argument(
        "--shop_csv",
        type=str,
        default="data/processed/sales_data_shop.csv",
        help="Путь к обработанным данным shop-level CSV"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/test_configs",
        help="Директория для сохранения тестовых моделей"
    )
    
    args = parser.parse_args()
    
    test_model_configurations(args.shop_csv, args.output_dir)

