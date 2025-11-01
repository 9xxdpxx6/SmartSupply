"""
Скрипт для автоматической настройки гиперпараметров Prophet.
Использует grid search для поиска оптимальных параметров.
"""
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.train import train_prophet
import json
import logging

logging.basicConfig(level=logging.WARNING)  # Уменьшаем логирование
logger = logging.getLogger(__name__)

def hyperparameter_tuning(shop_csv_path: str, output_dir: str = "models/tuned"):
    """
    Автоматическая настройка гиперпараметров Prophet.
    """
    print("=" * 80)
    print("АВТОМАТИЧЕСКАЯ НАСТРОЙКА ГИПЕРПАРАМЕТРОВ PROPHET")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Определяем сетку параметров для поиска
    param_grid = {
        'changepoint_prior_scale': [0.005, 0.01, 0.05, 0.1],
        'seasonality_prior_scale': [5.0, 10.0, 15.0, 20.0],
        'seasonality_mode': ['additive'],  # Только additive, т.к. multiplicative хуже
        'log_transform': [False],  # Без log-transform, т.к. на улучшенных данных хуже
    }
    
    print(f"\nИщу оптимальные параметры...")
    print(f"Вариантов для проверки: {np.prod([len(v) for v in param_grid.values()])}")
    
    results = []
    best_mape = float('inf')
    best_params = None
    best_result = None
    
    # Генерируем все комбинации
    keys = param_grid.keys()
    values = param_grid.values()
    
    for i, combination in enumerate(product(*values), 1):
        params = dict(zip(keys, combination))
        
        print(f"\n[{i}/{np.prod([len(v) for v in values])}] Тестирую: {params}")
        
        try:
            model_path = os.path.join(output_dir, f"model_{i}.pkl")
            
            result = train_prophet(
                shop_csv_path=shop_csv_path,
                model_out_path=model_path,
                include_regressors=False,
                log_transform=params['log_transform'],
                interval_width=0.95,
                holdout_frac=0.2,
                changepoint_prior_scale=params['changepoint_prior_scale'],
                seasonality_prior_scale=params['seasonality_prior_scale'],
                seasonality_mode=params['seasonality_mode']
            )
            
            mape = result['metrics']['mape']
            
            print(f"    MAPE: {mape:.2f}%")
            
            results.append({
                'params': params,
                'mape': mape,
                'mae': result['metrics']['mae'],
                'rmse': result['metrics']['rmse'],
                'model_path': model_path
            })
            
            if mape < best_mape:
                best_mape = mape
                best_params = params
                best_result = result
                print(f"    *** НОВЫЙ ЛУЧШИЙ РЕЗУЛЬТАТ! ***")
                
        except Exception as e:
            print(f"    ОШИБКА: {str(e)}")
            continue
    
    # Выводим результаты
    print(f"\n{'=' * 80}")
    print("РЕЗУЛЬТАТЫ НАСТРОЙКИ")
    print(f"{'=' * 80}")
    
    # Сортируем по MAPE
    results.sort(key=lambda x: x['mape'])
    
    print(f"\nТОП-5 ЛУЧШИХ КОНФИГУРАЦИЙ:")
    print(f"{'№':<4} {'MAPE':<10} {'MAE':<10} {'RMSE':<10} {'Параметры'}")
    print("-" * 80)
    
    for idx, res in enumerate(results[:5], 1):
        params_str = f"chp={res['params']['changepoint_prior_scale']:.3f}, seas={res['params']['seasonality_prior_scale']:.1f}"
        print(f"{idx:<4} {res['mape']:<10.2f} {res['mae']:<10.2f} {res['rmse']:<10.2f} {params_str}")
    
    print(f"\n{'=' * 80}")
    print("ЛУЧШАЯ КОНФИГУРАЦИЯ:")
    print(f"{'=' * 80}")
    print(f"MAPE: {best_mape:.2f}%")
    print(f"MAE:  {best_result['metrics']['mae']:.2f}")
    print(f"RMSE: {best_result['metrics']['rmse']:.2f}")
    print(f"\nПараметры:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    # Сохраняем результаты
    best_config_path = os.path.join(output_dir, "best_config.json")
    with open(best_config_path, 'w', encoding='utf-8') as f:
        json.dump({
            'params': best_params,
            'metrics': best_result['metrics'],
            'model_path': best_result.get('model_path', '')
        }, f, indent=2, ensure_ascii=False)
    
    all_results_path = os.path.join(output_dir, "all_results.json")
    with open(all_results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Результаты сохранены:")
    print(f"   Лучшая конфигурация: {best_config_path}")
    print(f"   Все результаты: {all_results_path}")
    
    return best_params, best_result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Настройка гиперпараметров Prophet")
    parser.add_argument("--shop_csv", default="data/processed/sales_data_shop.csv")
    parser.add_argument("--output_dir", default="models/tuned")
    
    args = parser.parse_args()
    
    hyperparameter_tuning(args.shop_csv, args.output_dir)

