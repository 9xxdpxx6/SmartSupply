#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для автоматического тестирования всех датасетов и подбора оптимальных параметров.
"""

import os
import sys
import pandas as pd
import json
from datetime import datetime
from typing import Dict, Any, List
import logging

# Добавляем путь к app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.preprocessing import parse_and_process
from app.train import train_prophet
from app.predict import predict_prophet

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_dataset_characteristics(csv_path: str) -> Dict[str, Any]:
    """Анализирует характеристики датасета."""
    logger.info(f"\n{'='*80}")
    logger.info(f"📊 АНАЛИЗ ДАТАСЕТА: {csv_path}")
    logger.info(f"{'='*80}")
    
    df = pd.read_csv(csv_path, nrows=1000)  # Читаем первые 1000 строк для анализа
    
    # Определяем формат
    has_order_date = 'order_date' in df.columns
    has_sale_date = 'Sale_Date' in df.columns
    
    if has_order_date:
        date_col = 'order_date'
        qty_col = 'qty_ordered'
        cat_col = 'category'
    elif has_sale_date:
        date_col = 'Sale_Date'
        qty_col = 'Quantity_Sold'
        cat_col = 'Product_Category'
    else:
        logger.error(f"❌ Неизвестный формат датасета: {csv_path}")
        return {}
    
    # Читаем полный датасет
    df_full = pd.read_csv(csv_path)
    df_full[date_col] = pd.to_datetime(df_full[date_col], errors='coerce')
    df_full = df_full.dropna(subset=[date_col])
    
    # Анализ
    stats = {
        'total_rows': len(df_full),
        'date_min': df_full[date_col].min().isoformat(),
        'date_max': df_full[date_col].max().isoformat(),
        'date_span_days': (df_full[date_col].max() - df_full[date_col].min()).days,
        'unique_dates': df_full[date_col].nunique(),
        'total_sales': float(df_full[qty_col].sum()),
        'mean_sales': float(df_full[qty_col].mean()),
        'std_sales': float(df_full[qty_col].std()),
        'cv': float(df_full[qty_col].std() / df_full[qty_col].mean()) if df_full[qty_col].mean() > 0 else 0.0,
    }
    
    if cat_col in df_full.columns:
        stats['unique_categories'] = df_full[cat_col].nunique()
        stats['categories'] = df_full[cat_col].value_counts().head(10).to_dict()
    
    logger.info(f"   Всего строк: {stats['total_rows']:,}")
    logger.info(f"   Период: {stats['date_min']} - {stats['date_max']} ({stats['date_span_days']} дней)")
    logger.info(f"   Уникальных дат: {stats['unique_dates']}")
    logger.info(f"   Всего продаж: {stats['total_sales']:,.0f}")
    logger.info(f"   Среднее: {stats['mean_sales']:.2f}, Std: {stats['std_sales']:.2f}")
    logger.info(f"   Коэффициент вариации (CV): {stats['cv']:.2f}")
    
    if stats['cv'] > 1.0:
        logger.warning(f"   ⚠️ ВЫСОКАЯ ВОЛАТИЛЬНОСТЬ (CV={stats['cv']:.2f})")
    elif stats['cv'] > 0.5:
        logger.info(f"   ℹ️ Умеренная волатильность (CV={stats['cv']:.2f})")
    else:
        logger.info(f"   ✓ Стабильные данные (CV={stats['cv']:.2f})")
    
    if 'unique_categories' in stats:
        logger.info(f"   Категорий: {stats['unique_categories']}")
    
    return stats


def test_dataset(
    csv_path: str,
    dataset_name: str,
    output_dir: str = "test_results"
) -> Dict[str, Any]:
    """Тестирует один датасет с автоматическим подбором параметров."""
    logger.info(f"\n{'='*80}")
    logger.info(f"🧪 ТЕСТИРОВАНИЕ: {dataset_name}")
    logger.info(f"{'='*80}")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    results = {
        'dataset_name': dataset_name,
        'csv_path': csv_path,
        'timestamp': datetime.now().isoformat(),
        'preprocessing': {},
        'training': {},
        'errors': []
    }
    
    try:
        # 1. Предобработка
        logger.info("\n📝 ШАГ 1: Предобработка данных...")
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        out_shop_csv = f"data/processed/{base_name}_shop.csv"
        out_category_csv = f"data/processed/{base_name}_category.csv"
        out_product_csv = f"data/processed/{base_name}_product.csv"
        
        preprocess_result = parse_and_process(
            csv_path,
            out_shop_csv,
            out_category_csv,
            out_product_csv=out_product_csv,
            force_weekly=False
        )
        
        results['preprocessing'] = {
            'success': True,
            'shop_csv': preprocess_result['shop_csv'],
            'category_csv': preprocess_result['category_csv'],
            'stats': preprocess_result['stats']
        }
        logger.info(f"✓ Предобработка завершена")
        logger.info(f"  Shop-level: {preprocess_result['stats']['shop_data_rows']} строк")
        logger.info(f"  Category-level: {preprocess_result['stats']['category_data_rows']} строк")
        
        # 2. Обучение shop-level модели с auto_tune
        logger.info("\n🎯 ШАГ 2: Обучение shop-level модели с auto_tune...")
        shop_model_path = f"models/test_{dataset_name}_shop.pkl"
        os.makedirs(os.path.dirname(shop_model_path), exist_ok=True)
        
        train_result = train_prophet(
            shop_csv_path=out_shop_csv,
            model_out_path=shop_model_path,
            include_regressors=False,
            log_transform=False,
            interval_width=0.95,
            holdout_frac=0.2,
            changepoint_prior_scale=0.01,
            seasonality_prior_scale=10.0,
            seasonality_mode='additive',
            auto_tune=True,  # АВТОМАТИЧЕСКИЙ ПОДБОР ПАРАМЕТРОВ
            skip_holdout=False,
            filter_column=None,
            filter_value=None
        )
        
        results['training'] = {
            'success': True,
            'model_path': train_result['model_path'],
            'metrics': train_result['metrics'],
            'train_range': train_result['train_range'],
            'test_range': train_result['test_range'],
            'n_train': train_result['n_train'],
            'n_test': train_result['n_test']
        }
        
        logger.info(f"✓ Обучение завершено")
        if train_result['metrics'].get('mape') is not None:
            logger.info(f"  MAPE: {train_result['metrics']['mape']:.2f}%")
            logger.info(f"  MAE: {train_result['metrics']['mae']:.2f}")
            logger.info(f"  RMSE: {train_result['metrics']['rmse']:.2f}")
        
        # 3. Тестирование прогноза
        logger.info("\n🔮 ШАГ 3: Тестирование прогноза на 30 дней...")
        forecast_df = predict_prophet(
            model_path=shop_model_path,
            horizon_days=30,
            log_transform=train_result['metrics'].get('log_transform', False),
            regressor_fill_method='forward',
            smooth_transition=False
        )
        
        results['forecast'] = {
            'success': True,
            'n_predictions': len(forecast_df),
            'forecast_mean': float(forecast_df['yhat'].mean()),
            'forecast_std': float(forecast_df['yhat'].std()),
            'forecast_min': float(forecast_df['yhat'].min()),
            'forecast_max': float(forecast_df['yhat'].max())
        }
        
        logger.info(f"✓ Прогноз создан: {len(forecast_df)} дней")
        logger.info(f"  Среднее: {results['forecast']['forecast_mean']:.2f}")
        
        # 4. Тестирование категорий (если есть)
        if preprocess_result['stats']['unique_categories'] > 0:
            logger.info("\n📦 ШАГ 4: Тестирование категорий...")
            
            # Берем топ-3 категории по объему продаж
            category_df = pd.read_csv(out_category_csv)
            category_totals = category_df.groupby('category')['y'].sum().sort_values(ascending=False)
            top_categories = category_totals.head(3).index.tolist()
            
            results['category_tests'] = []
            
            for cat_name in top_categories:
                try:
                    logger.info(f"  Тестирую категорию: {cat_name}...")
                    cat_model_path = f"models/test_{dataset_name}_category_{cat_name.replace(' ', '_')}.pkl"
                    
                    cat_train_result = train_prophet(
                        shop_csv_path=out_category_csv,
                        model_out_path=cat_model_path,
                        include_regressors=False,
                        log_transform=False,
                        interval_width=0.95,
                        holdout_frac=0.2,
                        changepoint_prior_scale=0.01,
                        seasonality_prior_scale=10.0,
                        seasonality_mode='additive',
                        auto_tune=True,  # АВТОМАТИЧЕСКИЙ ПОДБОР
                        skip_holdout=False,
                        filter_column='category',
                        filter_value=cat_name
                    )
                    
                    cat_result = {
                        'category': cat_name,
                        'success': True,
                        'metrics': cat_train_result['metrics'],
                        'n_train': cat_train_result['n_train'],
                        'n_test': cat_train_result['n_test']
                    }
                    
                    if cat_train_result['metrics'].get('mape') is not None:
                        logger.info(f"    ✓ MAPE: {cat_train_result['metrics']['mape']:.2f}%")
                    
                    results['category_tests'].append(cat_result)
                except Exception as e:
                    logger.error(f"    ❌ Ошибка для категории {cat_name}: {str(e)}")
                    results['category_tests'].append({
                        'category': cat_name,
                        'success': False,
                        'error': str(e)
                    })
        
        logger.info(f"\n✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО: {dataset_name}")
        
    except Exception as e:
        logger.error(f"\n❌ ОШИБКА при тестировании {dataset_name}: {str(e)}", exc_info=True)
        results['errors'].append(str(e))
        results['success'] = False
    
    return results


def main():
    """Основная функция для тестирования всех датасетов."""
    logger.info("="*80)
    logger.info("🚀 АВТОМАТИЧЕСКОЕ ТЕСТИРОВАНИЕ ДАТАСЕТОВ С ПОДБОРОМ ПАРАМЕТРОВ")
    logger.info("="*80)
    
    # Список датасетов для тестирования
    datasets = [
        {
            'name': 'sales_06_FY2020-21',
            'path': 'sales_06_FY2020-21.csv'
        },
        {
            'name': 'retail_sales_dataset',
            'path': 'retail_sales_dataset.csv'
        },
        {
            'name': 'customer_shopping_data',
            'path': 'customer_shopping_data.csv'
        }
    ]
    
    all_results = {}
    
    # Анализ всех датасетов
    logger.info("\n" + "="*80)
    logger.info("📊 ЭТАП 1: АНАЛИЗ ДАТАСЕТОВ")
    logger.info("="*80)
    
    for dataset in datasets:
        if os.path.exists(dataset['path']):
            stats = analyze_dataset_characteristics(dataset['path'])
            all_results[dataset['name']] = {'stats': stats}
        else:
            logger.warning(f"⚠️ Файл не найден: {dataset['path']}")
    
    # Тестирование всех датасетов
    logger.info("\n" + "="*80)
    logger.info("🧪 ЭТАП 2: ТЕСТИРОВАНИЕ С АВТОПОДБОРОМ ПАРАМЕТРОВ")
    logger.info("="*80)
    
    for dataset in datasets:
        if not os.path.exists(dataset['path']):
            continue
        
        try:
            result = test_dataset(
                csv_path=dataset['path'],
                dataset_name=dataset['name'],
                output_dir="test_results"
            )
            all_results[dataset['name']]['test_result'] = result
        except Exception as e:
            logger.error(f"❌ Критическая ошибка для {dataset['name']}: {str(e)}", exc_info=True)
            all_results[dataset['name']]['error'] = str(e)
    
    # Сохранение результатов
    logger.info("\n" + "="*80)
    logger.info("💾 СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    logger.info("="*80)
    
    os.makedirs("test_results", exist_ok=True)
    results_file = f"test_results/test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"✓ Результаты сохранены: {results_file}")
    
    # Сводка результатов
    logger.info("\n" + "="*80)
    logger.info("📋 СВОДКА РЕЗУЛЬТАТОВ")
    logger.info("="*80)
    
    for dataset_name, data in all_results.items():
        logger.info(f"\n📊 {dataset_name}:")
        if 'test_result' in data:
            tr = data['test_result']
            if tr.get('training', {}).get('success'):
                metrics = tr['training'].get('metrics', {})
                if metrics.get('mape') is not None:
                    logger.info(f"  Shop-level MAPE: {metrics['mape']:.2f}%")
                    logger.info(f"  MAE: {metrics['mae']:.2f}")
                    logger.info(f"  RMSE: {metrics['rmse']:.2f}")
                else:
                    logger.info(f"  ✓ Модель обучена (skip_holdout использован)")
            
            if 'category_tests' in tr:
                logger.info(f"  Категории протестированы: {len(tr['category_tests'])}")
                for cat_test in tr['category_tests']:
                    if cat_test.get('success') and cat_test.get('metrics', {}).get('mape'):
                        logger.info(f"    - {cat_test['category']}: MAPE={cat_test['metrics']['mape']:.2f}%")
        elif 'error' in data:
            logger.error(f"  ❌ Ошибка: {data['error']}")
    
    logger.info("\n" + "="*80)
    logger.info("✅ ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ")
    logger.info("="*80)


if __name__ == "__main__":
    main()

