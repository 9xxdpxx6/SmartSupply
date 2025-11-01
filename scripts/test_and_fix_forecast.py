"""
Автоматический тест и исправление проблем прогнозирования.
Проверяет завышение в начале прогноза и coverage CI.
"""
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import json
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.train import train_prophet
from app.predict import predict_prophet
from app.diagnostics import diagnose_model, save_diagnostics, calculate_coverage
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_forecast_issues(
    df_history: pd.DataFrame,
    df_forecast: pd.DataFrame,
    model_path: str
) -> dict:
    """Анализирует проблемы в прогнозе."""
    issues = {
        'overestimation_start': False,
        'overestimation_severity': 0.0,
        'low_coverage': False,
        'coverage_rate': 0.0,
        'first_day_overestimation': 0.0,
        'trend_mismatch': False
    }
    
    # Находим пересечение дат
    df_combined = df_history.merge(
        df_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
        on='ds',
        how='inner'
    )
    
    if len(df_combined) == 0:
        return issues
    
    # Проверка завышения в начале
    first_5_days = df_combined.head(5)
    if len(first_5_days) > 0:
        # Проверяем первые 3 дня
        for idx, row in first_5_days.head(3).iterrows():
            if row['yhat'] > row['y'] * 1.15:  # Завышение > 15%
                issues['overestimation_start'] = True
                severity = (row['yhat'] - row['y']) / row['y'] * 100
                if severity > issues['overestimation_severity']:
                    issues['overestimation_severity'] = severity
        
        # Первый день
        first_row = first_5_days.iloc[0]
        if 'y' in first_row and 'yhat' in first_row:
            issues['first_day_overestimation'] = (first_row['yhat'] - first_row['y']) / first_row['y'] * 100 if first_row['y'] > 0 else 0
    
    # Coverage
    if 'yhat_lower' in df_combined.columns and 'yhat_upper' in df_combined.columns:
        coverage_result = calculate_coverage(
            df_combined['y'].values,
            df_combined['yhat_lower'].values,
            df_combined['yhat_upper'].values
        )
        # calculate_coverage возвращает dict
        if isinstance(coverage_result, dict):
            coverage_rate = coverage_result.get('coverage_rate', 0.0)
        else:
            coverage_rate = float(coverage_result) if isinstance(coverage_result, (int, float)) else 0.0
        
        issues['coverage_rate'] = coverage_rate
        issues['low_coverage'] = coverage_rate < 0.85
    
    # Trend mismatch
    if len(df_combined) >= 10:
        from scipy import stats
        dates_num = pd.to_datetime(df_combined['ds']).astype(int) / 1e9
        y_trend = stats.linregress(dates_num, df_combined['y'].values)[0]
        yhat_trend = stats.linregress(dates_num, df_combined['yhat'].values)[0]
        
        if abs(y_trend) > 1e-10:
            trend_diff_pct = abs((yhat_trend - y_trend) / y_trend) * 100
            issues['trend_mismatch'] = trend_diff_pct > 20
    
    return issues


def test_forecast_cycle(
    shop_csv_path: str,
    max_iterations: int = 3
):
    """Тестирует и исправляет проблемы циклически."""
    logger.info("=" * 80)
    logger.info("АВТОМАТИЧЕСКОЕ ТЕСТИРОВАНИЕ И ИСПРАВЛЕНИЕ ПРОГНОЗА")
    logger.info("=" * 80)
    
    # Загружаем данные
    logger.info(f"\n1. Загрузка данных: {shop_csv_path}")
    df_data = pd.read_csv(shop_csv_path)
    df_data['ds'] = pd.to_datetime(df_data['ds'])
    df_data = df_data.sort_values('ds').reset_index(drop=True)
    
    last_date = df_data['ds'].max()
    logger.info(f"   Последняя дата: {last_date.date()}")
    logger.info(f"   Всего записей: {len(df_data)}")
    
    # Разделяем на train/test для тестирования
    holdout_frac = 0.2
    n_train = int(len(df_data) * (1 - holdout_frac))
    df_train = df_data.iloc[:n_train].copy()
    df_test = df_data.iloc[n_train:].copy()
    
    logger.info(f"   Train: {len(df_train)} samples ({df_train['ds'].min().date()} to {df_train['ds'].max().date()})")
    logger.info(f"   Test: {len(df_test)} samples ({df_test['ds'].min().date()} to {df_test['ds'].max().date()})")
    
    # Сохраняем train данные во временный файл
    temp_train_csv = "data/processed/temp_train_test.csv"
    os.makedirs(os.path.dirname(temp_train_csv), exist_ok=True)
    df_train.to_csv(temp_train_csv, index=False)
    
    # Тестируем с разными конфигурациями
    best_result = None
    best_score = float('inf')
    
    configs_to_test = [
        # Попытка 1: Базовый с auto_tune
        {
            'name': 'auto_tune_baseline',
            'auto_tune': True,
            'include_regressors': False,
            'log_transform': False,
            'interval_width': 0.95,
            'changepoint_prior_scale': 0.01,
            'seasonality_prior_scale': 10.0
        },
        # Попытка 2: С регрессорами
        {
            'name': 'auto_tune_with_regressors',
            'auto_tune': True,
            'include_regressors': True,
            'log_transform': False,
            'interval_width': 0.95
        },
        # Попытка 3: С log transform
        {
            'name': 'auto_tune_log_transform',
            'auto_tune': True,
            'include_regressors': False,
            'log_transform': True,
            'interval_width': 0.95
        },
    ]
    
    for iteration, config in enumerate(configs_to_test[:max_iterations], 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"ИТЕРАЦИЯ {iteration}: Тестирование конфигурации '{config['name']}'")
        logger.info(f"{'='*80}")
        
        try:
            model_path = f"models/test_fix_iter{iteration}.pkl"
            
            # Обучение
            logger.info(f"\n2. Обучение модели...")
            result = train_prophet(
                shop_csv_path=temp_train_csv,
                model_out_path=model_path,
                include_regressors=config.get('include_regressors', False),
                log_transform=config.get('log_transform', False),
                interval_width=config.get('interval_width', 0.95),
                holdout_frac=0.0,  # Не используем holdout здесь, уже разделили
                changepoint_prior_scale=config.get('changepoint_prior_scale', 0.01),
                seasonality_prior_scale=config.get('seasonality_prior_scale', 10.0),
                seasonality_mode=config.get('seasonality_mode', 'additive'),
                auto_tune=config.get('auto_tune', False)
            )
            
            logger.info(f"   MAPE: {result['metrics']['mape']:.2f}%")
            
            # Генерируем прогноз на тестовый период
            logger.info(f"\n3. Генерация прогноза с smooth_transition=True...")
            
            # Сначала получаем прогноз на исторические даты для сравнения с тестом
            model = joblib.load(model_path)
            
            # Генерируем прогноз ТОЛЬКО на будущие даты (как в реальном использовании)
            # Используем predict_prophet для правильного применения smooth_transition
            horizon_days = len(df_test)
            
            # Получаем регрессоры если нужно
            regressors_csv = None
            if config.get('include_regressors', False) and 'avg_price' in df_data.columns:
                # Сохраняем train данные с регрессорами во временный файл для predict_prophet
                temp_regressors_csv = "data/processed/temp_regressors.csv"
                df_train.to_csv(temp_regressors_csv, index=False)
                regressors_csv = temp_regressors_csv
            
            # Генерируем прогноз через predict_prophet (правильно применяет smooth_transition)
            forecast_test_smooth = predict_prophet(
                model_path=model_path,
                horizon_days=horizon_days,
                last_known_regressors_csv=regressors_csv,
                log_transform=config.get('log_transform', False),
                regressor_fill_method='forward',
                smooth_transition=True,
                smooth_days=21,
                smooth_alpha=0.6,
                max_change_pct=0.01
            )
            
            # Обрезаем до нужных дат теста (могут быть лишние дни из-за округления)
            test_start = df_test['ds'].min()
            test_end = df_test['ds'].max()
            forecast_test_smooth = forecast_test_smooth[
                (forecast_test_smooth['ds'] >= test_start) & 
                (forecast_test_smooth['ds'] <= test_end)
            ].copy()
            
            # Убеждаемся, что даты совпадают
            if len(forecast_test_smooth) != len(df_test):
                logger.warning(f"   Размеры не совпадают: прогноз={len(forecast_test_smooth)}, тест={len(df_test)}")
                # Мерджим по датам
                forecast_test_smooth = forecast_test_smooth.merge(
                    df_test[['ds', 'y']],
                    on='ds',
                    how='inner'
                )
                forecast_test_smooth = forecast_test_smooth[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'y']].copy()
            
            # Анализ проблем
            logger.info(f"\n4. Анализ проблем...")
            issues = analyze_forecast_issues(df_test, forecast_test_smooth, model_path)
            
            logger.info(f"   Завышение в начале: {'ДА' if issues['overestimation_start'] else 'НЕТ'}")
            if issues['overestimation_start']:
                logger.info(f"   Серьезность завышения: {issues['overestimation_severity']:.1f}%")
            logger.info(f"   Завышение первого дня: {issues['first_day_overestimation']:.1f}%")
            logger.info(f"   Coverage CI: {issues['coverage_rate']*100:.1f}% {'⚠️ НИЗКОЕ' if issues['low_coverage'] else '✅'}")
            logger.info(f"   Несоответствие тренда: {'ДА' if issues['trend_mismatch'] else 'НЕТ'}")
            
            # Вычисляем score (меньше = лучше)
            score = 0
            if issues['overestimation_start']:
                score += issues['overestimation_severity'] * 2  # Штраф за завышение
            if issues['low_coverage']:
                score += (0.85 - issues['coverage_rate']) * 500  # Очень большой штраф
            score += abs(issues['first_day_overestimation']) * 1.5  # Штраф за первый день
            
            logger.info(f"   Общий score: {score:.2f} (меньше = лучше)")
            
            # Сохраняем лучший результат
            if score < best_score:
                best_score = score
                best_result = {
                    'config': config,
                    'model_path': model_path,
                    'issues': issues,
                    'score': score,
                    'metrics': result['metrics'],
                    'forecast': forecast_test_smooth
                }
                logger.info(f"   ✅ НОВЫЙ ЛУЧШИЙ РЕЗУЛЬТАТ! Score: {score:.2f}")
            
            # Если проблемы решены - останавливаемся
            if not issues['overestimation_start'] and not issues['low_coverage'] and abs(issues['first_day_overestimation']) < 5:
                logger.info(f"\n✅ ПРОБЛЕМЫ РЕШЕНЫ! Останавливаем тестирование.")
                break
                
        except Exception as e:
            logger.error(f"   ❌ Ошибка в итерации {iteration}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Итоговый отчет
    logger.info(f"\n{'='*80}")
    logger.info("ИТОГОВЫЙ ОТЧЕТ")
    logger.info(f"{'='*80}")
    
    if best_result:
        logger.info(f"\nЛучшая конфигурация: {best_result['config']['name']}")
        logger.info(f"   Model path: {best_result['model_path']}")
        logger.info(f"   MAPE: {best_result['metrics']['mape']:.2f}%")
        logger.info(f"   Coverage: {best_result['issues']['coverage_rate']*100:.1f}%")
        logger.info(f"   Завышение первого дня: {best_result['issues']['first_day_overestimation']:.1f}%")
        logger.info(f"   Score: {best_result['score']:.2f}")
        
        # Сохраняем итоговый прогноз
        output_path = "data/processed/fixed_forecast.csv"
        best_result['forecast'].to_csv(output_path, index=False)
        logger.info(f"\n   Итоговый прогноз сохранен: {output_path}")
        
        return best_result
    else:
        logger.error("❌ Не удалось найти рабочую конфигурацию")
        return None


if __name__ == "__main__":
    shop_csv = "data/processed/sales_data_shop.csv"
    
    if not os.path.exists(shop_csv):
        logger.error(f"Файл не найден: {shop_csv}")
        sys.exit(1)
    
    result = test_forecast_cycle(shop_csv, max_iterations=3)
    
    if result:
        logger.info("\n✅ Тестирование завершено успешно!")
    else:
        logger.error("\n❌ Тестирование не дало результатов")
        sys.exit(1)

