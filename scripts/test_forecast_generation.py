"""
Скрипт для тестирования генерации прогноза.
Проверяет, что прогноз правильно генерируется на будущие даты.
"""
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.predict import predict_prophet
from app.train import train_prophet
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_forecast_generation(shop_csv_path: str, model_path: str = None, horizon_days: int = 30):
    """
    Тестирует генерацию прогноза.
    
    Args:
        shop_csv_path: Путь к обработанным данным
        model_path: Путь к модели (если None, обучит новую)
        horizon_days: Количество дней для прогноза
    """
    logger.info("=" * 80)
    logger.info("ТЕСТИРОВАНИЕ ГЕНЕРАЦИИ ПРОГНОЗА")
    logger.info("=" * 80)
    
    # Шаг 1: Загрузить данные и проверить последнюю дату
    logger.info(f"\n1. Загрузка данных: {shop_csv_path}")
    df_data = pd.read_csv(shop_csv_path)
    df_data['ds'] = pd.to_datetime(df_data['ds'])
    df_data = df_data.sort_values('ds').reset_index(drop=True)
    
    last_date = df_data['ds'].max()
    first_date = df_data['ds'].min()
    logger.info(f"   Дата начала данных: {first_date.date()}")
    logger.info(f"   Последняя дата в данных: {last_date.date()}")
    logger.info(f"   Всего записей: {len(df_data)}")
    
    # Шаг 2: Обучить модель если нужно
    if model_path is None or not os.path.exists(model_path):
        logger.info(f"\n2. Обучение модели...")
        model_path = "models/test_forecast_model.pkl"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        result = train_prophet(
            shop_csv_path=shop_csv_path,
            model_out_path=model_path,
            include_regressors=False,
            log_transform=True,
            interval_width=0.95,
            holdout_frac=0.2,
            changepoint_prior_scale=0.10,
            seasonality_prior_scale=18.0,
            seasonality_mode="additive"
        )
        logger.info(f"   ✅ Модель обучена: {model_path}")
        logger.info(f"   Метрики: MAE={result['metrics']['mae']:.2f}, MAPE={result['metrics']['mape']:.2f}%")
        
        # Получаем последнюю дату обучения из результата
        train_end_str = result['train_range']['end']
        train_end_date = pd.to_datetime(train_end_str).date()
        logger.info(f"   Последняя дата обучения: {train_end_date}")
    else:
        logger.info(f"\n2. Использование существующей модели: {model_path}")
        # Нужно получить последнюю дату обучения из данных
        train_end_date = last_date.date()  # Приблизительно
        logger.info(f"   (Предполагаемая последняя дата обучения: {train_end_date})")
    
    # Шаг 3: Генерация прогноза
    logger.info(f"\n3. Генерация прогноза на {horizon_days} дней...")
    try:
        forecast_df = predict_prophet(
            model_path=model_path,
            horizon_days=horizon_days,
            log_transform=True,
            regressor_fill_method='forward'
        )
        
        logger.info(f"   ✅ Прогноз сгенерирован")
        logger.info(f"   Количество прогнозов: {len(forecast_df)}")
        
    except Exception as e:
        logger.error(f"   ❌ Ошибка генерации прогноза: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Шаг 4: Проверка результатов
    logger.info(f"\n4. Проверка результатов прогноза...")
    
    # Проверка наличия обязательных колонок
    required_cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
    missing_cols = [col for col in required_cols if col not in forecast_df.columns]
    if missing_cols:
        logger.error(f"   ❌ Отсутствуют колонки: {missing_cols}")
        return False
    else:
        logger.info(f"   ✅ Все обязательные колонки присутствуют")
    
    # Проверка типов данных
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
    
    # Проверка дат - должны быть в будущем относительно последней даты данных
    forecast_start = forecast_df['ds'].min()
    forecast_end = forecast_df['ds'].max()
    
    logger.info(f"   Дата начала прогноза: {forecast_start.date()}")
    logger.info(f"   Дата конца прогноза: {forecast_end.date()}")
    logger.info(f"   Количество дней прогноза: {(forecast_end - forecast_start).days + 1}")
    
    # Проверка, что прогноз начинается после последней даты данных
    if forecast_start.date() <= last_date.date():
        logger.warning(f"   ⚠️  ВНИМАНИЕ: Прогноз начинается ({forecast_start.date()}) не после последней даты данных ({last_date.date()})")
        logger.warning(f"   ⚠️  Возможно, включены исторические даты вместо будущих")
    else:
        logger.info(f"   ✅ Прогноз начинается ПОСЛЕ последней даты данных (корректно)")
    
    # Проверка значений
    logger.info(f"\n5. Статистика прогноза:")
    logger.info(f"   Среднее прогноза (yhat): {forecast_df['yhat'].mean():.2f}")
    logger.info(f"   Медиана прогноза: {forecast_df['yhat'].median():.2f}")
    logger.info(f"   Минимум: {forecast_df['yhat'].min():.2f}")
    logger.info(f"   Максимум: {forecast_df['yhat'].max():.2f}")
    
    # Проверка на отрицательные значения
    negative_count = (forecast_df['yhat'] < 0).sum()
    if negative_count > 0:
        logger.warning(f"   ⚠️  Найдено {negative_count} отрицательных прогнозов")
    else:
        logger.info(f"   ✅ Нет отрицательных прогнозов")
    
    # Проверка confidence intervals
    logger.info(f"\n6. Проверка confidence intervals:")
    logger.info(f"   Среднее нижней границы: {forecast_df['yhat_lower'].mean():.2f}")
    logger.info(f"   Среднее верхней границы: {forecast_df['yhat_upper'].mean():.2f}")
    
    # Проверка, что yhat_lower <= yhat <= yhat_upper
    invalid_intervals = (
        (forecast_df['yhat'] < forecast_df['yhat_lower']) | 
        (forecast_df['yhat'] > forecast_df['yhat_upper'])
    ).sum()
    
    if invalid_intervals > 0:
        logger.error(f"   ❌ Найдено {invalid_intervals} некорректных интервалов (yhat вне границ)")
        logger.error(f"   Первые некорректные:")
        invalid_rows = forecast_df[invalid_intervals].head(5)
        for _, row in invalid_rows.iterrows():
            logger.error(f"     {row['ds'].date()}: yhat={row['yhat']:.2f}, lower={row['yhat_lower']:.2f}, upper={row['yhat_upper']:.2f}")
    else:
        logger.info(f"   ✅ Все интервалы корректны (yhat_lower <= yhat <= yhat_upper)")
    
    # Проверка ширины интервалов
    forecast_df['interval_width'] = forecast_df['yhat_upper'] - forecast_df['yhat_lower']
    avg_width = forecast_df['interval_width'].mean()
    max_width = forecast_df['interval_width'].max()
    
    logger.info(f"   Средняя ширина интервала: {avg_width:.2f}")
    logger.info(f"   Максимальная ширина интервала: {max_width:.2f}")
    
    if max_width > forecast_df['yhat'].mean() * 10:
        logger.warning(f"   ⚠️  Очень широкие интервалы! (max_width = {max_width:.2f}, mean_yhat = {forecast_df['yhat'].mean():.2f})")
    else:
        logger.info(f"   ✅ Ширина интервалов разумная")
    
    # Шаг 7: Сохранение для проверки
    output_csv = "data/processed/test_forecast_output.csv"
    forecast_df.to_csv(output_csv, index=False)
    logger.info(f"\n7. Прогноз сохранен в: {output_csv}")
    
    # Финальная проверка
    logger.info(f"\n{'=' * 80}")
    logger.info("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    logger.info(f"{'=' * 80}")
    
    all_checks = [
        ("Обязательные колонки", len(missing_cols) == 0),
        ("Прогноз в будущем", forecast_start.date() > last_date.date()),
        ("Нет отрицательных значений", negative_count == 0),
        ("Корректные интервалы", invalid_intervals == 0),
        ("Разумная ширина интервалов", max_width <= forecast_df['yhat'].mean() * 10),
    ]
    
    all_passed = True
    for check_name, passed in all_checks:
        status = "✅" if passed else "❌"
        logger.info(f"{status} {check_name}: {'ПРОЙДЕН' if passed else 'ПРОВАЛЕН'}")
        if not passed:
            all_passed = False
    
    logger.info(f"\n{'=' * 80}")
    if all_passed:
        logger.info("✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ - прогноз работает корректно!")
    else:
        logger.warning("⚠️  НЕКОТОРЫЕ ПРОВЕРКИ НЕ ПРОЙДЕНЫ - есть проблемы")
    logger.info(f"{'=' * 80}")
    
    return all_passed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Тестирование генерации прогноза")
    parser.add_argument(
        "--shop_csv",
        type=str,
        default="data/processed/sales_data_shop.csv",
        help="Путь к обработанным данным"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Путь к модели (если None, обучит новую)"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=30,
        help="Количество дней для прогноза"
    )
    
    args = parser.parse_args()
    
    test_forecast_generation(args.shop_csv, args.model_path, args.horizon)

