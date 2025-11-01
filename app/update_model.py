# file: app/update_model.py
"""
Модуль для дообучения/обновления модели на новых данных
"""
import pandas as pd
import numpy as np
from prophet import Prophet
import joblib
import logging
from typing import Dict, Any, Optional
import os

logger = logging.getLogger(__name__)


def update_model_with_new_data(
    model_path: str,
    new_data_csv: str,
    output_model_path: Optional[str] = None,
    include_regressors: bool = False,
    max_history_days: Optional[int] = None
) -> Dict[str, Any]:
    """
    Дообучить существующую модель на новых данных.
    
    Args:
        model_path: Путь к существующей обученной модели
        new_data_csv: Путь к CSV с новыми данными (в формате processed: ds, y, avg_price, avg_discount, ...)
        output_model_path: Путь для сохранения обновленной модели (если None, перезаписывает model_path)
        include_regressors: Использовать ли регрессоры (должно совпадать с исходной моделью)
        max_history_days: Максимальное количество дней истории для обучения (None = использовать все)
    
    Returns:
        Dict с метриками и информацией об обновлении
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not os.path.exists(new_data_csv):
        raise FileNotFoundError(f"New data file not found: {new_data_csv}")
    
    logger.info(f"Loading existing model from: {model_path}")
    old_model: Prophet = joblib.load(model_path)
    
    # Проверяем, какие параметры использовались в старой модели
    old_history = old_model.history.copy() if hasattr(old_model, 'history') and old_model.history is not None else None
    old_regressors = len(old_model.extra_regressors) > 0
    
    if old_regressors != include_regressors:
        logger.warning(f"Regressor mismatch: old model has regressors={old_regressors}, "
                      f"but include_regressors={include_regressors}. "
                      f"Using old model's setting: {old_regressors}")
        include_regressors = old_regressors
    
    logger.info(f"Loading new data from: {new_data_csv}")
    new_data = pd.read_csv(new_data_csv)
    new_data['ds'] = pd.to_datetime(new_data['ds'])
    new_data = new_data.sort_values('ds')
    
    # Объединяем старые данные (если доступны) с новыми
    if old_history is not None and len(old_history) > 0:
        logger.info(f"Old model has history: {len(old_history)} days "
                   f"({old_history['ds'].min().date()} to {old_history['ds'].max().date()})")
        
        # Берем старые данные
        old_df = old_history[['ds', 'y']].copy()
        
        # Если есть регрессоры в старых данных
        if include_regressors and 'avg_price' in old_history.columns:
            old_df['avg_price'] = old_history['avg_price']
            old_df['avg_discount'] = old_history['avg_discount']
        
        # Объединяем старые и новые данные
        combined_data = pd.concat([old_df, new_data], ignore_index=True)
        combined_data = combined_data.drop_duplicates(subset=['ds'], keep='last')  # Новые данные перезаписывают старые
        combined_data = combined_data.sort_values('ds')
        
        logger.info(f"Combined data: {len(combined_data)} days "
                   f"({combined_data['ds'].min().date()} to {combined_data['ds'].max().date()})")
    else:
        logger.info("Old model has no history, using only new data")
        combined_data = new_data.copy()
    
    # Ограничиваем историю если нужно
    if max_history_days is not None:
        last_date = combined_data['ds'].max()
        cutoff_date = last_date - pd.Timedelta(days=max_history_days)
        combined_data = combined_data[combined_data['ds'] >= cutoff_date]
        logger.info(f"Limited history to {max_history_days} days: {len(combined_data)} days")
    
    # Определяем параметры для новой модели (из старой модели)
    days_span = (combined_data['ds'].max() - combined_data['ds'].min()).days
    use_yearly = days_span >= 730
    
    # Создаем новую модель с теми же параметрами что и старая
    # Но переобучаем на всех данных
    new_model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=use_yearly,
        interval_width=getattr(old_model, 'interval_width', 0.95),
        changepoint_prior_scale=getattr(old_model, 'changepoint_prior_scale', 0.01),
        seasonality_prior_scale=getattr(old_model, 'seasonality_prior_scale', 10.0),
        seasonality_mode=getattr(old_model, 'seasonality_mode', 'additive')
    )
    
    # Добавляем месячную сезонность если нужно
    if days_span >= 365 and days_span < 730:
        logger.info("Adding monthly seasonality")
        new_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    
    # Добавляем регрессоры если нужно
    if include_regressors:
        logger.info("Adding regressors: avg_price, avg_discount")
        new_model.add_regressor('avg_price')
        new_model.add_regressor('avg_discount')
    
    # Подготавливаем данные для обучения
    df_train = combined_data[['ds', 'y']].copy()
    if include_regressors:
        df_train['avg_price'] = combined_data['avg_price']
        df_train['avg_discount'] = combined_data['avg_discount']
    
    # Обучаем новую модель
    logger.info(f"Training updated model on {len(df_train)} days of data...")
    new_model.fit(df_train)
    logger.info("Model updated successfully")
    
    # Сохраняем обновленную модель
    output_path = output_model_path if output_model_path else model_path
    joblib.dump(new_model, output_path)
    logger.info(f"Updated model saved to: {output_path}")
    
    return {
        'success': True,
        'model_path': output_path,
        'old_data_days': len(old_history) if old_history is not None else 0,
        'new_data_days': len(new_data),
        'combined_data_days': len(combined_data),
        'date_range': {
            'start': combined_data['ds'].min().isoformat(),
            'end': combined_data['ds'].max().isoformat()
        },
        'yearly_seasonality': use_yearly,
        'monthly_seasonality': (days_span >= 365 and days_span < 730),
        'has_regressors': include_regressors
    }


def predict_from_last_date(
    data_csv: str,
    model_path: str,
    horizon_days: int,
    last_known_regressors_csv: Optional[str] = None,
    smooth_transition: bool = False
) -> pd.DataFrame:
    """
    Сделать прогноз от последней даты в данных на horizon_days дней вперед.
    Использует ВСЕ данные из data_csv (без разделения на train/test).
    
    Args:
        data_csv: Путь к CSV с ВСЕМИ доступными данными (processed format)
        model_path: Путь к обученной модели (будет использована как есть, БЕЗ переобучения)
        horizon_days: Количество дней для прогноза
        last_known_regressors_csv: Путь к CSV с регрессорами (опционально)
        smooth_transition: Применить сглаживание перехода
    
    Returns:
        DataFrame с прогнозом на будущее (только даты после последней даты в data_csv)
    """
    from app.predict import predict_prophet
    
    # Загружаем данные чтобы определить последнюю дату
    data = pd.read_csv(data_csv)
    data['ds'] = pd.to_datetime(data['ds'])
    last_date = data['ds'].max()
    
    logger.info(f"Data loaded: {len(data)} days, last date: {last_date.date()}")
    logger.info(f"Making forecast for {horizon_days} days starting from {last_date.date()}")
    
    # Используем стандартную функцию прогноза
    # Она автоматически сделает прогноз от последней даты в model.history
    # НО нам нужно убедиться что модель обучена на всех данных
    
    # Проверяем модель
    model: Prophet = joblib.load(model_path)
    model_last_date = model.history['ds'].max() if hasattr(model, 'history') and model.history is not None else None
    
    if model_last_date is None:
        logger.warning("Model has no history, using model as-is")
    elif model_last_date < last_date:
        logger.warning(f"Model was trained on data up to {model_last_date.date()}, "
                      f"but new data goes until {last_date.date()}. "
                      f"Consider updating the model with update_model_with_new_data() first.")
    
    # Делаем прогноз
    forecast = predict_prophet(
        model_path=model_path,
        horizon_days=horizon_days,
        last_known_regressors_csv=last_known_regressors_csv or data_csv,
        smooth_transition=smooth_transition
    )
    
    # Фильтруем только прогноз после последней даты в данных
    forecast_future = forecast[forecast['ds'] > last_date].copy()
    
    if len(forecast_future) == 0:
        logger.warning(f"Forecast starts at {forecast['ds'].min().date()}, "
                      f"but data ends at {last_date.date()}. "
                      f"This might indicate the model needs to be retrained.")
    
    return forecast_future

