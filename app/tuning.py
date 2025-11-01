# file: app/tuning.py
"""
Модуль автоматической настройки модели с grid search.
Поддерживает:
- Prophet с разными параметрами
- Prophet с/без регрессоров
- Prophet с growth='logistic'
- Prophet без weekly_seasonality
- LSTM модель
- Гибридная модель (Prophet + LSTM для остатков)
"""
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, Any, List, Optional, Tuple
import logging
import json
import os
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Попытка импортировать keras/tensorflow для LSTM
try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    HAS_KERAS = True
except ImportError:
    HAS_KERAS = False
    logger.warning("TensorFlow/Keras not available. LSTM models will be skipped.")


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error."""
    mask = actual != 0
    if mask.any():
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    return 0.0


def calculate_coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Calculate CI coverage rate."""
    valid_mask = np.isfinite(actual) & np.isfinite(lower) & np.isfinite(upper)
    if valid_mask.sum() == 0:
        return 0.0
    in_range = (actual[valid_mask] >= lower[valid_mask]) & (actual[valid_mask] <= upper[valid_mask])
    return float(in_range.sum() / valid_mask.sum())


def calculate_trend_difference(df: pd.DataFrame, y_col: str = 'y', yhat_col: str = 'yhat', ds_col: str = 'ds') -> float:
    """Calculate trend difference between actual and predicted."""
    try:
        df = df.copy()
        df[ds_col] = pd.to_datetime(df[ds_col])
        df = df.sort_values(ds_col).reset_index(drop=True)
        
        dates_numeric = df[ds_col].astype(int) / 1e9
        y_values = df[y_col].values
        yhat_values = df[yhat_col].values
        
        from scipy import stats
        
        valid_mask_y = np.isfinite(y_values) & np.isfinite(dates_numeric)
        valid_mask_yhat = np.isfinite(yhat_values) & np.isfinite(dates_numeric)
        
        if valid_mask_y.sum() > 2 and valid_mask_yhat.sum() > 2:
            slope_y, _, _, _, _ = stats.linregress(dates_numeric[valid_mask_y], y_values[valid_mask_y])
            slope_yhat, _, _, _, _ = stats.linregress(dates_numeric[valid_mask_yhat], yhat_values[valid_mask_yhat])
            return float(abs(slope_yhat - slope_y))
    except:
        pass
    return 0.0


def train_prophet_variant(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Обучает вариант Prophet модели.
    
    Args:
        df_train: Training DataFrame (ds, y, возможно avg_price, avg_discount)
        df_test: Test DataFrame (ds, y, возможно avg_price, avg_discount)
        config: Конфигурация модели
        
    Returns:
        Словарь с метриками и моделью
    """
    model_name = config.get('name', 'prophet')
    logger.info(f"Training {model_name}...")
    
    try:
        # Подготовка данных
        prophet_cols = ['ds', 'y']
        include_regressors = config.get('include_regressors', False)
        
        if include_regressors and 'avg_price' in df_train.columns and 'avg_discount' in df_train.columns:
            prophet_cols.extend(['avg_price', 'avg_discount'])
            include_regressors = True
        else:
            include_regressors = False
        
        df_prophet_train = df_train[prophet_cols].copy()
        
        # Параметры Prophet
        seasonality_prior_scale = config.get('seasonality_prior_scale', 10.0)
        changepoint_prior_scale = config.get('changepoint_prior_scale', 0.05)
        interval_width = config.get('interval_width', 0.95)
        seasonality_mode = config.get('seasonality_mode', 'additive')
        growth = config.get('growth', 'linear')
        weekly_seasonality = config.get('weekly_seasonality', True)
        
        # Logistic growth требует cap
        if growth == 'logistic':
            # Используем максимальное значение y * 1.5 как cap
            cap_value = df_prophet_train['y'].max() * 1.5
            df_prophet_train['cap'] = cap_value
        
        # Определяем, достаточно ли данных для yearly seasonality
        days_span = (df_train['ds'].max() - df_train['ds'].min()).days
        use_yearly = days_span >= 730  # Только если данных >= 2 лет
        
        if not use_yearly:
            logger.info(f"{model_name}: Data span ({days_span} days) < 730 days, disabling yearly_seasonality")
        
        # Создание модели
        model = Prophet(
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=use_yearly,  # Автоматически отключаем для коротких данных
            interval_width=interval_width,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            seasonality_mode=seasonality_mode,
            growth=growth
        )
        
        # Для данных >= 365 дней (но < 730): добавляем месячную сезонность
        if days_span >= 365 and days_span < 730:
            logger.info(f"{model_name}: Adding monthly seasonality (data span {days_span} days >= 365 but < 730)")
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        
        if include_regressors:
            model.add_regressor('avg_price')
            model.add_regressor('avg_discount')
        
        # Обучение
        model.fit(df_prophet_train)
        
        # Прогноз
        periods = len(df_test)
        future = model.make_future_dataframe(periods=periods, freq='D')
        
        if include_regressors:
            all_regressors = pd.concat([
                df_train[['ds', 'avg_price', 'avg_discount']],
                df_test[['ds', 'avg_price', 'avg_discount']]
            ], ignore_index=True)
            future = future.merge(all_regressors, on='ds', how='left')
            for col in ['avg_price', 'avg_discount']:
                future[col] = future[col].ffill().fillna(all_regressors[col].median())
        
        if growth == 'logistic':
            future['cap'] = cap_value
        
        forecast = model.predict(future)
        
        # Извлечение прогнозов для тестового периода
        test_mask = forecast['ds'] >= df_test['ds'].min()
        forecast_test = forecast[test_mask].copy()
        
        # Выравнивание дат
        merged = forecast_test[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].merge(
            df_test[['ds', 'y']],
            on='ds',
            how='inner'
        )
        
        if len(merged) == 0:
            logger.warning(f"{model_name}: No overlapping dates")
            return {
                'name': model_name,
                'success': False,
                'error': 'No overlapping dates'
            }
        
        actual = merged['y'].values
        predicted = merged['yhat'].values
        lower = merged['yhat_lower'].values
        upper = merged['yhat_upper'].values
        
        # Обеспечиваем неотрицательные значения
        predicted = np.clip(predicted, 0, None)
        lower = np.clip(lower, 0, None)
        upper = np.clip(upper, 0, None)
        upper = np.maximum(upper, lower)
        
        # Метрики
        mae_val = mean_absolute_error(actual, predicted)
        rmse_val = np.sqrt(mean_squared_error(actual, predicted))
        mape_val = mape(actual, predicted)
        coverage_result = calculate_coverage(actual, lower, upper)
        # calculate_coverage возвращает dict
        if isinstance(coverage_result, dict):
            coverage_val = coverage_result.get('coverage_rate', 0.0)
        else:
            coverage_val = float(coverage_result) if isinstance(coverage_result, (int, float)) else 0.0
        systematic_bias = float(np.mean(actual - predicted))
        trend_diff = calculate_trend_difference(merged)
        
        results = {
            'name': model_name,
            'success': True,
            'config': config,
            'metrics': {
                'mae': float(mae_val),
                'rmse': float(rmse_val),
                'mape': float(mape_val),
                'coverage': float(coverage_val),
                'systematic_bias': systematic_bias,
                'trend_difference': trend_diff
            },
            'n_test_samples': len(merged)
        }
        
        logger.info(f"{model_name}: MAPE={mape_val:.2f}%, Coverage={coverage_val*100:.1f}%")
        
        return results
        
    except Exception as e:
        logger.error(f"{model_name} failed: {str(e)}")
        return {
            'name': model_name,
            'success': False,
            'error': str(e)
        }


def train_lstm_model(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Обучает LSTM модель.
    
    Args:
        df_train: Training DataFrame
        df_test: Test DataFrame
        config: Конфигурация
        
    Returns:
        Словарь с метриками
    """
    model_name = config.get('name', 'lstm')
    
    if not HAS_KERAS:
        return {
            'name': model_name,
            'success': False,
            'error': 'Keras/TensorFlow not available'
        }
    
    logger.info(f"Training {model_name}...")
    
    try:
        # Подготовка данных для LSTM
        df_all = pd.concat([df_train, df_test], ignore_index=True)
        df_all = df_all.sort_values('ds').reset_index(drop=True)
        df_all['ds'] = pd.to_datetime(df_all['ds'])
        
        # Нормализация
        y_values = df_all['y'].values
        y_mean = np.mean(y_values)
        y_std = np.std(y_values)
        y_normalized = (y_values - y_mean) / (y_std + 1e-8)
        
        # Создание последовательностей (lookback window = 30 дней)
        lookback = 30
        X, y = [], []
        
        for i in range(lookback, len(y_normalized)):
            X.append(y_normalized[i-lookback:i])
            y.append(y_normalized[i])
        
        X = np.array(X)
        y = np.array(y)
        
        # Разделение на train/test
        split_idx = len(df_train) - lookback
        if split_idx < lookback:
            logger.warning(f"{model_name}: Insufficient data for LSTM")
            return {
                'name': model_name,
                'success': False,
                'error': 'Insufficient data'
            }
        
        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_test = X[split_idx:]
        y_test = y[split_idx:]
        
        if len(X_test) == 0:
            return {
                'name': model_name,
                'success': False,
                'error': 'No test data'
            }
        
        # Reshape для LSTM
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # Создание модели
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Обучение (упрощенное, без валидации для скорости)
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
        
        # Прогноз
        y_pred_normalized = model.predict(X_test, verbose=0).flatten()
        
        # Обратная нормализация
        y_pred = y_pred_normalized * y_std + y_mean
        y_actual = y_test * y_std + y_mean
        
        # Обеспечиваем неотрицательные значения
        y_pred = np.clip(y_pred, 0, None)
        
        # Метрики
        mae_val = mean_absolute_error(y_actual, y_pred)
        rmse_val = np.sqrt(mean_squared_error(y_actual, y_pred))
        mape_val = mape(y_actual, y_pred)
        
        # Для LSTM coverage недоступен (нужна другая архитектура для интервалов)
        # Используем эмпирический интервал на основе RMSE
        rmse_for_ci = rmse_val
        lower = y_pred - 1.96 * rmse_for_ci  # Примерный 95% CI
        upper = y_pred + 1.96 * rmse_for_ci
        lower = np.clip(lower, 0, None)
        coverage_val = calculate_coverage(y_actual, lower, upper)
        
        systematic_bias = float(np.mean(y_actual - y_pred))
        
        # Trend difference (нужно восстановить DataFrame с датами)
        test_dates = df_test['ds'].values[:len(y_pred)]
        df_lstm = pd.DataFrame({
            'ds': test_dates,
            'y': y_actual,
            'yhat': y_pred
        })
        trend_diff = calculate_trend_difference(df_lstm)
        
        results = {
            'name': model_name,
            'success': True,
            'config': config,
            'metrics': {
                'mae': float(mae_val),
                'rmse': float(rmse_val),
                'mape': float(mape_val),
                'coverage': float(coverage_val),
                'systematic_bias': systematic_bias,
                'trend_difference': trend_diff
            },
            'n_test_samples': len(y_actual)
        }
        
        logger.info(f"{model_name}: MAPE={mape_val:.2f}%, Coverage={coverage_val*100:.1f}%")
        
        return results
        
    except Exception as e:
        logger.error(f"{model_name} failed: {str(e)}")
        return {
            'name': model_name,
            'success': False,
            'error': str(e)
        }


def train_hybrid_model(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Обучает гибридную модель: Prophet + LSTM для остатков.
    
    Args:
        df_train: Training DataFrame
        df_test: Test DataFrame
        config: Конфигурация
        
    Returns:
        Словарь с метриками
    """
    model_name = config.get('name', 'hybrid')
    
    if not HAS_KERAS:
        return {
            'name': model_name,
            'success': False,
            'error': 'Keras/TensorFlow not available'
        }
    
    logger.info(f"Training {model_name}...")
    
    try:
        # Шаг 1: Обучаем Prophet
        prophet_config = config.get('prophet_config', {})
        prophet_result = train_prophet_variant(df_train, df_train, {
            'name': 'prophet_base',
            'include_regressors': prophet_config.get('include_regressors', False),
            'seasonality_prior_scale': prophet_config.get('seasonality_prior_scale', 10.0),
            'changepoint_prior_scale': prophet_config.get('changepoint_prior_scale', 0.05),
            'interval_width': prophet_config.get('interval_width', 0.95),
            'seasonality_mode': prophet_config.get('seasonality_mode', 'additive')
        })
        
        if not prophet_result.get('success', False):
            return {
                'name': model_name,
                'success': False,
                'error': 'Prophet base model failed'
            }
        
        # Получаем прогнозы Prophet на train
        prophet_cols = ['ds', 'y']
        if prophet_config.get('include_regressors', False) and 'avg_price' in df_train.columns:
            prophet_cols.extend(['avg_price', 'avg_discount'])
        
        # Создаем временную модель для получения прогнозов
        from prophet import Prophet as ProphetModel
        temp_model = ProphetModel(
            weekly_seasonality=True,
            yearly_seasonality=True,
            interval_width=prophet_config.get('interval_width', 0.95),
            changepoint_prior_scale=prophet_config.get('changepoint_prior_scale', 0.05),
            seasonality_prior_scale=prophet_config.get('seasonality_prior_scale', 10.0),
            seasonality_mode=prophet_config.get('seasonality_mode', 'additive')
        )
        
        if prophet_config.get('include_regressors', False):
            temp_model.add_regressor('avg_price')
            temp_model.add_regressor('avg_discount')
        
        temp_model.fit(df_train[prophet_cols])
        
        # Прогноз на train для получения остатков
        future_train = temp_model.make_future_dataframe(periods=0, freq='D')
        if prophet_config.get('include_regressors', False):
            regressors_train = df_train[['ds', 'avg_price', 'avg_discount']]
            future_train = future_train.merge(regressors_train, on='ds', how='left')
            for col in ['avg_price', 'avg_discount']:
                future_train[col] = future_train[col].ffill().fillna(regressors_train[col].median())
        
        forecast_train = temp_model.predict(future_train)
        merged_train = forecast_train[['ds', 'yhat']].merge(
            df_train[['ds', 'y']],
            on='ds',
            how='inner'
        )
        
        residuals_train = (merged_train['y'] - merged_train['yhat']).values
        
        # Шаг 2: Обучаем LSTM на остатках
        # Нормализация остатков
        res_mean = np.mean(residuals_train)
        res_std = np.std(residuals_train)
        res_normalized = (residuals_train - res_mean) / (res_std + 1e-8)
        
        lookback = 30
        if len(res_normalized) < lookback + 10:
            return {
                'name': model_name,
                'success': False,
                'error': 'Insufficient data for LSTM on residuals'
            }
        
        X_res, y_res = [], []
        for i in range(lookback, len(res_normalized)):
            X_res.append(res_normalized[i-lookback:i])
            y_res.append(res_normalized[i])
        
        X_res = np.array(X_res).reshape((len(X_res), lookback, 1))
        y_res = np.array(y_res)
        
        # LSTM для остатков
        lstm_residual = Sequential([
            LSTM(32, return_sequences=False, input_shape=(lookback, 1)),
            Dropout(0.2),
            Dense(1)
        ])
        lstm_residual.compile(optimizer='adam', loss='mse')
        lstm_residual.fit(X_res, y_res, epochs=15, batch_size=32, verbose=0)
        
        # Шаг 3: Прогноз на test
        future_test = temp_model.make_future_dataframe(periods=len(df_test), freq='D')
        if prophet_config.get('include_regressors', False):
            all_regressors = pd.concat([
                df_train[['ds', 'avg_price', 'avg_discount']],
                df_test[['ds', 'avg_price', 'avg_discount']]
            ], ignore_index=True)
            future_test = future_test.merge(all_regressors, on='ds', how='left')
            for col in ['avg_price', 'avg_discount']:
                future_test[col] = future_test[col].ffill().fillna(all_regressors[col].median())
        
        forecast_test_prophet = temp_model.predict(future_test)
        
        # Извлекаем прогнозы Prophet для test периода
        test_mask = forecast_test_prophet['ds'] >= df_test['ds'].min()
        forecast_test_prophet = forecast_test_prophet[test_mask].copy()
        
        # Получаем остатки на train для последних lookback дней
        last_residuals = residuals_train[-lookback:]
        last_residuals_norm = (last_residuals - res_mean) / (res_std + 1e-8)
        
        # Прогнозируем остатки LSTM
        residuals_pred_normalized = []
        current_window = last_residuals_norm.copy()
        
        for _ in range(len(df_test)):
            X_window = current_window[-lookback:].reshape((1, lookback, 1))
            pred_res_norm = lstm_residual.predict(X_window, verbose=0)[0, 0]
            residuals_pred_normalized.append(pred_res_norm)
            current_window = np.append(current_window, pred_res_norm)
        
        residuals_pred = np.array(residuals_pred_normalized) * res_std + res_mean
        
        # Финальный прогноз = Prophet + LSTM остатки
        prophet_test = forecast_test_prophet['yhat'].values[:len(residuals_pred)]
        y_pred = prophet_test + residuals_pred
        
        # Выравнивание с фактическими значениями
        merged = forecast_test_prophet[['ds', 'yhat_lower', 'yhat_upper']].merge(
            df_test[['ds', 'y']],
            on='ds',
            how='inner'
        )
        
        if len(merged) == 0 or len(merged) != len(y_pred):
            return {
                'name': model_name,
                'success': False,
                'error': 'Alignment failed'
            }
        
        y_actual = merged['y'].values
        lower = merged['yhat_lower'].values
        upper = merged['yhat_upper'].values
        
        # Обеспечиваем неотрицательные значения
        y_pred = np.clip(y_pred, 0, None)
        lower = np.clip(lower, 0, None)
        upper = np.clip(upper, 0, None)
        upper = np.maximum(upper, lower)
        
        # Метрики
        mae_val = mean_absolute_error(y_actual, y_pred)
        rmse_val = np.sqrt(mean_squared_error(y_actual, y_pred))
        mape_val = mape(y_actual, y_pred)
        coverage_val = calculate_coverage(y_actual, lower, upper)
        systematic_bias = float(np.mean(y_actual - y_pred))
        
        df_hybrid = pd.DataFrame({
            'ds': merged['ds'].values,
            'y': y_actual,
            'yhat': y_pred
        })
        trend_diff = calculate_trend_difference(df_hybrid)
        
        results = {
            'name': model_name,
            'success': True,
            'config': config,
            'metrics': {
                'mae': float(mae_val),
                'rmse': float(rmse_val),
                'mape': float(mape_val),
                'coverage': float(coverage_val),
                'systematic_bias': systematic_bias,
                'trend_difference': trend_diff
            },
            'n_test_samples': len(y_actual)
        }
        
        logger.info(f"{model_name}: MAPE={mape_val:.2f}%, Coverage={coverage_val*100:.1f}%")
        
        return results
        
    except Exception as e:
        logger.error(f"{model_name} failed: {str(e)}")
        return {
            'name': model_name,
            'success': False,
            'error': str(e)
        }


def grid_search_models(
    shop_csv_path: str,
    holdout_frac: float = 0.2,
    output_dir: str = "analysis"
) -> Dict[str, Any]:
    """
    Проводит grid search по различным конфигурациям моделей.
    
    Args:
        shop_csv_path: Путь к shop-level CSV
        holdout_frac: Доля данных для теста
        output_dir: Директория для сохранения результатов
        
    Returns:
        Словарь с результатами всех моделей
    """
    logger.info("Starting grid search for model optimization...")
    
    # Загрузка данных
    df = pd.read_csv(shop_csv_path)
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds').reset_index(drop=True)
    
    # Разделение на train/test
    n_total = len(df)
    n_train = int(n_total * (1 - holdout_frac))
    
    # Убеждаемся, что есть достаточно данных для теста
    if n_train >= n_total:
        logger.warning(f"Holdout fraction too small, using last 20% for test")
        n_train = int(n_total * 0.8)
    
    df_train = df.iloc[:n_train].copy()
    df_test = df.iloc[n_train:].copy()
    
    # Минимум 7 дней для теста
    if len(df_test) < 7:
        logger.warning(f"Test set too small ({len(df_test)}), extending to 7 days")
        n_train = max(30, n_total - 7)  # Минимум 30 для train
        df_train = df.iloc[:n_train].copy()
        df_test = df.iloc[n_train:].copy()
    
    logger.info(f"Train: {len(df_train)} samples ({df_train['ds'].min().date()} to {df_train['ds'].max().date()})")
    logger.info(f"Test: {len(df_test)} samples ({df_test['ds'].min().date()} to {df_test['ds'].max().date()})")
    
    # Список конфигураций для тестирования
    configs = []
    
    # 1. Prophet базовые варианты
    # Анализ показывает: данных 365 дней - это минимум, yearly_seasonality может переобучаться
    # Используем консервативные настройки для стабильности
    prophet_base_configs = [
        {'seasonality_prior_scale': 1.0, 'changepoint_prior_scale': 0.01, 'interval_width': 0.95},  # Самый консервативный
        {'seasonality_prior_scale': 5.0, 'changepoint_prior_scale': 0.01, 'interval_width': 0.95},  # Консервативный
        {'seasonality_prior_scale': 10.0, 'changepoint_prior_scale': 0.01, 'interval_width': 0.95},  # Баланс
        {'seasonality_prior_scale': 1.0, 'changepoint_prior_scale': 0.01, 'interval_width': 0.9},
        {'seasonality_prior_scale': 10.0, 'changepoint_prior_scale': 0.01, 'interval_width': 0.9},
        # Варианты с чуть большей гибкостью (но все еще консервативно)
        {'seasonality_prior_scale': 5.0, 'changepoint_prior_scale': 0.05, 'interval_width': 0.95},  # Умеренный
        {'seasonality_prior_scale': 10.0, 'changepoint_prior_scale': 0.05, 'interval_width': 0.95},  # Умеренный
        # Добавляем вариант с log_transform для стабилизации волатильности
        {'seasonality_prior_scale': 10.0, 'changepoint_prior_scale': 0.01, 'interval_width': 0.95, 'log_transform': True},
    ]
    
    for i, base_config in enumerate(prophet_base_configs):
        # Prophet без регрессоров
        configs.append({
            'name': f'prophet_baseline_{i+1}',
            'type': 'prophet',
            'include_regressors': False,
            **base_config
        })
        
        # Prophet с регрессорами
        if 'avg_price' in df.columns:
            configs.append({
                'name': f'prophet_with_regressors_{i+1}',
                'type': 'prophet',
                'include_regressors': True,
                **base_config
            })
    
    # 2. Prophet без weekly_seasonality
    configs.append({
        'name': 'prophet_no_weekly',
        'type': 'prophet',
        'include_regressors': False,
        'weekly_seasonality': False,
        'seasonality_prior_scale': 10.0,
        'changepoint_prior_scale': 0.05,
        'interval_width': 0.95
    })
    
    # 3. Prophet с multiplicative seasonality
    configs.append({
        'name': 'prophet_multiplicative',
        'type': 'prophet',
        'include_regressors': False,
        'seasonality_mode': 'multiplicative',
        'seasonality_prior_scale': 10.0,
        'changepoint_prior_scale': 0.05,
        'interval_width': 0.95
    })
    
    # 4. Prophet с logistic growth (только если данные подходят)
    if df['y'].min() >= 0 and df['y'].max() / df['y'].mean() < 10:
        configs.append({
            'name': 'prophet_logistic',
            'type': 'prophet',
            'include_regressors': False,
            'growth': 'logistic',
            'seasonality_prior_scale': 10.0,
            'changepoint_prior_scale': 0.05,
            'interval_width': 0.95
        })
    
    # 5. LSTM (если доступен)
    if HAS_KERAS:
        configs.append({
            'name': 'lstm_basic',
            'type': 'lstm',
            'lookback': 30
        })
    
    # 6. Гибридная модель (если доступен Keras)
    if HAS_KERAS:
        configs.append({
            'name': 'hybrid_prophet_lstm',
            'type': 'hybrid',
            'prophet_config': {
                'include_regressors': False,
                'seasonality_prior_scale': 10.0,
                'changepoint_prior_scale': 0.05,
                'interval_width': 0.95,
                'seasonality_mode': 'additive'
            }
        })
    
    # Обучение всех моделей
    results = []
    
    for config in configs:
        try:
            if config['type'] == 'prophet':
                result = train_prophet_variant(df_train, df_test, config)
            elif config['type'] == 'lstm':
                result = train_lstm_model(df_train, df_test, config)
            elif config['type'] == 'hybrid':
                result = train_hybrid_model(df_train, df_test, config)
            else:
                continue
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error training {config.get('name', 'unknown')}: {str(e)}")
            results.append({
                'name': config.get('name', 'unknown'),
                'success': False,
                'error': str(e)
            })
    
    # Выбор лучшей модели
    successful_results = [r for r in results if r.get('success', False)]
    
    if len(successful_results) == 0:
        logger.error("No successful models!")
        return {
            'success': False,
            'error': 'No successful models',
            'all_results': results
        }
    
    # Сортировка по комбинации MAPE и coverage
    # Цель: минимизировать MAPE, максимизировать coverage (>=85% критично)
    def score_model(result):
        mape_val = result['metrics']['mape']
        coverage_val = result['metrics']['coverage']
        
        # Строгий штраф за низкое coverage
        if coverage_val < 0.85:
            # Очень большой штраф за coverage < 85%
            penalty = (0.85 - coverage_val) * 200  # Штраф 200 за каждый процент ниже 85%
        elif coverage_val < 0.90:
            # Небольшой штраф за coverage 85-90%
            penalty = (0.90 - coverage_val) * 20
        else:
            penalty = 0
        
        # Также учитываем systematic bias
        bias_penalty = abs(result['metrics'].get('systematic_bias', 0)) * 0.5
        
        return mape_val + penalty + bias_penalty
    
    successful_results.sort(key=score_model)
    best_model = successful_results[0]
    
    logger.info(f"Best model: {best_model['name']} - MAPE={best_model['metrics']['mape']:.2f}%, "
                f"Coverage={best_model['metrics']['coverage']*100:.1f}%")
    
    # Сохранение результатов
    os.makedirs(output_dir, exist_ok=True)
    
    # CSV с результатами всех моделей
    comparison_data = []
    for result in successful_results:
        comparison_data.append({
            'model_name': result['name'],
            'mape': result['metrics']['mape'],
            'rmse': result['metrics']['rmse'],
            'mae': result['metrics']['mae'],
            'coverage': result['metrics']['coverage'],
            'systematic_bias': result['metrics']['systematic_bias'],
            'trend_difference': result['metrics']['trend_difference'],
            'n_test_samples': result['n_test_samples']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_csv = os.path.join(output_dir, 'model_comparison.csv')
    comparison_df.to_csv(comparison_csv, index=False)
    logger.info(f"Model comparison saved to {comparison_csv}")
    
    # JSON с лучшей моделью
    best_model_json = os.path.join(output_dir, 'best_model.json')
    with open(best_model_json, 'w') as f:
        json.dump(best_model, f, indent=2, default=str)
    logger.info(f"Best model saved to {best_model_json}")
    
    # Создание графика сравнения моделей
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        
        # Используем backend без GUI для серверов
        matplotlib.use('Agg')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        model_names = [r['name'] for r in successful_results]
        mape_values = [r['metrics']['mape'] for r in successful_results]
        coverage_values = [r['metrics']['coverage'] * 100 for r in successful_results]
        
        x_pos = range(len(model_names))
        bars = ax.bar(x_pos, mape_values, color='lightblue', alpha=0.7, edgecolor='black')
        
        # Подсветка лучшей модели
        best_idx = 0  # Уже отсортировано
        bars[best_idx].set_color('green')
        bars[best_idx].set_alpha(0.8)
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('MAPE (%)', fontsize=12, fontweight='bold')
        ax.set_title('Model Comparison: MAPE Across Configurations', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Добавить значения на столбцы
        for i, (name, mape_val, cov_val) in enumerate(zip(model_names, mape_values, coverage_values)):
            ax.text(i, mape_val + max(mape_values) * 0.01, f'{mape_val:.1f}%', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
            # Coverage в скобках
            ax.text(i, mape_val + max(mape_values) * 0.04, f'({cov_val:.0f}%)', 
                   ha='center', va='bottom', fontsize=8, style='italic', color='gray')
        
        plt.tight_layout()
        
        tuning_plot_path = os.path.join(output_dir, 'tuning_plot.png')
        plt.savefig(tuning_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Comparison plot saved to {tuning_plot_path}")
    except Exception as e:
        logger.warning(f"Could not create comparison plot: {str(e)}")
        tuning_plot_path = None
    
    return {
        'success': True,
        'best_model': best_model,
        'all_results': results,
        'comparison_csv': comparison_csv,
        'best_model_json': best_model_json,
        'tuning_plot_path': tuning_plot_path,
        'n_models_tested': len(results),
        'n_successful': len(successful_results)
    }

