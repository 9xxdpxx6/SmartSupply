# file: app/diagnostics.py
"""
Модуль диагностики модели для выявления систематических проблем:
- Анализ распределения остатков
- Проверка мультиколлинеарности регрессоров
- Анализ смещения тренда
- Покрытие Confidence Interval
- Сдвиг локальных минимумов
"""
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro, normaltest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, Any, Optional, Tuple
import logging
import json
import os

logger = logging.getLogger(__name__)


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error."""
    mask = actual != 0
    if mask.any():
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    return 0.0


def calculate_residuals(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Calculate residuals (actual - predicted)."""
    return actual - predicted


def analyze_residuals(
    residuals: np.ndarray,
    dates: Optional[pd.Series] = None
) -> Dict[str, Any]:
    """
    Анализирует распределение остатков.
    
    Args:
        residuals: Массив остатков
        dates: Опциональные даты для анализа тренда ошибки
        
    Returns:
        Словарь с метриками остатков
    """
    residuals_clean = residuals[np.isfinite(residuals)]
    
    if len(residuals_clean) == 0:
        return {
            'mean': 0.0,
            'std': 0.0,
            'median': 0.0,
            'skewness': 0.0,
            'kurtosis': 0.0,
            'normality_test_pvalue': 1.0,
            'is_normal': False,
            'has_trend': False,
            'trend_slope': 0.0
        }
    
    mean_residual = float(np.mean(residuals_clean))
    std_residual = float(np.std(residuals_clean))
    median_residual = float(np.median(residuals_clean))
    
    # Проверка нормальности (Shapiro-Wilk для малых выборок, D'Agostino для больших)
    if len(residuals_clean) < 50:
        try:
            _, normality_pvalue = shapiro(residuals_clean)
        except:
            normality_pvalue = 1.0
    else:
        try:
            _, normality_pvalue = normaltest(residuals_clean)
        except:
            normality_pvalue = 1.0
    
    # Skewness and kurtosis
    skewness = float(stats.skew(residuals_clean))
    kurtosis = float(stats.kurtosis(residuals_clean))
    
    # Проверка тренда ошибки по времени
    has_trend = False
    trend_slope = 0.0
    
    if dates is not None and len(dates) == len(residuals):
        # Convert dates to numeric for regression
        dates_numeric = pd.to_datetime(dates).astype(int) / 1e9  # Convert to seconds
        valid_mask = np.isfinite(residuals) & np.isfinite(dates_numeric)
        
        if valid_mask.sum() > 2:
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    dates_numeric[valid_mask], residuals[valid_mask]
                )
                trend_slope = float(slope)
                has_trend = p_value < 0.05  # Значимый тренд
            except:
                pass
    
    return {
        'mean': mean_residual,
        'std': std_residual,
        'median': median_residual,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'normality_test_pvalue': float(normality_pvalue),
        'is_normal': normality_pvalue > 0.05,
        'has_trend': has_trend,
        'trend_slope': trend_slope,
        'n_samples': len(residuals_clean)
    }


def check_multicollinearity(
    df: pd.DataFrame,
    regressor_cols: list = ['avg_price', 'avg_discount']
) -> Dict[str, Any]:
    """
    Проверяет мультиколлинеарность регрессоров.
    
    Args:
        df: DataFrame с регрессорами
        regressor_cols: Список колонок регрессоров
        
    Returns:
        Словарь с метриками мультиколлинеарности
    """
    available_cols = [col for col in regressor_cols if col in df.columns]
    
    if len(available_cols) < 2:
        return {
            'correlation_matrix': {},
            'max_correlation': 0.0,
            'vif_scores': {},
            'has_multicollinearity': False,
            'warning': 'Insufficient regressors for multicollinearity check'
        }
    
    # Корреляционная матрица
    corr_matrix = df[available_cols].corr()
    
    # Найти максимальную корреляцию (кроме диагонали)
    max_corr = 0.0
    for i in range(len(available_cols)):
        for j in range(len(available_cols)):
            if i != j:
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > max_corr:
                    max_corr = corr_val
    
    # VIF (Variance Inflation Factor) - упрощенный вариант
    # VIF = 1 / (1 - R²) для каждого регрессора
    vif_scores = {}
    for col in available_cols:
        try:
            # Регрессия col на остальные регрессоры
            other_cols = [c for c in available_cols if c != col]
            if len(other_cols) > 0:
                from sklearn.linear_model import LinearRegression
                X = df[other_cols].values
                y = df[col].values
                
                # Удалить NaN
                mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
                if mask.sum() > len(other_cols):
                    X_clean = X[mask]
                    y_clean = y[mask]
                    
                    model = LinearRegression()
                    model.fit(X_clean, y_clean)
                    r2 = model.score(X_clean, y_clean)
                    
                    if r2 < 0.99:  # Избежать деления на ноль
                        vif = 1.0 / (1.0 - r2)
                    else:
                        vif = 999.0  # Высокая мультиколлинеарность
                    
                    vif_scores[col] = float(vif)
                else:
                    vif_scores[col] = 1.0
            else:
                vif_scores[col] = 1.0
        except:
            vif_scores[col] = 1.0
    
    # Правило: VIF > 5 или корреляция > 0.8 - мультиколлинеарность
    has_multicollinearity = max_corr > 0.8 or any(v > 5.0 for v in vif_scores.values())
    
    return {
        'correlation_matrix': corr_matrix.to_dict(),
        'max_correlation': float(max_corr),
        'vif_scores': vif_scores,
        'has_multicollinearity': has_multicollinearity,
        'warning': None
    }


def analyze_trend_bias(
    df: pd.DataFrame,
    y_col: str = 'y',
    yhat_col: str = 'yhat',
    ds_col: str = 'ds'
) -> Dict[str, Any]:
    """
    Анализирует систематическое смещение тренда.
    Сравнивает средние наклоны y и yhat.
    
    Args:
        df: DataFrame с фактическими и предсказанными значениями
        y_col: Колонка с фактическими значениями
        yhat_col: Колонка с предсказанными значениями
        ds_col: Колонка с датами
        
    Returns:
        Словарь с метриками смещения тренда
    """
    if y_col not in df.columns or yhat_col not in df.columns:
        return {
            'y_trend_slope': 0.0,
            'yhat_trend_slope': 0.0,
            'trend_difference': 0.0,
            'trend_bias_pct': 0.0,
            'has_bias': False
        }
    
    df = df.copy()
    df[ds_col] = pd.to_datetime(df[ds_col])
    df = df.sort_values(ds_col).reset_index(drop=True)
    
    # Конвертируем даты в числовой формат
    dates_numeric = df[ds_col].astype(int) / 1e9
    
    y_values = df[y_col].values
    yhat_values = df[yhat_col].values
    
    # Вычисляем наклон тренда для y и yhat
    y_trend_slope = 0.0
    yhat_trend_slope = 0.0
    
    try:
        valid_mask = np.isfinite(y_values) & np.isfinite(dates_numeric)
        if valid_mask.sum() > 2:
            slope, _, _, p_value, _ = stats.linregress(
                dates_numeric[valid_mask], y_values[valid_mask]
            )
            y_trend_slope = float(slope)
    except:
        pass
    
    try:
        valid_mask = np.isfinite(yhat_values) & np.isfinite(dates_numeric)
        if valid_mask.sum() > 2:
            slope, _, _, p_value, _ = stats.linregress(
                dates_numeric[valid_mask], yhat_values[valid_mask]
            )
            yhat_trend_slope = float(slope)
    except:
        pass
    
    trend_difference = yhat_trend_slope - y_trend_slope
    
    # Процентное смещение
    if abs(y_trend_slope) > 1e-10:
        trend_bias_pct = (trend_difference / abs(y_trend_slope)) * 100
    else:
        trend_bias_pct = 0.0 if abs(trend_difference) < 1e-10 else 999.0
    
    # Систематическое смещение если разница > 10% от фактического тренда
    has_bias = abs(trend_bias_pct) > 10.0
    
    return {
        'y_trend_slope': y_trend_slope,
        'yhat_trend_slope': yhat_trend_slope,
        'trend_difference': trend_difference,
        'trend_bias_pct': trend_bias_pct,
        'has_bias': has_bias
    }


def calculate_coverage(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray
) -> Dict[str, Any]:
    """
    Вычисляет покрытие Confidence Interval.
    
    Args:
        actual: Фактические значения
        lower: Нижние границы CI
        upper: Верхние границы CI
        
    Returns:
        Словарь с метриками покрытия
    """
    valid_mask = np.isfinite(actual) & np.isfinite(lower) & np.isfinite(upper)
    
    if valid_mask.sum() == 0:
        return {
            'coverage_rate': 0.0,
            'n_in_range': 0,
            'n_total': 0,
            'expected_coverage': 0.95
        }
    
    actual_clean = actual[valid_mask]
    lower_clean = lower[valid_mask]
    upper_clean = upper[valid_mask]
    
    # Проверяем, сколько значений попадает в интервал
    in_range = (actual_clean >= lower_clean) & (actual_clean <= upper_clean)
    n_in_range = int(in_range.sum())
    n_total = len(actual_clean)
    
    coverage_rate = float(n_in_range / n_total) if n_total > 0 else 0.0
    
    return {
        'coverage_rate': coverage_rate,
        'n_in_range': n_in_range,
        'n_total': n_total,
        'expected_coverage': 0.95
    }


def find_local_minima_shift(
    df: pd.DataFrame,
    y_col: str = 'y',
    yhat_col: str = 'yhat',
    ds_col: str = 'ds',
    window: int = 7
) -> Dict[str, Any]:
    """
    Находит сдвиг локальных минимумов между фактическими и предсказанными значениями.
    
    Args:
        df: DataFrame с данными
        y_col: Колонка с фактическими значениями
        yhat_col: Колонка с предсказанными значениями
        ds_col: Колонка с датами
        window: Окно для поиска минимумов
        
    Returns:
        Словарь с метриками сдвига
    """
    if y_col not in df.columns or yhat_col not in df.columns:
        return {
            'mean_shift_days': 0.0,
            'median_shift_days': 0.0,
            'std_shift_days': 0.0,
            'n_minima_found': 0
        }
    
    df = df.copy()
    df[ds_col] = pd.to_datetime(df[ds_col])
    df = df.sort_values(ds_col).reset_index(drop=True)
    
    y_values = df[y_col].values
    yhat_values = df[yhat_col].values
    
    # Находим локальные минимумы для y и yhat
    def find_local_minima(values, window=window):
        """Находит индексы локальных минимумов."""
        minima = []
        for i in range(window, len(values) - window):
            if all(values[i] <= values[i+j] for j in range(-window, window+1) if j != 0):
                minima.append(i)
        return minima
    
    y_minima_idx = find_local_minima(y_values, window)
    yhat_minima_idx = find_local_minima(yhat_values, window)
    
    if len(y_minima_idx) == 0 or len(yhat_minima_idx) == 0:
        return {
            'mean_shift_days': 0.0,
            'median_shift_days': 0.0,
            'std_shift_days': 0.0,
            'n_minima_found': min(len(y_minima_idx), len(yhat_minima_idx))
        }
    
    # Находим ближайшие пары минимумов
    shifts = []
    for y_idx in y_minima_idx:
        # Ищем ближайший минимум в yhat
        nearest_yhat_idx = min(yhat_minima_idx, key=lambda x: abs(x - y_idx))
        shift_days = nearest_yhat_idx - y_idx
        shifts.append(shift_days)
    
    if len(shifts) == 0:
        return {
            'mean_shift_days': 0.0,
            'median_shift_days': 0.0,
            'std_shift_days': 0.0,
            'n_minima_found': 0
        }
    
    return {
        'mean_shift_days': float(np.mean(shifts)),
        'median_shift_days': float(np.median(shifts)),
        'std_shift_days': float(np.std(shifts)),
        'n_minima_found': len(shifts)
    }


def diagnose_model(
    df_history: pd.DataFrame,
    df_forecast: pd.DataFrame,
    model_name: str = "unknown",
    include_regressors: bool = False
) -> Dict[str, Any]:
    """
    Полная диагностика модели.
    
    Args:
        df_history: DataFrame с историческими данными (ds, y, возможно avg_price, avg_discount)
        df_forecast: DataFrame с прогнозами (ds, yhat, yhat_lower, yhat_upper)
        model_name: Имя модели для лога
        include_regressors: Использовались ли регрессоры
        
    Returns:
        Словарь с результатами диагностики
    """
    logger.info(f"Starting model diagnostics for {model_name}")
    
    # Объединяем историю и прогноз для анализа
    df_combined = df_history.merge(
        df_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
        on='ds',
        how='inner'
    )
    
    if len(df_combined) == 0:
        logger.warning("No overlapping dates between history and forecast for diagnostics")
        return {
            'model_name': model_name,
            'error': 'No overlapping dates'
        }
    
    df_combined = df_combined.sort_values('ds').reset_index(drop=True)
    
    # 1. Анализ остатков
    residuals = calculate_residuals(
        df_combined['y'].values,
        df_combined['yhat'].values
    )
    residuals_analysis = analyze_residuals(residuals, df_combined['ds'])
    
    # 2. Проверка мультиколлинеарности
    multicollinearity = {}
    if include_regressors:
        multicollinearity = check_multicollinearity(df_history)
    
    # 3. Анализ смещения тренда
    trend_bias = analyze_trend_bias(df_combined)
    
    # 4. Покрытие CI
    coverage = calculate_coverage(
        df_combined['y'].values,
        df_combined['yhat_lower'].values,
        df_combined['yhat_upper'].values
    )
    
    # 5. Сдвиг локальных минимумов
    minima_shift = find_local_minima_shift(df_combined)
    
    # 6. Общие метрики
    mae_val = mean_absolute_error(df_combined['y'].values, df_combined['yhat'].values)
    rmse_val = np.sqrt(mean_squared_error(df_combined['y'].values, df_combined['yhat'].values))
    mape_val = mape(df_combined['y'].values, df_combined['yhat'].values)
    
    # Систематический bias (средний остаток)
    systematic_bias = float(np.mean(residuals))
    
    results = {
        'model_name': model_name,
        'metrics': {
            'mae': float(mae_val),
            'rmse': float(rmse_val),
            'mape': float(mape_val),
            'systematic_bias': systematic_bias
        },
        'residuals_analysis': residuals_analysis,
        'multicollinearity': multicollinearity,
        'trend_bias': trend_bias,
        'coverage': coverage,
        'minima_shift': minima_shift,
        'n_samples': len(df_combined)
    }
    
    logger.info(f"Diagnostics completed for {model_name}")
    logger.info(f"  MAPE: {mape_val:.2f}%, Coverage: {coverage['coverage_rate']*100:.1f}%, "
                f"Trend bias: {trend_bias['trend_bias_pct']:.1f}%")
    
    return results


def save_diagnostics(
    diagnostics: Dict[str, Any],
    output_path: str
) -> None:
    """
    Сохраняет результаты диагностики в JSON.
    
    Args:
        diagnostics: Результаты диагностики
        output_path: Путь для сохранения
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(diagnostics, f, indent=2, default=str)
    
    logger.info(f"Diagnostics saved to {output_path}")

