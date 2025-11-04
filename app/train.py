# file: app/train.py
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import warnings
import json
import os
import logging
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        
    Returns:
        MAPE value as percentage
    """
    mask = actual != 0
    if mask.any():
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    else:
        return 0.0


def train_prophet(
    shop_csv_path: str, 
    model_out_path: str, 
    include_regressors: bool = False,
    log_transform: bool = False,
    interval_width: float = 0.95,
    holdout_frac: float = 0.2,
    changepoint_prior_scale: float = 0.01,  # Консервативное значение для стабильности
    seasonality_prior_scale: float = 10.0,
    seasonality_mode: str = 'additive',
    auto_tune: bool = False,
    skip_holdout: bool = False,  # Если True, использует ВСЕ данные для обучения (без теста)
    filter_column: Optional[str] = None,  # 'category' или 'product_id' для фильтрации
    filter_value: Optional[str] = None  # Значение для фильтрации (название категории или ID товара)
) -> Dict[str, Any]:
    """
    Train a Prophet model on sales data (shop-level, category-level, or product-level).
    
    Args:
        shop_csv_path: Path to the CSV file (shop-level, category-level, or product-level)
        model_out_path: Path to save the trained model using joblib
        include_regressors: Whether to include avg_price and avg_discount as regressors
        log_transform: If True, apply log1p transformation to y before training
        interval_width: Confidence interval width for Prophet (default 0.95)
        holdout_frac: Fraction of data to use for testing (default 0.2)
        changepoint_prior_scale: Flexibility of automatic changepoint detection (default 0.05, higher = more flexible)
        seasonality_prior_scale: Strength of seasonality components (default 10.0, higher = stronger seasonality)
        seasonality_mode: 'additive' or 'multiplicative' (default 'additive')
        auto_tune: If True, perform automatic grid search to find best configuration
        filter_column: Optional column name for filtering ('category' or 'product_id')
        filter_value: Optional value to filter by (category name or product_id)
        
    Returns:
        Dictionary containing model path, metrics, and data ranges
    """
    logger.info(f"Starting model training from: {shop_csv_path}")
    
    # Если указана фильтрация, логируем
    if filter_column and filter_value:
        logger.info(f"Filtering data: {filter_column} = '{filter_value}'")
    
    # Auto-tuning: perform grid search
    if auto_tune:
        logger.info("Auto-tuning enabled: performing grid search...")
        try:
            from app.tuning import grid_search_models
            
            analysis_dir = os.path.join(os.path.dirname(model_out_path) or 'models', '..', 'analysis')
            analysis_dir = os.path.normpath(analysis_dir)
            
            tuning_results = grid_search_models(
                shop_csv_path=shop_csv_path,
                holdout_frac=holdout_frac,
                output_dir=analysis_dir
            )
            
            if tuning_results.get('success', False):
                best_model = tuning_results['best_model']
                best_config = best_model.get('config', {})
                
                logger.info(f"Best model from grid search: {best_model['name']}")
                logger.info(f"Best metrics: MAPE={best_model['metrics']['mape']:.2f}%, "
                           f"Coverage={best_model['metrics']['coverage']*100:.1f}%")
                
                # Обновляем параметры из лучшей конфигурации
                if 'seasonality_prior_scale' in best_config:
                    seasonality_prior_scale = best_config['seasonality_prior_scale']
                if 'changepoint_prior_scale' in best_config:
                    changepoint_prior_scale = best_config['changepoint_prior_scale']
                if 'interval_width' in best_config:
                    interval_width = best_config['interval_width']
                if 'seasonality_mode' in best_config:
                    seasonality_mode = best_config['seasonality_mode']
                if 'include_regressors' in best_config:
                    include_regressors = best_config['include_regressors']
                
                logger.info(f"Using optimized parameters: seasonality_prior_scale={seasonality_prior_scale}, "
                           f"changepoint_prior_scale={changepoint_prior_scale}, interval_width={interval_width}, "
                           f"seasonality_mode={seasonality_mode}, include_regressors={include_regressors}")
            else:
                logger.warning("Grid search failed or returned no results, using default parameters")
        except Exception as e:
            logger.error(f"Auto-tuning failed: {str(e)}, falling back to default parameters")
            logger.error(f"Error details: {str(e)}", exc_info=True)
    
    # Read the shop CSV file
    if not os.path.exists(shop_csv_path):
        raise FileNotFoundError(f"Shop CSV file not found: {shop_csv_path}")
    
    df = pd.read_csv(shop_csv_path)
    logger.info(f"Loaded {len(df)} rows from CSV")
    
    # Verify required columns exist
    if 'ds' not in df.columns or 'y' not in df.columns:
        raise ValueError("CSV must contain 'ds' and 'y' columns")
    
    # Apply filtering if specified
    if filter_column and filter_value:
        if filter_column not in df.columns:
            raise ValueError(f"Filter column '{filter_column}' not found in CSV. Available columns: {list(df.columns)}")
        
        # Convert filter_value to appropriate type if needed
        if filter_column == 'product_id':
            # Try to match product_id as string or convert to match data type
            df_filtered = df[df[filter_column].astype(str) == str(filter_value)].copy()
        else:
            # For category, match as string
            df_filtered = df[df[filter_column].astype(str) == str(filter_value)].copy()
        
        if len(df_filtered) == 0:
            available_values = df[filter_column].unique()[:10]  # Show first 10
            raise ValueError(f"No data found for {filter_column}='{filter_value}'. "
                           f"Available values (first 10): {list(available_values)}")
        
        df = df_filtered
        logger.info(f"After filtering: {len(df)} rows for {filter_column}='{filter_value}'")
        
        # Проверяем минимальное количество данных после фильтрации
        # Для недельной агрегации требуется меньше строк, чем для дневной
        min_required = 8 if 'ds' in df.columns and len(df) > 0 and (pd.to_datetime(df['ds'].max()) - pd.to_datetime(df['ds'].min())).days < 100 else 30
        if len(df) < min_required:
            raise ValueError(f"⚠️ НЕДОСТАТОЧНО ДАННЫХ для {filter_column}='{filter_value}': {len(df)} строк "
                           f"(минимум {min_required} требуется). "
                           f"Для категорий с малым количеством данных попробуйте:\n"
                           f"1. Использовать shop-level прогноз (более стабильный)\n"
                           f"2. Увеличить период данных\n"
                           f"3. Использовать skip_holdout=True для обучения на всех данных")
    
    # Prepare the data
    df = df.copy()
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds').reset_index(drop=True)
    
    # Save original y values before any transformation
    original_y = df['y'].copy()
    
    # Split data by time: train on first (1-holdout_frac), test on last holdout_frac
    # Разделение данных на train/test или использование всех данных
    if skip_holdout:
        # Используем ВСЕ данные для обучения (без тестового набора)
        logger.info("skip_holdout=True: Using ALL data for training (no test set split)")
        df_train = df.copy()
        df_test = pd.DataFrame(columns=df.columns)  # Пустой test set
        n_train = len(df_train)
        n_test = 0
        if n_train < 30:
            raise ValueError(f"Insufficient data for training: {n_train} rows (minimum 30 required)")
        logger.info(f"Training on ALL {n_train} days ({df_train['ds'].min().date()} to {df_train['ds'].max().date()})")
        logger.info("⚠️ No test set - metrics will not be calculated. Use for production forecasts.")
    else:
        n_total = len(df)
        n_train = int(n_total * (1 - holdout_frac))
        n_test = n_total - n_train
        
        if n_train < 30:
            raise ValueError(f"Insufficient data for training: {n_train} rows (minimum 30 required)")
        
        df_train = df.iloc[:n_train].copy()
        df_test = df.iloc[n_train:].copy()
    
    # Apply log transformation if requested (after split to preserve original test values)
    # Always save original test y values for metrics calculation
    df_test_original_y = df_test['y'].copy()
    
    if log_transform:
        logger.info("Applying log1p transformation to target variable")
        
        # Проверяем, что после log_transform останется достаточно данных
        # log1p(0) = 0, так что нули остаются нулями, но Prophet может не работать с большим количеством нулей
        non_zero_before = (df_train['y'] > 0).sum()
        if non_zero_before < 2:
            raise ValueError(f"⚠️ КРИТИЧЕСКАЯ ОШИБКА: После фильтрации по {filter_column}='{filter_value}' осталось "
                           f"только {non_zero_before} ненулевых значений в обучающем наборе ({len(df_train)} всего). "
                           f"Prophet не может обучаться на таких данных.\n"
                           f"Рекомендации:\n"
                           f"1. Используйте shop-level прогноз вместо категорийного\n"
                           f"2. Попробуйте без log-transform (снимите галочку)\n"
                           f"3. Используйте skip_holdout=True для обучения на всех данных\n"
                           f"4. Убедитесь, что выбранная категория имеет достаточно продаж")
        
        df_train['y'] = np.log1p(df_train['y'])
        df_test['y'] = np.log1p(df_test['y'])
        
        # Проверяем, что после log_transform не появилось NaN
        nan_count_train = df_train['y'].isna().sum()
        if nan_count_train > 0:
            logger.warning(f"После log_transform появилось {nan_count_train} NaN значений, заменяем на 0")
            df_train['y'] = df_train['y'].fillna(0)
        
        nan_count_test = df_test['y'].isna().sum()
        if nan_count_test > 0 and len(df_test) > 0:
            logger.warning(f"После log_transform в тесте появилось {nan_count_test} NaN значений, заменяем на 0")
            df_test['y'] = df_test['y'].fillna(0)
    
    
    train_range = {
        'start': df_train['ds'].min().isoformat(),
        'end': df_train['ds'].max().isoformat()
    }
    
    # Обработка test_range: если skip_holdout=True, df_test пустой
    if skip_holdout or len(df_test) == 0:
        test_range = {
            'start': None,
            'end': None
        }
        logger.info(f"Train period: {train_range['start']} to {train_range['end']} ({n_train} rows)")
        logger.info(f"Test period: N/A (skip_holdout=True, using all data for training)")
    else:
        test_range = {
            'start': df_test['ds'].min().isoformat(),
            'end': df_test['ds'].max().isoformat()
        }
        logger.info(f"Train period: {train_range['start']} to {train_range['end']} ({n_train} rows)")
        logger.info(f"Test period: {test_range['start']} to {test_range['end']} ({n_test} rows)")
    
    # Prepare data for Prophet
    prophet_cols = ['ds', 'y']
    if include_regressors:
        if 'avg_price' not in df.columns or 'avg_discount' not in df.columns:
            logger.warning("Regressors requested but avg_price/avg_discount not found in CSV. Proceeding without regressors.")
            include_regressors = False
        else:
            prophet_cols.extend(['avg_price', 'avg_discount'])
    
    df_prophet_train = df_train[prophet_cols].copy()
    
    # АВТОМАТИЧЕСКОЕ ОПРЕДЕЛЕНИЕ ВОЛАТИЛЬНОСТИ ДАННЫХ (перед обработкой)
    # Вычисляем коэффициент вариации (CV) для оценки волатильности
    train_values = df_prophet_train['y'].values
    train_values_nonzero = train_values[train_values > 0]
    
    if len(train_values_nonzero) > 1:
        mean_val = np.mean(train_values_nonzero)
        std_val = np.std(train_values_nonzero)
        cv = (std_val / mean_val) if mean_val > 0 else 0.0  # Коэффициент вариации
    else:
        cv = 0.0
        mean_val = 0.0
        std_val = 0.0
    
    # Определяем уровень волатильности
    is_highly_volatile = cv > 1.0  # CV > 1.0 означает очень высокую волатильность
    is_moderately_volatile = cv > 0.5  # CV > 0.5 означает умеренную волатильность
    
    logger.info(f"📊 Анализ волатильности данных:")
    logger.info(f"   Коэффициент вариации (CV): {cv:.2f}")
    logger.info(f"   Среднее значение: {mean_val:.2f}")
    logger.info(f"   Стандартное отклонение: {std_val:.2f}")
    
    if is_highly_volatile:
        logger.warning(f"⚠️ ОБНАРУЖЕНА ВЫСОКАЯ ВОЛАТИЛЬНОСТЬ (CV={cv:.2f} > 1.0)")
        logger.warning("   Применяем специальные настройки для волатильных данных...")
    elif is_moderately_volatile:
        logger.info(f"ℹ️ Обнаружена умеренная волатильность (CV={cv:.2f} > 0.5)")
    
    # Для категорий/товаров: проверяем и обрабатываем много нулевых значений
    if filter_column is not None:
        zero_count = (df_prophet_train['y'] == 0).sum()
        zero_percent = (zero_count / len(df_prophet_train)) * 100 if len(df_prophet_train) > 0 else 0
        
        # Более агрессивная обработка для категорий с >30% нулей
        # НО для волатильных данных сглаживание может ухудшить прогноз - применяем минимальное сглаживание
        if zero_percent > 30:
            logger.warning(f"⚠️ МНОГО НУЛЕВЫХ ЗНАЧЕНИЙ: {zero_percent:.1f}% ({zero_count} из {len(df_prophet_train)})")
            logger.warning("Это может сильно ухудшить прогноз. Prophet плохо работает с разреженными данными.")
            
            # Определяем, нужен ли агрессивный режим
            use_aggressive = zero_percent > 50
            
            # ДЛЯ ВОЛАТИЛЬНЫХ ДАННЫХ: минимизируем сглаживание
            if is_highly_volatile:
                logger.info("⚠️ ВОЛАТИЛЬНЫЕ ДАННЫЕ: минимизирую сглаживание для сохранения волатильности")
                use_aggressive = False  # Не применяем агрессивное сглаживание для волатильных данных
            
            if use_aggressive:
                logger.warning("⚠️ ОЧЕНЬ РАЗРЕЖЕННЫЕ ДАННЫЕ! Применяем АГРЕССИВНУЮ обработку...")
            
            # Применяем улучшение качества данных (замена нулей на медианы, сглаживание)
            # Но только если данные не очень волатильны
            if not is_highly_volatile or zero_percent > 70:
                try:
                    from app.preprocessing import _improve_data_quality
                    df_prophet_train = _improve_data_quality(df_prophet_train, aggressive=use_aggressive)
                    logger.info("Улучшение качества данных применено успешно")
                    
                    # Проверяем результат
                    zero_count_after = (df_prophet_train['y'] == 0).sum()
                    zero_percent_after = (zero_count_after / len(df_prophet_train)) * 100 if len(df_prophet_train) > 0 else 0
                    logger.info(f"После обработки: {zero_percent_after:.1f}% нулей (было {zero_percent:.1f}%)")
                except Exception as e:
                    logger.warning(f"Не удалось применить улучшение качества данных: {str(e)}")
            else:
                logger.info("⚠️ Пропускаю сглаживание для волатильных данных - сохраняю оригинальную волатильность")
        
        if zero_percent > 70:
            logger.error(f"⚠️ КРИТИЧЕСКИ МНОГО НУЛЕЙ: {zero_percent:.1f}%!")
            logger.error("Prophet может дать плохой прогноз даже без сезонности.")
            logger.error("РЕКОМЕНДАЦИЯ: Используйте shop-level прогноз и распределите его по категориям пропорционально.")
        
        # Предупреждение о необходимости skip_holdout для разреженных категорий
        if zero_percent > 40 and not skip_holdout:
            logger.warning(f"⚠️ ВАЖНО: Для категории с {zero_percent:.1f}% нулей рекомендуется:")
            logger.warning("  1. Включить 'skip_holdout=True' (обучение на ВСЕХ данных)")
            logger.warning("  2. Или уменьшить holdout_frac до 0.05-0.1")
            logger.warning("  3. Иначе может остаться слишком мало данных для обучения!")
    
    # Validate seasonality_mode
    if seasonality_mode not in ['additive', 'multiplicative']:
        raise ValueError(f"seasonality_mode must be 'additive' or 'multiplicative', got '{seasonality_mode}'")
    
    # Определяем, достаточно ли данных для yearly seasonality
    # Prophet рекомендует минимум 730 дней (2 года) для надежной yearly seasonality
    days_span = (df['ds'].max() - df['ds'].min()).days
    use_yearly = days_span >= 730  # Только если данных >= 2 лет
    
    # Определяем, является ли агрегация weekly или daily
    # Проверяем средний интервал между датами
    if len(df) > 1:
        df_sorted = df.sort_values('ds')
        time_diffs = df_sorted['ds'].diff().dropna()
        avg_days_between = time_diffs.median().total_seconds() / (24 * 3600) if len(time_diffs) > 0 else 1.0
        is_weekly_aggregated = avg_days_between >= 5.0  # Если средний интервал >= 5 дней, это weekly
    else:
        is_weekly_aggregated = False
        avg_days_between = 1.0
    
    # Для shop-level данных тоже применяем адаптивные настройки для волатильности
    # Для категорий/товаров применяем более агрессивные настройки по умолчанию
    # если пользователь не указал явно другие значения
    is_category_or_product = filter_column is not None
    if is_category_or_product:
        # Для категорий увеличиваем гибкость changepoints еще больше для улавливания волатильности
        # Используем только changepoints без seasonality для избежания циклических паттернов
        
        # АДАПТИВНАЯ НАСТРОЙКА ДЛЯ ВОЛАТИЛЬНЫХ ДАННЫХ
        if is_highly_volatile:
            # Для очень волатильных данных - максимальная гибкость
            if changepoint_prior_scale <= 0.01:
                changepoint_prior_scale = 0.5  # Очень высокая гибкость для волатильных данных
                logger.info(f"🔥 ВЫСОКАЯ ВОЛАТИЛЬНОСТЬ: увеличиваю changepoint_prior_scale до 0.5 для лучшего улавливания всплесков")
            elif changepoint_prior_scale < 0.3:
                changepoint_prior_scale = max(changepoint_prior_scale * 3.0, 0.3)
                logger.info(f"🔥 ВОЛАТИЛЬНОСТЬ: увеличиваю changepoint_prior_scale до {changepoint_prior_scale}")
            
            # Для волатильных данных используем multiplicative режим если он не был явно указан
            if seasonality_mode == 'additive' and not auto_tune:
                logger.info("💡 Рекомендация: для волатильных данных лучше использовать multiplicative режим")
                # Не меняем автоматически, только предупреждаем
        elif is_moderately_volatile:
            # Для умеренно волатильных данных - средняя гибкость
            if changepoint_prior_scale <= 0.01:
                changepoint_prior_scale = 0.3
                logger.info(f"📈 Умеренная волатильность: увеличиваю changepoint_prior_scale до 0.3")
            elif changepoint_prior_scale < 0.2:
                changepoint_prior_scale = max(changepoint_prior_scale * 2.0, 0.2)
                logger.info(f"📈 Умеренная волатильность: увеличиваю changepoint_prior_scale до {changepoint_prior_scale}")
        else:
            # Для стабильных данных - стандартные настройки
            if changepoint_prior_scale <= 0.01:
                changepoint_prior_scale = 0.25  # Высокая гибкость для улавливания всплесков и падений
                logger.info("Category/product data: increasing changepoint_prior_scale to 0.25 for better volatility capture")
            elif changepoint_prior_scale < 0.2:
                changepoint_prior_scale = max(changepoint_prior_scale * 2.0, 0.2)  # Увеличиваем если пользователь указал низкое значение
                logger.info(f"Category/product data: increasing changepoint_prior_scale to {changepoint_prior_scale} for volatility")
        
        # Для категорий полностью отключаем seasonality - используем только changepoints
        seasonality_prior_scale = 0.1  # Минимальное значение (сезонность будет отключена)
        logger.info("Category/product data: disabling seasonality completely, using only trend + flexible changepoints")
        
        # Уменьшаем interval_width для более узкого доверительного интервала
        if interval_width >= 0.95:
            interval_width = 0.80  # Более узкий интервал для категорий
            logger.info("Category/product data: reducing interval_width to 0.80 for narrower confidence interval")
    else:
        # ДЛЯ SHOP-LEVEL ДАННЫХ: также применяем адаптивные настройки для волатильности
        if is_highly_volatile:
            # Для волатильных shop-level данных увеличиваем changepoint_prior_scale
            if changepoint_prior_scale <= 0.01:
                changepoint_prior_scale = 0.1  # Умеренная гибкость для shop-level
                logger.info(f"🔥 ВОЛАТИЛЬНЫЕ SHOP-LEVEL ДАННЫЕ: увеличиваю changepoint_prior_scale до 0.1")
            elif changepoint_prior_scale < 0.05:
                changepoint_prior_scale = max(changepoint_prior_scale * 2.0, 0.05)
                logger.info(f"🔥 ВОЛАТИЛЬНОСТЬ: увеличиваю changepoint_prior_scale до {changepoint_prior_scale}")
        elif is_moderately_volatile:
            if changepoint_prior_scale <= 0.01:
                changepoint_prior_scale = 0.05
                logger.info(f"📈 Умеренная волатильность shop-level: увеличиваю changepoint_prior_scale до 0.05")
    
    if not use_yearly and days_span < 730:
        logger.warning(f"Data span ({days_span} days) < 730 days. Disabling yearly_seasonality for stability.")
        logger.info("Using only weekly_seasonality. This is recommended for datasets < 2 years.")
        logger.warning("⚠️ LONG-TERM FORECAST WARNING: For forecasts > 90 days with data < 2 years, "
                      "the model may show flat/cyclical patterns due to missing yearly_seasonality.")
    
    # Определяем, использовать ли weekly seasonality
    # Для категорий/товаров полностью отключаем seasonality - используем только changepoints
    if filter_column is not None:
        # Для категорий полностью отключаем weekly и yearly seasonality
        # Волатильность будет улавливаться через гибкие changepoints
        use_weekly_seasonality = False
        use_yearly = False  # Yearly отключаем для категорий
        logger.info(f"⚠️ Для категорий/товаров полностью отключаем сезонность")
        logger.info("Волатильность будет улавливаться через гибкие changepoints (changepoint_prior_scale={:.2f})".format(changepoint_prior_scale))
        logger.info("Это должно устранить циклические паттерны при сохранении способности улавливать всплески и падения")
    else:
        use_weekly_seasonality = True
        # use_yearly уже определен выше
    
    # Initialize Prophet model with configurable hyperparameters
    model = Prophet(
        weekly_seasonality=use_weekly_seasonality,  # Отключено для категорий
        yearly_seasonality=use_yearly,  # Автоматически отключаем для коротких данных
        interval_width=interval_width,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        seasonality_mode=seasonality_mode
    )
    
    # Для weekly агрегации категорий НЕ добавляем monthly seasonality - она создает циклические паттерны
    # Для данных >= 365 дней (но < 730): добавляем месячную сезонность как компромисс
    # НО только для daily агрегации или shop-level данных
    if days_span >= 365 and days_span < 730:
        # Для категорий/товаров НЕ добавляем monthly seasonality
        if filter_column is not None:
            logger.info("Category/product data: skipping monthly seasonality (using only trend)")
        else:
            # Для shop-level добавляем monthly seasonality
            logger.info(f"Data span ({days_span} days) >= 365 but < 730. Adding monthly seasonality.")
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    
    logger.info(f"Prophet model config: changepoint_prior_scale={changepoint_prior_scale}, "
                f"seasonality_prior_scale={seasonality_prior_scale}, seasonality_mode={seasonality_mode}")
    
    # Add regressors if requested
    if include_regressors:
        logger.info("Adding regressors: avg_price, avg_discount")
        model.add_regressor('avg_price')
        model.add_regressor('avg_discount')
    
    # Финальная проверка перед обучением модели
    # Проверяем количество валидных (non-NaN) строк после всех преобразований
    valid_rows = df_prophet_train['y'].notna().sum()
    total_rows = len(df_prophet_train)
    
    if valid_rows < 2:
        error_msg = (
            f"⚠️ КРИТИЧЕСКАЯ ОШИБКА: После всех преобразований осталось меньше 2 валидных строк "
            f"для обучения ({valid_rows} из {total_rows} строк).\n\n"
            f"Возможные причины:\n"
            f"1. Слишком много нулевых значений в категории/товаре (>90%)\n"
            f"2. Недельная агрегация + фильтрация оставила слишком мало данных "
            f"   (нужно минимум 8-10 недель для недельной агрегации)\n"
            f"3. log_transform в сочетании с разреженными данными создал проблемы\n"
            f"4. holdout_frac слишком большой для маленького датасета\n\n"
            f"Решения (в порядке приоритета):\n"
            f"1. ❌ ОТКЛЮЧИТЕ log-transform (снимите галочку) - это часто решает проблему\n"
            f"2. ✅ Используйте skip_holdout=True (обучение на ВСЕХ данных без разделения)\n"
            f"3. ✅ Попробуйте daily агрегацию вместо weekly (больше данных)\n"
            f"4. ✅ Используйте shop-level прогноз (работает стабильнее для разреженных категорий)\n"
            f"5. ✅ Выберите другую категорию с большим количеством продаж"
        )
        
        if filter_column is not None:
            error_msg += f"\n\n💡 Для категории '{filter_value}': проверьте статистику данных перед обучением."
        
        raise ValueError(error_msg)
    
    # Дополнительная проверка для недельной агрегации
    if total_rows < 8 and filter_column is not None:
        logger.warning(f"⚠️ Очень мало данных: {total_rows} строк. Для недельной агрегации рекомендуется минимум 15-20 недель.")
    
    # Fit the model
    logger.info(f"Fitting Prophet model on {valid_rows} valid rows ({total_rows} total)...")
    try:
        model.fit(df_prophet_train)
        logger.info("Model fitted successfully")
    except Exception as e:
        error_str = str(e)
        if "less than 2 non-NaN rows" in error_str or "Dataframe has less than 2" in error_str:
            raise ValueError(
                f"⚠️ Prophet не может обучить модель: недостаточно данных.\n"
                f"Валидных строк: {valid_rows}, Всего строк: {total_rows}\n"
                f"Ошибка Prophet: {error_str}\n\n"
                f"Попробуйте:\n"
                f"1. ❌ ОТКЛЮЧИТЕ log-transform (снимите галочку)\n"
                f"2. ✅ Включите skip_holdout=True (обучение на всех данных)\n"
                f"3. ✅ Используйте daily агрегацию вместо weekly\n"
                f"4. ✅ Выберите категорию с большим количеством данных\n"
                f"5. ✅ Используйте shop-level прогноз (работает лучше)"
            ) from e
        else:
            raise
    
    # Вычисление метрик (пропускается если skip_holdout)
    if skip_holdout:
        logger.info("Skipping metrics calculation (skip_holdout=True)")
        use_cv = False
        metrics_dict = {
            'mae': None,
            'rmse': None,
            'mape': None,
            'coverage': None,
            'log_transform': log_transform,
            'interval_width': interval_width,
            'holdout_frac': holdout_frac,
            'used_cross_validation': False,
            'changepoint_prior_scale': changepoint_prior_scale,
            'seasonality_prior_scale': seasonality_prior_scale,
            'seasonality_mode': seasonality_mode,
            'auto_tune': auto_tune,
            'skip_holdout': skip_holdout,
            'note': 'Model trained on ALL data. No test set metrics available. Ready for production forecasts.'
        }
    elif n_test < 7:
        # Check if test set is too small for proper evaluation
        use_cv = True
        logger.warning(f"Test set too small ({n_test} < 7). Using time-series cross-validation instead.")
    else:
        use_cv = False
    
    # Calculate metrics (только если не skip_holdout)
    if not skip_holdout:
        if use_cv:
            # Time-series cross-validation
            logger.info("Performing time-series cross-validation...")
            from prophet.diagnostics import cross_validation, performance_metrics
            
            # Perform cross-validation with minimum training period of 30 days
            min_periods = min(30, n_train)
            cv_results = cross_validation(
                model, 
                initial=f'{min_periods} days',
                period='7 days',
                horizon='7 days'
            )
            
            cv_metrics = performance_metrics(cv_results)
            
            # Get actual and predicted values from CV
            actual_values = cv_results['y'].values
            predicted_values = cv_results['yhat'].values
            
            # Apply inverse transform if log_transform was used
            if log_transform:
                logger.info("Applying inverse log1p (expm1) transformation to predictions")
                actual_values = np.expm1(actual_values)
                predicted_values = np.expm1(predicted_values)
            
            # Ensure non-negative values for predictions (realistic constraint)
            # Sales cannot be negative, so clip negative predictions to 0
            n_negative_pred = (predicted_values < 0).sum()
            if n_negative_pred > 0:
                logger.info(f"Clamping {n_negative_pred} negative CV predictions to 0 (sales cannot be negative)")
                predicted_values = np.clip(predicted_values, 0, None)
            
            # Also clip actual values if they're negative (data quality issue)
            n_negative_actual = (actual_values < 0).sum()
            if n_negative_actual > 0:
                logger.warning(f"Found {n_negative_actual} negative actual values in CV (data quality issue)")
                actual_values = np.clip(actual_values, 0, None)
            
            # Calculate metrics
            mae_val = mean_absolute_error(actual_values, predicted_values)
            rmse_val = np.sqrt(mean_squared_error(actual_values, predicted_values))
            mape_val = mape(actual_values, predicted_values)
            
            logger.info(f"Cross-validation metrics: MAE={mae_val:.2f}, RMSE={rmse_val:.2f}, MAPE={mape_val:.2f}%")
        
        else:
            # Standard test set evaluation
            # Create future dataframe for test period
            periods = len(df_test)
            future = model.make_future_dataframe(periods=periods, freq='D')
            
            # Add regressors for future dates if needed
            if include_regressors:
                # Combine train and test regressors for complete coverage
                all_regressors = pd.concat([
                    df_train[['ds', 'avg_price', 'avg_discount']],
                    df_test[['ds', 'avg_price', 'avg_discount']]
                ], ignore_index=True)
                
                # Merge regressors
                future = future.merge(
                    all_regressors,
                    on='ds',
                    how='left'
                )
                
                # Forward-fill regressors for any missing dates
                for col in ['avg_price', 'avg_discount']:
                    if future[col].isna().any():
                        # Forward fill, then use last known value if still NaN
                        future[col] = future[col].ffill()
                        if future[col].isna().any():
                            last_known_value = all_regressors[col].iloc[-1] if not all_regressors.empty else 0
                            future[col] = future[col].fillna(last_known_value)
            
            # Make predictions
            logger.info(f"Generating predictions for {periods} periods...")
            forecast = model.predict(future)
            
            # Extract predictions for test period only
            test_mask = forecast['ds'] >= df_test['ds'].min()
            forecast_test = forecast[test_mask].copy()
            
            # Align actual and predicted values
            forecast_test = forecast_test.sort_values('ds')
            df_test_aligned = df_test.sort_values('ds')
            
            # Merge on ds to ensure alignment
            merged = forecast_test[['ds', 'yhat']].merge(
                df_test_aligned[['ds']],
                on='ds',
                how='inner'
            )
            
            # Get actual y values for metrics (original values if log_transform, transformed otherwise)
            if log_transform:
                # Use original y values before log transform
                merged = merged.merge(
                    pd.DataFrame({'ds': df_test['ds'].values, 'y_original': df_test_original_y.values}),
                    on='ds',
                    how='inner'
                )
                actual_values = merged['y_original'].values
                logger.info("Applying inverse log1p (expm1) transformation to predictions")
                predicted_values = np.expm1(merged['yhat'].values)
            else:
                # Use transformed y values
                merged = merged.merge(
                    df_test_aligned[['ds', 'y']],
                    on='ds',
                    how='inner'
                )
                actual_values = merged['y'].values
                predicted_values = merged['yhat'].values
            
            # Ensure non-negative values for predictions (realistic constraint)
            # Sales cannot be negative, so clip negative predictions to 0
            n_negative_pred = (predicted_values < 0).sum()
            if n_negative_pred > 0:
                logger.info(f"Clamping {n_negative_pred} negative predictions to 0 (sales cannot be negative)")
                predicted_values = np.clip(predicted_values, 0, None)
            
            # Also clip actual values if they're negative (data quality issue)
            n_negative_actual = (actual_values < 0).sum()
            if n_negative_actual > 0:
                logger.warning(f"Found {n_negative_actual} negative actual values (data quality issue)")
                actual_values = np.clip(actual_values, 0, None)
            
            # Calculate metrics
            mae_val = mean_absolute_error(actual_values, predicted_values)
            rmse_val = np.sqrt(mean_squared_error(actual_values, predicted_values))
            mape_val = mape(actual_values, predicted_values)
            
            logger.info(f"Test metrics: MAE={mae_val:.2f}, RMSE={rmse_val:.2f}, MAPE={mape_val:.2f}%")
            
            # Prepare metrics dictionary for test case
            metrics_dict = {
                'mae': float(mae_val),
                'rmse': float(rmse_val),
                'mape': float(mape_val),
                'log_transform': log_transform,
                'interval_width': interval_width,
                'holdout_frac': holdout_frac,
                'used_cross_validation': use_cv,
                'changepoint_prior_scale': changepoint_prior_scale,
                'seasonality_prior_scale': seasonality_prior_scale,
                'seasonality_mode': seasonality_mode,
                'auto_tune': auto_tune,
                'skip_holdout': skip_holdout
            }
    
    # Save the trained model
    model_dir = os.path.dirname(model_out_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    
    logger.info(f"Saving model to: {model_out_path}")
    joblib.dump(model, model_out_path)
    
    # Save metrics to JSON file
    metrics_path = model_out_path.replace('.pkl', '_metrics.json')
    logger.info(f"Saving metrics to: {metrics_path}")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    # Prepare return dictionary
    results = {
        'model_path': model_out_path,
        'metrics': metrics_dict,
        'train_range': train_range,
        'test_range': test_range,
        'n_train': n_train,
        'n_test': n_test if not use_cv else len(cv_results) if use_cv else n_test
    }
    
    logger.info("Training completed successfully")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Prophet model on shop-level sales data")
    parser.add_argument("shop_csv", help="Path to shop-level CSV file")
    parser.add_argument("model_out", help="Output path for trained model")
    parser.add_argument("--include-regressors", action="store_true", 
                       help="Include avg_price and avg_discount as regressors")
    parser.add_argument("--log-transform", action="store_true",
                       help="Apply log1p transformation to target variable")
    parser.add_argument("--interval-width", type=float, default=0.95,
                       help="Confidence interval width (default: 0.95)")
    parser.add_argument("--holdout-frac", type=float, default=0.2,
                       help="Fraction of data for testing (default: 0.2)")
    parser.add_argument("--auto-tune", action="store_true",
                       help="Perform automatic grid search to find best configuration")
    
    args = parser.parse_args()
    
    result = train_prophet(
        shop_csv_path=args.shop_csv,
        model_out_path=args.model_out,
        include_regressors=args.include_regressors,
        log_transform=args.log_transform,
        interval_width=args.interval_width,
        holdout_frac=args.holdout_frac,
        auto_tune=args.auto_tune
    )
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)
    print(f"\nModel saved to: {result['model_path']}")
    print(f"Metrics saved to: {result['model_path'].replace('.pkl', '_metrics.json')}")
    print(f"\nTraining period: {result['train_range']['start']} to {result['train_range']['end']}")
    print(f"Training samples: {result['n_train']}")
    
    # Проверка test_range на None (когда skip_holdout=True)
    test_start = result['test_range'].get('start') if result.get('test_range') else None
    test_end = result['test_range'].get('end') if result.get('test_range') else None
    if test_start and test_end:
        print(f"Test period: {test_start} to {test_end}")
        print(f"Test samples: {result['n_test']}")
    else:
        print(f"Test period: N/A (skip_holdout=True)")
        print(f"Test samples: 0")
    print(f"\nMetrics:")
    mae_val = result['metrics'].get('mae')
    rmse_val = result['metrics'].get('rmse')
    mape_val = result['metrics'].get('mape')
    
    if mae_val is not None:
        print(f"  MAE:  {mae_val:.4f}")
    else:
        print(f"  MAE:  N/A (skip_holdout=True)")
    
    if rmse_val is not None:
        print(f"  RMSE: {rmse_val:.4f}")
    else:
        print(f"  RMSE: N/A (skip_holdout=True)")
    
    if mape_val is not None:
        print(f"  MAPE: {mape_val:.2f}%")
    else:
        print(f"  MAPE: N/A (skip_holdout=True)")
    
    print(f"  Log transform: {result['metrics'].get('log_transform', False)}")
    print(f"  Used cross-validation: {result['metrics']['used_cross_validation']}")
