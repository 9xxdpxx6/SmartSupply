# file: streamlit_app/app.py
import streamlit as st
import pandas as pd
import requests
import io
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
import logging

# Set up logging
logger = logging.getLogger(__name__)


# Set up the page configuration
st.set_page_config(
    page_title="Приложение прогнозирования продаж",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Приложение прогнозирования продаж")

# Initialize session state
if 'uploaded_file_path' not in st.session_state:
    st.session_state.uploaded_file_path = None
if 'preprocessed_shop_csv' not in st.session_state:
    st.session_state.preprocessed_shop_csv = None
if 'preprocessed_category_csv' not in st.session_state:
    st.session_state.preprocessed_category_csv = None
if 'preprocessed_product_csv' not in st.session_state:
    st.session_state.preprocessed_product_csv = None
if 'selected_aggregation_level' not in st.session_state:
    st.session_state.selected_aggregation_level = "shop"
if 'selected_filter_value' not in st.session_state:
    st.session_state.selected_filter_value = None
if 'preprocessing_stats' not in st.session_state:
    st.session_state.preprocessing_stats = None
if 'trained_model_path' not in st.session_state:
    st.session_state.trained_model_path = None
if 'training_metrics' not in st.session_state:
    st.session_state.training_metrics = None
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None
if 'forecast_csv_path' not in st.session_state:
    st.session_state.forecast_csv_path = None
if 'cv_results' not in st.session_state:
    st.session_state.cv_results = None
if 'pdf_data' not in st.session_state:
    st.session_state.pdf_data = None
if 'pdf_filename' not in st.session_state:
    st.session_state.pdf_filename = None
if 'log_transform_used' not in st.session_state:
    st.session_state.log_transform_used = False
if 'diagnostics' not in st.session_state:
    st.session_state.diagnostics = None
if 'model_comparison' not in st.session_state:
    st.session_state.model_comparison = None

# FastAPI backend URL
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8888")

# Sidebar: Mode selection and Health check
with st.sidebar:
    st.header("⚙️ Режим работы")
    
    # Режим работы: Обычный или Продвинутый
    if 'ui_mode' not in st.session_state:
        st.session_state.ui_mode = "simple"  # По умолчанию простой режим
    
    ui_mode = st.radio(
        "Выберите режим:",
        options=["simple", "advanced"],
        format_func=lambda x: {
            "simple": "📱 Обычный (минимум параметров)",
            "advanced": "🔧 Продвинутый (все параметры)"
        }[x],
        index=0 if st.session_state.ui_mode == "simple" else 1,
        help="Обычный режим: автоматизированный интерфейс с оптимальными настройками. Продвинутый режим: полный контроль над всеми параметрами."
    )
    st.session_state.ui_mode = ui_mode
    
    if ui_mode == "simple":
        st.info("💡 **Обычный режим**: Параметры настроены автоматически. Просто загрузите данные и получите прогноз!")
    else:
        st.info("💡 **Продвинутый режим**: Полный контроль над всеми параметрами модели.")
    
    st.markdown("---")
    
    st.header("Статус API")
    
    # Check API health with better error handling
    api_status = "unknown"
    api_message = ""
    
    # Try to get health status, but don't fail completely
    try:
        health_response = requests.get(f"{FASTAPI_URL}/health", timeout=3)
        if health_response.status_code == 200:
            api_status = "healthy"
            api_message = "✅ Backend API работает"
            st.success(api_message)
        else:
            api_status = "error"
            api_message = f"⚠️ Backend API вернул код {health_response.status_code}"
            st.warning(api_message)
    except requests.exceptions.ConnectionError:
        api_status = "connection_error"
        api_message = "⚠️ Не удалось подключиться к API"
        st.warning(api_message)
        st.info(f"💡 Убедитесь, что FastAPI запущен на {FASTAPI_URL}")
        st.info("💡 Запустите: `python -m app.main` или `uvicorn app.main:app --reload --port 8888`")
    except requests.exceptions.Timeout:
        api_status = "timeout"
        api_message = "⏱️ API не отвечает (таймаут)"
        st.warning(api_message)
    except Exception as e:
        api_status = "error"
        api_message = f"⚠️ Ошибка проверки API: {str(e)[:50]}"
        st.warning(api_message)
    
    # Button to retry health check
    if st.button("🔄 Проверить снова", help="Повторно проверить статус API"):
        st.rerun()
    
    st.header("Помощь")
    with st.expander("О приложении"):
        st.write("""
        Это приложение позволяет:
        1. Загружать данные о продажах (CSV)
        2. Обрабатывать и валидировать данные
        3. Обучать модели прогнозирования Prophet на трех уровнях:
           - 🏪 По всему магазину (агрегат всех продаж)
           - 📁 По категориям товаров
           - 📦 По конкретным товарам (product_id)
        4. Оценивать модели с помощью кросс-валидации
        5. Генерировать прогнозы
        6. Скачивать отчеты в формате PDF
        """)
    
    # Добавляем секцию с пояснениями к параметрам
    with st.expander("📚 Пояснения к параметрам и метрикам", expanded=False):
        st.write("""
        **Основные параметры данных:**
        
        - **y** - Фактическое значение продаж в исторических данных (target variable). Это реальные данные о продажах за прошлые периоды.
        
        - **yhat** - Прогнозируемое значение. Это предсказание модели для будущих периодов или тестового набора.
        
        - **ds** - Дата (date string). Столбец с датами в формате, который понимает Prophet.
        
        **Метрики качества модели:**
        
        - **MAPE** (Mean Absolute Percentage Error) - Средняя абсолютная процентная ошибка. 
          Показывает среднее процентное отклонение прогноза от фактических значений.
          - < 15% - Отличное качество
          - 15-20% - Хорошее качество
          - 20-30% - Удовлетворительное качество
          - > 30% - Требует улучшения
        
        - **MAE** (Mean Absolute Error) - Средняя абсолютная ошибка в единицах измерения.
          Показывает среднюю величину ошибки без учета направления (переоценка/недооценка).
        
        - **RMSE** (Root Mean Square Error) - Корень среднеквадратичной ошибки.
          Учитывает большие ошибки сильнее, чем MAE. Полезно для выявления выбросов.
        
        **Параметры модели:**
        
        - **Interval width** - Ширина доверительного интервала (0.95 = 95%).
          Показывает диапазон, в который с указанной вероятностью попадут фактические значения.
        
        - **Holdout fraction** - Доля данных для тестирования.
          Часть данных, которая не используется для обучения и служит для оценки качества.
        
        - **Changepoint flexibility** - Гибкость обнаружения точек изменения тренда.
          Выше значение = модель более гибкая, но риск переобучения.
        
        - **Seasonality strength** - Сила сезонных компонентов.
          Контролирует, насколько сильно модель учитывает сезонные паттерны.
        """)

# Determine UI mode
ui_mode = st.session_state.get('ui_mode', 'simple')

# Инициализируем распределенный прогноз по умолчанию для простого режима
if ui_mode == "simple":
    if 'use_distributed_forecast' not in st.session_state:
        st.session_state.use_distributed_forecast = True  # По умолчанию включен распределенный прогноз
    if 'selected_aggregation_level' not in st.session_state:
        st.session_state.selected_aggregation_level = "category"  # Устанавливаем уровень на category

# ОБЫЧНЫЙ РЕЖИМ: Упрощенный интерфейс
if ui_mode == "simple":
    st.header("📱 Обычный режим: Автоматизированный прогноз")
    st.info("💡 **Автоматический режим**: Все параметры настроены оптимально. Просто загрузите данные и получите прогноз!")
    
    # Шаг 1: Загрузка файла
    st.subheader("📁 Шаг 1: Загрузка данных")
    uploaded_file = st.file_uploader(
        "Загрузите CSV файл с данными о продажах", 
        type=["csv"],
        key="file_uploader_simple",
        help="CSV файл должен содержать колонки: Sale_Date, Product_ID, Product_Category, Unit_Price, Discount, Quantity_Sold"
    )
    
    if uploaded_file is not None:
        # Display raw data preview
        bytes_data = uploaded_file.getvalue()
        df_raw = pd.read_csv(io.StringIO(bytes_data.decode("utf-8")))
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Предпросмотр данных")
            st.dataframe(df_raw.head(10))
        
        with col2:
            st.subheader("Информация")
            st.write(f"Колонки: {', '.join(df_raw.columns.tolist())}")
            st.write(f"Всего строк: {len(df_raw)}")
        
        # Кнопка "Обучить" - выполняет все действия автоматически
        if st.button("🚀 Обучить модель (автоматически)", type="primary", help="Автоматически выполнит: загрузку в бэк, предобработку и обучение"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Шаг 1: Загрузка в бэк
                status_text.text("📤 Шаг 1/3: Загрузка файла в backend...")
                progress_bar.progress(10)
                
                files = {"file": (uploaded_file.name, bytes_data, "text/csv")}
                upload_response = requests.post(f"{FASTAPI_URL}/upload", files=files, timeout=30)
                
                if upload_response.status_code != 200:
                    st.error(f"❌ Ошибка загрузки файла: {upload_response.text}")
                    st.stop()
                
                upload_result = upload_response.json()
                st.session_state.uploaded_file_path = upload_result["file_path"]
                progress_bar.progress(30)
                
                # Шаг 2: Предобработка
                status_text.text("⚙️ Шаг 2/3: Предобработка данных...")
                progress_bar.progress(40)
                
                preprocess_payload = {
                    "file_path": st.session_state.uploaded_file_path,
                    "force_weekly": False  # Автоматически определяется
                }
                
                preprocess_response = requests.post(f"{FASTAPI_URL}/preprocess", json=preprocess_payload, timeout=120)
                
                if preprocess_response.status_code != 200:
                    st.error(f"❌ Ошибка предобработки: {preprocess_response.text}")
                    st.stop()
                
                preprocess_result = preprocess_response.json()
                st.session_state.preprocessed_shop_csv = preprocess_result["shop_csv"]
                st.session_state.preprocessed_category_csv = preprocess_result["category_csv"]
                st.session_state.preprocessed_product_csv = preprocess_result.get("product_csv")
                st.session_state.preprocessing_stats = preprocess_result["stats"]
                progress_bar.progress(60)
                
                # Шаг 3: Обучение shop-level модели
                status_text.text("🎯 Шаг 3/3: Обучение shop-level модели...")
                progress_bar.progress(70)
                
                train_payload = {
                    "shop_csv": st.session_state.preprocessed_shop_csv,
                    "model_out": "models/prophet_model.pkl",
                    "include_regressors": False,  # НЕ используем регрессоры в простом режиме
                    "log_transform": True,  # Включено для shop-level
                    "interval_width": 0.95,
                    "holdout_frac": 0.2,
                    "changepoint_prior_scale": 0.05,
                    "seasonality_prior_scale": 10.0,
                    "seasonality_mode": "additive",
                    "auto_tune": True,  # Включен автотюн для автоматического подбора параметров
                    "skip_holdout": True,  # Обучаем на ВСЕХ данных для прогноза на будущее
                    "filter_column": None,
                    "filter_value": None
                }
                
                train_response = requests.post(f"{FASTAPI_URL}/train", json=train_payload, timeout=1800)  # Увеличено до 30 минут для автотюна
                
                if train_response.status_code != 200:
                    st.error(f"❌ Ошибка обучения модели: {train_response.text}")
                    st.stop()
                
                train_result = train_response.json()
                st.session_state.trained_model_path = train_result["model_path"]
                st.session_state.training_metrics = train_result["metrics"]
                st.session_state.selected_aggregation_level = "shop"
                progress_bar.progress(100)
                status_text.text("✅ Обучение завершено!")
                
                st.success("✅ **Модель успешно обучена!** Shop-level модель готова для прогноза.")
                
                # Показываем краткую статистику
                stats = preprocess_result["stats"]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Всего строк", stats.get("n_rows_raw", "N/A"))
                with col2:
                    st.metric("Категорий", stats.get("unique_categories", "N/A"))
                with col3:
                    if train_result.get("metrics", {}).get("mape"):
                        mape_val = train_result["metrics"]["mape"]
                        st.metric("MAPE", f"{mape_val:.2f}%")
                    else:
                        st.metric("MAPE", "N/A")
                
            except Exception as e:
                st.error(f"❌ Ошибка: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                st.stop()
        
        # Кнопка "Получить прогноз" - распределенный прогноз по всем категориям
        if st.session_state.trained_model_path and st.session_state.preprocessed_category_csv:
            st.subheader("🔮 Шаг 2: Получить прогноз по всем категориям")
            
            horizon_auto = st.number_input(
                "Горизонт прогноза (дней)",
                min_value=1,
                max_value=365,
                value=30,  # По умолчанию 30 дней
                step=1,
                key="horizon_auto",
                help="На сколько дней вперед нужен прогноз"
            )
            
            if st.button("📊 Получить прогноз по всем категориям", type="primary", help="Генерирует распределенный прогноз для всех категорий"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Генерируем shop-level прогноз
                    status_text.text("🔮 Генерация shop-level прогноза...")
                    progress_bar.progress(20)
                    
                    shop_predict_payload = {
                        "model_path": st.session_state.trained_model_path,
                        "horizon": int(horizon_auto),
                        "log_transform": True,  # Включен log-transform для генерации прогноза
                        "smooth_transition": True,
                        "future_regressor_strategy": "ffill",
                        "last_known_regressors_csv": st.session_state.preprocessed_shop_csv,
                        "smooth_days": 21,
                        "smooth_alpha": 0.95,
                        "max_change_pct": 0.01
                    }
                    
                    shop_predict_response = requests.post(f"{FASTAPI_URL}/predict", json=shop_predict_payload, timeout=120)
                    
                    if shop_predict_response.status_code != 200:
                        st.error(f"❌ Ошибка генерации shop-level прогноза: {shop_predict_response.text}")
                        st.stop()
                    
                    shop_predict_result = shop_predict_response.json()
                    shop_forecast_csv = shop_predict_result["forecast_csv_path"]
                    progress_bar.progress(50)
                    
                    # Распределяем по категориям
                    status_text.text("📊 Распределение прогноза по категориям...")
                    progress_bar.progress(70)
                    
                    distributed_payload = {
                        "shop_forecast_csv": shop_forecast_csv,
                        "category_csv": st.session_state.preprocessed_category_csv,
                        "horizon_days": int(horizon_auto)
                    }
                    
                    distributed_response = requests.post(f"{FASTAPI_URL}/predict_category_distributed", json=distributed_payload, timeout=120)
                    
                    if distributed_response.status_code != 200:
                        st.error(f"❌ Ошибка распределения прогноза: {distributed_response.text}")
                        st.stop()
                    
                    distributed_result = distributed_response.json()
                    st.session_state.forecast_data = distributed_result["forecast"]
                    st.session_state.forecast_csv_path = distributed_result["forecast_csv_path"]
                    st.session_state.forecast_is_distributed = True
                    st.session_state.use_distributed_forecast = True
                    progress_bar.progress(100)
                    status_text.text("✅ Прогноз готов!")
                    
                    n_cats = distributed_result.get('n_categories', 0)
                    st.success(f"✅ **Прогноз успешно сгенерирован!** Прогноз для {n_cats} категорий на {horizon_auto} дней.")
                    
                    # Автоматически показываем визуализацию ниже
                    st.session_state.show_forecast_viz = True
                    
                except Exception as e:
                    st.error(f"❌ Ошибка: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.stop()
    
    # Показываем визуализацию если прогноз готов
    if st.session_state.get('show_forecast_viz', False) and st.session_state.forecast_data:
        st.markdown("---")
        st.subheader("📈 Визуализация прогноза")
        # Визуализация будет показана ниже в общем блоке
        
# ПРОДВИНУТЫЙ РЕЖИМ: Полный интерфейс
else:
    # File uploader (только для продвинутого режима)
    st.header("📁 Шаг 1: Загрузка данных")
    uploaded_file = st.file_uploader(
        "Загрузите CSV файл с данными о продажах", 
        type=["csv"],
        key="file_uploader_advanced",
        help="CSV файл должен содержать колонки: Sale_Date, Product_ID, Product_Category, Unit_Price, Discount, Quantity_Sold"
    )
    
    if uploaded_file is not None:
        # Display raw data preview
        bytes_data = uploaded_file.getvalue()
        df_raw = pd.read_csv(io.StringIO(bytes_data.decode("utf-8")))
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Предпросмотр исходных данных")
            st.dataframe(df_raw.head(10))
        
        with col2:
            st.subheader("Обнаруженные колонки")
            st.write(f"Колонки: {', '.join(df_raw.columns.tolist())}")
            st.write(f"Всего строк: {len(df_raw)}")
        
        # Upload file to backend
        if st.button("📤 Загрузить в Backend", key="upload_button_advanced", help="Загружает CSV файл в backend API"):
            try:
                files = {"file": (uploaded_file.name, bytes_data, "text/csv")}
                response = requests.post(f"{FASTAPI_URL}/upload", files=files, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.uploaded_file_path = result["file_path"]
                    st.success(f"✅ Файл успешно загружен: {result['file_path']}")
                else:
                    st.error(f"❌ Ошибка загрузки: {response.text}")
            except Exception as e:
                st.error(f"❌ Ошибка при загрузке файла: {str(e)}")
    
    # Preprocess section (только для продвинутого режима)
    st.header("⚙️ Шаг 2: Предобработка данных")
    if st.session_state.uploaded_file_path:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"📄 Файл готов: {st.session_state.uploaded_file_path}")
        with col2:
            force_weekly = st.checkbox("Принудительная недельная агрегация", 
                                      help="Принудительно использовать недельную агрегацию независимо от плотности данных")
        
        if st.button("🔄 Предобработать данные", help="Обрабатывает загруженный CSV, валидирует данные и генерирует агрегаты по магазинам/категориям"):
            # Check API before processing
            try:
                health_check = requests.get(f"{FASTAPI_URL}/health", timeout=3)
                if health_check.status_code != 200:
                    st.error(f"❌ API недоступен (код {health_check.status_code}). Убедитесь, что FastAPI запущен.")
                    st.stop()
            except requests.exceptions.ConnectionError:
                st.error(f"❌ Не удалось подключиться к API на {FASTAPI_URL}")
                st.info("💡 Запустите FastAPI: `python -m app.main` или `uvicorn app.main:app --reload --port 8888`")
                st.stop()
            except Exception as e:
                st.warning(f"⚠️ Не удалось проверить статус API: {str(e)}")
                st.info("💡 Продолжаем попытку предобработки...")
            
            try:
                payload = {
                    "file_path": st.session_state.uploaded_file_path,
                    "force_weekly": force_weekly
                }
                with st.spinner("Предобработка данных... Это может занять некоторое время."):
                    response = requests.post(f"{FASTAPI_URL}/preprocess", json=payload, timeout=120)
                
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.preprocessed_shop_csv = result["shop_csv"]
                    st.session_state.preprocessed_category_csv = result["category_csv"]
                    st.session_state.preprocessed_product_csv = result.get("product_csv")
                    st.session_state.preprocessing_stats = result["stats"]
                    
                    st.success("✅ Данные успешно предобработаны!")
                    
                    # Show stats
                    st.subheader("📊 Статистика предобработки")
                    
                    stats = result["stats"]
                    agg_suggestion = result.get("aggregation_suggestion", {})
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Всего строк (исходных)", stats.get("n_rows_raw", "N/A"))
                        st.metric("Строк после очистки", stats.get("n_rows_clean", "N/A"))
                        st.metric("Уникальных дат", stats.get("n_unique_dates", "N/A"))
                    
                    with col2:
                        st.metric("Начало периода", stats.get("date_min", "N/A")[:10] if stats.get("date_min") else "N/A")
                        st.metric("Конец периода", stats.get("date_max", "N/A")[:10] if stats.get("date_max") else "N/A")
                        st.metric("Дубликатов удалено", stats.get("duplicates_removed", 0))
                    
                    with col3:
                        freq_used = stats.get("freq_used", "D")
                        freq_icon = "📅" if freq_used == "D" else "📆"
                        st.metric("Частота агрегации", f"{freq_icon} {freq_used}")
                        
                        if agg_suggestion:
                            st.info(f"💡 Рекомендация: {agg_suggestion.get('freq', 'D')} - {agg_suggestion.get('reason', '')}")
                    
                    # Show available aggregation levels
                    if stats.get("unique_categories", 0) > 0 or stats.get("unique_products", 0) > 0:
                        st.info(f"📊 Доступные уровни агрегации: Магазин (1), Категории ({stats.get('unique_categories', 0)}), Товары ({stats.get('unique_products', 0)})")
                    
                    if stats.get("warning"):
                        st.warning(f"⚠️ {stats['warning']}")
                    
                    # Show detailed stats
                    with st.expander("📋 Детальная статистика"):
                        st.json(stats)
                elif response.status_code == 404:
                    st.error(f"❌ Файл не найден: {response.text}")
                elif response.status_code == 400:
                    st.error(f"❌ Ошибка валидации данных: {response.text}")
                    try:
                        error_detail = response.json().get("detail", response.text)
                        st.error(f"Детали: {error_detail}")
                    except:
                        st.error(f"Ответ сервера: {response.text}")
                elif response.status_code == 500:
                    st.error(f"❌ Внутренняя ошибка сервера: {response.text}")
                    st.info("💡 Проверьте логи FastAPI для подробностей")
                else:
                    st.error(f"❌ Ошибка предобработки (код {response.status_code}): {response.text}")
            except requests.exceptions.Timeout:
                st.error("❌ Таймаут при предобработке (превышено 120 секунд)")
                st.info("💡 Попробуйте снова или проверьте размер файла")
            except requests.exceptions.ConnectionError:
                st.error(f"❌ Потеряно соединение с API во время предобработки")
                st.info(f"💡 Убедитесь, что FastAPI все еще работает на {FASTAPI_URL}")
            except Exception as e:
                st.error(f"❌ Ошибка при предобработке данных: {str(e)}")
                st.info("💡 Проверьте, что API запущен и доступен")

    # Train section (только для продвинутого режима)
    st.header("🎯 Шаг 3: Обучение модели")
    if st.session_state.preprocessed_shop_csv:
        # Level of aggregation selection
        aggregation_level = st.session_state.selected_aggregation_level
    with st.expander("📊 Выбор уровня агрегации", expanded=True):
        aggregation_level = st.radio(
            "Выберите уровень агрегации для обучения:",
            options=["shop", "category", "product"],
            format_func=lambda x: {
                "shop": "🏪 По всему магазину (агрегат всех продаж)",
                "category": "📁 По категории товаров",
                "product": "📦 По конкретному товару"
            }[x],
            index=0 if st.session_state.selected_aggregation_level == "shop" else 
                  1 if st.session_state.selected_aggregation_level == "category" else 2,
            help="Выберите, на каком уровне делать прогноз: для всего магазина, для категории или для конкретного товара"
        )
        st.session_state.selected_aggregation_level = aggregation_level
    
    # Filter selection based on aggregation level
    selected_csv = st.session_state.preprocessed_shop_csv
    filter_column = None
    filter_value = None
    
    if aggregation_level == "category":
        selected_csv = st.session_state.preprocessed_category_csv
        if not selected_csv:
            st.error("❌ Файл с данными по категориям не найден. Выполните предобработку данных.")
            st.stop()
        
        # Get available categories - use raw CSV if available for real transaction counts
        try:
            # Priority: use raw CSV for real transaction counts
            params = {}
            if st.session_state.uploaded_file_path:
                params["raw_csv"] = st.session_state.uploaded_file_path
            else:
                params["category_csv"] = selected_csv
            
            categories_response = requests.get(
                f"{FASTAPI_URL}/categories",
                params=params,
                timeout=10
            )
            if categories_response.status_code == 200:
                categories_data = categories_response.json()
                categories_list = categories_data.get("categories", [])
                
                if categories_list:
                    # Format options with counts: "Category Name (123 записей)"
                    category_options = []
                    category_dict = {}  # Map display text to actual name
                    for cat_item in categories_list:
                        if isinstance(cat_item, dict):
                            cat_name = cat_item.get("name", "")
                            cat_count = cat_item.get("count", 0)
                            display_text = f"{cat_name} ({cat_count} транзакций)"
                            category_options.append(display_text)
                            category_dict[display_text] = cat_name
                        else:
                            # Fallback for old format (just strings)
                            category_options.append(cat_item)
                            category_dict[cat_item] = cat_item
                    
                    selected_category_display = st.selectbox(
                        "Выберите категорию:",
                        options=category_options,
                        help="Выберите категорию товаров для обучения модели. Количество в скобках показывает число реальных транзакций (продаж) в исходном датасете."
                    )
                    selected_category = category_dict.get(selected_category_display, selected_category_display)
                    filter_column = "category"
                    filter_value = selected_category
                    st.session_state.selected_filter_value = selected_category
                    
                    # Extract count for info message
                    for cat_item in categories_list:
                        if isinstance(cat_item, dict) and cat_item.get("name") == selected_category:
                            cat_count = cat_item.get("count", 0)
                            st.info(f"📁 Будет обучена модель для категории: **{selected_category}** ({cat_count} транзакций в датасете)")
                            break
                    else:
                        st.info(f"📁 Будет обучена модель для категории: **{selected_category}**")
                else:
                    st.warning("⚠️ Категории не найдены в данных")
            else:
                st.warning("⚠️ Не удалось загрузить список категорий")
        except Exception as e:
            st.warning(f"⚠️ Ошибка при загрузке категорий: {str(e)}")
    
    elif aggregation_level == "product":
        selected_csv = st.session_state.preprocessed_product_csv
        if not selected_csv:
            st.error("❌ Файл с данными по товарам не найден. Выполните предобработку данных.")
            st.stop()
        
        # Get available products - use raw CSV if available for real transaction counts
        try:
            # Priority: use raw CSV for real transaction counts
            params = {"limit": 500}
            if st.session_state.uploaded_file_path:
                params["raw_csv"] = st.session_state.uploaded_file_path
            else:
                params["product_csv"] = selected_csv
            
            products_response = requests.get(
                f"{FASTAPI_URL}/products",
                params=params,
                timeout=30  # Increased timeout
            )
            if products_response.status_code == 200:
                products_data = products_response.json()
                products_list = products_data.get("products", [])
                
                if products_list:
                    # Format options with counts: "Product ID (123 записей)"
                    product_options = []
                    product_dict = {}  # Map display text to actual product_id
                    for prod_item in products_list:
                        if isinstance(prod_item, dict):
                            prod_id = str(prod_item.get("name", ""))
                            prod_count = prod_item.get("count", 0)
                            display_text = f"{prod_id} ({prod_count} транзакций)"
                            product_options.append(display_text)
                            product_dict[display_text] = prod_id
                        else:
                            # Fallback for old format (just strings)
                            product_options.append(str(prod_item))
                            product_dict[str(prod_item)] = str(prod_item)
                    
                    # Show products (API already limits to 500 top products by count)
                    total_available = products_data.get("total_available", len(product_options))
                    is_limited = products_data.get("limited", False)
                    
                    selected_product_display = st.selectbox(
                        "Выберите товар (product_id):",
                        options=product_options,
                        help="Выберите товар для обучения модели. Количество в скобках показывает число реальных транзакций (продаж) в исходном датасете. Товары отсортированы по количеству транзакций (больше = лучше для прогноза)."
                    )
                    selected_product = product_dict.get(selected_product_display, selected_product_display)
                    filter_column = "product_id"
                    filter_value = str(selected_product)
                    st.session_state.selected_filter_value = selected_product
                    
                    # Extract count for info message
                    for prod_item in products_list:
                        if isinstance(prod_item, dict) and str(prod_item.get("name", "")) == str(selected_product):
                            prod_count = prod_item.get("count", 0)
                            st.info(f"📦 Будет обучена модель для товара: **{selected_product}** ({prod_count} транзакций в датасете)")
                            if prod_count < 30:
                                st.warning(f"⚠️ **Мало данных!** Товар имеет только {prod_count} транзакций. Прогноз может быть неточным. Рекомендуется минимум 30+ транзакций для надежного прогноза.")
                            break
                    else:
                        st.info(f"📦 Будет обучена модель для товара: **{selected_product}**")
                    
                    if is_limited:
                        st.info(f"💡 Показано топ {len(product_options)} товаров (из {total_available} всего). Отсортировано по количеству записей. Для просмотра остальных используйте поиск по ID.")
                else:
                    st.warning("⚠️ Товары не найдены в данных")
            else:
                st.warning("⚠️ Не удалось загрузить список товаров")
        except Exception as e:
            st.warning(f"⚠️ Ошибка при загрузке товаров: {str(e)}")
    else:
        # Shop-level обучение
        st.success("✅ **SHOP-LEVEL ОБУЧЕНИЕ:** Будет обучена модель для всего магазина (агрегат всех продаж)")
        st.info(f"📊 Используются данные магазинов: `{selected_csv}`")
        st.info("💡 **Рекомендации для shop-level:**\n"
                "- ✅ Log-transform: ВКЛЮЧЕНО (рекомендуется)\n"
                "- ✅ Interval width: 0.95\n"
                "- ✅ Holdout fraction: 0.20\n"
                "- ✅ Seasonality mode: additive\n"
                "- ✅ Changepoint flexibility: 0.01-0.05\n"
                "- ✅ Seasonality strength: 10.0")
    
    # Show category/product specific recommendations
    if aggregation_level in ["category", "product"]:
        with st.expander("💡 Специальные рекомендации для категорий/товаров", expanded=True):
            st.write("""
            **⚠️ Важно для категорий и товаров:**
            
            Прогнозы по категориям/товарам часто получаются слишком циклическими из-за:
            - Меньшего объема данных по сравнению с shop-level
            - Большей волатильности
            - Monthly seasonality может создавать регулярные циклы
            
            **Рекомендуемые настройки:**
            
            1. ✅ **Гибкость точек изменения**: **0.05-0.15** (вместо 0.01)
               - Позволяет модели адаптироваться к изменениям
            
            2. ✅ **Сила сезонности**: **5.0-7.5** (вместо 10.0)
               - Меньше переобучения на циклические паттерны
            
            3. ✅ **Режим сезонности**: 
               - **Additive** - если log-transform включен
               - **Multiplicative** - если log-transform выключен
            
            4. ⚠️ **Log-transform**: 
               - Для weekly агрегации категорий может быть проблематично
               - Если получаете ошибку "less than 2 rows" - СНИМИТЕ галочку
            
            5. ✅ **Автоматический подбор (auto_tune)**: Включите для лучших результатов
            
            **Что исправлено автоматически:**
            - Monthly seasonality использует меньший fourier_order (3 вместо 5) для категорий
            """)
    
    # Show recommended settings for best results
    with st.expander("💡 Рекомендуемые настройки для лучшего качества", expanded=False):
        st.write("""
        **Оптимальные параметры (показали MAPE ~39% в тестах):**
        
        ✅ **Model Configuration:**
        - Use regressors: ❌ ВЫКЛЮЧЕНО
        - Log-transform: ❌ ВЫКЛЮЧЕНО
        - Interval width: 0.95
        - Holdout fraction: 0.20
        
        ✅ **Advanced Hyperparameters:**
        - Seasonality mode: **additive**
        - Changepoint flexibility: **0.01** (консервативный)
        - Seasonality strength: **10.0** (стандартный)
        
        ⚠️ **Важно:** После переобработки данных (Step 1) эти параметры дают лучшее качество!
        """)
    
    # Analyze data and provide recommendations
    try:
        df_preview = pd.read_csv(st.session_state.preprocessed_shop_csv)
        if 'y' in df_preview.columns:
            mean_sales = df_preview['y'].mean()
            std_sales = df_preview['y'].std()
            cv = std_sales / mean_sales if mean_sales > 0 else 0  # Coefficient of variation
            min_sales = df_preview['y'].min()
            max_sales = df_preview['y'].max()
            
            # Generate recommendations
            recommendations = []
            if cv > 1.0:  # High volatility
                recommendations.append("🔴 Высокая волатильность данных (CV > 1.0) - рекомендуется включить **log-transform**")
                recommendations.append("🔴 Рекомендуется попробовать **multiplicative** seasonality mode")
            elif cv > 0.5:
                recommendations.append("🟡 Умеренная волатильность - можно попробовать **log-transform**")
            
            if min_sales >= 0 and max_sales / mean_sales > 10:
                recommendations.append("🟡 Большой разброс значений - рекомендуется **log-transform**")
            
            if recommendations:
                with st.expander("💡 Автоматические рекомендации на основе данных", expanded=True):
                    st.write("**Статистика данных:**")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Среднее", f"{mean_sales:.2f}")
                    with col2:
                        st.metric("Стд. откл.", f"{std_sales:.2f}")
                    with col3:
                        st.metric("CV", f"{cv:.2f}")
                    with col4:
                        st.metric("Min/Max", f"{min_sales:.0f} / {max_sales:.0f}")
                    
                    st.write("**Рекомендации:**")
                    for rec in recommendations:
                        st.write(f"- {rec}")
    except Exception as e:
        pass  # Skip recommendations if data can't be loaded
    
    # Model configuration
    with st.expander("⚙️ Конфигурация модели", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            include_regressors = st.checkbox(
                "Использовать регрессоры (цена/скидка)",
                value=False,
                help="Включить среднюю цену и среднюю скидку как дополнительные факторы в модель Prophet"
            )
            
            log_transform = st.checkbox(
                "Применить log-transform к целевому показателю",
                value=False,
                help="⚠️ РЕКОМЕНДУЕТСЯ для данных с высокой волатильностью! Применяет преобразование log1p к переменной y (полезно для асимметричных данных). "
                     "⚠️ Для категорий с weekly агрегацией может вызвать ошибку 'less than 2 rows' - в таком случае снимите галочку!"
            )
            
            # Предупреждение для категорий с weekly агрегацией
            if aggregation_level in ["category", "product"] and log_transform:
                st.info("💡 **Для категорий/товаров:** Если получите ошибку 'less than 2 rows', попробуйте снять галочку log-transform.")
        
        with col2:
            interval_width = st.slider(
                "Ширина доверительного интервала",
                min_value=0.5,
                max_value=0.99,
                value=0.95,
                step=0.01,
                help="Ширина доверительного интервала для прогнозов (0.95 = 95% уверенности). Показывает диапазон, в который с указанной вероятностью попадут фактические значения."
            )
            
            holdout_frac = st.slider(
                "Доля данных для тестирования",
                min_value=0.05,
                max_value=0.5,
                value=0.2,
                step=0.05,
                help="Доля данных для тестового набора (например, 0.2 = 20% данных пойдут на тест). Эти данные не используются для обучения и служат для оценки качества модели."
            )
        
        # Skip holdout option (для прогноза на будущее)
        skip_holdout = st.checkbox(
            "🚀 Обучить на ВСЕХ данных (пропустить holdout) - для прогноза на будущее",
            value=False if aggregation_level == "shop" else True,  # Для категорий/товаров по умолчанию True
            help="Если включено: модель обучается на ВСЕХ данных без разделения на train/test. "
                 "Используйте для продакшн-прогнозов на реальное будущее (не на тестовый период). "
                 "⚠️ Метрики качества (MAPE, MAE, RMSE) не будут вычислены, так как нет тестового набора. "
                 "✅ РЕКОМЕНДУЕТСЯ для категорий/товаров с малым количеством данных!"
        )
        
        if skip_holdout:
            st.info("💡 **Режим прогноза на будущее:** Модель обучится на всех данных. "
                   "После обучения используйте раздел 'Generate Forecast' для прогноза на реальные будущие даты. "
                   "Holdout fraction будет проигнорирован.")
        
        # Специальное предупреждение для категорий/товаров с weekly агрегацией
        if aggregation_level in ["category", "product"]:
            if not skip_holdout and holdout_frac >= 0.2:
                st.warning("⚠️ **Для категорий/товаров рекомендуется:** "
                          "Включите 'Обучить на ВСЕХ данных' или уменьшите 'Доля данных для тестирования' до 0.1. "
                          "Иначе может остаться слишком мало данных для обучения, особенно при weekly агрегации!")
        
        # Advanced hyperparameters
        with st.expander("🔧 Продвинутые гиперпараметры (для улучшения качества)", expanded=False):
            # Warning about log_transform + multiplicative combination
            if log_transform:
                st.info("💡 **Совет**: При включенном log-transform обычно лучше использовать **additive** seasonality. Multiplicative + log-transform могут конфликтовать и давать слишком широкие доверительные интервалы.")
            
            with st.expander("📚 Пояснения к гиперпараметрам", expanded=False):
                st.write("""
                **Режим сезонности (Seasonality mode):**
                - **Additive**: Сезонность добавляется к тренду. Подходит для данных с постоянной амплитудой сезонных колебаний.
                - **Multiplicative**: Сезонность умножается на тренд. Подходит для данных, где сезонные колебания растут вместе с трендом. 
                  ⚠️ Не рекомендуется использовать вместе с log-transform!
                
                **Гибкость точек изменения (Changepoint flexibility):**
                - Контролирует, насколько гибко модель обнаруживает изменения тренда
                - Низкие значения (0.001-0.01): Консервативный подход, меньше точек изменения, более плавный тренд
                - Высокие значения (0.1-0.5): Больше гибкости, больше точек изменения, риск переобучения
                - Рекомендуется: 0.005-0.01 для стабильных данных, 0.01-0.05 для волатильных
                
                **Сила сезонности (Seasonality strength):**
                - Контролирует, насколько сильно модель учитывает сезонные паттерны
                - Низкие значения (1-5): Слабый эффект сезонности
                - Стандартные значения (10-15): Умеренный эффект
                - Высокие значения (20-50): Сильный эффект сезонности
                """)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                seasonality_mode = st.selectbox(
                    "Режим сезонности",
                    options=["additive", "multiplicative"],
                    index=0 if log_transform else 0,  # Suggest additive if log_transform is on
                    help="Additive: сезонность добавляется к тренду. Multiplicative: сезонность умножается на тренд (лучше для высокой волатильности, но БЕЗ log-transform)"
                )
                
                # Предупреждение о проблемных комбинациях
                if log_transform and seasonality_mode == 'multiplicative':
                    st.warning("⚠️ **ВНИМАНИЕ**: Log-transform + Multiplicative seasonality могут конфликтовать! "
                              "Рекомендуется использовать **Additive** режим при включенном log-transform.")
            
            with col2:
                changepoint_prior_scale = st.slider(
                    "Гибкость точек изменения тренда",
                    min_value=0.001,
                    max_value=0.5,
                    value=0.01,
                    step=0.001,
                    format="%.3f",
                    help="Гибкость детекции точек изменения тренда (выше = больше гибкости, риск переобучения). Рекомендуется: 0.005-0.01 для стабильных данных"
                )
            
            with col3:
                seasonality_prior_scale = st.slider(
                    "Сила сезонности",
                    min_value=0.01,
                    max_value=100.0,
                    value=10.0,
                    step=1.0,
                    help="Сила сезонных компонентов (выше = сильнее сезонность)"
                )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        model_out_path = st.text_input(
            "Путь сохранения модели",
            value="models/prophet_model.pkl",
            help="Путь, по которому будет сохранена обученная модель"
        )
    
    # Auto-tune option
    auto_tune = st.checkbox(
        "🔍 Автоматический подбор параметров (Grid Search)",
        value=False,
        help="Автоматически находит лучшую конфигурацию модели через grid search (Prophet варианты, LSTM, Hybrid). Это займет больше времени, но даст лучшие результаты."
    )
    
    if auto_tune:
        st.info("💡 При включенном auto-tune будут протестированы различные конфигурации Prophet, LSTM и гибридные модели. Результаты сохранятся в analysis/model_comparison.csv")
    
    if st.button("🚀 Обучить модель", help="Обучает модель Prophet с выбранной конфигурацией"):
        try:
            # Determine model output path based on aggregation level
            base_model_path = model_out_path
            if aggregation_level == "category" and filter_value:
                # Create category-specific model path
                category_name_safe = str(filter_value).replace("/", "_").replace("\\", "_")
                base_model_path = f"models/model_category_{category_name_safe}.pkl"
            elif aggregation_level == "product" and filter_value:
                # Create product-specific model path
                product_id_safe = str(filter_value).replace("/", "_").replace("\\", "_")
                base_model_path = f"models/model_product_{product_id_safe}.pkl"
            
            payload = {
                "shop_csv": selected_csv,  # Use selected CSV (shop/category/product)
                "model_out": base_model_path,
                "include_regressors": include_regressors,
                "log_transform": log_transform,
                "interval_width": interval_width,
                "holdout_frac": holdout_frac,
                "changepoint_prior_scale": changepoint_prior_scale,
                "seasonality_prior_scale": seasonality_prior_scale,
                "seasonality_mode": seasonality_mode,
                "auto_tune": auto_tune,
                "skip_holdout": skip_holdout,
                "filter_column": filter_column,  # Add filter column
                "filter_value": filter_value  # Add filter value
            }
            
            spinner_text = "Обучение модели с автоматическим подбором параметров (это может занять несколько минут)..." if auto_tune else "Обучение модели... Это может занять некоторое время."
            timeout_val = 1800 if auto_tune else 300  # 30 minutes for auto-tune, 5 minutes for regular
            
            with st.spinner(spinner_text):
                response = requests.post(f"{FASTAPI_URL}/train", json=payload, timeout=timeout_val)
            
            if response.status_code == 200:
                result = response.json()
                st.session_state.trained_model_path = result["model_path"]
                st.session_state.training_metrics = result["metrics"]
                
                if skip_holdout:
                    st.success("✅ Модель успешно обучена на ВСЕХ данных! Готова для продакшн-прогнозов.")
                    st.info("💡 **Режим прогноза на будущее:** Метрики не вычислены (нет тестового набора). "
                            "Используйте раздел 'Генерация прогноза' для прогноза на реальные будущие даты.")
                else:
                    st.success("✅ Модель успешно обучена!")
                
                # Display metrics
                st.subheader("📈 Метрики обучения")
                
                # Добавляем expander с пояснениями к метрикам
                with st.expander("📚 Пояснения к метрикам качества", expanded=False):
                    st.write("""
                    **MAPE (Mean Absolute Percentage Error) - Средняя абсолютная процентная ошибка**
                    
                    Показывает среднее процентное отклонение прогноза от фактических значений.
                    
                    Интерпретация:
                    - **< 15%** - ✅ Отличное качество, модель готова к продакшену
                    - **15-20%** - ✅ Хорошее качество
                    - **20-30%** - 🟡 Удовлетворительное качество, можно улучшить
                    - **> 30%** - ⚠️ Низкое качество, требуется настройка параметров
                    - **> 50%** - 🚨 Критически плохое качество, модель не готова к использованию
                    
                    Формула: MAPE = (1/n) × Σ|y_actual - y_predicted| / |y_actual| × 100%
                    
                    ---
                    
                    **MAE (Mean Absolute Error) - Средняя абсолютная ошибка**
                    
                    Показывает среднюю величину ошибки в единицах измерения (например, в единицах продаж).
                    Не учитывает направление ошибки (переоценка или недооценка).
                    
                    Формула: MAE = (1/n) × Σ|y_actual - y_predicted|
                    
                    ---
                    
                    **RMSE (Root Mean Square Error) - Корень среднеквадратичной ошибки**
                    
                    Учитывает большие ошибки сильнее, чем MAE. Полезно для выявления выбросов и сильных отклонений.
                    Всегда >= MAE.
                    
                    Формула: RMSE = √[(1/n) × Σ(y_actual - y_predicted)²]
                    
                    ---
                    
                    **CI Coverage (Coverage Rate) - Покрытие доверительного интервала**
                    
                    Показывает процент фактических значений, которые попали в предсказанный доверительный интервал.
                    Хорошее покрытие: >= 85% (для 95% интервала).
                    """)
                
                metrics = result["metrics"]
                
                # Проверяем skip_holdout
                if skip_holdout or metrics.get('mape') is None:
                    st.info("ℹ️ Метрики не вычислены: модель обучена на всех данных (skip_holdout=True). "
                           "Готова для прогноза на реальное будущее!")
                else:
                    col1, col2, col3 = st.columns(3)
                    mape_val = metrics.get('mape')
                    mae_val = metrics.get('mae')
                    rmse_val = metrics.get('rmse')
                    
                    # Determine metric status
                    if isinstance(mape_val, (int, float)):
                        if mape_val > 50:
                            mape_delta = "❌ Критично плохо"
                            mape_color = "off"
                        elif mape_val > 30:
                            mape_delta = "⚠️ Плохо"
                            mape_color = "off"
                        elif mape_val > 20:
                            mape_delta = "🟡 Удовлетворительно"
                            mape_color = "normal"
                        elif mape_val > 15:
                            mape_delta = "✅ Хорошо"
                            mape_color = "normal"
                        else:
                            mape_delta = "✅ Отлично"
                            mape_color = "normal"
                    else:
                        mape_delta = None
                        mape_color = "normal"
                    
                    with col1:
                        st.metric("MAE (Средняя абсолютная ошибка)", f"{mae_val:.2f}" if mae_val is not None else "N/A", 
                                 help="Средняя величина ошибки в единицах измерения")
                    with col2:
                        st.metric("RMSE (Корень среднеквадратичной ошибки)", f"{rmse_val:.2f}" if rmse_val is not None else "N/A",
                                 help="Учитывает большие ошибки сильнее, чем MAE")
                    with col3:
                        if mape_val is not None:
                            st.metric("MAPE (Средняя абсолютная процентная ошибка)", f"{mape_val:.2f}%", 
                                     delta=mape_delta if isinstance(mape_val, (int, float)) else None,
                                     help="Среднее процентное отклонение прогноза от фактических значений")
                        else:
                            st.metric("MAPE", "N/A", delta="Режим продакшена")
                    
                    # Show quality warnings and recommendations (только если метрики доступны)
                    if isinstance(mape_val, (int, float)):
                        if mape_val > 50:
                            st.error(f"🚨 КРИТИЧЕСКОЕ КАЧЕСТВО: MAPE = {mape_val:.2f}% слишком высокий! Модель не готова к использованию.")
                            
                            # Check current configuration
                            current_log = metrics.get('log_transform', False)
                            current_mode = metrics.get('seasonality_mode', 'additive')
                            
                            with st.expander("💡 Рекомендации по улучшению"):
                                if current_log and current_mode == 'multiplicative':
                                    st.warning("""
                                    ⚠️ **Обнаружен потенциальный конфликт**: log-transform + multiplicative seasonality
                                    
                                    **Попробуйте один из вариантов:**
                                    
                                    **Вариант А (рекомендуется):**
                                    - ✅ Log-transform: ВКЛЮЧЕНО
                                    - ✅ Seasonality mode: **ADDITIVE** (вместо multiplicative)
                                    - ✅ Seasonality strength: 15-20
                                    - ✅ Changepoint flexibility: 0.10-0.15
                                    
                                    **Вариант Б:**
                                    - ❌ Log-transform: ВЫКЛЮЧЕНО  
                                    - ✅ Seasonality mode: **MULTIPLICATIVE**
                                    - ✅ Seasonality strength: 20-25
                                    - ✅ Changepoint flexibility: 0.15-0.20
                                    """)
                                else:
                                    st.write("""
                                    **Немедленные действия:**
                                    1. ✅ Если log-transform ВЫКЛЮЧЕН - включите его И используйте **additive** seasonality
                                    2. ✅ Если log-transform ВКЛЮЧЕН - попробуйте **additive** вместо multiplicative
                                    3. ✅ Увеличьте **seasonality_prior_scale** до 20-25
                                    4. ✅ Увеличьте **changepoint_prior_scale** до 0.15-0.20 (больше гибкости)
                                    5. ✅ Попробуйте включить **regressors** (price/discount)
                                    6. ⚠️ Проверьте данные на выбросы и аномалии
                                    
                                    **Целевые значения:**
                                    - MAPE < 15-20% для продакшена
                                    - Текущее значение слишком высоко для практического использования
                                    """)
                        elif mape_val > 30:
                            st.warning(f"⚠️ Качество модели ниже среднего: MAPE = {mape_val:.2f}%. Рекомендуется улучшить параметры.")
                            
                            current_log = metrics.get('log_transform', False)
                            current_mode = metrics.get('seasonality_mode', 'additive')
                            
                            with st.expander("💡 Рекомендации по улучшению"):
                                if current_log and current_mode == 'multiplicative':
                                    st.write("""
                                    ⚠️ **Совет**: Log-transform + multiplicative могут конфликтовать. Попробуйте:
                                    1. Оставить log-transform, изменить на **additive** seasonality
                                    2. Или выключить log-transform, использовать **multiplicative**
                                    3. Увеличить **changepoint_prior_scale** до 0.12-0.15
                                    """)
                                else:
                                    st.write("""
                                    **Рекомендуемые действия:**
                                    1. Если log-transform выключен - включите его (с additive)
                                    2. Настройте **changepoint_prior_scale** (0.10-0.15) и **seasonality_prior_scale** (20-25)
                                    3. Проведите кросс-валидацию для оценки стабильности
                                    """)
                        elif mape_val > 20:
                            st.info(f"ℹ️ Качество модели удовлетворительное: MAPE = {mape_val:.2f}%. Можно попробовать улучшить для лучших результатов.")
                        elif mape_val <= 15:
                            st.success(f"✅ Отличное качество модели! MAPE = {mape_val:.2f}% - модель готова к продакшену.")
                
                # Show training info
                with st.expander("📊 Детали обучения"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Период обучения:**")
                        st.write(f"- Начало: {result['train_range']['start'][:10]}")
                        st.write(f"- Конец: {result['train_range']['end'][:10]}")
                        st.write(f"- Образцов: {result['n_train']}")
                    
                    with col2:
                        if skip_holdout or result.get('test_range', {}).get('start') is None:
                            st.write("**⚠️ Режим продакшена:**")
                            st.write("- Тестовый период: N/A (skip_holdout=True)")
                            st.write("- Образцов в тесте: 0")
                            st.info("💡 Модель обучена на всех данных. Готова для прогноза на реальное будущее!")
                        else:
                            st.write("**Тестовый период:**")
                            test_start = result.get('test_range', {}).get('start', 'N/A')
                            test_end = result.get('test_range', {}).get('end', 'N/A')
                            if test_start and test_start != 'N/A':
                                st.write(f"- Начало: {test_start[:10] if isinstance(test_start, str) else test_start}")
                            if test_end and test_end != 'N/A':
                                st.write(f"- Конец: {test_end[:10] if isinstance(test_end, str) else test_end}")
                            st.write(f"- Образцов: {result.get('n_test', 0)}")
                    
                    st.write("**Конфигурация:**")
                    st.write(f"- Log transform: {metrics.get('log_transform', False)}")
                    st.write(f"- Ширина интервала: {metrics.get('interval_width', 0.95)}")
                    st.write(f"- Режим сезонности: {metrics.get('seasonality_mode', 'additive')}")
                    st.write(f"- Гибкость точек изменения: {metrics.get('changepoint_prior_scale', 0.05)}")
                    st.write(f"- Сила сезонности: {metrics.get('seasonality_prior_scale', 10.0)}")
                    st.write(f"- Использована кросс-валидация: {metrics.get('used_cross_validation', False)}")
                    st.write(f"- Использован auto-tune: {metrics.get('auto_tune', False)}")
                
                # Show auto-tune results if available (проверяем skip_holdout через metrics)
                if response.status_code == 200 and metrics.get('auto_tune', False):
                    try:
                        import os
                        comparison_csv = "analysis/model_comparison.csv"
                        if os.path.exists(comparison_csv):
                            df_comparison = pd.read_csv(comparison_csv)
                            st.subheader("📊 Сравнение моделей (Результаты auto-tune)")
                            
                            # Сортировка по MAPE
                            df_comparison_sorted = df_comparison.sort_values('mape')
                            st.dataframe(df_comparison_sorted, use_container_width=True)
                            
                            # Plot comparison с цветовой индикацией лучшей модели
                            fig_comparison = go.Figure()
                            
                            # Цвета: зеленый для лучшей, синий для остальных
                            colors = ['green' if i == 0 else 'lightblue' for i in range(len(df_comparison_sorted))]
                            
                            fig_comparison.add_trace(go.Bar(
                                x=df_comparison_sorted['model_name'],
                                y=df_comparison_sorted['mape'],
                                name='MAPE (%)',
                                marker_color=colors,
                                text=[f"{m:.1f}%" for m in df_comparison_sorted['mape']],
                                textposition='outside',
                                hovertemplate='<b>%{x}</b><br>MAPE: %{y:.2f}%<br>Coverage: %{customdata:.1f}%<extra></extra>',
                                customdata=df_comparison_sorted['coverage'] * 100
                            ))
                            fig_comparison.update_layout(
                                title="Сравнение MAPE моделей (Зеленый = Лучшая модель)",
                                xaxis_title="Модель",
                                yaxis_title="MAPE (%)",
                                height=500,
                                showlegend=False
                            )
                            st.plotly_chart(fig_comparison, use_container_width=True)
                            
                            # Сохраняем сравнение в session state
                            st.session_state.model_comparison = df_comparison_sorted
                            
                            # Информация о лучшей модели
                            best_model_name = df_comparison_sorted.iloc[0]['model_name']
                            best_mape = df_comparison_sorted.iloc[0]['mape']
                            best_coverage = df_comparison_sorted.iloc[0]['coverage'] * 100
                            
                            st.success(f"🏆 Лучшая модель: **{best_model_name}** (MAPE: {best_mape:.2f}%, Coverage: {best_coverage:.1f}%)")
                            st.info(f"💡 Текущий прогноз использует модель из пути: {model_out_path}. "
                                   f"Для использования другой модели из списка переобучите модель или выберите модель вручную.")
                    except Exception as e:
                        st.warning(f"Не удалось загрузить результаты сравнения auto-tune: {str(e)}")
            else:
                st.error(f"❌ Ошибка обучения: {response.text}")
        except Exception as e:
            st.error(f"❌ Ошибка при обучении модели: {str(e)}")
    
    # Diagnostics section
    if st.session_state.trained_model_path and st.session_state.preprocessed_shop_csv:
        st.subheader("🔍 Диагностика модели")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info("Диагностика модели поможет выявить систематические проблемы: переоценка тренда, низкое покрытие CI, смещение минимумов и др.")
        with col2:
            if st.button("🔍 Запустить диагностику", help="Запускает полную диагностику модели"):
                try:
                    # Получаем include_regressors из метрик, если доступно
                    include_regressors_diag = False
                    if st.session_state.training_metrics:
                        # Проверяем, использовались ли регрессоры (можно проверить через наличие avg_price в данных)
                        try:
                            df_check = pd.read_csv(st.session_state.preprocessed_shop_csv)
                            include_regressors_diag = 'avg_price' in df_check.columns
                        except:
                            pass
                    
                    payload = {
                        "shop_csv": st.session_state.preprocessed_shop_csv,
                        "model_path": st.session_state.trained_model_path,
                        "include_regressors": include_regressors_diag
                    }
                    
                    with st.spinner("Выполняется диагностика..."):
                        response = requests.post(f"{FASTAPI_URL}/diagnose", json=payload, timeout=120)
                    
                    if response.status_code == 200:
                        diagnostics = response.json()
                        st.session_state.diagnostics = diagnostics
                        
                        st.success("✅ Диагностика завершена!")
                        
                        # Display diagnostics
                        st.subheader("📊 Результаты диагностики")
                        
                        metrics_diag = diagnostics.get('metrics', {})
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("MAPE", f"{metrics_diag.get('mape', 0):.2f}%", 
                                     help="Средняя абсолютная процентная ошибка")
                        with col2:
                            st.metric("Систематическое смещение", f"{metrics_diag.get('systematic_bias', 0):.2f}",
                                     help="Систематическая ошибка модели (положительное = переоценка, отрицательное = недооценка)")
                        with col3:
                            coverage = diagnostics.get('coverage', {}).get('coverage_rate', 0) * 100
                            st.metric("Покрытие CI", f"{coverage:.1f}%",
                                     help="Процент фактических значений, попавших в доверительный интервал")
                        
                        # Trend bias
                        trend_bias = diagnostics.get('trend_bias', {})
                        if trend_bias.get('has_bias', False):
                            st.warning(f"⚠️ Обнаружено систематическое смещение тренда: {trend_bias.get('trend_bias_pct', 0):.1f}%")
                        else:
                            st.success("✅ Смещения тренда не обнаружено")
                        
                        # Coverage warning
                        if coverage < 85:
                            st.warning(f"⚠️ Покрытие CI слишком низкое ({coverage:.1f}%). Желательно >= 85%")
                        
                        # Residuals analysis
                        residuals = diagnostics.get('residuals_analysis', {})
                        with st.expander("📈 Анализ остатков"):
                            st.write(f"Средний остаток: {residuals.get('mean', 0):.2f}")
                            st.write(f"Стд. откл. остатка: {residuals.get('std', 0):.2f}")
                            st.write(f"P-value теста на нормальность: {residuals.get('normality_test_pvalue', 0):.4f}")
                            if residuals.get('has_trend', False):
                                st.warning(f"⚠️ Обнаружен тренд в остатках: slope={residuals.get('trend_slope', 0):.6f}")
                        
                        # Multicollinearity
                        multicollinearity = diagnostics.get('multicollinearity', {})
                        if multicollinearity.get('has_multicollinearity', False):
                            st.error("🚨 Обнаружена мультиколлинеарность регрессоров!")
                            st.write(f"Макс. корреляция: {multicollinearity.get('max_correlation', 0):.2f}")
                            st.write(f"VIF scores: {multicollinearity.get('vif_scores', {})}")
                        
                        # Minima shift
                        minima_shift = diagnostics.get('minima_shift', {})
                        mean_shift = minima_shift.get('mean_shift_days', 0)
                        if abs(mean_shift) > 3:
                            st.warning(f"⚠️ Локальные минимумы сдвинуты на {mean_shift:.1f} дней")
                        
                    else:
                        st.error(f"❌ Ошибка диагностики: {response.text}")
                except Exception as e:
                    st.error(f"❌ Ошибка при выполнении диагностики: {str(e)}")

# Evaluate section (только для продвинутого режима)
if ui_mode == "advanced":
    st.header("📊 Шаг 4: Оценка модели (Кросс-валидация)")
    if st.session_state.preprocessed_shop_csv:
        with st.expander("🔍 Конфигурация кросс-валидации", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                initial_days = st.number_input(
                    "Начальный период обучения (дней)",
                    min_value=30,
                    value=180,
                    step=30,
                    help="Количество дней для начального периода обучения в скользящей кросс-валидации"
                )
            
            with col2:
                horizon_days = st.number_input(
                    "Горизонт прогноза (дней)",
                    min_value=1,
                    value=30,
                    step=5,
                    help="Количество дней для прогноза в каждом шаге кросс-валидации"
                )
            
            with col3:
                period_days = st.number_input(
                    "Период сдвига окна (дней)",
                    min_value=1,
                    value=30,
                    step=5,
                    help="Количество дней, на которое сдвигается окно между шагами кросс-валидации"
                )
            
            cv_include_regressors = st.checkbox(
                "Использовать регрессоры для CV",
                value=False,
                help="Включить регрессоры в кросс-валидацию (должно совпадать с конфигурацией обучения)"
            )
            
            cv_log_transform = st.checkbox(
                "Применить log-transform для CV",
                value=False,
                help="Применить log-transform в кросс-валидации (должно совпадать с конфигурацией обучения)"
            )
            
            if st.button("📈 Запустить кросс-валидацию", help="Выполняет скользящую кросс-валидацию для оценки производительности модели"):
                try:
                    payload = {
                        "shop_csv": st.session_state.preprocessed_shop_csv,
                        "initial_days": initial_days,
                        "horizon_days": horizon_days,
                        "period_days": period_days,
                        "include_regressors": cv_include_regressors,
                        "log_transform": cv_log_transform
                    }
                    
                    with st.spinner("Выполняется кросс-валидация... Это может занять несколько минут."):
                        response = requests.post(f"{FASTAPI_URL}/evaluate", json=payload, timeout=600)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.cv_results = result
                        
                        st.success("✅ Кросс-валидация завершена!")
                        
                        # Display aggregate metrics
                        st.subheader("📊 Результаты кросс-валидации")
                        
                        metrics = result["metrics"]
                        summary = result["summary"]
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("MAE", f"{summary['mae_mean']:.2f}", f"±{summary['mae_std']:.2f}")
                        with col2:
                            st.metric("RMSE", f"{summary['rmse_mean']:.2f}", f"±{summary['rmse_std']:.2f}")
                        with col3:
                            st.metric("MAPE", f"{summary['mape_mean']:.2f}%", f"±{summary['mape_std']:.2f}%")
                        
                        st.info(f"📊 Количество шагов CV: {result['n_cv_steps']}")
                        st.info(f"💾 Прогнозы сохранены в: {result['cv_predictions_csv']}")
                        
                        # Plot CV results
                        try:
                            df_cv = pd.read_csv(result['cv_predictions_csv'])
                            df_cv['ds'] = pd.to_datetime(df_cv['ds'])
                            df_cv = df_cv.sort_values('ds')
                            
                            fig = go.Figure()
                            
                            # Plot actual
                            fig.add_trace(go.Scatter(
                                x=df_cv['ds'],
                                y=df_cv['actual'],
                                mode='lines+markers',
                                name='Фактические продажи',
                                line=dict(color='blue', width=2),
                                marker=dict(size=4)
                            ))
                            
                            # Plot predictions (grouped by CV step)
                            if 'cv_step' in df_cv.columns:
                                for step in sorted(df_cv['cv_step'].unique()):
                                    step_data = df_cv[df_cv['cv_step'] == step]
                                    fig.add_trace(go.Scatter(
                                        x=step_data['ds'],
                                        y=step_data['predicted'],
                                        mode='lines+markers',
                                        name=f'Прогнозы (Шаг {step})',
                                        line=dict(color='red', width=1, dash='dash'),
                                        marker=dict(size=3)
                                    ))
                            else:
                                fig.add_trace(go.Scatter(
                                    x=df_cv['ds'],
                                    y=df_cv['predicted'],
                                    mode='lines+markers',
                                    name='Прогнозы',
                                    line=dict(color='red', width=1, dash='dash'),
                                    marker=dict(size=3)
                                ))
                            
                            fig.update_layout(
                                title="Результаты кросс-валидации: Фактические vs Прогнозируемые",
                                xaxis_title="Дата",
                                yaxis_title="Продажи",
                                hovermode='x unified',
                                height=500,
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Не удалось построить график результатов CV: {str(e)}")
                        
                    else:
                        st.error(f"❌ Ошибка кросс-валидации: {response.text}")
                except Exception as e:
                    st.error(f"❌ Ошибка при выполнении кросс-валидации: {str(e)}")

    # Predict section (только для продвинутого режима)
    st.header("🔮 Шаг 5: Генерация прогноза")
    st.info("💡 **Прогноз на будущее:** Если модель была обучена с 'skip_holdout=True', прогноз будет сделан на даты **после** последней даты в обучающих данных. "
           "Для использования сохраненной модели укажите путь к `.pkl` файлу ниже.")

# Заметный раздел для прогноза по категориям (только для продвинутого режима)
if ui_mode == "advanced" and st.session_state.preprocessed_category_csv:
    with st.expander("📁 ПРОГНОЗ ПО КАТЕГОРИЯМ (Распределенный метод - РЕКОМЕНДУЕТСЯ)", expanded=True):
        st.write("""
        **🎯 Распределенный прогноз категорий:**
        
        Использует shop-level прогноз и распределяет его по категориям пропорционально их историческим долям.
        Это **РЕКОМЕНДУЕТСЯ** для категорий, так как дает более точные результаты и избегает циклических паттернов.
        
        **📋 Что нужно сделать:**
        1. Обучьте shop-level модель (в Шаге 3 выберите "По всему магазину")
        2. Включите опцию ниже и следуйте инструкциям
        """)
        
        use_distributed = st.checkbox(
            "✅ Использовать распределенный прогноз категорий",
            value=False,
            key="use_distributed_main",
            help="Включить альтернативный метод прогнозирования категорий через shop-level прогноз"
        )
        
        if use_distributed:
            st.session_state.use_distributed_forecast = True
            st.session_state.selected_aggregation_level = "category"  # Устанавливаем уровень на category
            st.success("💡 **Режим распределенного прогноза активирован!** Прокрутите вниз до раздела '🆕 Распределенный прогноз категорий' для настройки.")
        else:
            st.session_state.use_distributed_forecast = False

# Инициализируем model_info заранее
model_info = ""

if st.session_state.trained_model_path:
    # Инициализируем model_info
    model_info = f"🤖 Используется модель: `{st.session_state.trained_model_path}`"
    
    # Показываем информацию о текущей модели (только для продвинутого режима)
    if ui_mode == "advanced":
        # Показываем предупреждение если модель обучена с skip_holdout
        if st.session_state.training_metrics and st.session_state.training_metrics.get('skip_holdout', False):
            st.success("✅ Модель обучена на всех данных - готова для прогноза на реальное будущее!")
else:
    # Позволяем загрузить сохраненную модель вручную
    st.subheader("📂 Использование сохраненной модели")
    
    # Сохраняем путь к сохраненной модели отдельно, чтобы не сбрасывался при загрузке файла
    if 'saved_model_path' not in st.session_state:
        st.session_state.saved_model_path = ""
    
    saved_model_path = st.text_input(
        "Путь к сохраненной модели (.pkl)",
        value=st.session_state.saved_model_path,
        help="Укажите путь к ранее обученной модели для прогноза без переобучения"
    )
    
    if saved_model_path and saved_model_path.endswith('.pkl'):
        import os
        if os.path.exists(saved_model_path):
            st.session_state.trained_model_path = saved_model_path
            st.session_state.saved_model_path = saved_model_path  # Сохраняем в session_state
            st.success(f"✅ Модель загружена: {saved_model_path}")
            model_info = f"🤖 Используется сохраненная модель: `{saved_model_path}`"
            
            # Проверяем, требует ли модель регрессоры
            try:
                import joblib
                model = joblib.load(saved_model_path)
                requires_regressors = len(model.extra_regressors) > 0 if hasattr(model, 'extra_regressors') else False
                if requires_regressors:
                    st.warning("⚠️ **Модель требует регрессоры** (avg_price, avg_discount). Укажите путь к CSV с регрессорами ниже.")
            except Exception as e:
                pass  # Если не можем проверить модель, продолжаем
        else:
            st.error(f"❌ Файл не найден: {saved_model_path}")
            st.stop()
    elif saved_model_path:
        st.warning("⚠️ Путь должен указывать на файл .pkl")
        st.stop()
    else:
        # Если модель не загружена, устанавливаем пустую модель_info
        model_info = "ℹ️ Модель не загружена. Обучите модель в разделе 'Шаг 3' или загрузите сохраненную модель выше."

# Проверяем уровень агрегации для распределенного прогноза (только для продвинутого режима)
if ui_mode == "advanced":
    aggregation_level = st.session_state.get('selected_aggregation_level', 'shop')
    
    # Если есть сравнение моделей, показываем информацию о выбранной
    if st.session_state.trained_model_path:
        # model_info уже определена выше, но переопределяем если нужно
        if not model_info:
            model_info = f"🤖 Используется модель: `{st.session_state.trained_model_path}`"
        
        if 'model_comparison' in st.session_state and st.session_state.model_comparison is not None:
            df_comp = st.session_state.model_comparison
            # Пытаемся определить, какая модель используется по пути
            model_name_from_path = os.path.basename(st.session_state.trained_model_path).replace('.pkl', '')
            
            # Ищем метрики текущей модели
            matching_models = df_comp[df_comp['model_name'].str.contains(model_name_from_path, case=False, na=False)]
            if len(matching_models) == 0:
                # Показываем лучшую модель
                best_model = df_comp.iloc[0]
                model_info += f"\n\n📊 **Лучшая модель из auto-tune**: {best_model['model_name']} "
                model_info += f"(MAPE: {best_model['mape']:.2f}%, Coverage: {best_model['coverage']*100:.1f}%)"
            else:
                current_model = matching_models.iloc[0]
                model_info += f"\n\n📊 **Метрики текущей модели**: MAPE: {current_model['mape']:.2f}%, "
                model_info += f"Coverage: {current_model['coverage']*100:.1f}%"
        else:
            # Если есть метрики обучения, показываем их
            if st.session_state.training_metrics:
                mape_val = st.session_state.training_metrics.get('mape', 'N/A')
                if isinstance(mape_val, (int, float)):
                    model_info += f"\n📊 MAPE: {mape_val:.2f}%"
        
        # Показываем информацию о модели только если она загружена и в продвинутом режиме
        st.info(model_info)
    
    col1, col2 = st.columns(2)
    
    with col1:
        horizon = st.number_input(
            "Горизонт прогноза (дней)",
            min_value=1,
            max_value=365,
            value=30,
            step=1,
            help="Количество дней для прогноза в будущее"
        )
        
        log_transform_predict = st.checkbox(
            "Применить log-transform (обратное)",
            value=st.session_state.training_metrics.get('log_transform', False) if st.session_state.training_metrics else False,
            help="Применить обратное преобразование log1p к прогнозам (должно совпадать с настройкой обучения)"
        )
        
        smooth_transition = st.checkbox(
            "⚠️ Smooth transition (РЕКОМЕНДУЕТСЯ: уменьшает завышение в начале)",
            value=True,
            help="Применяет агрессивное сглаживание первых дней прогноза для уменьшения завышения. Использует последнее фактическое значение как якорь."
        )
        
        if not smooth_transition:
            st.warning("⚠️ Без smooth transition прогноз может быть завышен в начале периода!")
    
    with col2:
        # Проверяем, требует ли модель регрессоры
        model_requires_regressors = False
        regressors_csv_value = st.session_state.preprocessed_shop_csv if st.session_state.preprocessed_shop_csv else ""
        
        # Если есть сохраненная модель, проверяем её
        if st.session_state.trained_model_path:
            try:
                import joblib
                import os
                if os.path.exists(st.session_state.trained_model_path):
                    model = joblib.load(st.session_state.trained_model_path)
                    model_requires_regressors = len(model.extra_regressors) > 0 if hasattr(model, 'extra_regressors') else False
            except Exception:
                pass  # Если не можем загрузить, продолжаем
        
        regressor_strategy = st.selectbox(
            "Стратегия заполнения регрессоров",
            options=["ffill", "median"],
            help="Стратегия заполнения регрессоров на будущие даты: 'ffill' использует последние известные значения, 'median' использует медиану",
            disabled=not model_requires_regressors  # Отключаем, если регрессоры не нужны
        )
        
        regressors_csv = st.text_input(
            "CSV с регрессорами (опционально)",
            value=regressors_csv_value,
            help="Путь к CSV файлу с регрессорами (avg_price, avg_discount). Обязательно, если модель использует регрессоры!",
            disabled=not model_requires_regressors  # Отключаем, если регрессоры не нужны
        )
        
        if model_requires_regressors:
            if not regressors_csv:
                st.error("❌ **Обязательно**: Модель требует регрессоры! Укажите путь к CSV файлу с колонками avg_price и avg_discount.")
            else:
                import os
                if not os.path.exists(regressors_csv):
                    st.error(f"❌ Файл не найден: {regressors_csv}")
                else:
                    # Проверяем наличие нужных колонок
                    try:
                        import pandas as pd
                        df_check = pd.read_csv(regressors_csv)
                        if 'avg_price' not in df_check.columns or 'avg_discount' not in df_check.columns:
                            st.warning("⚠️ CSV файл не содержит колонок avg_price и/или avg_discount!")
                        else:
                            st.success("✅ CSV с регрессорами найден")
                    except Exception as e:
                        st.warning(f"⚠️ Не удалось проверить CSV: {str(e)}")
    
    # Параметры сглаживания (показываем только если включено)
    if smooth_transition:
        with st.expander("🔧 Параметры сглаживания", expanded=False):
            smooth_days = st.slider(
                "Дней для сглаживания",
                min_value=1,
                max_value=30,
                value=21,  # Увеличено до 21 дня для более плавного перехода
                step=1,
                help="Количество первых дней прогноза, к которым применяется сглаживание (рекомендуется 21-30)"
            )
            smooth_alpha = st.slider(
                "Вес истории для первого дня (alpha)",
                min_value=0.0,
                max_value=1.0,
                value=0.6,  # UI значение, но в коде используется фиксированное 0.95
                step=0.05,
                format="%.2f",
                disabled=True,  # Отключаем - теперь используется автоматическое значение 95%
                help="⚠️ АВТОМАТИЧЕСКИ: Используется 95% веса истории для первого дня (фиксировано для максимальной эффективности)"
            )
            st.info("💡 Первый день автоматически использует 95% последнего фактического значения + 5% прогноза")
            max_change_pct = st.slider(
                "Макс. изменение день-день (%)",
                min_value=0.5,
                max_value=5.0,
                value=1.0,  # Снижено до 1% по умолчанию
                step=0.1,
                format="%.1f",
                help="Максимальное изменение между днями (1% = очень плавный). Первые 3 дня: 0.5%, дни 4-7: 1%"
            )
    else:
        smooth_days = 14
        smooth_alpha = 0.6
        max_change_pct = 0.015
    
    # Проверяем уровень агрегации для распределенного прогноза
    aggregation_level = st.session_state.get('selected_aggregation_level', 'shop')
    use_distributed = st.session_state.get('use_distributed_forecast', False)
    
    if use_distributed and aggregation_level == "category":
        # Распределенный прогноз категорий
        st.subheader("🆕 Распределенный прогноз категорий")
        
        # Параметры прогноза для распределенного метода
        col_params, col_inputs = st.columns([1, 1])
        
        with col_params:
            horizon_distributed = st.number_input(
                "Горизонт прогноза (дней)",
                min_value=1,
                max_value=365,
                value=30,
                step=1,
                key="horizon_distributed",
                help="Количество дней для прогноза в будущее"
            )
            
            log_transform_distributed = st.checkbox(
                "Применить log-transform для shop-level прогноза",
                value=True,
                key="log_transform_distributed",
                help="Применить log-transform при генерации shop-level прогноза (рекомендуется)"
            )
            
            smooth_transition_distributed = st.checkbox(
                "Smooth transition для shop-level прогноза",
                value=True,
                key="smooth_transition_distributed",
                help="Применить smooth transition при генерации shop-level прогноза"
            )
        
        with col_inputs:
            shop_model_path = st.text_input(
                "Путь к shop-level модели (.pkl)",
                value="models/prophet_model.pkl",
                help="Путь к обученной shop-level модели (обученной на shop CSV). По умолчанию: models/prophet_model.pkl"
            )
            
            # Проверяем, требует ли модель регрессоры
            shop_model_requires_regressors = False
            if shop_model_path and shop_model_path.endswith('.pkl'):
                import os
                if os.path.exists(shop_model_path):
                    try:
                        import joblib
                        model = joblib.load(shop_model_path)
                        shop_model_requires_regressors = len(model.extra_regressors) > 0 if hasattr(model, 'extra_regressors') else False
                    except Exception:
                        pass
            
            shop_forecast_csv = st.text_input(
                "Путь к shop-level прогнозу (CSV)",
                value="",
                help="Путь к уже сгенерированному shop-level прогнозу (CSV с колонками: ds, yhat, yhat_lower, yhat_upper). "
                     "Если оставить пустым, будет сгенерирован автоматически."
            )
            
            category_csv = st.text_input(
                "Путь к CSV категорий",
                value=st.session_state.preprocessed_category_csv if st.session_state.preprocessed_category_csv else "",
                help="Путь к CSV файлу с историческими данными категорий (category, ds, y)"
            )
            
            # Поле для регрессоров если модель их требует
            if shop_model_requires_regressors:
                st.warning("⚠️ **Shop-level модель требует регрессоры!** Укажите путь к CSV с регрессорами ниже.")
                regressors_csv_distributed = st.text_input(
                    "CSV с регрессорами (avg_price, avg_discount) - ОБЯЗАТЕЛЬНО",
                    value=st.session_state.preprocessed_shop_csv if st.session_state.preprocessed_shop_csv else "",
                    help="Путь к CSV файлу с регрессорами (должен содержать колонки: ds, avg_price, avg_discount). "
                         "Обычно используется shop CSV из предобработки."
                )
            else:
                regressors_csv_distributed = None
        
        if st.button("🔮 Сгенерировать распределенный прогноз", help="Генерирует распределенный прогноз категорий через shop-level прогноз"):
            # Проверяем наличие shop-level модели
            if not shop_model_path or not shop_model_path.endswith('.pkl'):
                st.error("❌ Укажите путь к shop-level модели (.pkl)")
                st.stop()
            
            import os
            if not os.path.exists(shop_model_path):
                st.error(f"❌ Shop-level модель не найдена: {shop_model_path}")
                st.stop()
            
            # Если shop-level прогноз не указан, генерируем его
            if not shop_forecast_csv:
                st.info("💡 Shop-level прогноз не указан. Генерируем автоматически...")
                
                # Проверяем наличие регрессоров если модель их требует
                if shop_model_requires_regressors:
                    if not regressors_csv_distributed or not regressors_csv_distributed.strip():
                        st.error("❌ **Ошибка:** Shop-level модель требует регрессоры, но CSV файл не указан!")
                        st.info("💡 Укажите путь к CSV с регрессорами (обычно это shop CSV из предобработки)")
                        st.stop()
                    
                    import os
                    if not os.path.exists(regressors_csv_distributed):
                        st.error(f"❌ Файл с регрессорами не найден: {regressors_csv_distributed}")
                        st.stop()
                    
                    # Проверяем наличие нужных колонок
                    try:
                        import pandas as pd
                        df_check = pd.read_csv(regressors_csv_distributed)
                        if 'avg_price' not in df_check.columns or 'avg_discount' not in df_check.columns:
                            st.error("❌ CSV файл не содержит колонок avg_price и/или avg_discount!")
                            st.info("💡 Убедитесь, что CSV содержит колонки: ds, avg_price, avg_discount")
                            st.stop()
                    except Exception as e:
                        st.error(f"❌ Ошибка проверки CSV: {str(e)}")
                        st.stop()
                
                try:
                    shop_payload = {
                        "model_path": shop_model_path,
                        "horizon": int(horizon_distributed),
                        "log_transform": log_transform_distributed,
                        "smooth_transition": smooth_transition_distributed
                    }
                    
                    # Добавляем регрессоры если нужны
                    if shop_model_requires_regressors:
                        shop_payload["last_known_regressors_csv"] = regressors_csv_distributed
                        shop_payload["future_regressor_strategy"] = "ffill"  # Используем forward fill по умолчанию
                    
                    with st.spinner("Генерация shop-level прогноза..."):
                        shop_response = requests.post(f"{FASTAPI_URL}/predict", json=shop_payload, timeout=120)
                    
                    if shop_response.status_code == 200:
                        shop_result = shop_response.json()
                        shop_forecast_csv = shop_result["forecast_csv_path"]
                        st.success(f"✅ Shop-level прогноз создан: {shop_forecast_csv}")
                    else:
                        error_text = shop_response.text
                        st.error(f"❌ Ошибка генерации shop-level прогноза: {error_text}")
                        # Даем полезные советы
                        if "regressors" in error_text.lower():
                            st.info("💡 **Решение:** Убедитесь, что:\n"
                                   "1. Shop-level модель была обучена с регрессорами (avg_price, avg_discount)\n"
                                   "2. Указан правильный путь к CSV с регрессорами выше\n"
                                   "3. CSV содержит колонки: ds, avg_price, avg_discount\n"
                                   "4. Обычно используется shop CSV из предобработки (data/processed/sales_06_FY2020-21_shop.csv)")
                        st.stop()
                except Exception as e:
                    st.error(f"❌ Ошибка при генерации shop-level прогноза: {str(e)}")
                    st.stop()
            
            # Генерируем распределенный прогноз
            try:
                payload = {
                    "shop_forecast_csv": shop_forecast_csv,
                    "category_csv": category_csv if category_csv else st.session_state.preprocessed_category_csv,
                    "horizon_days": int(horizon_distributed)
                }
                
                with st.spinner("Распределение shop-level прогноза по категориям..."):
                    response = requests.post(f"{FASTAPI_URL}/predict_category_distributed", json=payload, timeout=120)
                
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.forecast_data = result["forecast"]
                    st.session_state.forecast_csv_path = result["forecast_csv_path"]
                    st.session_state.log_transform_used = False  # Распределенный прогноз не использует log transform
                    # Исправляем ошибку - используем правильное поле
                    n_cats = result.get('n_categories', len(set([f.get('category', '') for f in result.get('forecast', [])])))
                    st.success(f"✅ Распределенный прогноз успешно сгенерирован! ({n_cats} категорий, {result.get('n_predictions', 0)} прогнозов)")
                    # Сохраняем информацию о том, что это распределенный прогноз
                    st.session_state.forecast_is_distributed = True
                else:
                    st.error(f"❌ Ошибка генерации распределенного прогноза: {response.text}")
            except Exception as e:
                st.error(f"❌ Ошибка при генерации распределенного прогноза: {str(e)}")
    else:
        # Обычный прогноз
        if st.button("🔮 Сгенерировать прогноз", help="Генерирует прогноз на указанный горизонт"):
            # Проверяем, требует ли модель регрессоры перед отправкой запроса
            if model_requires_regressors and not regressors_csv:
                st.error("❌ **Ошибка**: Модель требует регрессоры, но CSV файл не указан. Пожалуйста, укажите путь к CSV с колонками avg_price и avg_discount.")
                st.stop()
            
            if model_requires_regressors and regressors_csv:
                import os
                if not os.path.exists(regressors_csv):
                    st.error(f"❌ **Ошибка**: Файл с регрессорами не найден: {regressors_csv}")
                    st.stop()
            
            try:
                payload = {
                    "model_path": st.session_state.trained_model_path,
                    "horizon": int(horizon),
                    "log_transform": log_transform_predict,
                    "future_regressor_strategy": regressor_strategy,
                    "last_known_regressors_csv": regressors_csv if (regressors_csv and model_requires_regressors) else None,
                    "smooth_transition": smooth_transition,
                    "smooth_days": smooth_days,
                    "smooth_alpha": smooth_alpha,
                    "max_change_pct": max_change_pct / 100.0
                }
                
                with st.spinner("Генерация прогноза..."):
                    response = requests.post(f"{FASTAPI_URL}/predict", json=payload, timeout=120)
                
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.forecast_data = result["forecast"]
                    st.session_state.forecast_csv_path = result["forecast_csv_path"]
                    st.session_state.log_transform_used = log_transform_predict
                    st.success(f"✅ Прогноз успешно сгенерирован! ({result['n_predictions']} прогнозов)")
                else:
                    st.error(f"❌ Ошибка генерации прогноза: {response.text}")
            except Exception as e:
                st.error(f"❌ Ошибка при генерации прогноза: {str(e)}")

# Display forecast visualization and table if forecast data exists (для ОБОИХ режимов)
if st.session_state.forecast_data is not None and st.session_state.forecast_csv_path is not None:
    # Загружаем прогноз из CSV для правильной обработки распределенного прогноза
    try:
        df_forecast_full = pd.read_csv(st.session_state.forecast_csv_path)
        df_forecast_full['ds'] = pd.to_datetime(df_forecast_full['ds'])
        
        # Проверяем, является ли это распределенным прогнозом категорий
        is_distributed_category_forecast = 'category' in df_forecast_full.columns
        
        if is_distributed_category_forecast:
            # Распределенный прогноз категорий - показываем отдельные графики
            st.subheader("📈 Визуализация прогноза по категориям")
            
            # Получаем список категорий
            categories = sorted(df_forecast_full['category'].unique())
            
            # Выбор категорий для отображения
            col1, col2 = st.columns([2, 1])
            with col1:
                display_mode = st.radio(
                    "Режим отображения:",
                    options=["top5", "selected", "all"],
                    format_func=lambda x: {
                        "top5": "📊 Топ-5 категорий (по среднему прогнозу)",
                        "selected": "✅ Выбранные категории (чекбоксы)",
                        "all": "📋 Все категории"
                    }[x],
                    index=2,  # По умолчанию показываем все категории
                    help="Выберите какие категории показывать на графиках"
                )
            
            with col2:
                if display_mode == "all":
                    st.info(f"Показано категорий: {len(categories)}")
                elif display_mode == "top5":
                    st.info("Показано категорий: 5")
            
            # Определяем какие категории показывать
            if display_mode == "top5":
                # Вычисляем средний прогноз для каждой категории
                category_means = df_forecast_full.groupby('category')['yhat'].mean().sort_values(ascending=False)
                selected_categories = category_means.head(5).index.tolist()
                st.info(f"📊 Топ-5 категорий по среднему прогнозу: {', '.join(selected_categories)}")
            elif display_mode == "selected":
                # Чекбоксы для выбора категорий
                st.write("**Выберите категории для отображения:**")
                selected_categories = []
                n_cols = 3
                cols = st.columns(n_cols)
                for idx, cat in enumerate(categories):
                    col_idx = idx % n_cols
                    with cols[col_idx]:
                        if st.checkbox(cat, value=(idx < 5), key=f"cat_checkbox_{cat}"):
                            selected_categories.append(cat)
                if not selected_categories:
                    st.warning("⚠️ Не выбрано ни одной категории! Выберите хотя бы одну.")
                    selected_categories = categories[:5]  # Fallback на первые 5
            else:
                selected_categories = categories
            
            # Показываем графики для каждой выбранной категории
            for category in selected_categories:
                cat_forecast = df_forecast_full[df_forecast_full['category'] == category].copy()
                cat_forecast = cat_forecast.sort_values('ds')
                
                # Загружаем историю для этой категории
                cat_history = None
                if st.session_state.preprocessed_category_csv:
                    try:
                        df_history_all = pd.read_csv(st.session_state.preprocessed_category_csv)
                        df_history_all['ds'] = pd.to_datetime(df_history_all['ds'])
                        cat_history = df_history_all[df_history_all['category'] == category].copy()
                        cat_history = cat_history.sort_values('ds')
                        
                        # Разделяем на train/test
                        forecast_start = cat_forecast['ds'].min()
                        cat_history_train = cat_history[cat_history['ds'] < forecast_start].copy()
                        cat_history_test = cat_history[cat_history['ds'] >= forecast_start].copy()
                    except Exception as e:
                        logger.warning(f"Could not load history for {category}: {str(e)}")
                        cat_history_train = None
                        cat_history_test = None
                else:
                    cat_history_train = None
                    cat_history_test = None
                
                # Создаем график для категории
                fig = go.Figure()
                
                # Исторические данные (период обучения)
                if cat_history_train is not None and not cat_history_train.empty:
                    fig.add_trace(go.Scatter(
                        x=cat_history_train['ds'],
                        y=cat_history_train['y'],
                        mode='lines',
                        name='Исторические продажи (период обучения)',
                        line=dict(color='blue', width=2)
                    ))
                
                # Фактические продажи (тестовый период)
                if cat_history_test is not None and not cat_history_test.empty:
                    fig.add_trace(go.Scatter(
                        x=cat_history_test['ds'],
                        y=cat_history_test['y'],
                        mode='lines',
                        name='Фактические продажи (тестовый период)',
                        line=dict(color='green', width=2, dash='dash')
                    ))
                
                # Прогноз
                fig.add_trace(go.Scatter(
                    x=cat_forecast['ds'],
                    y=cat_forecast['yhat'],
                    mode='lines',
                    name='Прогноз (будущее)',
                    line=dict(color='red', width=2)
                ))
                
                # Доверительный интервал
                if 'yhat_lower' in cat_forecast.columns and 'yhat_upper' in cat_forecast.columns:
                    fig.add_trace(go.Scatter(
                        x=cat_forecast['ds'],
                        y=cat_forecast['yhat_upper'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    fig.add_trace(go.Scatter(
                        x=cat_forecast['ds'],
                        y=cat_forecast['yhat_lower'],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(255, 0, 0, 0.2)',
                        name='Доверительный интервал',
                        showlegend=True,
                        hoverinfo='skip'
                    ))
                
                fig.update_layout(
                    title=f"Прогноз продаж: {category}",
                    xaxis_title="Дата",
                    yaxis_title="Продажи",
                    hovermode='x unified',
                    height=400,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Краткая статистика по категории
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(f"{category}: Средний прогноз", f"{cat_forecast['yhat'].mean():.1f}")
                with col2:
                    st.metric("Минимум", f"{cat_forecast['yhat'].min():.1f}")
                with col3:
                    st.metric("Максимум", f"{cat_forecast['yhat'].max():.1f}")
                with col4:
                    st.metric("Дней прогноза", len(cat_forecast))
                
                st.markdown("---")
            
            # Общая таблица со всеми категориями
            with st.expander("📋 Таблица прогноза для всех категорий", expanded=False):
                st.dataframe(df_forecast_full, use_container_width=True)
                
        else:
            # Обычный прогноз (shop-level или одна категория/товар)
            # Используем существующую логику визуализации
            df_forecast = df_forecast_full.copy()
            
            # Ensure non-negative values
            if 'yhat' in df_forecast.columns:
                df_forecast['yhat'] = df_forecast['yhat'].clip(lower=0.0)
            if 'yhat_lower' in df_forecast.columns:
                df_forecast['yhat_lower'] = df_forecast['yhat_lower'].clip(lower=0.0)
            if 'yhat_upper' in df_forecast.columns:
                df_forecast['yhat_upper'] = df_forecast['yhat_upper'].clip(lower=0.0)
            if 'yhat_lower' in df_forecast.columns and 'yhat_upper' in df_forecast.columns:
                df_forecast['yhat_upper'] = df_forecast[['yhat_upper', 'yhat_lower']].max(axis=1)
            
            # Plot forecast
            st.subheader("📈 Визуализация прогноза")
            
            # Load history if available
            df_history_train = None
            df_history_test = None
            aggregation_level = st.session_state.get('selected_aggregation_level', 'shop')
            
            if aggregation_level == "category":
                history_csv = st.session_state.preprocessed_category_csv
                history_filter_column = "category"
                history_filter_value = st.session_state.get('selected_filter_value')
            elif aggregation_level == "product":
                history_csv = st.session_state.preprocessed_product_csv
                history_filter_column = "product_id"
                history_filter_value = st.session_state.get('selected_filter_value')
            else:
                history_csv = st.session_state.preprocessed_shop_csv
                history_filter_column = None
                history_filter_value = None
            
            if history_csv:
                try:
                    df_history = pd.read_csv(history_csv)
                    df_history['ds'] = pd.to_datetime(df_history['ds'])
                    
                    if history_filter_column and history_filter_value and history_filter_column in df_history.columns:
                        df_history = df_history[df_history[history_filter_column].astype(str) == str(history_filter_value)].copy()
                    
                    df_history = df_history.sort_values('ds')
                    forecast_start = df_forecast['ds'].min()
                    df_history_train = df_history[df_history['ds'] < forecast_start].copy()
                    df_history_test = df_history[df_history['ds'] >= forecast_start].copy()
                except Exception as e:
                    logger.error(f"Error loading history: {str(e)}")
            
            fig = go.Figure()
            
            if df_history_train is not None and not df_history_train.empty:
                fig.add_trace(go.Scatter(
                    x=df_history_train['ds'],
                    y=df_history_train['y'],
                    mode='lines',
                    name='Исторические продажи (период обучения)',
                    line=dict(color='blue', width=2)
                ))
            
            if df_history_test is not None and not df_history_test.empty:
                fig.add_trace(go.Scatter(
                    x=df_history_test['ds'],
                    y=df_history_test['y'],
                    mode='lines',
                    name='Фактические продажи (тестовый период)',
                    line=dict(color='green', width=2, dash='dash')
                ))
            
            fig.add_trace(go.Scatter(
                x=df_forecast['ds'],
                y=df_forecast['yhat'],
                mode='lines',
                name='Прогноз (будущее)',
                line=dict(color='red', width=2)
            ))
            
            if 'yhat_lower' in df_forecast.columns and 'yhat_upper' in df_forecast.columns:
                fig.add_trace(go.Scatter(
                    x=df_forecast['ds'],
                    y=df_forecast['yhat_upper'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                fig.add_trace(go.Scatter(
                    x=df_forecast['ds'],
                    y=df_forecast['yhat_lower'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    name='Доверительный интервал',
                    showlegend=True,
                    hoverinfo='skip'
                ))
            
            title = "Прогноз продаж"
            if st.session_state.get('log_transform_used', False):
                title += " (Применен Log Transform)"
            
            fig.update_layout(
                title=title,
                xaxis_title="Дата",
                yaxis_title="Продажи",
                hovermode='x unified',
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📋 Таблица прогноза")
            st.dataframe(df_forecast, use_container_width=True)
    except Exception as e:
        st.error(f"❌ Ошибка при загрузке данных прогноза: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
    
    # PDF отчет (только для продвинутого режима)
    if ui_mode == "advanced":
        st.subheader("📥 Скачать PDF отчет")
        
        # Button to generate PDF
        if st.button("📥 Сгенерировать PDF отчет", help="Генерирует PDF отчет с визуализацией прогноза и статистикой"):
            try:
                params = {"path": st.session_state.forecast_csv_path}
                with st.spinner("Генерация PDF отчета..."):
                    response = requests.get(f"{FASTAPI_URL}/forecast/download", params=params, timeout=120)
                
                if response.status_code == 200:
                    st.session_state.pdf_data = response.content
                    st.session_state.pdf_filename = "forecast_report.pdf"
                    st.success("✅ PDF отчет сгенерирован! Нажмите кнопку скачивания ниже.")
                else:
                    st.error(f"❌ Ошибка генерации PDF: {response.text}")
                    st.session_state.pdf_data = None
                    st.session_state.pdf_filename = None
            except Exception as e:
                st.error(f"❌ Ошибка при генерации PDF: {str(e)}")
                st.session_state.pdf_data = None
                st.session_state.pdf_filename = None
        
        # Show download button if PDF data is available
        if st.session_state.pdf_data is not None:
            st.download_button(
                label="💾 Скачать PDF",
                data=st.session_state.pdf_data,
                file_name=st.session_state.pdf_filename,
                mime="application/pdf",
                key='download_pdf'
            )
