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


# Set up the page configuration
st.set_page_config(
    page_title="Sales Forecasting App",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Sales Forecasting App")

# Initialize session state
if 'uploaded_file_path' not in st.session_state:
    st.session_state.uploaded_file_path = None
if 'preprocessed_shop_csv' not in st.session_state:
    st.session_state.preprocessed_shop_csv = None
if 'preprocessed_category_csv' not in st.session_state:
    st.session_state.preprocessed_category_csv = None
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

# Sidebar: Health check
with st.sidebar:
    st.header("API Status")
    try:
        health_response = requests.get(f"{FASTAPI_URL}/health", timeout=5)
        if health_response.status_code == 200:
            st.success("✅ Backend API is running")
        else:
            st.error("❌ Backend API is not responding")
    except:
        st.error("❌ Unable to connect to backend API")
    
    st.header("Help")
    with st.expander("About this app"):
        st.write("""
        This application allows you to:
        1. Upload sales data (CSV)
        2. Preprocess and validate data
        3. Train Prophet forecasting models
        4. Evaluate models with cross-validation
        5. Generate predictions
        6. Download reports as PDF
        """)

# File uploader
st.header("📁 Step 1: Upload Data")
uploaded_file = st.file_uploader("Upload your sales CSV file", type=["csv"], 
                                 help="CSV file must contain columns: Sale_Date, Product_ID, Product_Category, Unit_Price, Discount, Quantity_Sold")

if uploaded_file is not None:
    # Display raw data preview
    bytes_data = uploaded_file.getvalue()
    df_raw = pd.read_csv(io.StringIO(bytes_data.decode("utf-8")))
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Raw Data Preview")
        st.dataframe(df_raw.head(10))
    
    with col2:
        st.subheader("Detected Columns")
        st.write(f"Columns: {', '.join(df_raw.columns.tolist())}")
        st.write(f"Total rows: {len(df_raw)}")
    
    # Upload file to backend
    if st.button("📤 Upload to Backend", help="Uploads the CSV file to the backend API"):
        try:
            files = {"file": (uploaded_file.name, bytes_data, "text/csv")}
            response = requests.post(f"{FASTAPI_URL}/upload", files=files, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                st.session_state.uploaded_file_path = result["file_path"]
                st.success(f"✅ File uploaded successfully: {result['file_path']}")
            else:
                st.error(f"❌ Upload failed: {response.text}")
        except Exception as e:
            st.error(f"❌ Error uploading file: {str(e)}")

# Preprocess section
st.header("⚙️ Step 2: Preprocess Data")
if st.session_state.uploaded_file_path:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"📄 File ready: {st.session_state.uploaded_file_path}")
    with col2:
        force_weekly = st.checkbox("Force weekly aggregation", 
                                  help="Force weekly aggregation regardless of data density")
    
    if st.button("🔄 Preprocess Data", help="Preprocesses the uploaded CSV, validates data, and generates shop/category aggregates"):
        try:
            payload = {
                "file_path": st.session_state.uploaded_file_path,
                "force_weekly": force_weekly
            }
            response = requests.post(f"{FASTAPI_URL}/preprocess", json=payload, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                st.session_state.preprocessed_shop_csv = result["shop_csv"]
                st.session_state.preprocessed_category_csv = result["category_csv"]
                st.session_state.preprocessing_stats = result["stats"]
                
                st.success("✅ Data preprocessed successfully!")
                
                # Show stats
                st.subheader("📊 Preprocessing Statistics")
                
                stats = result["stats"]
                agg_suggestion = result.get("aggregation_suggestion", {})
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Rows (Raw)", stats.get("n_rows_raw", "N/A"))
                    st.metric("Rows After Cleaning", stats.get("n_rows_clean", "N/A"))
                    st.metric("Unique Dates", stats.get("n_unique_dates", "N/A"))
                
                with col2:
                    st.metric("Date Range Start", stats.get("date_min", "N/A")[:10] if stats.get("date_min") else "N/A")
                    st.metric("Date Range End", stats.get("date_max", "N/A")[:10] if stats.get("date_max") else "N/A")
                    st.metric("Duplicates Removed", stats.get("duplicates_removed", 0))
                
                with col3:
                    freq_used = stats.get("freq_used", "D")
                    freq_icon = "📅" if freq_used == "D" else "📆"
                    st.metric("Aggregation Frequency", f"{freq_icon} {freq_used}")
                    
                    if agg_suggestion:
                        st.info(f"💡 Suggestion: {agg_suggestion.get('freq', 'D')} - {agg_suggestion.get('reason', '')}")
                
                if stats.get("warning"):
                    st.warning(f"⚠️ {stats['warning']}")
                
                # Show detailed stats
                with st.expander("📋 Detailed Statistics"):
                    st.json(stats)
            else:
                st.error(f"❌ Preprocessing failed: {response.text}")
        except Exception as e:
            st.error(f"❌ Error preprocessing data: {str(e)}")

# Train section
st.header("🎯 Step 3: Train Model")
if st.session_state.preprocessed_shop_csv:
    st.info(f"📊 Using shop data: {st.session_state.preprocessed_shop_csv}")
    
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
    with st.expander("⚙️ Model Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            include_regressors = st.checkbox(
                "Use regressors (price/discount)",
                value=False,
                help="Include average price and average discount as regressors in the Prophet model"
            )
            
            log_transform = st.checkbox(
                "Apply log-transform to target",
                value=False,
                help="⚠️ РЕКОМЕНДУЕТСЯ для данных с высокой волатильностью! Apply log1p transformation to target variable (useful for skewed data)"
            )
        
        with col2:
            interval_width = st.slider(
                "Interval width",
                min_value=0.5,
                max_value=0.99,
                value=0.95,
                step=0.01,
                help="Confidence interval width for predictions (0.95 = 95% confidence)"
            )
            
            holdout_frac = st.slider(
                "Holdout fraction",
                min_value=0.05,
                max_value=0.5,
                value=0.2,
                step=0.05,
                help="Fraction of data to use for testing (e.g., 0.2 = 20% for test set)"
            )
        
        # Skip holdout option (для прогноза на будущее)
        skip_holdout = st.checkbox(
            "🚀 Train on ALL data (skip holdout) - для прогноза на будущее",
            value=False,
            help="Если включено: модель обучается на ВСЕХ данных без разделения на train/test. "
                 "Используйте для продакшн-прогнозов на реальное будущее (не на тестовый период). "
                 "⚠️ Метрики (MAPE, MAE) не будут вычислены, так как нет тестового набора."
        )
        
        if skip_holdout:
            st.info("💡 **Режим прогноза на будущее:** Модель обучится на всех данных. "
                   "После обучения используйте раздел 'Generate Forecast' для прогноза на реальные будущие даты. "
                   "Holdout fraction будет проигнорирован.")
        
        # Advanced hyperparameters
        with st.expander("🔧 Advanced Hyperparameters (для улучшения качества)", expanded=False):
            # Warning about log_transform + multiplicative combination
            if log_transform:
                st.info("💡 **Совет**: При включенном log-transform обычно лучше использовать **additive** seasonality. Multiplicative + log-transform могут конфликтовать и давать слишком широкие confidence intervals.")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                seasonality_mode = st.selectbox(
                    "Seasonality mode",
                    options=["additive", "multiplicative"],
                    index=0 if log_transform else 0,  # Suggest additive if log_transform is on
                    help="additive: сезонность добавляется к тренду. multiplicative: сезонность умножается на тренд (лучше для высокой волатильности, но БЕЗ log-transform)"
                )
            
            with col2:
                changepoint_prior_scale = st.slider(
                    "Changepoint flexibility",
                    min_value=0.001,
                    max_value=0.5,
                    value=0.01,
                    step=0.001,
                    format="%.3f",
                    help="Гибкость детекции точек изменения тренда (выше = больше гибкости, риск переобучения). Рекомендуется: 0.005-0.01 для стабильных данных"
                )
            
            with col3:
                seasonality_prior_scale = st.slider(
                    "Seasonality strength",
                    min_value=0.01,
                    max_value=100.0,
                    value=10.0,
                    step=1.0,
                    help="Сила сезонных компонентов (выше = сильнее сезонность)"
                )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        model_out_path = st.text_input(
            "Model output path",
            value="models/prophet_model.pkl",
            help="Path where the trained model will be saved"
        )
    
    # Auto-tune option
    auto_tune = st.checkbox(
        "🔍 Auto-tune model (Grid Search)",
        value=False,
        help="Автоматически находит лучшую конфигурацию модели через grid search (Prophet варианты, LSTM, Hybrid). Это займет больше времени, но даст лучшие результаты."
    )
    
    if auto_tune:
        st.info("💡 При включенном auto-tune будут протестированы различные конфигурации Prophet, LSTM и гибридные модели. Результаты сохранятся в analysis/model_comparison.csv")
    
    if st.button("🚀 Train Model", help="Trains a Prophet model with the selected configuration"):
        try:
            payload = {
                "shop_csv": st.session_state.preprocessed_shop_csv,
                "model_out": model_out_path,
                "include_regressors": include_regressors,
                "log_transform": log_transform,
                "interval_width": interval_width,
                "holdout_frac": holdout_frac,
                "changepoint_prior_scale": changepoint_prior_scale,
                "seasonality_prior_scale": seasonality_prior_scale,
                "seasonality_mode": seasonality_mode,
                "auto_tune": auto_tune,
                "skip_holdout": skip_holdout  # Новый параметр
            }
            
            spinner_text = "Training model with auto-tuning (this may take several minutes)..." if auto_tune else "Training model... This may take a while."
            timeout_val = 1800 if auto_tune else 300  # 30 minutes for auto-tune, 5 minutes for regular
            
            with st.spinner(spinner_text):
                response = requests.post(f"{FASTAPI_URL}/train", json=payload, timeout=timeout_val)
            
            if response.status_code == 200:
                result = response.json()
                st.session_state.trained_model_path = result["model_path"]
                st.session_state.training_metrics = result["metrics"]
                
                if skip_holdout:
                    st.success("✅ Model trained successfully on ALL data! Ready for production forecasts.")
                    st.info("💡 **Режим прогноза на будущее:** Метрики не вычислены (нет тестового набора). "
                            "Используйте раздел 'Generate Forecast' для прогноза на реальные будущие даты.")
                else:
                    st.success("✅ Model trained successfully!")
                
                # Display metrics
                st.subheader("📈 Training Metrics")
                
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
                        st.metric("MAE", f"{mae_val:.2f}" if mae_val is not None else "N/A")
                    with col2:
                        st.metric("RMSE", f"{rmse_val:.2f}" if rmse_val is not None else "N/A")
                    with col3:
                        if mape_val is not None:
                            st.metric("MAPE", f"{mape_val:.2f}%", delta=mape_delta if isinstance(mape_val, (int, float)) else None)
                        else:
                            st.metric("MAPE", "N/A", delta="Production mode")
                    
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
                with st.expander("📊 Training Details"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Training Period:**")
                        st.write(f"- Start: {result['train_range']['start'][:10]}")
                        st.write(f"- End: {result['train_range']['end'][:10]}")
                        st.write(f"- Samples: {result['n_train']}")
                    
                    with col2:
                        if skip_holdout or result.get('test_range', {}).get('start') is None:
                            st.write("**⚠️ Production Mode:**")
                            st.write("- Test Period: N/A (skip_holdout=True)")
                            st.write("- Samples in test: 0")
                            st.info("💡 Модель обучена на всех данных. Готова для прогноза на реальное будущее!")
                        else:
                            st.write("**Test Period:**")
                            test_start = result.get('test_range', {}).get('start', 'N/A')
                            test_end = result.get('test_range', {}).get('end', 'N/A')
                            if test_start and test_start != 'N/A':
                                st.write(f"- Start: {test_start[:10] if isinstance(test_start, str) else test_start}")
                            if test_end and test_end != 'N/A':
                                st.write(f"- End: {test_end[:10] if isinstance(test_end, str) else test_end}")
                            st.write(f"- Samples: {result.get('n_test', 0)}")
                    
                    st.write("**Configuration:**")
                    st.write(f"- Log transform: {metrics.get('log_transform', False)}")
                    st.write(f"- Interval width: {metrics.get('interval_width', 0.95)}")
                    st.write(f"- Seasonality mode: {metrics.get('seasonality_mode', 'additive')}")
                    st.write(f"- Changepoint prior scale: {metrics.get('changepoint_prior_scale', 0.05)}")
                    st.write(f"- Seasonality prior scale: {metrics.get('seasonality_prior_scale', 10.0)}")
                    st.write(f"- Used cross-validation: {metrics.get('used_cross_validation', False)}")
                    st.write(f"- Auto-tune used: {metrics.get('auto_tune', False)}")
                
                # Show auto-tune results if available (проверяем skip_holdout через metrics)
                if response.status_code == 200 and metrics.get('auto_tune', False):
                    try:
                        import os
                        comparison_csv = "analysis/model_comparison.csv"
                        if os.path.exists(comparison_csv):
                            df_comparison = pd.read_csv(comparison_csv)
                            st.subheader("📊 Model Comparison (Auto-tune Results)")
                            
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
                                title="MAPE Comparison Across Models (Green = Best Model)",
                                xaxis_title="Model",
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
                        st.warning(f"Could not load auto-tune comparison results: {str(e)}")
            else:
                st.error(f"❌ Training failed: {response.text}")
        except Exception as e:
            st.error(f"❌ Error training model: {str(e)}")
    
    # Diagnostics section
    if st.session_state.trained_model_path and st.session_state.preprocessed_shop_csv:
        st.subheader("🔍 Model Diagnostics")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info("Диагностика модели поможет выявить систематические проблемы: переоценка тренда, низкое покрытие CI, смещение минимумов и др.")
        with col2:
            if st.button("🔍 Run Diagnostics", help="Запускает полную диагностику модели"):
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
                    
                    with st.spinner("Running diagnostics..."):
                        response = requests.post(f"{FASTAPI_URL}/diagnose", json=payload, timeout=120)
                    
                    if response.status_code == 200:
                        diagnostics = response.json()
                        st.session_state.diagnostics = diagnostics
                        
                        st.success("✅ Diagnostics completed!")
                        
                        # Display diagnostics
                        st.subheader("📊 Diagnostic Results")
                        
                        metrics_diag = diagnostics.get('metrics', {})
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("MAPE", f"{metrics_diag.get('mape', 0):.2f}%")
                        with col2:
                            st.metric("Systematic Bias", f"{metrics_diag.get('systematic_bias', 0):.2f}")
                        with col3:
                            coverage = diagnostics.get('coverage', {}).get('coverage_rate', 0) * 100
                            st.metric("CI Coverage", f"{coverage:.1f}%")
                        
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
                        with st.expander("📈 Residuals Analysis"):
                            st.write(f"Mean residual: {residuals.get('mean', 0):.2f}")
                            st.write(f"Std residual: {residuals.get('std', 0):.2f}")
                            st.write(f"Normality test p-value: {residuals.get('normality_test_pvalue', 0):.4f}")
                            if residuals.get('has_trend', False):
                                st.warning(f"⚠️ Обнаружен тренд в остатках: slope={residuals.get('trend_slope', 0):.6f}")
                        
                        # Multicollinearity
                        multicollinearity = diagnostics.get('multicollinearity', {})
                        if multicollinearity.get('has_multicollinearity', False):
                            st.error("🚨 Обнаружена мультиколлинеарность регрессоров!")
                            st.write(f"Max correlation: {multicollinearity.get('max_correlation', 0):.2f}")
                            st.write(f"VIF scores: {multicollinearity.get('vif_scores', {})}")
                        
                        # Minima shift
                        minima_shift = diagnostics.get('minima_shift', {})
                        mean_shift = minima_shift.get('mean_shift_days', 0)
                        if abs(mean_shift) > 3:
                            st.warning(f"⚠️ Локальные минимумы сдвинуты на {mean_shift:.1f} дней")
                        
                    else:
                        st.error(f"❌ Diagnostics failed: {response.text}")
                except Exception as e:
                    st.error(f"❌ Error running diagnostics: {str(e)}")

# Evaluate section
st.header("📊 Step 4: Evaluate Model (Cross-Validation)")
if st.session_state.preprocessed_shop_csv:
    with st.expander("🔍 Cross-Validation Configuration", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            initial_days = st.number_input(
                "Initial training days",
                min_value=30,
                value=180,
                step=30,
                help="Number of days for initial training period in rolling CV"
            )
        
        with col2:
            horizon_days = st.number_input(
                "Forecast horizon (days)",
                min_value=1,
                value=30,
                step=5,
                help="Number of days to forecast ahead in each CV step"
            )
        
        with col3:
            period_days = st.number_input(
                "Window slide period (days)",
                min_value=1,
                value=30,
                step=5,
                help="Number of days to slide the window forward between CV steps"
            )
        
        cv_include_regressors = st.checkbox(
            "Use regressors for CV",
            value=False,
            help="Include regressors in cross-validation (must match training configuration)"
        )
        
        cv_log_transform = st.checkbox(
            "Apply log-transform for CV",
            value=False,
            help="Apply log-transform in cross-validation (must match training configuration)"
        )
    
    if st.button("📈 Run Cross-Validation", help="Performs rolling cross-validation to evaluate model performance"):
        try:
            payload = {
                "shop_csv": st.session_state.preprocessed_shop_csv,
                "initial_days": initial_days,
                "horizon_days": horizon_days,
                "period_days": period_days,
                "include_regressors": cv_include_regressors,
                "log_transform": cv_log_transform
            }
            
            with st.spinner("Running cross-validation... This may take several minutes."):
                response = requests.post(f"{FASTAPI_URL}/evaluate", json=payload, timeout=600)
            
            if response.status_code == 200:
                result = response.json()
                st.session_state.cv_results = result
                
                st.success("✅ Cross-validation completed!")
                
                # Display aggregate metrics
                st.subheader("📊 Cross-Validation Results")
                
                metrics = result["metrics"]
                summary = result["summary"]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("MAE", f"{summary['mae_mean']:.2f}", f"±{summary['mae_std']:.2f}")
                with col2:
                    st.metric("RMSE", f"{summary['rmse_mean']:.2f}", f"±{summary['rmse_std']:.2f}")
                with col3:
                    st.metric("MAPE", f"{summary['mape_mean']:.2f}%", f"±{summary['mape_std']:.2f}%")
                
                st.info(f"📊 Number of CV steps: {result['n_cv_steps']}")
                st.info(f"💾 Predictions saved to: {result['cv_predictions_csv']}")
                
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
                        name='Actual Sales',
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
                                name=f'Predictions (Step {step})',
                                line=dict(color='red', width=1, dash='dash'),
                                marker=dict(size=3)
                            ))
                    else:
                        fig.add_trace(go.Scatter(
                            x=df_cv['ds'],
                            y=df_cv['predicted'],
                            mode='lines+markers',
                            name='Predictions',
                            line=dict(color='red', width=1, dash='dash'),
                            marker=dict(size=3)
                        ))
                    
                    fig.update_layout(
                        title="Cross-Validation Results: Actual vs Predicted",
                        xaxis_title="Date",
                        yaxis_title="Sales",
                        hovermode='x unified',
                        height=500,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not plot CV results: {str(e)}")
                
            else:
                st.error(f"❌ Cross-validation failed: {response.text}")
        except Exception as e:
            st.error(f"❌ Error running cross-validation: {str(e)}")

# Predict section
st.header("🔮 Step 5: Generate Forecast")
st.info("💡 **Прогноз на будущее:** Если модель была обучена с 'skip_holdout=True', прогноз будет сделан на даты **после** последней даты в обучающих данных. "
       "Для использования сохраненной модели укажите путь к `.pkl` файлу ниже.")

if st.session_state.trained_model_path:
    # Показываем информацию о текущей модели
    model_info = f"🤖 Using model: `{st.session_state.trained_model_path}`"
    
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
            model_info = f"🤖 Using saved model: `{saved_model_path}`"
            
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

if st.session_state.trained_model_path:
    
    # Если есть сравнение моделей, показываем информацию о выбранной
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
            model_info += f"\n📊 MAPE: {mape_val:.2f}%" if isinstance(mape_val, (int, float)) else ""
    
    st.info(model_info)
    
    col1, col2 = st.columns(2)
    
    with col1:
        horizon = st.number_input(
            "Forecast horizon (days)",
            min_value=1,
            max_value=365,
            value=30,
            step=1,
            help="Number of days to forecast into the future"
        )
        
        log_transform_predict = st.checkbox(
            "Apply log-transform (inverse)",
            value=st.session_state.training_metrics.get('log_transform', False) if st.session_state.training_metrics else False,
            help="Apply inverse log1p transformation to predictions (should match training setting)"
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
            "Regressor fill strategy",
            options=["ffill", "median"],
            help="Strategy for filling regressors on future dates: 'ffill' uses last known values, 'median' uses median",
            disabled=not model_requires_regressors  # Отключаем, если регрессоры не нужны
        )
        
        regressors_csv = st.text_input(
            "Regressors CSV (optional)",
            value=regressors_csv_value,
            help="Path to CSV with regressors (avg_price, avg_discount). Обязательно, если модель использует регрессоры!",
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
    
    if st.button("🔮 Generate Forecast", help="Generates forecast for the specified horizon"):
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
            
            with st.spinner("Generating forecast..."):
                response = requests.post(f"{FASTAPI_URL}/predict", json=payload, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                st.session_state.forecast_data = result["forecast"]
                st.session_state.forecast_csv_path = result["forecast_csv_path"]
                st.session_state.log_transform_used = log_transform_predict
                st.success(f"✅ Forecast generated successfully! ({result['n_predictions']} predictions)")
            else:
                st.error(f"❌ Prediction failed: {response.text}")
        except Exception as e:
            st.error(f"❌ Error generating forecast: {str(e)}")
    
    # Display forecast visualization and table if forecast data exists
    if st.session_state.forecast_data is not None and st.session_state.forecast_csv_path is not None:
        # Load forecast data
        df_forecast = pd.DataFrame(st.session_state.forecast_data)
        df_forecast['ds'] = pd.to_datetime(df_forecast['ds'])
        
        # Ensure non-negative values (safety check for visualization)
        if 'yhat' in df_forecast.columns:
            n_neg = (df_forecast['yhat'] < 0).sum()
            if n_neg > 0:
                st.warning(f"⚠️ Found {n_neg} negative forecast values, clamping to 0")
                df_forecast['yhat'] = df_forecast['yhat'].clip(lower=0.0)
        
        if 'yhat_lower' in df_forecast.columns:
            n_neg = (df_forecast['yhat_lower'] < 0).sum()
            if n_neg > 0:
                st.warning(f"⚠️ Found {n_neg} negative lower bounds, clamping to 0")
                df_forecast['yhat_lower'] = df_forecast['yhat_lower'].clip(lower=0.0)
        
        if 'yhat_upper' in df_forecast.columns:
            n_neg = (df_forecast['yhat_upper'] < 0).sum()
            if n_neg > 0:
                st.warning(f"⚠️ Found {n_neg} negative upper bounds, clamping to 0")
                df_forecast['yhat_upper'] = df_forecast['yhat_upper'].clip(lower=0.0)
        
        # Ensure yhat_upper >= yhat_lower
        if 'yhat_lower' in df_forecast.columns and 'yhat_upper' in df_forecast.columns:
            df_forecast['yhat_upper'] = df_forecast[['yhat_upper', 'yhat_lower']].max(axis=1)
        
        # Plot forecast
        st.subheader("📈 Forecast Visualization")
        
        # Load history if available
        df_history = None
        train_end_date = None
        
        # Try to get training range from session state (stored after training)
        if 'training_metrics' in st.session_state and st.session_state.training_metrics:
            try:
                # Check if train_range is stored in metrics response
                # We need to get it from the training response, but for now use forecast start date
                forecast_start = df_forecast['ds'].min()
                # Assume training ended 1 day before forecast starts (typical case)
                train_end_date = forecast_start - pd.Timedelta(days=1)
            except:
                pass
        
        if st.session_state.preprocessed_shop_csv:
            try:
                df_history = pd.read_csv(st.session_state.preprocessed_shop_csv)
                df_history['ds'] = pd.to_datetime(df_history['ds'])
                df_history = df_history.sort_values('ds')
                
                # If we know training end date, split history into train/test periods
                if train_end_date is not None:
                    df_history_train = df_history[df_history['ds'] <= train_end_date].copy()
                    df_history_test = df_history[df_history['ds'] > train_end_date].copy()
                else:
                    # Use forecast start date as approximation
                    forecast_start = df_forecast['ds'].min()
                    df_history_train = df_history[df_history['ds'] < forecast_start].copy()
                    df_history_test = df_history[df_history['ds'] >= forecast_start].copy()
            except:
                df_history_train = None
                df_history_test = None
        else:
            df_history_train = None
            df_history_test = None
        
        fig = go.Figure()
        
        # Plot training period data
        if df_history_train is not None and not df_history_train.empty:
            fig.add_trace(go.Scatter(
                x=df_history_train['ds'],
                y=df_history_train['y'],
                mode='lines',
                name='Historical Sales (Training Period)',
                line=dict(color='blue', width=2)
            ))
        
        # Plot test period data (if available) - это реальные данные для сравнения
        if df_history_test is not None and not df_history_test.empty:
            fig.add_trace(go.Scatter(
                x=df_history_test['ds'],
                y=df_history_test['y'],
                mode='lines',
                name='Actual Sales (Test Period)',
                line=dict(color='green', width=2, dash='dash')
            ))
        
        # Plot forecast (future predictions)
        fig.add_trace(go.Scatter(
            x=df_forecast['ds'],
            y=df_forecast['yhat'],
            mode='lines',
            name='Forecast (Future)',
            line=dict(color='red', width=2)
        ))
        
        # Plot confidence intervals
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
                name='Confidence Interval',
                showlegend=True,
                hoverinfo='skip'
            ))
        
        title = "Sales Forecast"
        if st.session_state.get('log_transform_used', False):
            title += " (Log Transform Applied)"
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Sales",
            hovermode='x unified',
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display forecast table
        st.subheader("📋 Forecast Table")
        st.dataframe(df_forecast, use_container_width=True)
        
        # Download PDF section
        st.subheader("📥 Download PDF Report")
        
        # Button to generate PDF
        if st.button("📥 Generate PDF Report", help="Generates a PDF report with forecast visualization and statistics"):
            try:
                params = {"path": st.session_state.forecast_csv_path}
                with st.spinner("Generating PDF report..."):
                    response = requests.get(f"{FASTAPI_URL}/forecast/download", params=params, timeout=120)
                
                if response.status_code == 200:
                    st.session_state.pdf_data = response.content
                    st.session_state.pdf_filename = "forecast_report.pdf"
                    st.success("✅ PDF report generated! Click the download button below.")
                else:
                    st.error(f"❌ PDF generation failed: {response.text}")
                    st.session_state.pdf_data = None
                    st.session_state.pdf_filename = None
            except Exception as e:
                st.error(f"❌ Error generating PDF: {str(e)}")
                st.session_state.pdf_data = None
                st.session_state.pdf_filename = None
        
        # Show download button if PDF data is available
        if st.session_state.pdf_data is not None:
            st.download_button(
                label="💾 Download PDF",
                data=st.session_state.pdf_data,
                file_name=st.session_state.pdf_filename,
                mime="application/pdf",
                key='download_pdf'
            )
