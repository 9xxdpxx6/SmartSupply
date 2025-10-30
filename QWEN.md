# QWEN Project: Forecasting service (FastAPI) + Streamlit GUI — руководство

## Цель

Построить локально тестируемый сервис прогнозирования продаж для **всего магазина** (агрегат по всем товарам) с возможностью:

* загрузить CSV,
* провести предобработку,
* обучить модель,
* получить прогноз (horizon, например 30 дней),
* визуализировать результаты в Streamlit,
* скачать PDF отчёт с графиками и таблицами.

## Стек

* Python 3.10+
* pandas, numpy
* prophet
* scikit-learn
* matplotlib
* joblib
* FastAPI + uvicorn
* Streamlit
* matplotlib.backends.backend_pdf.PdfPages
* optional: lightgbm/xgboost (если захочешь feature-based моделирование)

## Структура проекта (рекомендуемая)

```
sales_forecast_project/
├── data/
│   ├── raw/                       # загруженные CSV
│   │   └── sales_data.csv
│   ├── processed/
│   │   ├── shop_daily.csv         # агрегат по дате (ds, y)
│   │   └── category_daily.csv     # агрегат по категория (category, ds, y)
├── notebooks/
│   ├── explore.ipynb
├── models/
│   ├── prophet_shop.pkl
│   └── prophet_category_<cat>.pkl  # optional
├── app/
│   ├── main.py                    # FastAPI приложение
│   ├── preprocessing.py           # parser + converter + aggregate
│   ├── train.py                   # функции тренировки модели
│   ├── predict.py                 # функции предсказания
│   ├── utils.py                   # метрики, pdf export
├── streamlit_app/
│   └── app.py                     # Streamlit GUI
├── requirements.txt
├── README.md
└── QWEN.md
```

## Каноническая схема данных

При парсинге входного CSV все столбцы приводятся к каноническим именам:

* `Sale_Date` → `ds` (datetime)
* `Product_ID` → `product_id` (string/int)
* `Product_Category` → `category` (string)
* `Unit_Price` → `price` (float)
* `Discount` → `discount` (float, 0..1)
* `Quantity_Sold` → `y` (int/float) — **целевое значение**

### Обязательные столбцы для парсера:

`Sale_Date, Product_ID, Product_Category, Unit_Price, Discount, Quantity_Sold`

### Минимальные выходные файлы после preprocessing:

* `data/processed/shop_daily.csv` — колонки: `ds`, `y` (sum of `Quantity_Sold` per ds)
* `data/processed/category_daily.csv` — колонки: `category`, `ds`, `y` (sum per category per ds)

**Примечание**: заполнять пропущенные даты до непрерывного диапазона (fill zeros) — важно для временных рядов.

## Предобработка (parsing + feature engineering)

1. Прочитать CSV в pandas, задать `parse_dates=["Sale_Date"]`.
2. Убедиться, что `Quantity_Sold` числовое; пропуски → 0.
3. Привести `ds` к дневной частоте (если данные внутри дня — агрегировать по дню).
4. Агрегировать:

   * shop-level: `df_shop = df.groupby("ds")["y"].sum().reset_index()`
   * category-level: `df_cat = df.groupby(["category", "ds"])["y"].sum().reset_index()`
5. Добавить извлекаемые признаки (для feature-based модели):

   * `day_of_week`, `month`, `is_weekend`
   * скользящие средние: `MA7`, `MA30` (для аналитики, необязательно для Prophet)
6. Сохранить csv'ы в `data/processed/`.

## Модель (рекомендация)

* **Primary model**: Prophet (удобен, объясним, прост в настройке)

  * Обучение: на `shop_daily.csv` (ds, y)
  * horizon: параметризуем в API (default 30 days)
  * сезонности: weekly + yearly (Prophet автоматически)
  * внешние регрессоры: можно добавить `price`, `discount` (если агрегировать соответствующим образом) — опционально
* **Optional**: LightGBM/XGBoost — для регулярной feature-based модели, если есть много дополнительных регрессоров и cross-sectional данные.

## Метрики оценки

* MAE, RMSE, MAPE — считать на holdout-период (например, последние 20% по датам)

## FastAPI — Endpoints (спецификация)

* `POST /upload` — multipart/form-data, file: CSV → сохраняет в `data/raw/`, возвращает `{status: ok, path: ...}`.
* `POST /preprocess` — body: `{file_path: str}` → запускает preprocessing, сохраняет processed csv, возвращает `{shop_csv, category_csv}`.
* `POST /train` — body: `{model: "prophet_shop", training_range: {from, to}}` → тренирует модель, сохраняет `models/prophet_shop.pkl`, возвращает метрики.
* `POST /predict` — body: `{model: "prophet_shop", horizon: int}` → возвращает прогноз (json + saves csv `data/processed/forecast_shop.csv`).
* `GET /forecast/download?format=pdf` — возвращает PDF отчёт (графики и таблицы).
* `GET /health` — проверка статуса сервиса.

## Streamlit GUI (функционал)

* Загрузка CSV (отправка на `/upload`)
* Кнопка Preprocess (вызывает `/preprocess`)
* Показать статистику: записи, даты, пропуски
* Кнопка Train (вызывает `/train`, отображает метрики)
* Настройка прогнозного горизонта (input)
* Кнопка Predict (вызов `/predict`), вывод:

  * График `fact vs forecast` (matplotlib/plotly)
  * Таблица прогнозов (с возможностью сортировки)
  * Кнопка Download PDF (call `/forecast/download`)
* Логирование/статусы (notify user)

## PDF-отчёт

* Содержит:

  * Краткую статистику данных (период, total sales, avg per day)
  * График “fact vs forecast”
  * Таблицу с ближайшими N (например 30) предсказаниями
  * Метрики качества на holdout
* Технически: использовать `matplotlib.backends.backend_pdf.PdfPages` или `reportlab` (PdfPages проще).

## Заметки по реализации (качество/производительность)

* Для локального тестирования — train может занимать 10–60s в зависимости от данных. Model save через `joblib.dump`.
* Добавь простой логгер, обработку ошибок и валидацию входных данных.
* Подготовь `requirements.txt` с фиксированными версиями (чтобы потом удобно деплоить).

## Нюансы по агрегированию признаков (если хочешь учитывать price/discount)

* Если хочешь использовать `price` и `discount` как регрессоры в Prophet:

  * Сначала агрегируешь среднюю цену и среднюю скидку по дате (shop-level):

    * `avg_price = df.groupby("ds")["price"].mean()`
    * `avg_discount = df.groupby("ds")["discount"].mean()`
  * Затем в Prophet `m.add_regressor('avg_price')` и `m.add_regressor('avg_discount')`.
  * В predict нужно передать регрессоры на будущие даты (например, держать равными последним known values или задать сценарии).

## Security / production considerations (архитектура-ready)

* Конфиг переменных через env vars (порты, пути)
* Обёртка endpoint'ов в try/except + валидация
* Возможность асинхронной тренировки (task queue) — оставить hooks для future
* Логи и базовая аутентификация (опционально)

## Quick dev checklist

* [ ] Создать виртуальное окружение и requirements.txt
* [ ] Имплементировать `app/preprocessing.py`
* [ ] Реализовать `app/train.py` (Prophet)
* [ ] Реализовать `app/predict.py`
* [ ] Сделать FastAPI `app/main.py` с вышеописанными endpoint'ами
* [ ] Сделать `streamlit_app/app.py`
* [ ] Тесты: базовый smoke-test (загрузка CSV → preprocess → train → predict → download)
* [ ] README с инструкцией “run locally”

