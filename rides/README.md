# Анализ сервиса аренды самокатов

## Используемый стек

![Python](https://img.shields.io/badge/-Python-blue)
![Pandas](https://img.shields.io/badge/-Pandas-blue)
![NumPy](https://img.shields.io/badge/-NumPy-yellow)
![SciPy](https://img.shields.io/badge/-SciPy-blue)
![Requests](https://img.shields.io/badge/-Requests-green)
![PyArrow](https://img.shields.io/badge/-PyArrow-lightgrey)
![PowerShell](https://img.shields.io/badge/-PowerShell-5391FE)

## Цель проекта

Построить воспроизводимый аналитический пайплайн для сервиса аренды самокатов:
подготовить данные о пользователях и поездках, посчитать помесячную выручку и
проверить ключевые продуктовые гипотезы.

## Данные

Источник: `users_go.csv`, `rides_go.csv`, `subscriptions_go.csv`.

Основные сущности:

- пользователи (`age`, `city`, `subscription_type`);
- поездки (`distance`, `duration`, `date`);
- тарифы (`minute_price`, `start_ride_price`, `subscription_fee`).

## Этапы пайплайна

1. `ingest`  
   Скачивает исходные CSV в `data/raw`.
2. `validate`  
   Проверяет наличие файлов, обязательных колонок и базовую целостность.
3. `build_dataset`  
   Повторяет ключевые шаги ноутбука: конвертацию даты, месяц поездки, объединение
   таблиц, округление длительности, помесячную агрегацию и расчет выручки.
4. `analyze`  
   Считает метрики, сводки по подпискам и статистические тесты (`scipy.stats`).

## Структура проекта

```text
rides/
├── src/
│   ├── ingest.py
│   ├── validate.py
│   ├── build_dataset.py
│   └── analyze.py
├── data/
│   ├── raw/
│   ├── validated/
│   └── processed/
├── artifacts/
├── run_pipeline.ps1
├── requirements.txt
└── rides.ipynb
```

## Быстрый запуск

```powershell
pip install -r requirements.txt
./run_pipeline.ps1
```

## Итоговые артефакты

- `data/validated/quality_report.json` - отчет по проверкам исходных файлов.
- `data/processed/rides_enriched.parquet` - объединенный датасет поездок.
- `data/processed/monthly_user_metrics.parquet` - помесячные метрики пользователя.
- `artifacts/analysis_report.json` - итоговые бизнес-метрики и результаты гипотез.
