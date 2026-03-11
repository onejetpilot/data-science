# Определение стоимости автомобилей

## Используемый стек

![Python](https://img.shields.io/badge/-Python-blue)
![Pandas](https://img.shields.io/badge/-Pandas-blue)
![NumPy](https://img.shields.io/badge/-NumPy-yellow)
![scikit--learn](https://img.shields.io/badge/-scikit--learn-orange)
![CatBoost](https://img.shields.io/badge/-CatBoost-black)
![PyArrow](https://img.shields.io/badge/-PyArrow-lightgrey)
![Requests](https://img.shields.io/badge/-Requests-green)
![PowerShell](https://img.shields.io/badge/-PowerShell-5391FE)

## Цель проекта

Построить воспроизводимый пайплайн оценки рыночной стоимости подержанного
автомобиля по его характеристикам: от загрузки данных и проверки качества
до подготовки признаков, обучения модели и сохранения метрик.

## Данные

Источник данных: `autos.csv`.

Ключевые поля:

- `Price` - целевой признак (стоимость автомобиля).
- `VehicleType`, `Gearbox`, `Model`, `FuelType`, `Brand`, `Repaired` - категориальные признаки.
- `RegistrationYear`, `Power`, `Kilometer`, `RegistrationMonth` - числовые признаки.
- `DateCrawled`, `DateCreated`, `LastSeen` - служебные временные поля.

## Этапы пайплайна

1. `ingest`  
   Скачивает `autos.csv` в `data/raw` и создает `manifest.json`.
2. `validate`  
   Проверяет наличие обязательных колонок и базовую целостность данных.
3. `build_dataset`  
   Очищает аномалии, формирует признаки, делит данные на train/val.
4. `train`  
   Обучает `CatBoostRegressor` и считает метрики `RMSE` и `MAE`.

## Структура проекта

```text
car_price/
├── src/
│   ├── ingest.py
│   ├── validate.py
│   ├── build_dataset.py
│   └── train.py
├── data/
│   ├── raw/
│   ├── validated/
│   └── processed/
├── artifacts/
├── run_pipeline.ps1
├── requirements.txt
└── autos.ipynb
```

## Быстрый запуск

```powershell
pip install -r requirements.txt
./run_pipeline.ps1
```

Если нужен другой источник CSV:

```powershell
./run_pipeline.ps1 -DataUrl "https://your-url/autos.csv"
```

## Итоговые артефакты

- `data/validated/quality_report.json` - отчет по проверкам качества.
- `data/processed/train.parquet` и `data/processed/val.parquet` - подготовленные выборки.
- `data/processed/feature_manifest.json` - список признаков для обучения.
- `artifacts/model.cbm` - обученная CatBoost модель.
- `artifacts/metrics.json` - итоговые метрики запуска (`RMSE`, `MAE`).
