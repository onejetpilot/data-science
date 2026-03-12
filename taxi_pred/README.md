# Прогнозирование заказов такси

## Используемый стек

![Python](https://img.shields.io/badge/-Python-blue)
![Pandas](https://img.shields.io/badge/-Pandas-blue)
![NumPy](https://img.shields.io/badge/-NumPy-yellow)
![scikit--learn](https://img.shields.io/badge/-scikit--learn-orange)
![CatBoost](https://img.shields.io/badge/-CatBoost-black)
![Requests](https://img.shields.io/badge/-Requests-green)
![PyArrow](https://img.shields.io/badge/-PyArrow-lightgrey)
![PowerShell](https://img.shields.io/badge/-PowerShell-5391FE)

## Цель проекта

Построить воспроизводимый mini-pipeline прогноза количества заказов такси на
следующий час и проверить достижение целевого качества RMSE <= 48.

## Данные

Источник: `taxi.csv` с колонками:

- `datetime` - временная метка;
- `num_orders` - количество заказов.

## Этапы пайплайна

1. `ingest`  
   Загружает исходный CSV в `data/raw` и сохраняет `manifest.json`.
2. `validate`  
   Проверяет обязательные колонки, валидность `datetime` и `num_orders`.
3. `build_dataset`  
   Ресемплирует ряд по часу, создает лаги (`1,2,3,24,168`) и скользящие средние
   (`24,168`), формирует train/test без shuffle.
4. `train`  
   Сравнивает `CatBoost`, `RandomForest`, `LinearRegression` по CV RMSE
   (`TimeSeriesSplit`), выбирает лучшую и считает RMSE на test.

## Структура проекта

```text
taxi_pred/
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
└── taxi_pred.ipynb
```

## Быстрый запуск

```powershell
pip install -r requirements.txt
./run_pipeline.ps1
```

Если файл уже есть локально:

```powershell
./run_pipeline.ps1 -LocalFile ".\taxi.csv"
```

## Итоговые артефакты

- `data/validated/quality_report.json` - отчет по проверкам данных.
- `data/processed/train.parquet`, `data/processed/test.parquet` - подготовленные выборки.
- `data/processed/feature_manifest.json` - параметры признаков и split.
- `artifacts/model.joblib` - лучшая модель.
- `artifacts/metrics.json` - CV и test RMSE + проверка порога 48.
- `artifacts/test_predictions.csv` - факт/прогноз для test.
