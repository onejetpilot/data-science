# Выбор локации для разработки месторождений

## Используемый стек

![Python](https://img.shields.io/badge/-Python-blue)
![Pandas](https://img.shields.io/badge/-Pandas-blue)
![NumPy](https://img.shields.io/badge/-NumPy-yellow)
![scikit--learn](https://img.shields.io/badge/-scikit--learn-orange)
![Requests](https://img.shields.io/badge/-Requests-green)
![PyArrow](https://img.shields.io/badge/-PyArrow-lightgrey)
![PowerShell](https://img.shields.io/badge/-PowerShell-5391FE)

## Цель проекта

Построить воспроизводимый mini-pipeline выбора региона для бурения скважин:
обучить модель прогноза запасов по каждому региону, рассчитать прибыль и риск
убытков через bootstrap, выбрать лучший регион с контролем риска.

## Данные

Используются три файла:

- `geo_data_0.csv`
- `geo_data_1.csv`
- `geo_data_2.csv`

Каждый файл содержит `id`, `f0`, `f1`, `f2`, `product`.

## Этапы пайплайна

1. `ingest`  
   Загружает/копирует сырые CSV в `data/raw` и сохраняет `manifest.json`.
2. `validate`  
   Проверяет схему и корректность числовых колонок.
3. `build_dataset`  
   Очищает данные, убирает дубликаты и сохраняет подготовленные файлы по регионам.
4. `analyze`  
   По каждому региону обучает `LinearRegression`, считает RMSE, прибыль по
   топ-200 скважинам и bootstrap-оценку прибыли/риска.

## Структура проекта

```text
well_loc/
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
└── well_loc.ipynb
```

## Быстрый запуск

```powershell
pip install -r requirements.txt
./run_pipeline.ps1
```

Если файлы уже есть локально:

```powershell
./run_pipeline.ps1 `
  -LocalGeo0File ".\geo_data_0.csv" `
  -LocalGeo1File ".\geo_data_1.csv" `
  -LocalGeo2File ".\geo_data_2.csv"
```

## Итоговые артефакты

- `data/validated/quality_report.json` - отчет по проверкам данных.
- `data/processed/region_0.parquet`, `region_1.parquet`, `region_2.parquet` - подготовленные данные.
- `data/processed/feature_manifest.json` - описание датасетов по регионам.
- `artifacts/analysis_report.json` - RMSE, прибыль, доверительные интервалы и риск убытков.
