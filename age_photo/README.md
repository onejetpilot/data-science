# Определение возраста по фотографии

## Используемый стек

![Python](https://img.shields.io/badge/-Python-blue)
![Pandas](https://img.shields.io/badge/-Pandas-blue)
![NumPy](https://img.shields.io/badge/-NumPy-yellow)
![Pillow](https://img.shields.io/badge/-Pillow-green)
![scikit--learn](https://img.shields.io/badge/-scikit--learn-orange)
![PyArrow](https://img.shields.io/badge/-PyArrow-lightgrey)
![Joblib](https://img.shields.io/badge/-Joblib-teal)
![PowerShell](https://img.shields.io/badge/-PowerShell-5391FE)

## Цель проекта

Построить воспроизводимый пайплайн для предсказания возраста по фотографии:
от загрузки и проверки качества данных до подготовки признаков, обучения модели
и сохранения артефактов.

## Данные

Используется датасет:

```text
datasets/faces/
├── labels.csv
└── final_files/
```

`labels.csv` содержит основные поля:

- `file_name`
- `real_age`

## Этапы пайплайна

1. `ingest`  
   Копирует `labels.csv` в `data/raw` и сохраняет `manifest.json`.
2. `validate`  
   Выполняет проверки качества: null, дубликаты, диапазон возраста, наличие изображений.
3. `build_dataset`  
   Извлекает базовые признаки изображения (`img_width`, `img_height`, `pixel_mean`, `pixel_std`) и формирует train/val.
4. `train`  
   Обучает `RandomForestRegressor`, считает `MAE`, сохраняет модель и метрики.

## Быстрый запуск

```powershell
pip install -r requirements.txt
./run_pipeline.ps1 -DatasetDir datasets/faces
```

## Итоговые артефакты

- `data/validated/quality_report.json` - отчет по качеству данных.
- `data/processed/train.parquet` и `data/processed/val.parquet` - подготовленные выборки.
- `artifacts/model.joblib` - обученная baseline-модель.
- `artifacts/metrics.json` - итоговые метрики запуска (включая `MAE`).



