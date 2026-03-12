# Прогнозирование температуры звезды

## Используемый стек

![Python](https://img.shields.io/badge/-Python-blue)
![Pandas](https://img.shields.io/badge/-Pandas-blue)
![NumPy](https://img.shields.io/badge/-NumPy-yellow)
![scikit--learn](https://img.shields.io/badge/-scikit--learn-orange)
![PyTorch](https://img.shields.io/badge/-PyTorch-red)
![Requests](https://img.shields.io/badge/-Requests-green)
![PyArrow](https://img.shields.io/badge/-PyArrow-lightgrey)
![PowerShell](https://img.shields.io/badge/-PowerShell-5391FE)

## Цель проекта

Построить воспроизводимый пайплайн предсказания абсолютной температуры звезды:
от загрузки исходного CSV до обучения baseline и улучшенной нейросети с
сохранением итоговых метрик.

## Данные

Источник: `6_class_1.csv`.

Ключевые признаки:

- `Luminosity(L/Lo)` - относительная светимость.
- `Radius(R/Ro)` - относительный радиус.
- `Absolute magnitude(Mv)` - абсолютная звездная величина.
- `Star color` - цвет звезды.
- `Star type` - тип звезды.
- `Spectral Class` - спектральный класс.
- `Temperature (K)` - целевой признак.

## Этапы пайплайна

1. `ingest`  
   Скачивает исходный CSV в `data/raw` и сохраняет `manifest.json`.
2. `validate`  
   Проверяет обязательные колонки и базовые показатели качества.
3. `build_dataset`  
   Выполняет очистку категорий (`star_color`), split train/val и preprocessing
   (`StandardScaler + OneHotEncoder`), сохраняет подготовленные выборки.
4. `train`  
   Обучает набор baseline архитектур и улучшенную модель `MyNetCD` с подбором
   `dropout/batch_size`, сохраняет модель, метрики и прогнозы.

## Структура проекта

```text
star_temp/
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
└── star_temp.ipynb
```

## Быстрый запуск

```powershell
pip install -r requirements.txt
./run_pipeline.ps1
```

Если нужен другой источник CSV:

```powershell
./run_pipeline.ps1 -DataUrl "https://your-url/6_class_1.csv"
```

Если источник недоступен, можно использовать локальный файл:

```powershell
./run_pipeline.ps1 -LocalFile ".\6_class_1.csv"
```

## Итоговые артефакты

- `data/raw/6_class_1.csv` - исходный датасет.
- `data/validated/quality_report.json` - отчет по проверкам качества.
- `data/processed/train.parquet`, `data/processed/val.parquet` - подготовленные выборки.
- `data/processed/feature_manifest.json` - итоговый список признаков.
- `artifacts/model.pt` - веса лучшей улучшенной модели.
- `artifacts/metrics.json` - сводные метрики baseline/tuned.
- `artifacts/predictions.csv` - факт/прогноз на валидации.
