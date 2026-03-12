# Предсказание успешности стартапов

## Используемый стек

![Python](https://img.shields.io/badge/-Python-blue)
![Pandas](https://img.shields.io/badge/-Pandas-blue)
![NumPy](https://img.shields.io/badge/-NumPy-yellow)
![scikit--learn](https://img.shields.io/badge/-scikit--learn-orange)
![LightGBM](https://img.shields.io/badge/-LightGBM-green)
![CatBoost](https://img.shields.io/badge/-CatBoost-black)
![Requests](https://img.shields.io/badge/-Requests-green)
![PyArrow](https://img.shields.io/badge/-PyArrow-lightgrey)
![PowerShell](https://img.shields.io/badge/-PowerShell-5391FE)

## Цель проекта

Построить воспроизводимый mini-pipeline, который предсказывает статус стартапа
(`closed` / `operating`) по данным о финансировании, категориях, датах и
географии.

## Данные

Используются три файла:

- `kaggle_startups_train_28062024.csv`
- `kaggle_startups_test_28062024.csv`
- `worldcitiespop.csv`

## Этапы пайплайна

1. `ingest`  
   Загружает/копирует исходные CSV в `data/raw` и сохраняет `manifest.json`.
2. `validate`  
   Проверяет обязательные колонки и базовые метрики качества.
3. `build_dataset`  
   Повторяет ключевую логику ноутбука: `cat_1..cat_5`, временные признаки
   (`lifetime`, `fundingtime`, ...), merge с городами, отсечение колонок с
   высокой корреляцией.
4. `train`  
   Обучает pipeline с `CatBoostEncoder` и подбором `LightGBM/CatBoost` через
   `RandomizedSearchCV` (метрика `F1` для класса `closed`), сохраняет модель и
   предсказания для test.

## Структура проекта

```text
start_up/
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
└── start_up_pred.ipynb
```

## Быстрый запуск

```powershell
pip install -r requirements.txt
./run_pipeline.ps1
```

Если нужно запускать из локальных файлов:

```powershell
./run_pipeline.ps1 `
  -LocalTrainFile ".\kaggle_startups_train_28062024.csv" `
  -LocalTestFile ".\kaggle_startups_test_28062024.csv" `
  -LocalCitiesFile ".\worldcitiespop.csv"
```

## Итоговые артефакты

- `data/validated/quality_report.json` - отчет валидации сырых данных.
- `data/processed/train.parquet`, `data/processed/test.parquet` - готовые выборки.
- `data/processed/feature_manifest.json` - список итоговых признаков.
- `artifacts/model.joblib` - обученный pipeline.
- `artifacts/metrics.json` - CV и валидационные метрики.
- `artifacts/submit_predictions.csv` - предсказания статуса для test.
