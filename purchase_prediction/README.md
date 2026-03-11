# Предсказание покупок

## Используемый стек

![Python](https://img.shields.io/badge/-Python-blue)
![Pandas](https://img.shields.io/badge/-Pandas-blue)
![NumPy](https://img.shields.io/badge/-NumPy-yellow)
![scikit--learn](https://img.shields.io/badge/-scikit--learn-orange)
![Category Encoders](https://img.shields.io/badge/-Category_Encoders-2E8B57)
![LightGBM](https://img.shields.io/badge/-LightGBM-green)
![CatBoost](https://img.shields.io/badge/-CatBoost-black)
![PyArrow](https://img.shields.io/badge/-PyArrow-lightgrey)
![Joblib](https://img.shields.io/badge/-Joblib-teal)
![PowerShell](https://img.shields.io/badge/-PowerShell-5391FE)

## Цель проекта

Построить воспроизводимый ML-пайплайн, который предсказывает вероятность покупки
клиента в течение 90 дней после маркетинговой коммуникации.

## Данные

Источник: `filtered_data.zip` (Hugging Face), внутри:

- `apparel-messages.csv`
- `apparel-purchases.csv`
- `apparel-target_binary.csv`

Целевая переменная:

- `target` (`1` = была покупка, `0` = не было покупки).

## Этапы пайплайна

1. `ingest`  
   Скачивает архив и раскладывает исходные CSV в `data/raw`.
2. `validate`  
   Проверяет наличие файлов, обязательных колонок и базовую целостность.
3. `build_dataset`  
   Повторяет основную feature engineering-логику ноутбука:
   объединение источников, фильтры по `price/quantity`, top-100 `category_ids`,
   временные признаки и отбор части `cat_*` по корреляции.
4. `train`  
   Обучает и сравнивает `LogisticRegression`, `LGBMClassifier`, `CatBoostClassifier`
   через `GridSearchCV` с метрикой `roc_auc`.

## Структура проекта

```text
purchase_prediction/
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
└── purchase_prediction.ipynb
```

## Быстрый запуск

```powershell
pip install -r requirements.txt
./run_pipeline.ps1
```

## Итоговые артефакты

- `data/validated/quality_report.json` - отчет по проверкам сырья.
- `data/processed/dataset.parquet` - подготовленный датасет для обучения.
- `data/processed/feature_manifest.json` - описание признаков и схемы.
- `artifacts/model.joblib` - лучшая модель по `roc_auc`.
- `artifacts/metrics.json` - метрики всех моделей и параметры лучшей.



