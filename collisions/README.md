# Предсказание ДТП

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

Построить воспроизводимый пайплайн оценки вероятности ДТП:
от загрузки подготовленного датасета столкновений до обучения модели
классификации и сохранения метрик качества.

## Данные

Источник данных: `df_dtp.csv`.

Ключевые поля:

- `at_fault` - целевой признак (0/1, виновен ли водитель в ДТП).
- `weather_1`, `road_surface`, `lighting`, `location_type` - условия ДТП.
- `vehicle_type`, `vehicle_transmission`, `vehicle_age` - характеристики ТС.
- `cellphone_in_use` - факт использования телефона.
- `distance`, `insurance_premium` - дополнительные числовые факторы.

## Этапы пайплайна

1. `ingest`  
   Скачивает `df_dtp.csv` в `data/raw` и создает `manifest.json`.
2. `validate`  
   Проверяет обязательные колонки и бинарность целевого признака.
3. `build_dataset`  
   Очищает признаки, заполняет пропуски, делит данные на train/val.
4. `train`  
   Обучает `CatBoostClassifier` и считает `ROC-AUC`, `F1`, `Precision`, `Recall`.

## Структура проекта

```text
collisions/
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
└── collisions.ipynb
```

## Быстрый запуск

```powershell
pip install -r requirements.txt
./run_pipeline.ps1
```

Если нужен другой источник CSV:

```powershell
./run_pipeline.ps1 -DataUrl "https://your-url/df_dtp.csv"
```

## Итоговые артефакты

- `data/validated/quality_report.json` - отчет по проверкам данных.
- `data/processed/train.parquet` и `data/processed/val.parquet` - подготовленные выборки.
- `data/processed/feature_manifest.json` - список признаков и target.
- `artifacts/model.cbm` - обученная CatBoost модель.
- `artifacts/metrics.json` - метрики запуска (`ROC-AUC`, `F1`, `Precision`, `Recall`).
