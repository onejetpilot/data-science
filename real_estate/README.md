# Исследование объявлений о продаже квартир

## Используемый стек

![Python](https://img.shields.io/badge/-Python-blue)
![Pandas](https://img.shields.io/badge/-Pandas-blue)
![NumPy](https://img.shields.io/badge/-NumPy-yellow)
![PyArrow](https://img.shields.io/badge/-PyArrow-lightgrey)
![Requests](https://img.shields.io/badge/-Requests-green)
![PowerShell](https://img.shields.io/badge/-PowerShell-5391FE)

## Цель проекта

Собрать воспроизводимый пайплайн для данных объявлений о продаже квартир:
загрузить сырой датасет, провести валидацию, повторить ключевую очистку и
сформировать аналитические срезы по факторам цены.

## Данные

Источник: `real_estate_data.csv` (Hugging Face).

Ключевые поля:

- `last_price` - цена квартиры;
- `total_area`, `living_area`, `kitchen_area` - площади;
- `rooms`, `floor`, `floors_total` - характеристики планировки;
- `locality_name`, `city_centers_nearest` - локация;
- `first_day_exposition`, `days_exposition` - параметры публикации.

## Этапы пайплайна

1. `ingest`  
   Скачивает исходный CSV в `data/raw`.
2. `validate`  
   Проверяет наличие обязательных колонок и базовую целостность данных.
3. `build_dataset`  
   Повторяет шаги очистки и feature engineering из ноутбука:
   заполнение пропусков, нормализация признаков, фильтры выбросов, новые признаки.
4. `analyze`  
   Строит агрегаты и сохраняет итоговый аналитический отчет в JSON.

## Структура проекта

```text
real_estate/
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
└── real_estate.ipynb
```

## Быстрый запуск

```powershell
pip install -r requirements.txt
./run_pipeline.ps1
```

## Итоговые артефакты

- `data/validated/quality_report.json` - отчет по проверке сырого датасета.
- `data/processed/dataset.parquet` - очищенный датасет.
- `data/processed/feature_manifest.json` - описание признаков и размеров выборки.
- `artifacts/analysis_report.json` - итоговые агрегаты и факторы цены.
