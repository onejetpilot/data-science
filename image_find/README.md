# Поиск изображения по текстовому запросу

## Используемый стек

![Python](https://img.shields.io/badge/-Python-blue)
![Pandas](https://img.shields.io/badge/-Pandas-blue)
![NumPy](https://img.shields.io/badge/-NumPy-yellow)
![scikit--learn](https://img.shields.io/badge/-scikit--learn-orange)
![Pillow](https://img.shields.io/badge/-Pillow-green)
![SentenceTransformers](https://img.shields.io/badge/-SentenceTransformers-darkgreen)
![PyTorch](https://img.shields.io/badge/-PyTorch-red)
![TorchVision](https://img.shields.io/badge/-TorchVision-darkred)
![Joblib](https://img.shields.io/badge/-Joblib-teal)
![PyArrow](https://img.shields.io/badge/-PyArrow-lightgrey)
![PowerShell](https://img.shields.io/badge/-PowerShell-5391FE)

## Цель проекта

Построить воспроизводимый пайплайн для оценки соответствия пары
`текстовый запрос -> изображение`: от загрузки исходных таблиц и
формирования target до обучения baseline-модели и сохранения метрик.

## Данные

Источник данных: архив `find_image.zip` (распаковывается в `data/raw/find_image`).

Используемые таблицы:

- `train_dataset.csv` - пары `image`, `query_id`, `query_text`.
- `ExpertAnnotations.tsv` - экспертные оценки (`rate_1`, `rate_2`, `rate_3`).
- `CrowdAnnotations.tsv` - крауд-оценки (`share_confirmed` и счетчики подтверждений/отклонений).

Целевая метка:

- `target` строится как агрегат экспертных и крауд-оценок.
- `target_bin = 1`, если `target >= 0.5`, иначе `0`.

## Этапы пайплайна

1. `ingest`  
   Скачивает `find_image.zip`, распаковывает нужные файлы и создает `manifest.json`.
2. `validate`  
   Проверяет наличие обязательных файлов/колонок и базовую целостность оценок.
3. `build_dataset`  
   Объединяет train + expert + crowd, строит target и делит данные на train/val.
4. `train`  
   Обучает `MLP` на признаках `SBERT(query_text) + ResNet50(image)` и считает  
   `ROC-AUC`, `F1`, `Precision`, `Recall`.

## Структура проекта

```text
image_find/
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
└── image_find.ipynb
```

## Быстрый запуск

```powershell
pip install -r requirements.txt
./run_pipeline.ps1
```

Если нужен другой источник архива:

```powershell
./run_pipeline.ps1 -DataUrl "https://your-url/find_image.zip"
```

## Итоговые артефакты

- `data/validated/quality_report.json` - отчет по проверкам данных.
- `data/processed/train.parquet` и `data/processed/val.parquet` - подготовленные выборки.
- `data/processed/feature_manifest.json` - описание target и признаков.
- `artifacts/model.joblib` - обученная MLP-модель (state dict) и параметры признаков.
- `artifacts/metrics.json` - метрики запуска (`ROC-AUC`, `F1`, `Precision`, `Recall`).
