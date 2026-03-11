param(
    [string]$DataUrl = "https://huggingface.co/datasets/onejetpilot/alloy_temp/resolve/main/alloy_temp.db"
)

# Если любой шаг падает, пайплайн сразу останавливается.
$ErrorActionPreference = "Stop"

# Шаг 1: ingest (скачивание sqlite БД + manifest).
Write-Host "[pipeline] Step 1/4 ingest"
python .\src\ingest.py --data-url $DataUrl --output-dir data/raw
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Шаг 2: validate (проверка структуры таблиц и качества данных).
Write-Host "[pipeline] Step 2/4 validate"
python .\src\validate.py --db-path data/raw/alloy_temp.db --output-dir data/validated
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Шаг 3: build_dataset (агрегация признаков + train/val выборки).
Write-Host "[pipeline] Step 3/4 build_dataset"
python .\src\build_dataset.py --db-path data/raw/alloy_temp.db --output-dir data/processed
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Шаг 4: train (обучение baseline модели + метрики).
Write-Host "[pipeline] Step 4/4 train"
python .\src\train.py --processed-dir data/processed --artifacts-dir artifacts
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[pipeline] Done. See artifacts/metrics.json"
