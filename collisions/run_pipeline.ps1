param(
    [string]$DataUrl = "https://huggingface.co/datasets/onejetpilot/collisions/resolve/main/df_dtp.csv"
)

# Останавливаем пайплайн при первой ошибке.
$ErrorActionPreference = "Stop"

# Шаг 1: ingest (скачивание df_dtp.csv + manifest).
Write-Host "[pipeline] Step 1/4 ingest"
python .\src\ingest.py --data-url $DataUrl --output-dir data/raw
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Шаг 2: validate (проверка схемы и целевого признака).
Write-Host "[pipeline] Step 2/4 validate"
python .\src\validate.py --raw-csv data/raw/df_dtp.csv --output-dir data/validated
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Шаг 3: build_dataset (очистка, признаки, train/val).
Write-Host "[pipeline] Step 3/4 build_dataset"
python .\src\build_dataset.py --validated-csv data/validated/df_dtp_validated.csv --output-dir data/processed
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Шаг 4: train (CatBoost + метрики классификации).
Write-Host "[pipeline] Step 4/4 train"
python .\src\train.py --processed-dir data/processed --artifacts-dir artifacts
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[pipeline] Done. See artifacts/metrics.json"
