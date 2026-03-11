param(
    [string]$DatasetDir = "datasets/faces"
)

# Если любой шаг завершится с ошибкой, пайплайн останавливается.
$ErrorActionPreference = "Stop"

# Шаг 1: ingest (raw слой + manifest).
Write-Host "[pipeline] Step 1/4 ingest"
python .\src\ingest.py --source-dir $DatasetDir --output-dir data/raw
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Шаг 2: validate (quality checks + quality_report.json).
Write-Host "[pipeline] Step 2/4 validate"
python .\src\validate.py --raw-labels data/raw/labels.csv --images-dir "$DatasetDir/final_files" --output-dir data/validated
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Шаг 3: build_dataset (табличные признаки + train/val).
Write-Host "[pipeline] Step 3/4 build_dataset"
python .\src\build_dataset.py --validated-labels data/validated/labels_validated.csv --images-dir "$DatasetDir/final_files" --output-dir data/processed
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Шаг 4: train (baseline-модель + metrics.json).
Write-Host "[pipeline] Step 4/4 train"
python .\src\train.py --processed-dir data/processed --artifacts-dir artifacts
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Финальное сообщение с ключевым артефактом.
Write-Host "[pipeline] Done. See artifacts/metrics.json"
