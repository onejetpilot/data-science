param(
    [string]$DataUrl = "https://huggingface.co/datasets/onejetpilot/find_image/resolve/main/find_image.zip"
)

# Останавливаем пайплайн при первой ошибке.
$ErrorActionPreference = "Stop"

# Шаг 1: ingest (скачивание и распаковка архива).
Write-Host "[pipeline] Step 1/4 ingest"
python .\src\ingest.py --data-url $DataUrl --output-dir data/raw
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Шаг 2: validate (проверка таблиц train/expert/crowd).
Write-Host "[pipeline] Step 2/4 validate"
python .\src\validate.py --raw-dir data/raw/find_image/to_upload --output-dir data/validated
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Шаг 3: build_dataset (таргет + train/val split).
Write-Host "[pipeline] Step 3/4 build_dataset"
python .\src\build_dataset.py --validated-dir data/validated --output-dir data/processed
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Шаг 4: train (логистическая модель по тексту и image-id).
Write-Host "[pipeline] Step 4/4 train"
python .\src\train.py --processed-dir data/processed --artifacts-dir artifacts
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[pipeline] Done. See artifacts/metrics.json"
