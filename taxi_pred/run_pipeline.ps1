param(
    [string]$DataUrl = "https://huggingface.co/datasets/onejetpilot/taxi/resolve/main/taxi.csv",
    [string]$LocalFile = ""
)

# Останавливаем пайплайн при первой ошибке.
$ErrorActionPreference = "Stop"

# Шаг 1: ingest (скачивание исходного CSV).
Write-Host "[pipeline] Step 1/4 ingest"
if ([string]::IsNullOrWhiteSpace($LocalFile)) {
    python .\src\ingest.py --data-url $DataUrl --output-dir data/raw
} else {
    python .\src\ingest.py --data-url $DataUrl --local-file $LocalFile --output-dir data/raw
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Шаг 2: validate (проверка схемы и индекса времени).
Write-Host "[pipeline] Step 2/4 validate"
python .\src\validate.py --raw-dir data/raw --output-dir data/validated
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Шаг 3: build_dataset (ресемплирование и лаговые признаки).
Write-Host "[pipeline] Step 3/4 build_dataset"
python .\src\build_dataset.py --validated-dir data/validated --output-dir data/processed
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Шаг 4: train (сравнение моделей и финальная оценка).
Write-Host "[pipeline] Step 4/4 train"
python .\src\train.py --processed-dir data/processed --artifacts-dir artifacts
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[pipeline] Done. See artifacts/metrics.json"
