param(
    [string]$TrainUrl = "https://huggingface.co/datasets/onejetpilot/start_up/resolve/main/kaggle_startups_train_28062024.csv",
    [string]$TestUrl = "https://huggingface.co/datasets/onejetpilot/start_up/resolve/main/kaggle_startups_test_28062024.csv",
    [string]$CitiesUrl = "https://huggingface.co/datasets/onejetpilot/start_up/resolve/main/worldcitiespop.csv",
    [string]$LocalTrainFile = "",
    [string]$LocalTestFile = "",
    [string]$LocalCitiesFile = ""
)

# Останавливаем пайплайн при первой ошибке.
$ErrorActionPreference = "Stop"

# Шаг 1: ingest (скачивание исходных CSV).
Write-Host "[pipeline] Step 1/4 ingest"
python .\src\ingest.py `
  --train-url $TrainUrl `
  --test-url $TestUrl `
  --cities-url $CitiesUrl `
  --local-train-file $LocalTrainFile `
  --local-test-file $LocalTestFile `
  --local-cities-file $LocalCitiesFile `
  --output-dir data/raw
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Шаг 2: validate (проверка схемы и базового качества).
Write-Host "[pipeline] Step 2/4 validate"
python .\src\validate.py --raw-dir data/raw --output-dir data/validated
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Шаг 3: build_dataset (feature engineering и подготовка train/test).
Write-Host "[pipeline] Step 3/4 build_dataset"
python .\src\build_dataset.py --validated-dir data/validated --output-dir data/processed
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Шаг 4: train (подбор модели и предсказания для test).
Write-Host "[pipeline] Step 4/4 train"
python .\src\train.py --processed-dir data/processed --artifacts-dir artifacts
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[pipeline] Done. See artifacts/metrics.json and artifacts/submit_predictions.csv"
