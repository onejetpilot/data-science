param(
    [string]$Geo0Url = "https://huggingface.co/datasets/onejetpilot/well_loc/resolve/main/geo_data_0.csv",
    [string]$Geo1Url = "https://huggingface.co/datasets/onejetpilot/well_loc/resolve/main/geo_data_1.csv",
    [string]$Geo2Url = "https://huggingface.co/datasets/onejetpilot/well_loc/resolve/main/geo_data_2.csv",
    [string]$LocalGeo0File = "",
    [string]$LocalGeo1File = "",
    [string]$LocalGeo2File = ""
)

$ErrorActionPreference = "Stop"

Write-Host "[pipeline] Step 1/4 ingest"
python .\src\ingest.py `
  --geo0-url $Geo0Url `
  --geo1-url $Geo1Url `
  --geo2-url $Geo2Url `
  --local-geo0-file $LocalGeo0File `
  --local-geo1-file $LocalGeo1File `
  --local-geo2-file $LocalGeo2File `
  --output-dir data/raw
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[pipeline] Step 2/4 validate"
python .\src\validate.py --raw-dir data/raw --output-dir data/validated
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[pipeline] Step 3/4 build_dataset"
python .\src\build_dataset.py --validated-dir data/validated --output-dir data/processed
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[pipeline] Step 4/4 analyze"
python .\src\analyze.py --processed-dir data/processed --artifacts-dir artifacts
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[pipeline] Done. See artifacts/analysis_report.json"
