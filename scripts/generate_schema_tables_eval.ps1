param(
  [string]$SchemaDir = "",
  [string]$OutputPath = "<local_path_removed>"
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot
$env:PYTHONPATH = $repoRoot.Path

if (-not $SchemaDir) {
  if ($env:SCHEMA_DIR) {
    $SchemaDir = $env:SCHEMA_DIR
  } else {
    throw "SchemaDir is required."
  }
}

$env:SCHEMA_DIR = (Resolve-Path $SchemaDir).Path
$outputParent = Split-Path -Parent $OutputPath
if ($outputParent) {
  New-Item -ItemType Directory -Force -Path $outputParent | Out-Null
}
$env:OUTPUT_PATH = $OutputPath
& ".\.venv\Scripts\python.exe" (Join-Path $PSScriptRoot 'generate_schema_tables_eval.py')
