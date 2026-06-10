param(
  [string]$Cwd = (Get-Location).Path,
  [switch]$Bootstrap
)

$ErrorActionPreference = "Stop"

Write-Host "Running eval in $Cwd"

function Test-VenvUsable {
  param([string]$PythonPath)

  if (-not (Test-Path $PythonPath)) {
    return $false
  }

  try {
    & $PythonPath -c "import sys; print(sys.executable)" *> $null
  } catch {
    return $false
  }
  return $LASTEXITCODE -eq 0
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$bootstrapScript = Join-Path $PSScriptRoot "bootstrap_local_env.ps1"

if ($Bootstrap -or -not (Test-VenvUsable $venvPython)) {
  & $bootstrapScript
  if ($LASTEXITCODE -ne 0) {
    throw "Bootstrap failed with exit code $LASTEXITCODE"
  }
}

# promptfoo (requires node)
if (Get-Command npx -ErrorAction SilentlyContinue) {
  npx promptfoo eval -c configs/promptfoo.yaml
} else {
  Write-Host "npx not found, skipping promptfoo."
}

# ragas
& $venvPython -m pipelines.rag_pipeline --data-path data/fbtp_eval.jsonl --output-dir reports/latest
if ($LASTEXITCODE -ne 0) {
  throw "rag_pipeline failed with exit code $LASTEXITCODE"
}

# comparison
& $venvPython -m pipelines.compare --data-path data/fbtp_eval.jsonl --output-dir reports/latest
if ($LASTEXITCODE -ne 0) {
  throw "compare failed with exit code $LASTEXITCODE"
}
