param(
  [switch]$Bootstrap,
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

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

& $venvPython -m pipelines.model_sweep @ExtraArgs
if ($LASTEXITCODE -ne 0) {
  throw "model sweep failed with exit code $LASTEXITCODE"
}
