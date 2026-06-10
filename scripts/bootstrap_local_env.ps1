param(
  [string]$PythonExe = "python",
  [string]$VenvDir = ".venv",
  [switch]$Recreate
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
  param(
    [string]$Label,
    [scriptblock]$Action
  )

  Write-Host "==> $Label"
  & $Action
  if ($LASTEXITCODE -ne 0) {
    throw "Step failed: $Label (exit code $LASTEXITCODE)"
  }
}

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
$ragkbRoot = Resolve-Path (Join-Path $repoRoot "..\llm-rag-knowledge-base")
$venvPath = Join-Path $repoRoot $VenvDir
$venvPython = Join-Path $venvPath "Scripts\python.exe"

if ($Recreate -or -not (Test-VenvUsable $venvPython)) {
  if (Test-Path $venvPath) {
    Write-Host "Removing unusable virtual environment:" $venvPath
    Remove-Item -Recurse -Force $venvPath
  }

  Invoke-Step "Create virtual environment" { & $PythonExe -m venv $venvPath }
}

Invoke-Step "Upgrade pip" { & $venvPython -m pip --disable-pip-version-check install --upgrade pip setuptools wheel --quiet }
Invoke-Step "Install ragkb editable" { & $venvPython -m pip --disable-pip-version-check install --progress-bar off --quiet -e $ragkbRoot }
Invoke-Step "Install benchmark editable" { & $venvPython -m pip --disable-pip-version-check install --progress-bar off --quiet -e $repoRoot pytest }

Write-Host "Created virtual environment:" $venvPath
Write-Host "Installed editable packages:"
Write-Host "  -" $ragkbRoot
Write-Host "  -" $repoRoot
Write-Host "Use this interpreter:"
Write-Host "  $venvPython"
