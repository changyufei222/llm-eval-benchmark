$ErrorActionPreference = 'Stop'

function Assert-Contains {
    param(
        [string]$Path,
        [string]$Pattern,
        [string]$Message
    )

    $content = Get-Content -Raw $Path
    if ($content -notmatch $Pattern) {
        throw "$Message`nPath: $Path`nExpected pattern: $Pattern"
    }
}

$scriptsDir = Split-Path -Parent $PSScriptRoot
$bootstrapScript = Join-Path $scriptsDir 'bootstrap_local_env.ps1'
$runEvalScript = Join-Path $scriptsDir 'run_eval.ps1'
$runSweepScript = Join-Path $scriptsDir 'run_model_sweep.ps1'

Assert-Contains $bootstrapScript 'function\s+Test-VenvUsable' 'Bootstrap script should define Test-VenvUsable.'
Assert-Contains $runEvalScript 'Test-VenvUsable' 'run_eval.ps1 should verify that the local venv is usable, not just present.'
Assert-Contains $runSweepScript 'Test-VenvUsable' 'run_model_sweep.ps1 should verify that the local venv is usable, not just present.'

Write-Host 'bootstrap_script_tests: PASS'
