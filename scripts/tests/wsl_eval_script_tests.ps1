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
$wslScript = Join-Path $scriptsDir 'run_eval_wsl.sh'
$wslPs = Join-Path $scriptsDir 'run_eval_wsl.ps1'
$bootstrapScript = Join-Path $scriptsDir 'bootstrap_wsl_env.sh'

Assert-Contains $wslScript '\.venv_wsl' 'WSL eval script should use a project-local .venv_wsl runtime.'
Assert-Contains $wslScript 'PGHOST=127\.0\.0\.1' 'WSL eval script should use PostgreSQL through the local WSL host.'
Assert-Contains $wslScript '--eval-mode ragas' 'WSL eval script should force ragas compare mode.'
Assert-Contains $wslScript 'bootstrap_wsl_env\.sh' 'WSL eval script should call the dedicated bootstrap_wsl_env.sh helper.'
Assert-Contains $wslPs 'run_eval_wsl\.sh' 'PowerShell wrapper should call the WSL eval shell script.'

if (-not (Test-Path $bootstrapScript)) {
    throw "Missing bootstrap script: $bootstrapScript"
}

Write-Host 'wsl_eval_script_tests: PASS'
