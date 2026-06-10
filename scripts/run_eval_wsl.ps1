param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

function Convert-ToWslPath {
  param([string]$Path)

  $resolved = (Resolve-Path $Path).Path
  if ($resolved -match '^(?<drive>[A-Za-z]):\\(?<rest>.*)$') {
    $drive = $Matches['drive'].ToLowerInvariant()
    $rest = ($Matches['rest'] -replace '\\', '/')
    return "/mnt/$drive/$rest"
  }
  throw "Could not convert path to WSL path: $resolved"
}

$scriptPath = Convert-ToWslPath (Join-Path $PSScriptRoot 'run_eval_wsl.sh')
if ($ExtraArgs.Count -gt 0) {
  $quotedArgs = ($ExtraArgs | ForEach-Object { "'$_'" }) -join ' '
  wsl bash -lc "bash '$scriptPath' $quotedArgs"
} else {
  wsl bash -lc "bash '$scriptPath'"
}
