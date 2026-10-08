$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$mode = if ($args.Count -gt 0) { $args[0] } else { "verify" }
python (Join-Path $repo "scripts/reproduce_project.py") --project tds --mode $mode
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
