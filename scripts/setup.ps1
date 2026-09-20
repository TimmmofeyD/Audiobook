param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
& $Python -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Python 3.12+ is required. Pass -Python with the executable path.' }
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-dev.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
if (-not (Test-Path -LiteralPath '.env')) { Copy-Item -LiteralPath '.env.example' -Destination '.env' }
Write-Host 'Ready. Run .\start.ps1'
