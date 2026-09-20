param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    Write-Host 'First run: execute scripts\setup.ps1 to install dependencies.'
    exit 1
}
Write-Host "Golos audiobook studio: http://127.0.0.1:$Port"
& $pythonPath -m uvicorn app.main:app --host 127.0.0.1 --port $Port --workers 1
