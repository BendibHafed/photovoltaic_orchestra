# scripts/wiows_setup_dev.ps1
# PvOptiX - Development Environment Setup (Windows)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "PvOptiX - Development Environment Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`n[1] Creating virtual environment..." -ForegroundColor Yellow
python -m venv pvoptix_venv

Write-Host "`n[2] Activating virtual environment..." -ForegroundColor Yellow
.\pvoptix_venv\Scripts\Activate.ps1

Write-Host "`n[3] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

Write-Host "`n[4] Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt
pip install -r requirements-dev.txt

Write-Host "`n[5] Installing pvoptix in editable mode..." -ForegroundColor Yellow
pip install -e .

Write-Host "`n[6] Running smoke test..." -ForegroundColor Yellow
python scripts/smoke_test.py

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "`nTo activate the environment, run:" -ForegroundColor White
Write-Host "  .\pvoptix_venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "`nTo deactivate:" -ForegroundColor White
Write-Host "  deactivate" -ForegroundColor White