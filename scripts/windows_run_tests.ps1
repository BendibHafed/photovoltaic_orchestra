# scripts/windows_run_tests.ps1
# PvOptiX - Run all tests (Windows)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "PvOptiX - Running Tests" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Activate virtual environment
.\pvoptix_venv\Scripts\Activate.ps1

# Run tests with coverage
pytest tests/ -v --cov=pvoptix --cov-report=html --cov-report=term

Write-Host "`nCoverage report generated in htmlcov/index.html" -ForegroundColor Green