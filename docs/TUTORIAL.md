# PvOptiX Tutorial

## Step 1: Setup Development Environment

### Windows (PowerShell)
```powershell
# Clone the repository
git clone https://github.com/yourusername/photovoltaic_orchestra.git
cd photovoltaic_orchestra
```
# Run setup script
```powershell
.\scripts\setup_dev.ps1
```
### Linux/Mac
```powershell
# Clone the repository
git clone https://github.com/yourusername/photovoltaic_orchestra.git
cd photovoltaic_orchestra
```

# Make script executable and run
chmod +x scripts/setup_dev.sh
./scripts/setup_dev.sh
```
# Run all tests
```powershell
pytest tests/ -v
```
# Run with coverage
```powershell
pytest tests/ --cov=pvoptix --cov-report=html
```
# Run specific test
```powershell
pytest tests/test_double_diode.py -v
```