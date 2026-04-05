#!/bin/bash
# scripts/linux_setup_dev.sh
# PvOptiX - Development Environment Setup (Linux/Mac)

echo "========================================"
echo "PvOptiX - Development Environment Setup"
echo "========================================"

echo ""
echo "[1] Creating virtual environment..."
python3 -m venv pvoptix_venv

echo ""
echo "[2] Activating virtual environment..."
source pvoptix_venv/bin/activate

echo ""
echo "[3] Upgrading pip..."
pip install --upgrade pip

echo ""
echo "[4] Installing dependencies..."
pip install -r requirements.txt
pip install -r requirements-dev.txt

echo ""
echo "[5] Installing pvoptix in editable mode..."
pip install -e .

echo ""
echo "[6] Running smoke test..."
python scripts/smoke_test.py

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "To activate the environment, run:"
echo "  source pvoptix_venv/bin/activate"
echo ""
echo "To deactivate:"
echo "  deactivate"