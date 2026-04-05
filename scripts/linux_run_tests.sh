#!/bin/bash
# scripts/linux_run_tests.sh
# PvOptiX - Run all tests

echo "========================================"
echo "PvOptiX - Running Tests"
echo "========================================"

# Activate virtual environment
source pvoptix_venv/bin/activate

# Run tests with coverage
pytest tests/ -v --cov=pvoptix --cov-report=html --cov-report=term

echo ""
echo "Coverage report generated in htmlcov/index.html"