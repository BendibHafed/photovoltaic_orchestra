# Photovoltaic Orchestra

[![CI](https://github.com/yourusername/photovoltaic_orchestra/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/photovoltaic_orchestra/actions/workflows/ci.yml)
[![CD](https://github.com/yourusername/photovoltaic_orchestra/actions/workflows/cd.yml/badge.svg)](https://github.com/yourusername/photovoltaic_orchestra/actions/workflows/cd.yml)
[![PyPI version](https://badge.fury.io/py/pvoptix.svg)](https://badge.fury.io/py/pvoptix)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Photovoltaic Orchestra** is a DevOps-driven mono-repo for optimizing photovoltaic (PV) model parameters using measured I–V curves. The scientific engine is **pvoptix** (Python double-diode model with progressive multi-condition optimization).

## Features

- ✅ **Double-diode model** (7 parameters) with physics-based translation equations
- ✅ **Progressive optimization** – accumulate scans throughout the day to find true STC parameters
- ✅ **Multi-condition optimization** – simultaneous fitting across multiple irradiance/temperature conditions
- ✅ **Genetic Algorithm** with adaptive operators for global optimization
- ✅ **Backward compatible** with single-diode model
- ✅ **CI/CD ready** with GitHub Actions, pytest, ruff, mypy

## Repository Structure
photovoltaic_orchestra/
├── pvoptix/ # Scientific engine (Python package)
├── pv_backend/ # FastAPI backend (WIP)
├── pv_frontend/ # React frontend (WIP)
├── worker_optimizer/ # Background jobs (planned)
├── infra/ # Infrastructure (planned)
├── scripts/ # Utility scripts
├── tests/ # Unit tests
├── examples/ # Example usage
└── docs/ # Documentation
