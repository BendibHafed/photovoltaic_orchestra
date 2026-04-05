# PvOptiX - PV Optimization Engine

Scientific library for photovoltaic module parameter extraction using double-diode model with progressive multi-condition optimization.

## Installation

```bash
pip install pvoptix
```
## Quick Start

```bash
from pvoptix import optimize_double_progressive

# Our Hardware interface provides scans
scan_stream = get_measurements()  # Iterator of (V, I, T, G)

result = optimize_double_progressive(scan_stream, ns=36)
print(result.best_params)
```