#!/usr/bin/env python3
"""Debug the evaluation chain."""

import numpy as np
from pvoptix.pvoptix.optimization.ga.genome_mapping_double import encode_individual_double, decode_individual_double
from pvoptix.pvoptix.objectives.double import pv_rmse_objective_double
from pvoptix import create_virtual_stc_curve_double

print("=" * 50)
print("Debugging evaluation chain")
print("=" * 50)

# Create virtual curve
dataset = create_virtual_stc_curve_double()
print(f"\n1. Virtual curve created: {len(dataset['V'])} points")

# Parameters
params = {
    'Rs': 0.28,
    'Rsh': 3200.0,
    'I01': 6.5e-8,
    'I02': 1.2e-7,
    'Iph': 4.68,
    'n1': 1.3,
    'n2': 1.8
}
print(f"\n2. Parameters: {params}")

# Encode
individual = encode_individual_double(params)
print(f"\n3. Encoded individual (length {len(individual)}):")
print(f"   {individual}")

# Decode back
decoded = decode_individual_double(individual)
print(f"\n4. Decoded parameters:")
for k, v in decoded.items():
    print(f"   {k}: {v:.6e}")

# Check if decode matches original
print(f"\n5. Check if decode matches original:")
for k in params:
    if abs(params[k] - decoded[k]) / max(abs(params[k]), 1e-12) > 1e-6:
        print(f"   {k}: mismatch! {params[k]:.6e} vs {decoded[k]:.6e}")
    else:
        print(f"   {k}: OK")

# Evaluate
rmse = pv_rmse_objective_double(individual, [dataset], 36)
print(f"\n6. RMSE: {rmse:.10f}")

print("\n" + "=" * 50)