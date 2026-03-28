#!/usr/bin/env python3
"""Test evaluation function."""

from pvoptix import create_virtual_stc_curve_double, evaluate_double_parameters
import numpy as np

print("=" * 50)
print("Testing evaluate_double_parameters")
print("=" * 50)

# Create a virtual STC curve
dataset = create_virtual_stc_curve_double()
print(f"\n1. Created virtual STC curve:")
print(f"   Points: {len(dataset['V'])}")
print(f"   G = {dataset['G']:.0f} W/m²")
print(f"   T = {dataset['T'] - 273.15:.1f} °C")

# Use the SAME parameters that generated the curve
params = {
    'Rs': 0.28,
    'Rsh': 3200.0,
    'I01': 6.5e-8,
    'I02': 1.2e-7,
    'Iph': 4.68,
    'n1': 1.3,
    'n2': 1.8
}

print(f"\n2. Parameters used to generate the curve:")
for k, v in params.items():
    print(f"   {k}: {v:.4e}")

# Evaluate - should be very close to 0
rmse = evaluate_double_parameters(params, [dataset], ns=36)
print(f"\n3. Evaluation result:")
print(f"   RMSE = {rmse:.10f}")

if rmse < 0.01:
    print("-- Perfect fit (RMSE very small)")
else:
    print("-- Not a perfect fit (unexpected)")

# Test with different parameters
params_diff = {
    'Rs': 0.30,
    'Rsh': 3000.0,
    'I01': 7.0e-8,
    'I02': 1.3e-7,
    'Iph': 4.70,
    'n1': 1.32,
    'n2': 1.82
}

print(f"\n4. Testing with different parameters:")
rmse2 = evaluate_double_parameters(params_diff, [dataset], ns=36)
print(f"   RMSE = {rmse2:.10f}")

if rmse2 > rmse:
    print("-- Different parameters give larger error (expected)")
else:
    print("-- Different parameters gave smaller error (unexpected)")

print("\n" + "=" * 50)
print("Test complete!")