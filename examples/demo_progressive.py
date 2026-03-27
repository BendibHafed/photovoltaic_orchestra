# examples/demo_progressive.py
from pvoptix import optimize_double_progressive
import numpy as np

def generate_scans():
    """Simulate hardware scans every 10 minutes."""
    for hour in range(8, 17):
        for minute in range(0, 60, 10):
            # Simulate realistic conditions
            G = 200 + 800 * np.sin((hour - 8) * np.pi / 9)
            T = 20 + 15 * np.sin((hour - 8) * np.pi / 9) + 273.15
            
            # Generate synthetic I-V curve
            V = np.linspace(0, 21.6, 50)
            I = 4.68 * (1 - V/21.6) + np.random.randn(50) * 0.02
            
            yield {
                'V': V, 'I': I,
                'T': T, 'G': G
            }

# Run progressive optimization
result = optimize_double_progressive(
    scan_stream=generate_scans(),
    ns=36,
    pop_size=60,
    generations=80,
    verbose=True
)

print(f"Best parameters: {result.best_params}")
print(f"Final RMSE: {result.best_fitness:.6f}")