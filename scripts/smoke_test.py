#!/usr/bin/env python3
"""Quick smoke test to verify installation."""

import sys


def test_imports():
    """Test that all core imports work."""
    print("Testing imports...")
    
    try:
        import pvoptix
        print(f" pvoptix version: {pvoptix.__version__}")
        
        # Test double-diode functions
        assert hasattr(pvoptix, "optimize_double_multicondition")
        assert hasattr(pvoptix, "optimize_double_progressive")
        assert hasattr(pvoptix, "simulate_iv_curve_double")
        print(" Double-diode functions available")
        
        # Test legacy functions
        assert hasattr(pvoptix, "load_datasets_from_dir")
        assert hasattr(pvoptix, "OptimizationResult")
        print(" Legacy functions available")
        
        # Test analysis functions
        assert hasattr(pvoptix, "compute_power")
        assert hasattr(pvoptix, "find_mpp")
        print(" Analysis functions available")
        
        print("\n All imports successful!")
        return True
        
    except Exception as e:
        print(f"\n Import failed: {e}")
        return False


def test_basic_usage():
    """Test basic usage with synthetic data."""
    print("\nTesting basic usage...")
    
    try:
        from pvoptix import create_virtual_stc_curve_double, evaluate_double_parameters
        
        # Create a virtual STC curve
        dataset = create_virtual_stc_curve_double()
        print(f" Created virtual STC curve: {len(dataset['V'])} points")
        
        # Test evaluation (dummy parameters)
        dummy_params = {
            "Rs": 0.28, "Rsh": 3200.0,
            "I01": 6.5e-8, "I02": 1.2e-7,
            "Iph": 4.68, "n1": 1.3, "n2": 1.8
        }
        
        rmse = evaluate_double_parameters(dummy_params, [dataset], ns=36)
        print(f" Evaluation works (RMSE: {rmse:.6f})")
        
        print("\n Basic usage test passed!")
        return True
        
    except Exception as e:
        print(f"\n Basic usage test failed: {e}")
        return False


def main():
    print("=" * 50)
    print("PvOptiX - Smoke Test")
    print("=" * 50)
    
    success = test_imports() and test_basic_usage()
    
    print("\n" + "=" * 50)
    if success:
        print("SMOKE TEST PASSED")
        print("=" * 50)
        sys.exit(0)
    else:
        print("SMOKE TEST FAILED")
        print("=" * 50)
        sys.exit(1)


if __name__ == "__main__":
    main()