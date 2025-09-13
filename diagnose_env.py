#!/usr/bin/env python3
"""
Environment Diagnostic Script
Tests Python environment and dependencies
"""

import sys
import os

def test_import(module_name, description):
    """Test if a module can be imported"""
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {description}: {module_name} {version}")
        return True
    except ImportError as e:
        print(f"❌ {description}: {module_name} - ImportError: {e}")
        return False
    except Exception as e:
        print(f"❌ {description}: {module_name} - Error: {type(e).__name__}: {e}")
        return False

def main():
    print("🔍 Python Environment Diagnostic")
    print("=" * 50)

    # Basic Python info
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    print()

    # Test core dependencies
    print("📦 Testing Core Dependencies:")
    results = []

    results.append(test_import("numpy", "NumPy"))
    results.append(test_import("pandas", "Pandas"))
    results.append(test_import("scipy", "SciPy"))
    results.append(test_import("sklearn", "Scikit-learn"))
    results.append(test_import("tensorflow", "TensorFlow"))

    print()
    print("🎯 Results Summary:")
    working = sum(results)
    total = len(results)
    print(f"Working: {working}/{total} dependencies")

    if working == total:
        print("✅ All dependencies working!")
        print("🚀 Ready to test sophisticated strategies")
    else:
        print("❌ Some dependencies have issues")
        print("🔧 Need to fix environment before testing strategies")

    print()
    print("💡 Next Steps:")
    if working < total:
        print("1. Create clean virtual environment:")
        print("   python3 -m venv trading_env")
        print("   source trading_env/bin/activate")
        print("   pip install numpy pandas scipy scikit-learn")
        print("2. Test again: python3 diagnose_env.py")
    else:
        print("1. Test simple strategies: python3 backtests/quick_ma.py --mean-reversion")
        print("2. Test complex strategies if available")
        print("3. Validate with real data")

if __name__ == "__main__":
    main()
