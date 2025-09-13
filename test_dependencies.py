#!/usr/bin/env python3
"""
Dependency Reality Check - Tests critical dependencies without safety nets
"""

import importlib
import sys

CRITICAL_DEPS = [
    'pandas', 'numpy', 'scipy', 'sklearn', 'xgboost', 'lightgbm',
    'tensorflow', 'torch', 'ta', 'yfinance', 'backtrader',
    'pymongo', 'redis', 'fastapi', 'websockets'
]

SECONDARY_DEPS = [
    'matplotlib', 'seaborn', 'plotly', 'dash', 'streamlit',
    'requests', 'beautifulsoup4', 'lxml', 'selenium',
    'sqlalchemy', 'psycopg2', 'mysqlclient',
    'celery', 'rq', 'apscheduler'
]

working = []
missing = []
broken = []
versions = []

print("🔍 DEPENDENCY REALITY CHECK")
print("=" * 50)

def test_dependency(dep):
    """Test a single dependency"""
    try:
        mod = importlib.import_module(dep)
        version = getattr(mod, '__version__', 'unknown')
        working.append(dep)
        versions.append(f"{dep} ({version})")
        return True
    except ImportError:
        missing.append(dep)
        return False
    except Exception as e:
        broken.append(f"{dep}: {str(e)[:50]}...")
        return False

print("\n📦 TESTING CRITICAL DEPENDENCIES:")
for dep in CRITICAL_DEPS:
    print("2d", end='', flush=True)
    success = test_dependency(dep)
    status = "✅" if success else "❌"
    print(f" {status}")

print("\n📦 TESTING SECONDARY DEPENDENCIES:")
for dep in SECONDARY_DEPS:
    print("2d", end='', flush=True)
    success = test_dependency(dep)
    status = "✅" if success else "❌"
    print(f" {status}")

print("\n" + "=" * 50)
print("📊 RESULTS SUMMARY:")
print(f"✅ Working ({len(working)}):")
for i in range(0, len(versions), 3):
    print(f"   {', '.join(versions[i:i+3])}")

print(f"\n❌ Missing ({len(missing)}):")
for i in range(0, len(missing), 5):
    print(f"   {', '.join(missing[i:i+5])}")

print(f"\n🔥 Broken ({len(broken)}):")
for item in broken:
    print(f"   {item}")

print(f"\n📈 SUCCESS RATE: {len(working)}/{len(working) + len(missing) + len(broken)} ({len(working)/(len(working) + len(missing) + len(broken))*100:.1f}%)")

# Test environment info
print("
🖥️  ENVIRONMENT INFO:"    print(f"   Python: {sys.version}")
print(f"   Platform: {sys.platform}")

try:
    import pkg_resources
    installed = list(pkg_resources.working_set)
    print(f"   Total installed packages: {len(installed)}")
except:
    print("   Could not count installed packages")

print("\n" + "=" * 50)
