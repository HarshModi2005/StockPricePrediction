#!/usr/bin/env python3
"""
Stock Prediction Installation and Setup Script
This script helps set up the environment and test real-time data capabilities.
"""

import subprocess
import sys
import os
import importlib.util
from typing import List, Tuple

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def check_package(package_name: str, import_name: str = None) -> bool:
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def install_requirements():
    """Install requirements from requirements.txt."""
    print("\n📦 Installing required packages...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def test_data_pipeline():
    """Test the real-time data pipeline."""
    print("\n🧪 Testing real-time data pipeline...")
    
    try:
        # Import and test basic functionality
        sys.path.append('scripts')
        from data_pipeline import StockDataPipeline
        
        pipeline = StockDataPipeline()
        
        # Test data fetching
        print("   Testing data fetch for AAPL...")
        data = pipeline.fetch_yahoo_finance("AAPL", period="5d")
        
        if data is not None and len(data) > 0:
            print(f"   ✅ Successfully fetched {len(data)} records")
            
            # Test real-time quote
            print("   Testing real-time quote...")
            quote = pipeline.fetch_real_time_quote("AAPL")
            
            if quote and quote.get('current_price'):
                print(f"   ✅ Real-time quote: ${quote['current_price']:.2f}")
                return True
            else:
                print("   ⚠️ Real-time quote unavailable (may be market hours)")
                return True  # Still consider successful
        else:
            print("   ❌ Failed to fetch data")
            return False
            
    except Exception as e:
        print(f"   ❌ Error testing pipeline: {e}")
        return False

def show_next_steps():
    """Show what to do next."""
    print("\n🚀 Next Steps:")
    print("   1. Try the comprehensive demo:")
    print("      python scripts/realtime_demo.py")
    print()
    print("   2. Run a real-time prediction:")
    print("      python scripts/realtime_predictor.py --symbol AAPL --model lstm")
    print()
    print("   3. Analyze intraday data:")
    print("      python scripts/intraday_realtime.py --symbol MSFT --interval 5m")
    print()
    print("   4. List available stock symbols:")
    print("      python scripts/realtime_demo.py --list-symbols")

def main():
    """Main installation and setup function."""
    print("🎯 Stock Prediction Setup & Installation")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check if we're in the right directory
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt not found")
        print("   Please run this script from the StockPricePrediction directory")
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Test critical packages
    critical_packages = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("yfinance", "yfinance"),
        ("tensorflow", "tensorflow"),
        ("sklearn", "sklearn"),
        ("matplotlib", "matplotlib")
    ]
    
    print("\n🔍 Checking critical packages...")
    missing_packages = []
    
    for package_name, import_name in critical_packages:
        if check_package(package_name, import_name):
            print(f"   ✅ {package_name}")
        else:
            print(f"   ❌ {package_name}")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("   Try: pip install " + " ".join(missing_packages))
        return False
    
    # Test data pipeline
    if not test_data_pipeline():
        print("\n⚠️ Data pipeline test failed")
        print("   You may still be able to use CSV-based models")
    else:
        print("\n✅ Real-time data pipeline working!")
    
    # Show next steps
    show_next_steps()
    
    print("\n🎉 Setup completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 