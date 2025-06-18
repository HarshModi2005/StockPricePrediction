"""
Run All Stock Prediction Models
This utility script runs all available prediction models and provides
a comprehensive comparison of their performance.
"""

import os
import sys
import subprocess
import time
from datetime import datetime
import pandas as pd

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'sklearn', 
        'tensorflow', 'keras'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    return True

def check_data_files():
    """Check if required data files exist."""
    required_files = [
        '../data/HistoricalData.csv',
        '../data/stockdata.csv', 
        '../data/nifty.csv',
        '../data/data.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("Missing data files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease ensure all data files are in the ../data/ directory")
        print("See ../data/README.md for instructions on obtaining data")
        return False
    
    return True

def run_script(script_name, description):
    """Run a Python script and capture its output."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the script and capture output
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ SUCCESS ({duration:.1f}s)")
            if result.stdout:
                print("Output:")
                print(result.stdout[-500:])  # Last 500 characters
        else:
            print(f"❌ FAILED ({duration:.1f}s)")
            print("Error:")
            print(result.stderr)
        
        return {
            'script': script_name,
            'description': description,
            'success': result.returncode == 0,
            'duration': duration,
            'output': result.stdout,
            'error': result.stderr
        }
        
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT (>5 minutes)")
        return {
            'script': script_name,
            'description': description,
            'success': False,
            'duration': 300,
            'output': '',
            'error': 'Script timed out after 5 minutes'
        }
    except Exception as e:
        print(f"💥 ERROR: {e}")
        return {
            'script': script_name,
            'description': description,
            'success': False,
            'duration': 0,
            'output': '',
            'error': str(e)
        }

def generate_report(results):
    """Generate a summary report of all model runs."""
    print(f"\n{'='*80}")
    print("STOCK PREDICTION MODELS - EXECUTION REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    total_scripts = len(results)
    successful_scripts = sum(1 for r in results if r['success'])
    total_time = sum(r['duration'] for r in results)
    
    print(f"\nSUMMARY:")
    print(f"  Total Scripts: {total_scripts}")
    print(f"  Successful: {successful_scripts}")
    print(f"  Failed: {total_scripts - successful_scripts}")
    print(f"  Total Time: {total_time:.1f} seconds")
    
    print(f"\nDETAILED RESULTS:")
    print(f"{'Script':<35} {'Status':<10} {'Duration':<10} {'Description'}")
    print("-" * 80)
    
    for result in results:
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        duration = f"{result['duration']:.1f}s"
        description = result['description'][:30] + "..." if len(result['description']) > 30 else result['description']
        
        print(f"{result['script']:<35} {status:<10} {duration:<10} {description}")
    
    # Show failed scripts details
    failed_scripts = [r for r in results if not r['success']]
    if failed_scripts:
        print(f"\nFAILED SCRIPTS DETAILS:")
        for result in failed_scripts:
            print(f"\n{result['script']}:")
            print(f"  Error: {result['error'][:200]}...")
    
    print(f"\n{'='*80}")

def main():
    """Main execution function."""
    print("Stock Prediction Models - Batch Runner")
    print("=" * 50)
    
    # Check prerequisites
    print("Checking dependencies...")
    if not check_dependencies():
        return
    
    print("Checking data files...")
    if not check_data_files():
        return
    
    print("All prerequisites met! Starting model execution...")
    
    # Define scripts to run
    scripts_to_run = [
        {
            'script': 'prediction_microsoft_allfeatures.py',
            'description': 'Microsoft LSTM with all features (OHLCV)'
        },
        {
            'script': 'prediction_microsoft_lessfeatures.py', 
            'description': 'Microsoft LSTM with reduced features (OHL)'
        },
        {
            'script': 'prediction_snp500.py',
            'description': 'S&P 500 prediction'
        },
        {
            'script': 'nifty_prediction_open.py',
            'description': 'NIFTY opening price prediction'
        },
        {
            'script': 'next_day_open_microsoft.py',
            'description': 'Microsoft next-day opening prediction'
        },
        {
            'script': 'intraday.py',
            'description': 'Intraday analysis and comparison'
        }
    ]
    
    # Run all scripts
    results = []
    for script_info in scripts_to_run:
        if os.path.exists(script_info['script']):
            result = run_script(script_info['script'], script_info['description'])
            results.append(result)
        else:
            print(f"⚠️  Script not found: {script_info['script']}")
            results.append({
                'script': script_info['script'],
                'description': script_info['description'],
                'success': False,
                'duration': 0,
                'output': '',
                'error': 'Script file not found'
            })
    
    # Generate final report
    generate_report(results)
    
    # Save results to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f'model_execution_report_{timestamp}.txt'
    
    with open(report_file, 'w') as f:
        f.write("Stock Prediction Models - Execution Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        for result in results:
            f.write(f"Script: {result['script']}\n")
            f.write(f"Description: {result['description']}\n")
            f.write(f"Success: {result['success']}\n")
            f.write(f"Duration: {result['duration']:.1f}s\n")
            if result['error']:
                f.write(f"Error: {result['error']}\n")
            f.write("\n" + "-" * 40 + "\n\n")
    
    print(f"\nDetailed report saved to: {report_file}")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    successful_scripts = [r for r in results if r['success']]
    if successful_scripts:
        fastest_script = min(successful_scripts, key=lambda x: x['duration'])
        print(f"  - Fastest model: {fastest_script['script']} ({fastest_script['duration']:.1f}s)")
        
        print(f"  - For quick testing: Use {fastest_script['script']}")
        print(f"  - For comprehensive analysis: Compare all successful models")
        print(f"  - For visualization: Run plot.py with different datasets")
    
    print(f"\nNext steps:")
    print(f"  1. Review individual model outputs and performance metrics")
    print(f"  2. Use plot.py to visualize results")
    print(f"  3. Compare R² scores and RMSE values between models")
    print(f"  4. Consider ensemble methods combining multiple approaches")

if __name__ == "__main__":
    main() 