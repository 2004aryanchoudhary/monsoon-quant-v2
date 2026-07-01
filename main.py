import subprocess
import sys
import argparse
import time

def run_script(script_name: str, description: str):
    """Executes a python script inside the active virtual environment and mirrors stdout."""
    print(f"\n=====================================================================")
    print(f" STARTING STAGE: {description}")
    print(f" Running: uv run {script_name}")
    print(f"=====================================================================")
    
    start_time = time.time()
    
    # Use sys.executable to ensure it run inside the identical uv virtual environment
    result = subprocess.run([sys.executable, script_name], capture_output=False)
    
    elapsed_time = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\nCRITICAL ERROR: {script_name} failed with exit code {result.returncode}.")
        sys.exit(result.returncode)
        
    print(f"STAGE COMPLETE: Finished in {elapsed_time:.2f} seconds.")

def main():
    parser = argparse.ArgumentParser(
        description="Monsoon Quant V2: End-to-End Multi-Asset Algorithmic Trading Pipeline"
    )
    
    # CLI Flags for execution optimization
    parser.add_argument(
        "--skip-download", 
        action="store_true", 
        help="Skip downloading raw market data and historical weather index files."
    )
    
    args = parser.parse_args()
    pipeline_start = time.time()
    
    print("Initializing Unified Monsoon Quant Pipeline execution context...")
    
    # Stage 1: Alternative Data & Financial Data Ingestion
    if not args.skip_download:
        run_script("weather_loader.py", "Meteorological Archive Aggregation")
        run_script("market_loader.py", "Financial Panel Market Data Ingestion")
    else:
        print("\n[INFO] --skip-download flag active. Skipping API data retrieval stages.")
        
    # Stage 2: Panel Feature Engineering and Normalization
    run_script("feature_engineer.py", "Cross-Sectional Feature Generation & Scaling")
    
    # Stage 3: Centralized Model Training
    run_script("train_classifier.py", "XGBoost Chronological Panel Classification")
    
    # Stage 4: Cross-Sectional Vectorized Backtesting
    run_script("backtest_strategy.py", "Portfolio Simulation with Capital Preservation Filter")
    
    total_pipeline_time = time.time() - pipeline_start
    print(f"\n=====================================================================")
    print(f" PIPELINE EXECUTION SUCCESSFUL")
    print(f" Full Data-to-Alpha Cycle Completed in {total_pipeline_time:.2f} seconds.")
    print(f"=====================================================================\n")

if __name__ == "__main__":
    main()