import pandas as pd
import yfinance as yf
from datetime import datetime

def fetch_market_data(ticker: str, benchmark: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical daily close prices and volumes for a target ticker
    and a benchmark index, aligning them into a single clean DataFrame.
    """
    print(f"Downloading historical market data from {start_date} to {end_date}...")
    
    # Download target asset and benchmark data
    target_df = yf.download(ticker, start=start_date, end=end_date)
    benchmark_df = yf.download(benchmark, start=start_date, end=end_date)
    
    # Flatten multi-index columns if present in recent yfinance versions
    if isinstance(target_df.columns, pd.MultiIndex):
        target_df.columns = target_df.columns.get_level_values(0)
    if isinstance(benchmark_df.columns, pd.MultiIndex):
        benchmark_df.columns = benchmark_df.columns.get_level_values(0)
        
    # Extract clean target metrics
    df = pd.DataFrame(index=target_df.index)
    df['target_close'] = target_df['Close']
    df['target_volume'] = target_df['Volume']
    
    # Extract benchmark close and align timelines via inner join
    df = df.join(benchmark_df['Close'].rename('benchmark_close'), how='inner')
    
    # Drop rows with missing values to ensure numerical stability
    df.dropna(inplace=True)
    
    # Calculate simple daily percentage returns for baseline analysis
    df['target_return'] = df['target_close'].pct_change()
    df['benchmark_return'] = df['benchmark_close'].pct_change()
    
    # First row will contain NaN returns due to pct_change(); drop it
    df.dropna(inplace=True)
    
    print(f"Data ingestion successful. Total trading rows aligned: {len(df)}")
    return df

if __name__ == "__main__":
    # Define parameters
    TARGET_TICKER = "KSCL.NS"
    BENCHMARK_INDEX = "^NSEI"  # NIFTY 50 Index
    START = "2018-01-01"
    END = datetime.today().strftime('%Y-%m-%d')
    
    try:
        market_data = fetch_market_data(TARGET_TICKER, BENCHMARK_INDEX, START, END)
        
        # Save directly to the raw data folder you created
        output_filename = "data/raw/raw_market_data.csv"
        market_data.to_csv(output_filename)
        
        print(f"Saved synchronized market baseline to {output_filename}")
        print("\nFirst 5 rows of ingested data:")
        print(market_data.head())
        
    except Exception as e:
        print(f"Execution failed: {e}")