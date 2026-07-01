import pandas as pd
import yfinance as tf
from datetime import datetime

def download_sector_data(tickers: list, benchmark: str, start_date: str, end_date: str):
    print(f"Downloading market data from {start_date} to {end_date}...")
    
    # 1. Fetch benchmark data safely
    bench_raw = tf.download(benchmark, start=start_date, end=end_date)
    
    # Collapse any multi-index column headers if they exist
    if isinstance(bench_raw.columns, pd.MultiIndex):
        bench_raw.columns = bench_raw.columns.get_level_values(0)
        
    bench_df = pd.DataFrame(index=bench_raw.index)
    bench_df['benchmark_close'] = bench_raw['Close']
    bench_df['benchmark_return'] = bench_df['benchmark_close'].pct_change()
    
    all_ticker_dfs = []
    
    # 2. Fetch sector universe data safely
    for ticker in tickers:
        print(f" -> Fetching {ticker}...")
        asset_raw = tf.download(ticker, start=start_date, end=end_date)
        
        if asset_raw.empty:
            print(f"Warning: No data returned for {ticker}")
            continue
            
        # Collapse multi-index columns for the asset
        if isinstance(asset_raw.columns, pd.MultiIndex):
            asset_raw.columns = asset_raw.columns.get_level_values(0)
            
        df = pd.DataFrame(index=asset_raw.index)
        df['Ticker'] = ticker
        df['target_close'] = asset_raw['Close']
        df['target_volume'] = asset_raw['Volume']
        df['target_return'] = df['target_close'].pct_change()
        
        # Synchronize with benchmark columns
        df = df.join(bench_df, how='inner')
        all_ticker_dfs.append(df)
        
    # Stack all dataframes vertically to form the panel structure
    panel_df = pd.concat(all_ticker_dfs).reset_index()
    
    output_path = "data/raw/raw_market_data.csv"
    panel_df.to_csv(output_path, index=False)
    print(f"\nSuccessfully compiled multi-asset market panel data at: {output_path}")
    print(f"Total historical data rows: {len(panel_df)}")

if __name__ == "__main__":
    SECTOR_UNIVERSE = ["KSCL.NS", "COROMANDEL.NS", "RALLIS.NS", "DHANUKA.NS"]
    BENCHMARK_INDEX = "^NSEI"  # NIFTY 50
    START = "2018-01-01"
    END = datetime.today().strftime('%Y-%m-%d')
    
    download_sector_data(SECTOR_UNIVERSE, BENCHMARK_INDEX, START, END)