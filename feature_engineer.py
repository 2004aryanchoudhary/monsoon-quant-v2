import pandas as pd
import numpy as np

def engineer_features(market_path: str, weather_path: str) -> pd.DataFrame:
    print("Loading raw datasets...")
    market_df = pd.read_csv(market_path, parse_dates=['Date'], index_col='Date')
    weather_df = pd.read_csv(weather_path, parse_dates=['Date'], index_col='Date')
    
    # Inner join to synchronize trading timelines
    merged_df = market_df.join(weather_df, how='inner').sort_index()
    
    # --- STEP 1: FINANCIAL FEATURES (MONTHLY ROLLING) ---
    merged_df['alpha_return'] = merged_df['target_return'] - merged_df['benchmark_return']
    
    # 21-day technical momentum metrics
    merged_df['volatility_21'] = merged_df['target_return'].rolling(window=21).std()
    merged_df['target_return_21d'] = merged_df['target_close'].pct_change(periods=21)
    merged_df['bench_return_21d'] = merged_df['benchmark_close'].pct_change(periods=21)
    
    # --- STEP 2: AGRICULTURAL WEATHER FEATURES (MONTHLY CUMULATIVE) ---
    # Total mm of rain accumulated over the past month
    merged_df['weather_rain_cum_21d'] = merged_df['precip_mm'].rolling(window=21).sum()
    # Average max temperature over the past month
    merged_df['weather_temp_avg_21d'] = merged_df['max_temp_c'].rolling(window=21).mean()
    
    # --- STEP 3: CLEAN 21-DAY FORWARD TARGET VARIABLE ---
    # Calculate the continuous alpha outperformance over the NEXT 21 trading days
    # We look ahead by taking the 21-day future price change difference
    future_target_change = merged_df['target_close'].shift(-21) / merged_df['target_close'] - 1
    future_bench_change = merged_df['benchmark_close'].shift(-21) / merged_df['benchmark_close'] - 1
    future_alpha_21d = future_target_change - future_bench_change
    
    # Binary Target: 1 if the stock outperforms the index over the next 21 days, else 0
    merged_df['target_alpha_positive'] = (future_alpha_21d > 0).astype(int)
    
    # Drop rows with NaNs from rolling indicators and future lookaheads
    clean_df = merged_df.dropna().copy()
    print(f"Engineered Monthly Feature Matrix Shape: {clean_df.shape}")
    
    return clean_df

if __name__ == "__main__":
    MARKET_FILE = "data/raw/raw_market_data.csv"
    WEATHER_FILE = "data/raw/raw_weather_data.csv"
    OUTPUT_FILE = "data/raw/processed_features.csv"
    
    try:
        processed_data = engineer_features(MARKET_FILE, WEATHER_FILE)
        processed_data.to_csv(OUTPUT_FILE)
        print(f"Master feature matrix overwritten successfully at: {OUTPUT_FILE}")
    except Exception as e:
        print(f"Feature engineering failed: {e}")