import pandas as pd
import numpy as np

def engineer_panel_features(market_path: str, weather_path: str) -> pd.DataFrame:
    print("Loading raw datasets for domestic universe...")
    market_df = pd.read_csv(market_path, parse_dates=['Date'])
    weather_df = pd.read_csv(weather_path, parse_dates=['Date'], index_col='Date')
    
    processed_ticker_groups = []
    
    for ticker, group in market_df.groupby('Ticker'):
        print(f" -> Structuring normalized features for: {ticker}")
        group = group.sort_values('Date').set_index('Date')
        ticker_merged = group.join(weather_df, how='inner')
        
        # 1. THE MATH FIX: Relative Metrics (No raw prices allowed)
        # Distance from 21-day Moving Average (e.g., 1.05 means 5% above average)
        ticker_merged['price_vs_ma21'] = ticker_merged['target_close'] / ticker_merged['target_close'].rolling(window=21).mean()
        # Relative Volume (e.g., 1.5 means volume is 50% higher than average)
        ticker_merged['relative_volume_21d'] = ticker_merged['target_volume'] / ticker_merged['target_volume'].rolling(window=21).mean()
        
        # 2. Standard Momentum & Volatility
        ticker_merged['volatility_21'] = ticker_merged['target_return'].rolling(window=21).std()
        ticker_merged['target_return_21d'] = ticker_merged['target_close'].pct_change(periods=21)
        ticker_merged['bench_return_21d'] = ticker_merged['benchmark_close'].pct_change(periods=21)
        
        # 3. Macro Weather Features
        ticker_merged['weather_rain_cum_21d'] = ticker_merged['precip_mm'].rolling(window=21).sum()
        ticker_merged['weather_temp_avg_21d'] = ticker_merged['max_temp_c'].rolling(window=21).mean()
        
        # 4. Target Variables
        future_target_change = ticker_merged['target_close'].shift(-21) / ticker_merged['target_close'] - 1
        future_bench_change = ticker_merged['benchmark_close'].shift(-21) / ticker_merged['benchmark_close'] - 1
        ticker_merged['target_alpha_positive'] = (future_target_change - future_bench_change > 0).astype(int)
        
        processed_ticker_groups.append(ticker_merged.dropna())
        
    master_panel = pd.concat(processed_ticker_groups).sort_index()
    print(f"\nFinal Normalized Panel Matrix Shape: {master_panel.shape}")
    return master_panel

if __name__ == "__main__":
    engineer_panel_features("data/raw/raw_market_data.csv", "data/raw/raw_weather_data.csv").to_csv("data/raw/processed_features.csv")