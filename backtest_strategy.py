import pandas as pd
import numpy as np
from xgboost import XGBClassifier

def run_cross_sectional_backtest(features_path: str):
    print("Loading panel feature matrix for cross-sectional portfolio analysis...")
    df = pd.read_csv(features_path, parse_dates=['Date'], index_col='Date').sort_index()
    
    feature_cols = [
        'target_close', 'target_volume', 'volatility_21', 
        'target_return_21d', 'bench_return_21d', 
        'weather_rain_cum_21d', 'weather_temp_avg_21d'
    ]
    
    # Apply identical chronological train/test splitting boundaries
    unique_dates = df.index.unique().sort_values()
    split_date = unique_dates[int(len(unique_dates) * 0.8)]
    
    train_panel = df.loc[df.index < split_date]
    test_panel = df.loc[df.index >= split_date].copy()
    
    # Fit the structural model
    model = XGBClassifier(
        n_estimators=120, max_depth=2, learning_rate=0.01,
        subsample=0.6, colsample_bytree=0.6, random_state=42
    )
    model.fit(train_panel[feature_cols], train_panel['target_alpha_positive'])
    
    # Predict the precise probability of generating positive outperformance alpha
    test_panel['prob_outperform'] = model.predict_proba(test_panel[feature_cols])[:, 1]
    
   # Map future realizations cleanly within each specific ticker's timeline
    test_panel['actual_target_21d'] = test_panel.groupby('Ticker')['target_close'].shift(-21) / test_panel['target_close'] - 1
    test_panel['actual_bench_21d'] = test_panel.groupby('Ticker')['benchmark_close'].shift(-21) / test_panel['benchmark_close'] - 1
    
    # Isolate non-overlapping 21-day reallocation steps
    test_dates = test_panel.index.unique().sort_values()
    monthly_checkpoints = test_dates[::21]
    
    strategy_returns = []
    benchmark_returns = []
    equal_weight_returns = []
    
    print(f"\nSimulating Allocations Across {len(monthly_checkpoints)} Cross-Sectional Windows:")
    print("-" * 85)
    
    for checkpoint in monthly_checkpoints:
        cross_section = test_panel.loc[test_panel.index == checkpoint].copy()
        
        if cross_section.empty or len(cross_section) < 2:
            continue
            
        # Select the asset with the highest probability
        selected_allocation = cross_section.sort_values(by='prob_outperform', ascending=False).iloc[0]
        max_confidence = selected_allocation['prob_outperform']
        
        # THE FIX: The Absolute Confidence Cash Filter
        CONFIDENCE_THRESHOLD = 0.52
        
        if max_confidence >= CONFIDENCE_THRESHOLD:
            # Trade executed
            ticker_traded = selected_allocation['Ticker']
            strat_return = selected_allocation['actual_target_21d']
        else:
            # Setup is too risky, sit in cash
            ticker_traded = "CASH"
            strat_return = 0.0
            
        bench_return = selected_allocation['actual_bench_21d']
        ew_sector_return = cross_section['actual_target_21d'].mean()
        
        if not np.isnan(strat_return) and not np.isnan(bench_return):
            strategy_returns.append(strat_return)
            benchmark_returns.append(bench_return)
            equal_weight_returns.append(ew_sector_return)
            
            print(f"Date: {checkpoint.strftime('%Y-%m-%d')} | "
                  f"Top Pick: {ticker_traded.ljust(13)} | "
                  f"Confidence: {max_confidence*100:.2f}% | "
                  f"Forward Return: {strat_return*100:+.2f}%")

    # Compute compounding portfolio metrics over time
    cum_strategy = np.prod(1 + np.array(strategy_returns)) - 1
    cum_benchmark = np.prod(1 + np.array(benchmark_returns)) - 1
    cum_sector_ew = np.prod(1 + np.array(equal_weight_returns)) - 1
    
    print("\n================ CROSS-SECTIONAL STRATEGY PERFORMANCE ================")
    print(f"Total Reallocation Cycles:               {len(strategy_returns)}")
    print(f"Benchmark (NIFTY 50) Cumulative Return:  {cum_benchmark*100:.2f}%")
    print(f"Passive Equal-Weight Sector Basket:      {cum_sector_ew*100:.2f}%")
    print(f"Monsoon Cross-Sectional Alpha Strategy:  {cum_strategy*100:.2f}%")
    print("======================================================================")

if __name__ == "__main__":
    run_cross_sectional_backtest("data/raw/processed_features.csv")