import pandas as pd
import numpy as np
from xgboost import XGBClassifier

def run_monthly_backtest(features_path: str):
    print("Loading engineered monthly matrix for backtesting...")
    df = pd.read_csv(features_path, parse_dates=['Date'], index_col='Date').sort_index()
    
    feature_cols = [
        'target_close', 'target_volume', 'volatility_21', 
        'target_return_21d', 'bench_return_21d', 
        'weather_rain_cum_21d', 'weather_temp_avg_21d'
    ]
    
    X = df[feature_cols]
    y = df['target_alpha_positive']
    
    # Chronological Split (80% Train, 20% Test)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    
    backtest_df = pd.DataFrame(index=X_test.index)
    # Track the actual actual 21-day forward returns for evaluation
    backtest_df['actual_target_21d'] = df['target_close'].shift(-21) / df['target_close'] - 1
    backtest_df['actual_bench_21d'] = df['benchmark_close'].shift(-21) / df['benchmark_close'] - 1
    
    # Train the exact same model configuration
    model = XGBClassifier(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.01,
        subsample=0.6,
        colsample_bytree=0.6,
        random_state=42
    )
    model.fit(X_train, y_train := y.iloc[:split_idx])
    
    # Extract prediction probability for outperformance
    backtest_df['prob_positive'] = model.predict_proba(X_test)[:, 1]
    
    # Strategy Execution: Go long if probability > 52%
    # Because we are evaluating rolling 21-day forward returns, we sample every 21 days 
    # to avoid overlapping trade accounting errors.
    CONFIDENCE_THRESHOLD = 0.52
    backtest_df['signal'] = (backtest_df['prob_positive'] > CONFIDENCE_THRESHOLD).astype(int)
    
    # Filter for non-overlapping monthly trading checkpoints
    monthly_checkpoints = backtest_df.iloc[::21].dropna().copy()
    
    # Strategy monthly return matrix
    monthly_checkpoints['strat_return'] = monthly_checkpoints['signal'] * monthly_checkpoints['actual_target_21d']
    
    # Calculate Compounded Cumulative performance across checkpoints
    monthly_checkpoints['cum_bench'] = (1 + monthly_checkpoints['actual_bench_21d']).cumprod() - 1
    monthly_checkpoints['cum_bh'] = (1 + monthly_checkpoints['actual_target_21d']).cumprod() - 1
    monthly_checkpoints['cum_strat'] = (1 + monthly_checkpoints['strat_return']).cumprod() - 1
    
    print("\n================ MONTHLY BACKTEST PERFORMANCE ================")
    print(f"Total OOS Monthly Windows Tested: {len(monthly_checkpoints)}")
    print(f"Active Allocation Months:         {monthly_checkpoints['signal'].sum()} months")
    print(f"Benchmark (NIFTY 50) Return:      {monthly_checkpoints['cum_bench'].iloc[-1]*100:.2f}%")
    print(f"Buy & Hold (KSCL) Return:         {monthly_checkpoints['cum_bh'].iloc[-1]*100:.2f}%")
    print(f"Monsoon Quant Strategy Return:    {monthly_checkpoints['cum_strat'].iloc[-1]*100:.2f}%")
    print("==============================================================")

if __name__ == "__main__":
    run_monthly_backtest("data/raw/processed_features.csv")