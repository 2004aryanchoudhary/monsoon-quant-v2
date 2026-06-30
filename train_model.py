import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_predictive_model(features_path: str):
    print("Loading engineered feature matrix...")
    df = pd.read_csv(features_path, parse_dates=['Date'], index_col='Date')
    df.sort_index(inplace=True) # Ensure chronological order
    
    # --- STEP 1: DEFINE FEATURES AND TARGET ---
    # Independent variables (features)
    feature_cols = [
        'target_close', 'target_volume', 'benchmark_close', 
        'rsi_14', 'volatility_14', 
        'precip_mm', 'max_temp_c', 'precip_roll_7', 
        'precip_surprise', 'precip_cum_3', 'precip_cum_7'
    ]
    
    X = df[feature_cols]
    y = df['target_next_alpha'] # Dependent variable: next day's outperformance
    
    # --- STEP 2: TIME-SERIES TRAIN-TEST SPLIT ---
    # We use the final 20% of chronological data for evaluation to simulate real trading
    split_idx = int(len(df) * 0.8)
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training sample size: {len(X_train)} rows | Testing sample size: {len(X_test)} rows")
    
    # --- STEP 3: INITIALIZE AND TRAIN XGBOOST ---
    print("Fitting XGBoost Regressor...")
    model = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # --- STEP 4: EVALUATE MODEL PERFORMANCE ---
    predictions = model.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    print("\n================ MODEL METRICS ================")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"R-squared Score (R2):            {r2:.6f}")
    print("===============================================")
    
    # --- STEP 5: EXTRACT FEATURE IMPORTANCE ---
    # See exactly which features (market indicators vs weather metrics) drove predictions
    importance = model.feature_importances_
    feat_imp_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    print("\nFeature Importance Rankings:")
    print(feat_imp_df.to_string(index=False))
    
    # Create a small DataFrame with actuals vs predictions to verify behavior
    results_df = pd.DataFrame({
        'Actual_Next_Alpha': y_test,
        'Predicted_Next_Alpha': predictions
    }, index=X_test.index)
    
    return results_df

if __name__ == "__main__":
    FEATURES_FILE = "data/raw/processed_features.csv"
    
    try:
        test_results = train_predictive_model(FEATURES_FILE)
        print("\nFirst 5 prediction rows:")
        print(test_results.head())
        
    except Exception as e:
        print(f"Model training failed: {e}")