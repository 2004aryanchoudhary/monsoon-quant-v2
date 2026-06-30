import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_monthly_model(features_path: str):
    df = pd.read_csv(features_path, parse_dates=['Date'], index_col='Date').sort_index()
    
    feature_cols = [
        'target_close', 'target_volume', 'volatility_21', 
        'target_return_21d', 'bench_return_21d', 
        'weather_rain_cum_21d', 'weather_temp_avg_21d'
    ]
    
    X = df[feature_cols]
    y = df['target_alpha_positive']
    
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training Rows: {len(X_train)} | Testing Rows: {len(X_test)}")
    print(f"Base rate of outperformance in Train Set: {y_train.mean()*100:.2f}%")
    
    # Highly regularized shallow trees to force learning broad trends instead of daily noise
    model = XGBClassifier(
        n_estimators=100,
        max_depth=2, 
        learning_rate=0.01,
        subsample=0.6,
        colsample_bytree=0.6,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    print("\n================ CLASSIFIER METRICS ================")
    print(f"Out-of-Sample Accuracy: {accuracy_score(y_test, predictions)*100:.2f}%")
    print("====================================================")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    
    # Feature Importance
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': importance}).sort_values(by='Importance', ascending=False)
    print("\nFeature Importance Rankings:")
    print(feat_imp.to_string(index=False))

if __name__ == "__main__":
    train_monthly_model("data/raw/processed_features.csv")