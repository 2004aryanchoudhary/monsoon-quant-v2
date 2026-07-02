import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def train_panel_classifier(features_path: str):
    print("Loading master panel matrix for centralized training...")
    df = pd.read_csv(features_path, parse_dates=['Date'], index_col='Date').sort_index()
    
    feature_cols = [
        'price_vs_ma21', 'relative_volume_21d', 'volatility_21', 
        'target_return_21d', 'bench_return_21d', 
        'weather_rain_cum_21d', 'weather_temp_avg_21d'
    ]
    
    # Identify unique chronological timestamps to prevent time-leakage
    unique_dates = df.index.unique().sort_values()
    split_idx = int(len(unique_dates) * 0.8)
    split_date = unique_dates[split_idx]
    
    # Chronological panel split
    train_panel = df.loc[df.index < split_date]
    test_panel = df.loc[df.index >= split_date]
    
    X_train, y_train = train_panel[feature_cols], train_panel['target_alpha_positive']
    X_test, y_test = test_panel[feature_cols], test_panel['target_alpha_positive']
    
    print(f"Training Data points: {len(X_train)} | Testing Data points: {len(X_test)}")
    print(f"Global outperformance base rate (Train Set): {y_train.mean()*100:.2f}%")
    
    # Train centralized regularization framework
    model = XGBClassifier(
        n_estimators=120,
        max_depth=2,
        learning_rate=0.01,
        subsample=0.6,
        colsample_bytree=0.6,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    print("\n================ UNIVERSAL SECTOR CLASSIFIER METRICS ================")
    print(f"Out-of-Sample Accuracy: {accuracy_score(y_test, predictions)*100:.2f}%")
    print("=====================================================================")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    
    # Feature Importance
    feat_imp = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print("\nGlobal Sector Feature Importance Rankings:")
    print(feat_imp.to_string(index=False))

    # Save the trained model for the live Streamlit dashboard
    joblib.dump(model, "monsoon_xgboost.joblib")
    print("Model successfully exported to monsoon_xgboost.joblib")

if __name__ == "__main__":
    train_panel_classifier("data/raw/processed_features.csv")