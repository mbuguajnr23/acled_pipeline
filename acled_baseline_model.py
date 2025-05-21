import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

def train_baseline_model(data_path='data\acled_modeling_data.csv'):
    """
    Train a baseline model for conflict prediction using ACLED data
    
    Parameters:
    -----------
    data_path : str
        Path to the prepared modeling data
        
    Returns:
    --------
    Trained model and evaluation metrics
    """
    print("Loading prepared data...")
    df = pd.read_csv(data_path)
    
    # Convert date back to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Exclude target-related columns and non-feature columns from features
    non_feature_cols = ['country', 'admin1', 'date', 'year_month', 
                        'future_violent_events', 'conflict_occurs']
    
    # Also exclude the current month's metrics since they wouldn't be available for prediction
    current_month_cols = ['total_events', 'violent_events', 'fatalities', 
                         'event_diversity', 'actor1_count', 'actor2_count']
    
    feature_cols = [col for col in df.columns if col not in non_feature_cols and col not in current_month_cols]
    
    X = df[feature_cols]
    y = df['conflict_occurs']
    
    print(f"Number of features: {len(feature_cols)}")
    print(f"Feature list: {feature_cols}")
    
    # Create time-based splits for validation
    # We'll use TimeSeriesSplit to respect the temporal nature of the data
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    # Dictionary to store results
    results = {}
    
    # Create a date-based cut for temporal validation
    # Use 80% of data for training, 20% for testing
    cutoff_date = df['date'].sort_values().iloc[int(len(df) * 0.8)]
    print(f"Time-based validation: using data before {cutoff_date} for training, after for testing")
    
    train_mask = df['date'] < cutoff_date
    test_mask = df['date'] >= cutoff_date
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    # Scale features for better model performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        print("\nClassification report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion matrix:")
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(conf_matrix)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Conflict', 'Conflict'],
                   yticklabels=['No Conflict', 'Conflict'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {name}')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{name.replace(" ", "_").lower()}.png')
        
        # ROC curve
        roc_auc = roc_auc_score(y_test, y_prob)
        print(f"\nROC AUC: {roc_auc:.4f}")
        
        # Precision-Recall curve (important for imbalanced datasets)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)
        print(f"PR AUC: {pr_auc:.4f}")
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {name} (AUC: {pr_auc:.4f})')
        plt.grid(True)
        plt.savefig(f'pr_curve_{name.replace(" ", "_").lower()}.png')
        
        # Feature importance
        if name == 'Random Forest':
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(12, 8))
            plt.title(f'Feature Importances - {name}')
            plt.bar(range(X_train.shape[1]), importances[indices], align='center')
            plt.xticks(range(X_train.shape[1]), [feature_cols[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.savefig(f'feature_importance_{name.replace(" ", "_").lower()}.png')
            
            print("\nTop 10 important features:")
            for i in range(10):
                print(f"{feature_cols[indices[i]]}: {importances[indices[i]]:.4f}")
        
        # SHAP values for more detailed feature importance
        if name == 'XGBoost':
            explainer = shap.Explainer(model)
            shap_values = explainer(X_test)
            
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
            plt.tight_layout()
            plt.savefig('charts\shap_summary.png')
            print("\nSHAP summary plot saved as 'shap_summary.png'")
        
        # Store results
        results[name] = {
            'model': model,
            'conf_matrix': conf_matrix,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }
    
    # Save the best model
    best_model_name = max(results.items(), key=lambda x: x[1]['pr_auc'])[0]
    print(f"\nBest model based on PR-AUC: {best_model_name}")
    
    # Save feature importances to CSV for analysis
    if 'Random Forest' in results:
        rf_model = results['Random Forest']['model']
        feature_importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': rf_model.feature_importances_
        })
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
        feature_importance_df.to_csv('feature_importances.csv', index=False)
        print("Saved feature importances to 'feature_importances.csv'")
    
    return results

if __name__ == "__main__":
    train_baseline_model()
