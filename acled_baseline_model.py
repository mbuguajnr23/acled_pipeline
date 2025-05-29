import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.model_selection import TimeSeriesSplit # You used a manual time split, which is fine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap
import warnings
import os # For os.path.join and os.makedirs
import logging # For logging

warnings.filterwarnings('ignore')

# Get a logger for this module
logger_baseline = logging.getLogger("ACLED_Pipeline.Baseline")

def train_baseline_model(data_path,
                         output_charts_dir='charts_baseline', # Default if called standalone
                         prediction_window_months=1 # Keep for consistency if used, though not directly in this func
                        ):
    """
    Train baseline models for conflict prediction using ACLED data.
    Saves evaluation charts and feature importances to output_charts_dir.
    """
    logger_baseline.info(f"Starting baseline model training. Output charts and reports to: {os.path.abspath(output_charts_dir)}")
    os.makedirs(output_charts_dir, exist_ok=True) # Ensure directory exists

    logger_baseline.info(f"Loading prepared data from: {data_path}")
    try:
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
    except FileNotFoundError:
        logger_baseline.error(f"Data file not found for baseline model: {data_path}")
        return {}
    except Exception as e:
        logger_baseline.error(f"Error loading data for baseline model {data_path}: {e}")
        return {}

    if df.empty:
        logger_baseline.warning("Input DataFrame for baseline model is empty.")
        return {}

    # --- Feature Selection for Baseline ---
    # Exclude target-related, identifiers, current month's raw data, and spatial features.
    non_feature_cols = ['country', 'admin1', 'date', 'future_violent_events', 'conflict_occurs']
    if 'year_month' in df.columns:
        non_feature_cols.append('year_month')

    current_month_base_cols = [ # As defined in your feature engineering
        'total_events', 'fatalities', 'violent_events_count', 'battles_count',
        'vac_count', 'explosion_remote_count', 'riots_count', 'protests_count',
        'distinct_actors_count', 'distinct_actor_types_count',
        'sub_event_diversity', 'event_diversity', 'days_since_last_violent_event'
    ]
    current_month_cols_to_exclude = [col for col in current_month_base_cols if col in df.columns]

    # Keywords to identify and exclude spatial features if they are present in the input df
    spatial_feature_keywords = ['neighbor_', 'country_other_', 'country_conflict_density']
    
    cols_to_exclude = non_feature_cols + current_month_cols_to_exclude
    
    feature_cols = []
    for col in df.columns:
        if col not in cols_to_exclude and not any(keyword in col for keyword in spatial_feature_keywords):
            feature_cols.append(col)

    if not feature_cols:
        logger_baseline.error("No feature columns identified for baseline models after exclusions.")
        return {}

    X = df[feature_cols].copy() # Use .copy()
    y = df['conflict_occurs'].copy()
    
    # Handle NaNs before scaling
    X = X.fillna(-999) # Consistent NaN handling, or more sophisticated imputation
    
    logger_baseline.info(f"Number of features for baseline models: {len(feature_cols)}")
    # logger_baseline.debug(f"Baseline feature list: {feature_cols}") # Use debug for long lists

    # Temporal Split (80/20)
    unique_sorted_dates = np.sort(df['date'].unique())
    if len(unique_sorted_dates) < 5: # Ensure enough unique time points for a split
        logger_baseline.error("Not enough unique time points for baseline train/test split.")
        return {}
    cutoff_date = pd.to_datetime(unique_sorted_dates[int(len(unique_sorted_dates) * 0.8)])
    
    train_mask = df['date'] < cutoff_date
    test_mask = df['date'] >= cutoff_date
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    logger_baseline.info(f"Time-based validation for baselines: Training before {cutoff_date}, Testing from {cutoff_date}")
    logger_baseline.info(f"Baseline training data shape: X-{X_train.shape}, y-{y_train.shape}")
    logger_baseline.info(f"Baseline testing data shape: X-{X_test.shape}, y-{y_test.shape}")

    if X_train.empty or X_test.empty:
        logger_baseline.error("Training or testing set is empty for baseline. Check date cutoff/data.")
        return {}

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert scaled arrays back to DataFrames with original column names for SHAP
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)


    models_to_run = {
        'RandomForest_Baseline': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost_Baseline': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    results = {}

    for name, model in models_to_run.items():
        logger_baseline.info(f"Training {name}...")
        try:
            model.fit(X_train_scaled_df, y_train) # Use scaled DataFrame for SHAP compatibility
            
            y_pred = model.predict(X_test_scaled_df)
            y_prob = model.predict_proba(X_test_scaled_df)[:, 1]
            
            logger_baseline.info(f"\nClassification report for {name}:")
            logger_baseline.info(classification_report(y_test, y_pred, zero_division=0))
            
            conf_matrix = confusion_matrix(y_test, y_pred)
            # logger_baseline.info(f"\nConfusion matrix for {name}:\n{conf_matrix}")
            
            cm_path = os.path.join(output_charts_dir, f'confusion_matrix_{name.replace(" ", "_").lower()}.png')
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['No Conflict', 'Conflict'], yticklabels=['No Conflict', 'Conflict'])
            plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title(f'Confusion Matrix - {name}')
            plt.tight_layout(); plt.savefig(cm_path); plt.close()
            logger_baseline.info(f"Saved confusion matrix for {name} to {cm_path}")
            
            roc_auc = roc_auc_score(y_test, y_prob)
            logger_baseline.info(f"ROC AUC for {name}: {roc_auc:.4f}")
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            pr_auc = auc(recall, precision)
            logger_baseline.info(f"PR AUC for {name}: {pr_auc:.4f}")
            
            pr_curve_path = os.path.join(output_charts_dir, f'pr_curve_{name.replace(" ", "_").lower()}.png')
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, marker='.', label=f'{name} (PR AUC = {pr_auc:.2f})')
            plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'Precision-Recall Curve - {name}')
            plt.legend(); plt.grid(True); plt.savefig(pr_curve_path); plt.close()
            logger_baseline.info(f"Saved PR curve for {name} to {pr_curve_path}")
            
            # Feature importance / SHAP
            if name == 'RandomForest_Baseline' and hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                fi_path = os.path.join(output_charts_dir, f'feature_importance_{name.replace(" ", "_").lower()}.png')
                plt.figure(figsize=(12, max(8, len(X_train.columns) * 0.3))) # Adjust height for many features
                top_n_fi = min(20, len(X_train.columns))
                plt.title(f'Top {top_n_fi} Feature Importances - {name}')
                plt.barh([feature_cols[i] for i in indices[:top_n_fi]], importances[indices[:top_n_fi]], align='center')
                plt.gca().invert_yaxis()
                plt.tight_layout(); plt.savefig(fi_path); plt.close()
                logger_baseline.info(f"Saved feature importance plot for {name} to {fi_path}")
                
                # logger_baseline.info(f"\nTop 10 important features for {name}:")
                # for i in range(min(10, len(indices))):
                #     logger_baseline.info(f"  {feature_cols[indices[i]]}: {importances[indices[i]]:.4f}")

                # Save all feature importances to CSV
                fi_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
                fi_df = fi_df.sort_values('Importance', ascending=False)
                fi_csv_path = os.path.join(output_charts_dir, f'feature_importances_{name.replace(" ", "_").lower()}.csv')
                fi_df.to_csv(fi_csv_path, index=False)
                logger_baseline.info(f"Saved all feature importances for {name} to {fi_csv_path}")

            elif name == 'XGBoost_Baseline' or name == 'RandomForest_Baseline': # SHAP for both tree models
                logger_baseline.info(f"Generating SHAP summary plot for {name}...")
                try:
                    # Use X_test_scaled_df for SHAP values as model was trained on scaled data
                    explainer = shap.Explainer(model, X_train_scaled_df) # Pass training data for some explainers
                    shap_values = explainer(X_test_scaled_df) # Calculate SHAP values for test set
                    
                    # For binary classification with TreeExplainer, shap_values object often has .values for each class
                    # We typically plot for the positive class (class 1)
                    shap_values_to_plot = shap_values
                    if hasattr(shap_values, 'values') and isinstance(shap_values.values, np.ndarray) and shap_values.values.ndim == 3: # XGBoost SHAP structure
                        shap_values_to_plot = shap_values.values[..., 1] # Values for positive class
                    elif isinstance(shap_values.values, list) and len(shap_values.values) == 2: # Scikit-learn RF SHAP structure
                         shap_values_to_plot = shap_values.values[1]


                    plt.figure()
                    # Pass DataFrame to summary_plot for feature names
                    shap.summary_plot(shap_values_to_plot, X_test_scaled_df, show=False, max_display=15)
                    plt.tight_layout()
                    shap_plot_path = os.path.join(output_charts_dir, f'shap_summary_{name.replace(" ", "_").lower()}.png')
                    plt.savefig(shap_plot_path)
                    plt.close()
                    logger_baseline.info(f"Saved SHAP summary plot for {name} to {shap_plot_path}")
                except Exception as e_shap:
                    logger_baseline.error(f"Could not generate SHAP plot for {name}: {e_shap}", exc_info=True)

            results[name] = {'roc_auc': roc_auc, 'pr_auc': pr_auc, 'model_object': model}
        except Exception as e_model:
            logger_baseline.error(f"Error training or evaluating {name}: {e_model}", exc_info=True)
            results[name] = {'roc_auc': 0.0, 'pr_auc': 0.0} # Default on error

    best_model_name = ""
    if results: # Check if results is not empty
        best_model_name = max(results.items(), key=lambda x: x[1].get('pr_auc', 0))[0]
        logger_baseline.info(f"Best baseline model based on PR-AUC: {best_model_name}")
    else:
        logger_baseline.warning("No baseline models were successfully trained and evaluated.")

    return results