import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit # Using manual time split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap
import warnings
import os # For os.path.join and os.makedirs
import logging # For logging

warnings.filterwarnings('ignore')

# Get a logger for this module (assuming it's part of the ACLED_Pipeline)
logger_baseline = logging.getLogger("ACLED_Pipeline.Baseline")

def train_baseline_model(data_path,output_charts_dir
                        ):
    """
    Train baseline models for conflict prediction using ACLED data.
    Saves evaluation charts and feature importances to output_charts_dir.
    """
    logger_baseline.info(f"Starting baseline model training using data: {data_path}")
    logger_baseline.info(f"Output charts and reports will be saved to: {os.path.abspath(output_charts_dir)}")
    os.makedirs(output_charts_dir, exist_ok=True) # Ensure directory exists

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
    # Exclude target-related, identifiers, current month's raw data, and known spatial features.
    non_feature_cols = ['country', 'admin1', 'date', 'future_violent_events', 'conflict_occurs']
    if 'year_month' in df.columns: # year_month might be object/string type from CSV
        non_feature_cols.append('year_month')


    current_month_base_cols = [
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

    X = df[feature_cols].copy()
    y = df['conflict_occurs'].copy()
    
    # Handle NaNs before scaling
    X = X.fillna(-999) 
    
    logger_baseline.info(f"Number of features for baseline models: {len(feature_cols)}")
    # logger_baseline.debug(f"Baseline feature list: {feature_cols}")

    # Temporal Split (80/20)
    # Ensure 'date' column is present and datetime
    if 'date' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['date']):
        logger_baseline.error("'date' column missing or not datetime. Cannot perform temporal split.")
        return {}
        
    unique_sorted_dates = np.sort(df['date'].unique())
    if len(unique_sorted_dates) < 5:
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
        # 'LogisticRegression_Baseline': LogisticRegression(random_state=42, solver='liblinear', max_iter=1000), # Requires scaled data
        'RandomForest_Baseline': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost_Baseline': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    results = {} # Initialize results dictionary

    for name, model in models_to_run.items():
        logger_baseline.info(f"Training {name}...")
        try:
            # Logistic Regression benefits from scaled data, tree models don't strictly need it
            # but using scaled for consistency with SHAP on scaled data if desired
            current_X_train = X_train_scaled_df # if name == 'LogisticRegression_Baseline' else X_train
            current_X_test = X_test_scaled_df # if name == 'LogisticRegression_Baseline' else X_test
            
            model.fit(current_X_train, y_train)
            
            y_pred = model.predict(current_X_test)
            y_prob = model.predict_proba(current_X_test)[:, 1]
            
            logger_baseline.info(f"\nClassification report for {name}:")
            logger_baseline.info(classification_report(y_test, y_pred, zero_division=0))
            
            conf_matrix = confusion_matrix(y_test, y_pred)
            
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
                fi_df = pd.DataFrame({'Feature': current_X_train.columns, 'Importance': importances})
                fi_df = fi_df.sort_values('Importance', ascending=False)

                fi_path = os.path.join(output_charts_dir, f'feature_importance_{name.replace(" ", "_").lower()}.png')
                plt.figure(figsize=(12, max(8, len(fi_df) * 0.3)))
                top_n_fi = min(20, len(fi_df))
                plt.title(f'Top {top_n_fi} Feature Importances - {name}')
                plt.barh(fi_df['Feature'][:top_n_fi], fi_df['Importance'][:top_n_fi], align='center')
                plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig(fi_path); plt.close()
                logger_baseline.info(f"Saved feature importance plot for {name} to {fi_path}")
                
                fi_csv_path = os.path.join(output_charts_dir, f'feature_importances_{name.replace(" ", "_").lower()}.csv')
                fi_df.to_csv(fi_csv_path, index=False)
                logger_baseline.info(f"Saved all feature importances for {name} to {fi_csv_path}")

            elif name == 'XGBoost_Baseline' or (name == 'RandomForest_Baseline' and 'shap' in globals()):
                logger_baseline.info(f"Generating SHAP summary plot for {name}...")
                try:
                    explainer = shap.Explainer(model, current_X_train) # Pass training data for some explainers
                    shap_values_obj = explainer(current_X_test)
                    
                    shap_values_for_plot = shap_values_obj
                    if hasattr(shap_values_obj, 'values') and isinstance(shap_values_obj.values, np.ndarray) and shap_values_obj.values.ndim == 3:
                        shap_values_for_plot = shap_values_obj.values[..., 1]
                    elif hasattr(shap_values_obj, 'values') and isinstance(shap_values_obj.values, list) and len(shap_values_obj.values) == 2:
                         shap_values_for_plot = shap_values_obj.values[1]
                    # If shap_values_obj is already the array for the positive class, use it directly.

                    plt.figure()
                    shap.summary_plot(shap_values_for_plot, current_X_test, show=False, max_display=15)
                    plt.tight_layout()
                    shap_plot_path = os.path.join(output_charts_dir, f'shap_summary_{name.replace(" ", "_").lower()}.png')
                    plt.savefig(shap_plot_path)
                    plt.close()
                    logger_baseline.info(f"Saved SHAP summary plot for {name} to {shap_plot_path}")
                except Exception as e_shap:
                    logger_baseline.error(f"Could not generate SHAP plot for {name}: {e_shap}", exc_info=True)

            results[name] = {'roc_auc': roc_auc, 'pr_auc': pr_auc, 'model_object': model} # Storing model if needed later
        except Exception as e_model:
            logger_baseline.error(f"Error training or evaluating {name}: {e_model}", exc_info=True)
            results[name] = {'roc_auc': 0.0, 'pr_auc': 0.0}

    best_model_name = ""
    if results:
        # Filter out models that might have failed (pr_auc might not exist or be 0)
        valid_results = {k: v for k, v in results.items() if 'pr_auc' in v and v['pr_auc'] > 0}
        if valid_results:
            best_model_name = max(valid_results.items(), key=lambda x: x[1]['pr_auc'])[0]
            logger_baseline.info(f"Best baseline model based on PR-AUC: {best_model_name}")
        else:
            logger_baseline.warning("No baseline models were successfully trained and evaluated with valid PR-AUC.")
    else:
        logger_baseline.warning("No baseline models were run or results dictionary is empty.")

    return results