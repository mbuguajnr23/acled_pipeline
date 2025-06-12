import pandas as pd
import numpy as np
import os
import geopandas as gpd
import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestClassifier # Not used in train_spatial_model
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
import xgboost as xgb
# from shapely.geometry import Point # Indirectly used by geopandas
import warnings
import os # For path joining and directory creation
import pickle # For saving the model

warnings.filterwarnings('ignore')

def add_spatial_features(df, shapefile_path=None):
    """
    Add spatial features based on neighboring admin1 regions.
    Falls back to country-based proximity if shapefile is not provided or fails.
    """
    print("Adding spatial features...")
    if not shapefile_path:
        print("No shapefile path provided. Falling back to country-based proximity.")
        return add_country_based_spatial_features(df.copy()) # Pass a copy

    df_copy = df.copy() # Work on a copy

    try:
        gdf = gpd.read_file(shapefile_path)
        rename_map = {}
        for col_name in ['CNTRY_NAME', 'COUNTRY', 'NAME_0', 'ADM0_NAME', 'admin0Name', 'COUNTRYNAME']:
            if col_name in gdf.columns:
                rename_map[col_name] = 'country'
                break
        for col_name in ['ADMIN1', 'NAME_1', 'ADM1_NAME', 'admin1Name', 'province', 'state', 'ADM1_EN', 'shapeName']:
            if col_name in gdf.columns:
                rename_map[col_name] = 'admin1'
                break
        
        if 'country' not in rename_map or 'admin1' not in rename_map:
            # Try to guess if only one is missing and the other matches 'country' or 'admin1'
            if 'country' not in rename_map and 'country' in gdf.columns: rename_map['country'] = 'country'
            if 'admin1' not in rename_map and 'admin1' in gdf.columns: rename_map['admin1'] = 'admin1'

            if 'country' not in rename_map or 'admin1' not in rename_map:
                 raise ValueError(f"Could not find standard country/admin1 columns in shapefile. Found: {gdf.columns.tolist()}. Need to map to 'country' and 'admin1'.")

        gdf = gdf.rename(columns=rename_map)
        # Ensure no duplicate region identifiers if shapefile has multipart polygons as separate rows
        gdf = gdf[['country', 'admin1', 'geometry']].drop_duplicates(subset=['country', 'admin1'])


        if gdf.crs and gdf.crs.to_string() != "EPSG:4326": # More robust CRS check
            print(f"Reprojecting shapefile from {gdf.crs.to_string()} to EPSG:4326")
            gdf = gdf.to_crs("EPSG:4326")

        if not gdf.sindex:
             gdf.sindex # Build spatial index

        neighbors_map = {}
        print("Identifying neighbors using spatial index...")
        for idx, focal_row in gdf.iterrows():
            focal_geom = focal_row['geometry']
            focal_region_id = (focal_row['country'], focal_row['admin1'])
            neighbors_map[focal_region_id] = []
            possible_matches_indices = list(gdf.sindex.intersection(focal_geom.bounds))
            possible_matches = gdf.iloc[possible_matches_indices]
            precise_neighbors = possible_matches[possible_matches.geometry.touches(focal_geom) & (possible_matches.index != idx)]
            for _, neighbor_row in precise_neighbors.iterrows():
                neighbors_map[focal_region_id].append((neighbor_row['country'], neighbor_row['admin1']))
        
        print(f"Identified neighbors for {len(neighbors_map)} regions.")

        neighbor_feature_cols_new = ['neighbor_violent_events_count_lag1_avg', 'neighbor_fatalities_lag1_avg', 'neighbor_conflict_density_lag1']
        for col in neighbor_feature_cols_new:
            df_copy[col] = 0.0

        # --- Source columns for neighbor features (from df_copy) ---
        source_col_violent_lag1_for_neighbor = 'violent_events_count_lag1'
        source_col_fatalities_lag1_for_neighbor = 'fatalities_lag1'
        # --- End source columns ---

        missing_source_cols = []
        if source_col_violent_lag1_for_neighbor not in df_copy.columns:
            missing_source_cols.append(source_col_violent_lag1_for_neighbor)
        if source_col_fatalities_lag1_for_neighbor not in df_copy.columns:
            missing_source_cols.append(source_col_fatalities_lag1_for_neighbor)
        
        if missing_source_cols:
            print(f"WARNING: Source columns {missing_source_cols} for neighbor features not found in DataFrame. Shapefile-based spatial features might be incomplete or zero.")
            # Optionally fall back if critical:
            # print("Falling back to country-based proximity due to missing source columns for neighbor features.")
            # return add_country_based_spatial_features(df.copy())


        print("Calculating spatial features from neighbors...")
        # Using df.set_index for faster lookups in the loop
        # Ensure 'date' column in df_copy is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_copy['date']):
            df_copy['date'] = pd.to_datetime(df_copy['date'])
            
        df_indexed = df_copy.set_index(['date', 'country', 'admin1'])

        for original_idx, row_df in df_copy.iterrows(): # Iterate over df_copy to fill its rows
            current_date = row_df['date']
            focal_region_id = (row_df['country'], row_df['admin1'])

            if focal_region_id not in neighbors_map or not neighbors_map[focal_region_id]:
                continue

            current_neighbors = neighbors_map[focal_region_id]
            sum_n_violent_lag1, sum_n_fatalities_lag1, active_n_lag1, valid_n_count = 0, 0, 0, 0

            for neighbor_id_tuple in current_neighbors:
                try:
                    # Ensure neighbor_id_tuple is (country, admin1)
                    n_country, n_admin1 = neighbor_id_tuple
                    neighbor_data_row = df_indexed.loc[(current_date, n_country, n_admin1)]
                    
                    if source_col_violent_lag1_for_neighbor in neighbor_data_row.index:
                        sum_n_violent_lag1 += neighbor_data_row.get(source_col_violent_lag1_for_neighbor, 0)
                        if neighbor_data_row.get(source_col_violent_lag1_for_neighbor, 0) > 0:
                            active_n_lag1 += 1
                    if source_col_fatalities_lag1_for_neighbor in neighbor_data_row.index:
                         sum_n_fatalities_lag1 += neighbor_data_row.get(source_col_fatalities_lag1_for_neighbor, 0)
                    valid_n_count += 1
                except KeyError:
                    pass # Neighbor data not found for this specific date (could happen if neighbor not in original event data)

            if valid_n_count > 0:
                df_copy.at[original_idx, 'neighbor_violent_events_count_lag1_avg'] = sum_n_violent_lag1 / valid_n_count
                df_copy.at[original_idx, 'neighbor_fatalities_lag1_avg'] = sum_n_fatalities_lag1 / valid_n_count
                df_copy.at[original_idx, 'neighbor_conflict_density_lag1'] = active_n_lag1 / valid_n_count
        
        print("Added spatial features based on shapefile.")
        return df_copy

    except Exception as e:
        print(f"Error processing shapefile: {e}. Ensure shapefile is valid and has 'country'/'admin1' like columns.")
        print("Falling back to country-based proximity...")
        return add_country_based_spatial_features(df.copy())


def add_country_based_spatial_features(df):
    """
    Add simplified spatial features based on country proximity
    when a shapefile is not available. Uses lagged data.
    """
    print("Adding country-based spatial features (activity in other admin1 regions of same country)...")
    df_copy = df.copy()

    # --- VERIFY THESE SOURCE COLUMN NAMES AGAINST YOUR acled_modeling_data_prepared.csv ---
    source_col_violent_lag1 = 'violent_events_count_lag1'
    source_col_fatalities_lag1 = 'fatalities_lag1'
    # --- END VERIFICATION ---

    new_col_country_other_violent = 'country_other_admin1_violent_events_lag1'
    new_col_country_other_fatalities = 'country_other_admin1_fatalities_lag1'
    new_col_country_density = 'country_conflict_density_lag1'

    df_copy[new_col_country_other_violent] = 0.0
    df_copy[new_col_country_other_fatalities] = 0.0
    df_copy[new_col_country_density] = 0.0
    
    if source_col_violent_lag1 not in df_copy.columns:
        print(f"WARNING: Source column '{source_col_violent_lag1}' not found for country-based features.")
    else:
        country_total_violent_at_date = df_copy.groupby(['date', 'country'])[source_col_violent_lag1].transform('sum')
        df_copy[new_col_country_other_violent] = country_total_violent_at_date - df_copy[source_col_violent_lag1]

    if source_col_fatalities_lag1 not in df_copy.columns:
        print(f"WARNING: Source column '{source_col_fatalities_lag1}' not found for country-based features.")
    else:
        country_total_fatalities_at_date = df_copy.groupby(['date', 'country'])[source_col_fatalities_lag1].transform('sum')
        df_copy[new_col_country_other_fatalities] = country_total_fatalities_at_date - df_copy[source_col_fatalities_lag1]

    if source_col_violent_lag1 in df_copy.columns:
        df_copy['had_conflict_lag1_temp'] = (df_copy[source_col_violent_lag1] > 0).astype(int)
        country_conflict_metrics = df_copy.groupby(['date', 'country'])['had_conflict_lag1_temp'].agg(
            conflict_regions_lag1='sum', total_regions='count'
        ).reset_index()
        
        country_conflict_metrics['country_conflict_density_lag1_calc'] = 0.0
        mask_tr_gt_zero = country_conflict_metrics['total_regions'] > 0
        country_conflict_metrics.loc[mask_tr_gt_zero, 'country_conflict_density_lag1_calc'] = \
            country_conflict_metrics.loc[mask_tr_gt_zero, 'conflict_regions_lag1'] / \
            country_conflict_metrics.loc[mask_tr_gt_zero, 'total_regions']
        
        df_copy = pd.merge(df_copy,
                           country_conflict_metrics[['date', 'country', 'country_conflict_density_lag1_calc']],
                           on=['date', 'country'], how='left')
        df_copy[new_col_country_density] = df_copy['country_conflict_density_lag1_calc'].fillna(0)
        df_copy.drop(columns=['had_conflict_lag1_temp', 'country_conflict_density_lag1_calc'], inplace=True, errors='ignore')
    
    print("Added country-based spatial features.")
    return df_copy


def train_spatial_model(data_path, shapefile_path, output_model_path='models/spatial_conflict_model.pkl', # ADDED
                        output_charts_dir='charts'):
    """
    Train a model incorporating spatial features for conflict prediction
    
    Parameters:
    -----------
    data_path : str
        Path to the prepared modeling data
    shapefile_path : str
        Path to admin1 boundary shapefile (optional)
        
    Returns:
    --------
    Trained model and evaluation metrics
    """
    print(f"Loading prepared data from: {data_path}")
    df = pd.read_csv(data_path)
    # Ensure 'date' is datetime, as it might be read as string from CSV
    df['date'] = pd.to_datetime(df['date']) 
    
    # Add spatial features (this calls the functions above)
    df = add_spatial_features(df, shapefile_path)
    
    # --- Define columns to exclude from features ---
    non_feature_cols = ['country', 'admin1', 'date', 'future_violent_events', 'conflict_occurs']
    if 'year_month' in df.columns: # year_month might be string if read from CSV
        non_feature_cols.append('year_month')

    # Current month's raw aggregated metrics (from acled_feature_engineering.py)
    # These are the 'base' columns before lags/rolls were created.
    # !! CRITICAL: Ensure these names match your acled_modeling_data_prepared.csv !!
    current_month_base_cols = [
        'total_events', 'fatalities', 'violent_events_count', 'battles_count',
        'vac_count', 'explosion_remote_count', 'riots_count', 'protests_count',
        'distinct_actors_count', 'distinct_actor_types_count',
        'sub_event_diversity', 'event_diversity', 'days_since_last_violent_event' # current state
    ]
    current_month_cols_to_exclude = [col for col in current_month_base_cols if col in df.columns]
    cols_to_exclude = non_feature_cols + current_month_cols_to_exclude
    
    feature_cols = [col for col in df.columns if col not in cols_to_exclude]
    
    if not feature_cols:
        print("ERROR: No feature columns selected in train_spatial_model. Check column exclusion lists.")
        return None
        
    X = df[feature_cols].fillna(-999) # Fill NaNs that might arise from spatial features or other steps
    y = df['conflict_occurs']
    
    print(f"Number of features for model: {len(feature_cols)}")
    
    # Temporal validation split
    # Get unique sorted dates from the DataFrame to ensure split point is on an actual date boundary
    unique_sorted_dates = np.sort(df['date'].unique())
    if len(unique_sorted_dates) < 5: # Arbitrary small number, ensure enough unique dates for a split
        print("ERROR: Not enough unique time points in the data for a train/test split.")
        return None
    cutoff_date = pd.to_datetime(unique_sorted_dates[int(len(unique_sorted_dates) * 0.8)])
    
    train_mask = df['date'] < cutoff_date
    test_mask = df['date'] >= cutoff_date
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    print(f"Time-based validation: Training on data before {cutoff_date}, Testing on data from {cutoff_date}")
    print(f"Training data shape: X-{X_train.shape}, y-{y_train.shape}")
    print(f"Testing data shape: X-{X_test.shape}, y-{y_test.shape}")

    if X_train.empty or X_test.empty:
        print("ERROR: Training or testing set is empty. Check date cutoff, data range, and filters.")
        return None

    print("\nTraining XGBoost with spatial features...")
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                              n_estimators=200, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC AUC: {roc_auc:.4f}")
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    print(f"PR AUC: {pr_auc:.4f}")
    
    pr_curve_path = os.path.join(output_charts_dir, 'pr_curve_spatial_model.png')
    # Plot PR curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - Spatial Model (AUC: {pr_auc:.4f})')
    plt.grid(True)
    plt.savefig(pr_curve_path) # Use the constructed path
    plt.close()
    print(f"Saved PR curve to {pr_curve_path}")
    
    # Feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    fi_plot_path = os.path.join(output_charts_dir, 'feature_importance_spatial_model.png') 
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances - Spatial Model')
    plt.bar(range(min(20, X_train.shape[1])), importances[indices[:20]], align='center')
    plt.xticks(range(min(20, X_train.shape[1])), [feature_cols[i] for i in indices[:20]], rotation=90)
    plt.tight_layout()
    plt.savefig(fi_plot_path) # Use the constructed path
    plt.close()
    print(f"Saved feature importance plot to {fi_plot_path}")
    
    print("\nTop 10 important features:")
    for i in range(min(10, len(indices))):
        print(f"{feature_cols[indices[i]]}: {importances[indices[i]]:.4f}")
    
    # Save feature importances to CSV for analysis
    feature_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    })
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
    feature_importance_df.to_csv('spatial_feature_importances.csv', index=False)
    print("Saved feature importances to 'spatial_feature_importances.csv'")
    
    # Save the model
    import pickle
    if os.path.dirname(output_model_path): # Check if output_model_path includes a directory part
        os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    with open('spatial_conflict_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved model to {output_model_path}")
    
    results = {'model': model, 'roc_auc': roc_auc, 'pr_auc': pr_auc, 'feature_importances': feature_importance_df}
    return results

if __name__ == "__main__":
    # Example Usage:
    # Create dummy data and shapefile for testing if run directly
    # In a real pipeline, this __main__ block would typically not be used,
    # as the functions are called by acled_complete_pipeline.py

    print("Running acled_spatial_model.py directly (example usage / test mode)")

    # Create dummy 'data' directory and 'acled_modeling_data_prepared.csv'
    os.makedirs('data', exist_ok=True)
    sample_data_rows = 1000
    # Create a more realistic set of columns based on expected output from feature engineering
    # For simplicity, we'll just create a few key ones.
    # In reality, this dummy data should closely mirror your actual prepared data structure.
    feature_engineering_output_cols = [
        'date', 'country', 'admin1', 'violent_events_count_lag1', 'fatalities_lag1',
        'total_events', 'fatalities', 'violent_events_count', 'battles_count',
        'vac_count', 'explosion_remote_count', 'riots_count', 'protests_count',
        'distinct_actors_count', 'distinct_actor_types_count',
        'sub_event_diversity', 'event_diversity', 'days_since_last_violent_event',
        'some_other_lag_feature', 'some_roll_feature', # Add more representative features
        'future_violent_events', 'conflict_occurs'
    ]
    dummy_data = pd.DataFrame(np.random.randint(0,10,size=(sample_data_rows, len(feature_engineering_output_cols))),
                              columns=feature_engineering_output_cols)
    
    # Create a realistic date range for dummy data
    base_date = pd.to_datetime('2020-01-01')
    dummy_data['date'] = [base_date + pd.DateOffset(months=i // 50) for i in range(sample_data_rows)] # Spread dates
    dummy_data['country'] = [f'Country_{i%3}' for i in range(sample_data_rows)]
    dummy_data['admin1'] = [f'Admin1_{i%10}' for i in range(sample_data_rows)]
    dummy_data['conflict_occurs'] = np.random.randint(0,2,size=sample_data_rows)
    dummy_data['violent_events_count_lag1'] = np.random.randint(0,5,size=sample_data_rows)
    dummy_data['fatalities_lag1'] = np.random.randint(0,3,size=sample_data_rows)

    # Ensure current month base cols are also somewhat realistic for exclusion test
    for col in ['total_events', 'fatalities', 'violent_events_count']:
        if col in dummy_data.columns: # Check if already created, e.g. fatalities
             dummy_data[col] = np.random.randint(0,5,size=sample_data_rows)


    dummy_data_path = 'data/dummy_acled_modeling_data_prepared.csv'
    dummy_data.to_csv(dummy_data_path, index=False)
    print(f"Created dummy data at {dummy_data_path}")
    
    # Since shapefile processing is complex, we'll run with shapefile_path=None by default for this test
    # If you have a test shapefile, provide its path here.
    # e.g., test_shapefile_path = 'path/to/your/test_admin1.shp'
    test_shapefile_path = None 
    
    # Define output paths for the test run
    test_output_model_path = 'models/test_spatial_model.pkl'
    test_output_charts_dir = 'charts_test'

    # Ensure output directories for the test run exist
    os.makedirs(os.path.dirname(test_output_model_path), exist_ok=True)
    os.makedirs(test_output_charts_dir, exist_ok=True)

    results = train_spatial_model(
        data_path=dummy_data_path,
        shapefile_path=test_shapefile_path,
        output_model_path=test_output_model_path,
        output_charts_dir=test_output_charts_dir
    )
    if results:
        print("Test run completed. Model and charts saved to test paths.")
        print(f"Test Model PR-AUC: {results.get('pr_auc', 'N/A')}")