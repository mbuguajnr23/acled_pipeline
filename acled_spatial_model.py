import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
import xgboost as xgb
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')

def add_spatial_features(df, shapefile_path=None):
    print("Adding spatial features...")
    if not shapefile_path:
        print("No shapefile path provided. Falling back to country-based proximity.")
        return add_country_based_spatial_features(df)

    try:
        gdf = gpd.read_file(shapefile_path)
        # Standardize GDF columns (example, make this more robust or parametric)
        # Common admin level 0 (country) and 1 (province/state) names
        # This needs to be robust to different shapefile schemas
        rename_map = {}
        # Try common country name variants
        for col_name in ['CNTRY_NAME', 'COUNTRY', 'NAME_0', 'ADM0_NAME', 'admin0Name']:
            if col_name in gdf.columns:
                rename_map[col_name] = 'country'
                break
        # Try common admin1 name variants
        for col_name in ['ADMIN1', 'NAME_1', 'ADM1_NAME', 'admin1Name', 'province', 'state']:
            if col_name in gdf.columns:
                rename_map[col_name] = 'admin1'
                break
        
        if 'country' not in rename_map or 'admin1' not in rename_map:
            raise ValueError("Could not find standard country/admin1 columns in shapefile.")
        gdf = gdf.rename(columns=rename_map)
        gdf = gdf[['country', 'admin1', 'geometry']] # Keep only necessary columns

        # Ensure CRS is consistent or reproject (assuming WGS84 for ACLED)
        if gdf.crs and gdf.crs != "EPSG:4326":
            print(f"Reprojecting shapefile from {gdf.crs} to EPSG:4326")
            gdf = gdf.to_crs("EPSG:4326")

        # Create spatial index
        if not gdf.sindex: # Build it if it does not exist (geopandas might do this automatically on first access)
             gdf.sindex

        neighbors_map = {}
        print("Identifying neighbors using spatial index...")
        for idx, focal_row in gdf.iterrows():
            focal_geom = focal_row['geometry']
            focal_region_id = (focal_row['country'], focal_row['admin1'])
            neighbors_map[focal_region_id] = []

            # Query spatial index for potential neighbors (those whose bounding boxes intersect)
            possible_matches_indices = list(gdf.sindex.intersection(focal_geom.bounds))
            possible_matches = gdf.iloc[possible_matches_indices]
            
            # Perform precise check (touches) only on these candidates
            # Exclude self using the original index 'idx'
            precise_neighbors = possible_matches[possible_matches.geometry.touches(focal_geom) & (possible_matches.index != idx)]
            
            for _, neighbor_row in precise_neighbors.iterrows():
                neighbors_map[focal_region_id].append((neighbor_row['country'], neighbor_row['admin1']))
        
        print(f"Identified neighbors for {len(neighbors_map)} regions.")

        # Initialize new columns for neighbor features
        neighbor_feature_cols = ['neighbor_violent_events_lag1_avg', 'neighbor_fatalities_lag1_avg', 'neighbor_conflict_density_lag1']
        for col in neighbor_feature_cols:
            df[col] = 0.0

        # Lagged columns we need from neighbors
        neighbor_source_cols = {
            'violent_events_count_lag1': 'neighbor_violent_events_lag1_avg', 
            'fatalities_lag1': 'neighbor_fatalities_lag1_avg',
            # For density, we need violent_events_count_lag1
        }

        # Check if source lag columns exist
        for source_col in neighbor_source_cols.keys():
            if source_col not in df.columns:
                print(f"WARNING: Source column '{source_col}' for neighbor features not found in DataFrame. Spatial features might be all zeros.")
                # Fallback or skip if critical columns are missing
                # return add_country_based_spatial_features(df.copy()) # Pass a copy to avoid modifying original df in a failed attempt

        print("Calculating spatial features from neighbors...")
        # Group by date to process each time slice
        # Iterating this way can still be slow. A merge-based approach would be faster.

        
        # Create a temporary mapping for faster lookups within each date group
        # df needs to be indexed by ['date', 'country', 'admin1'] for faster lookup
        df_indexed = df.set_index(['date', 'country', 'admin1'])

        for idx_df, row_df in df.iterrows(): # Iterate over the original df to fill its rows
            current_date = row_df['date']
            focal_region_id = (row_df['country'], row_df['admin1'])

            if focal_region_id not in neighbors_map:
                continue # Region not in shapefile or no neighbors found

            current_neighbors = neighbors_map[focal_region_id]
            if not current_neighbors:
                continue

            sum_neighbor_violent_lag1 = 0
            sum_neighbor_fatalities_lag1 = 0
            active_neighbors_lag1 = 0
            valid_neighbors_count = 0

            for neighbor_id in current_neighbors:
                try:
                    # Lookup neighbor's data for the same date
                    neighbor_data_row = df_indexed.loc[(current_date, neighbor_id[0], neighbor_id[1])]
                    
                    # Accumulate based on *EXISTING LAG FEATURES* in neighbor_data_row
                    if 'violent_events_count_lag1' in neighbor_data_row: # Check if column exists
                        sum_neighbor_violent_lag1 += neighbor_data_row.get('violent_events_count_lag1', 0) # Use .get for safety
                    if 'fatalities_lag1' in neighbor_data_row:
                        sum_neighbor_fatalities_lag1 += neighbor_data_row.get('fatalities_lag1', 0)
                    if neighbor_data_row.get('violent_events_count_lag1', 0) > 0:
                        active_neighbors_lag1 += 1
                    valid_neighbors_count += 1
                except KeyError:
                    # Neighbor data not found for this date (shouldn't happen if grid is complete)
                    pass # Or log this

            if valid_neighbors_count > 0:
                df.at[idx_df, 'neighbor_violent_events_lag1_avg'] = sum_neighbor_violent_lag1 / valid_neighbors_count
                df.at[idx_df, 'neighbor_fatalities_lag1_avg'] = sum_neighbor_fatalities_lag1 / valid_neighbors_count
                df.at[idx_df, 'neighbor_conflict_density_lag1'] = active_neighbors_lag1 / valid_neighbors_count
        
        print("Added spatial features based on shapefile.")

    except Exception as e:
        print(f"Error processing shapefile: {e}")
        print("Falling back to country-based proximity...")
        df = add_country_based_spatial_features(df.copy()) # Pass a copy
    return df

def add_country_based_spatial_features(df):
    """
    Add simplified spatial features based on country proximity
    when a shapefile is not available
    """
    print("Adding country-based spatial features...")
    
    # Group by country and date
    country_date_groups = df.groupby(['date', 'country'])
    
    # Add country-level features
    country_features = {
        'country_violent_events_lag1': 'violent_events_count_lag1',
        'country_fatalities_lag1': 'fatalities_lag1'
    }
    
    for new_col, source_col in country_features.items():
        df[new_col] = 0.0
    
    # Calculate country aggregates
    for (date, country), group in country_date_groups:
        # Calculate country-level metrics
        country_violent = group[country_features['country_violent_events_lag1']].sum()
        country_fatalities = group[country_features['country_fatalities_lag1']].sum()
        
        # Update rows for this country and date
        for idx in group.index:
            admin1_violent = df.loc[idx, country_features['country_violent_events_lag1']]
            admin1_fatalities = df.loc[idx, country_features['country_fatalities_lag1']]
            
            # Country total excluding this admin1
            df.loc[idx, 'country_violent_events_lag1'] = country_violent - admin1_violent
            df.loc[idx, 'country_fatalities_lag1'] = country_fatalities - admin1_fatalities
    
    # Add country-level conflict density (percentage of admin1 regions with conflict)
    for (date, country), group in country_date_groups:
        total_regions = len(group)
        conflict_regions = sum(1 for _, row in group.iterrows() 
                              if row['violent_events_count_lag1'] > 0)
        
        conflict_density = conflict_regions / total_regions if total_regions > 0 else 0
        
        # Update all rows for this country and date
        for idx in group.index:
            df.loc[idx, 'country_conflict_density'] = conflict_density
    
    print("Added country-based spatial features")
    return df

def train_spatial_model(data_path, shapefile_path=None):
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
    print("Loading prepared data...")
    df = pd.read_csv(data_path)
    
    # Convert date back to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Add spatial features
    df = add_spatial_features(df, shapefile_path)
    
    # Exclude target-related columns and non-feature columns from features
    non_feature_cols = ['country', 'admin1', 'date', 'year_month', 
                        'future_violent_events', 'conflict_occurs']
    
    # Also exclude the current month's metrics
    current_month_cols = ['total_events', 'violent_events', 'fatalities', 
                         'event_diversity', 'actor1_count', 'actor2_count']
    
    feature_cols = [col for col in df.columns if col not in non_feature_cols 
                   and col not in current_month_cols]
    
    X = df[feature_cols]
    y = df['conflict_occurs']
    
    print(f"Number of features: {len(feature_cols)}")
    
    # Create a date-based cut for temporal validation
    cutoff_date = df['date'].sort_values().iloc[int(len(df) * 0.8)]
    print(f"Time-based validation: using data before {cutoff_date} for training, after for testing")
    
    train_mask = df['date'] < cutoff_date
    test_mask = df['date'] >= cutoff_date
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")
    
    # Train XGBoost model with spatial features
    print("\nTraining XGBoost with spatial features...")
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', 
                             n_estimators=200, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    
    # ROC and PR curves
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC AUC: {roc_auc:.4f}")
    
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    print(f"PR AUC: {pr_auc:.4f}")
    
    # Plot PR curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - Spatial Model (AUC: {pr_auc:.4f})')
    plt.grid(True)
    plt.savefig('charts/pr_curve_spatial_model.png')
    
    # Feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances - Spatial Model')
    plt.bar(range(min(20, X_train.shape[1])), importances[indices[:20]], align='center')
    plt.xticks(range(min(20, X_train.shape[1])), [feature_cols[i] for i in indices[:20]], rotation=90)
    plt.tight_layout()
    plt.savefig('charts/feature_importance_spatial_model.png')
    
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
    with open('spatial_conflict_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Saved model to 'spatial_conflict_model.pkl'")
    
    # Return results
    results = {
        'model': model,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'feature_importances': feature_importance_df
    }
    
    return results

if __name__ == "__main__":
    # Adjust paths as needed
    data_path='data/acled_modeling_data_prepared.csv'
    train_spatial_model(data_path, shapefile_path=None)