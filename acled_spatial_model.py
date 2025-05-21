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
    """
    Add spatial features to the dataset
    
    Parameters:
    -----------
    df : DataFrame
        Prepared ACLED data with features
    shapefile_path : str
        Path to a shapefile containing admin1 boundaries (optional)
        
    Returns:
    --------
    DataFrame with additional spatial features
    """
    print("Adding spatial features...")
    
    # If shapefile is provided, we can use it to find neighboring regions
    if shapefile_path:
        try:
            # Load shapefile with admin1 boundaries
            gdf = gpd.read_file(shapefile_path)
            
            # Ensure we have country and admin1 columns matching our data
            if 'COUNTRY' in gdf.columns and 'ADMIN1' in gdf.columns:
                gdf = gdf.rename(columns={'COUNTRY': 'country', 'ADMIN1': 'admin1'})
            
            # Find neighboring regions
            neighbors = {}
            for idx, row in gdf.iterrows():
                country = row['country']
                admin1 = row['admin1']
                geom = row['geometry']
                
                # Find regions that share a boundary
                neighbors[(country, admin1)] = []
                for idx2, row2 in gdf.iterrows():
                    if idx != idx2:
                        if geom.touches(row2['geometry']):
                            neighbors[(country, admin1)].append((row2['country'], row2['admin1']))
            
            print(f"Identified neighbors for {len(neighbors)} regions")
            
            # Group the data by date
            date_groups = df.groupby('date')
            
            # Initialize new columns for neighbor features
            neighbor_features = [
                'neighbor_violent_events', 
                'neighbor_fatalities',
                'neighbor_conflict_density'
            ]
            
            for col in neighbor_features:
                df[col] = 0.0
            
            # For each date, calculate spatial features
            for date, group in date_groups:
                # Create a dictionary for quick lookups
                region_data = {(row['country'], row['admin1']): row for _, row in group.iterrows()}
                
                # Update the main dataframe with neighbor information
                for idx, row in group.iterrows():
                    country = row['country']
                    admin1 = row['admin1']
                    
                    if (country, admin1) in neighbors:
                        neighbor_list = neighbors[(country, admin1)]
                        
                        # Calculate metrics from neighbors
                        if neighbor_list:
                            n_violent_events = 0
                            n_fatalities = 0
                            
                            for n_country, n_admin1 in neighbor_list:
                                if (n_country, n_admin1) in region_data:
                                    n_row = region_data[(n_country, n_admin1)]
                                    n_violent_events += n_row['violent_events_lag1']
                                    n_fatalities += n_row['fatalities_lag1']
                            
                            # Average neighbor metrics
                            n_count = len(neighbor_list)
                            df.at[idx, 'neighbor_violent_events'] = n_violent_events / n_count
                            df.at[idx, 'neighbor_fatalities'] = n_fatalities / n_count
                            df.at[idx, 'neighbor_conflict_density'] = sum(
                                1 for n in neighbor_list if (n in region_data and 
                                                            region_data[n]['violent_events_lag1'] > 0)
                            ) / n_count
            
            print("Added spatial features based on shapefile")
            
        except Exception as e:
            print(f"Error processing shapefile: {e}")
            print("Falling back to country-based proximity...")
            # Fall back to country-based proximity method
            add_country_based_spatial_features(df)
    else:
        # Use country-based proximity as a simpler approach
        add_country_based_spatial_features(df)
    
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
        'country_violent_events_lag1': 'violent_events_lag1',
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
                              if row['violent_events_lag1'] > 0)
        
        conflict_density = conflict_regions / total_regions if total_regions > 0 else 0
        
        # Update all rows for this country and date
        for idx in group.index:
            df.loc[idx, 'country_conflict_density'] = conflict_density
    
    print("Added country-based spatial features")
    return df

def train_spatial_model(data_path='data\acled_modeling_data.csv', shapefile_path=None):
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
    plt.savefig('charts\pr_curve_spatial_model.png')
    
    # Feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances - Spatial Model')
    plt.bar(range(min(20, X_train.shape[1])), importances[indices[:20]], align='center')
    plt.xticks(range(min(20, X_train.shape[1])), [feature_cols[i] for i in indices[:20]], rotation=90)
    plt.tight_layout()
    plt.savefig('charts\feature_importance_spatial_model.png')
    
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
    train_spatial_model(data_path='data\acled_modeling_data.csv', shapefile_path=None)