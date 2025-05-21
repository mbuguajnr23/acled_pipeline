import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import pickle
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import calendar
from datetime import datetime, timedelta

def visualize_conflict_risk(model_path, data_path, admin1_shapefile=None):
    """
    Generate and visualize conflict risk predictions across Africa
    
    Parameters:
    -----------
    model_path : str
        Path to the trained model pickle file
    data_path : str
        Path to the prepared modeling data
    admin1_shapefile : str
        Path to shapefile with admin1 boundaries (optional)
    """
    print("Loading model and data...")
    # Load the model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load the data
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Get the most recent date in the data
    latest_date = df['date'].max()
    
    # Filter to the latest data point for each region
    latest_data = df[df['date'] == latest_date].copy()
    
    # Define feature columns (excluding non-feature columns)
    non_feature_cols = ['country', 'admin1', 'date', 'year_month', 
                        'future_violent_events', 'conflict_occurs']
    current_month_cols = ['total_events', 'violent_events', 'fatalities', 
                         'event_diversity', 'actor1_count', 'actor2_count']
    
    feature_cols = [col for col in df.columns if col not in non_feature_cols 
                   and col not in current_month_cols]
    
    # Make predictions for the next time period
    X_predict = latest_data[feature_cols]
    
    # Handle any missing features that might be in the model but not in our data
    missing_cols = set(model.feature_names_in_) - set(X_predict.columns)
    for col in missing_cols:
        X_predict[col] = 0  # Add missing columns with default values
    
    # Ensure columns are in the right order
    X_predict = X_predict[model.feature_names_in_]
    
    # Generate predictions
    latest_data['conflict_probability'] = model.predict_proba(X_predict)[:, 1]
    
    # Define the prediction month (3 months from latest date by default)
    pred_month = latest_date + pd.DateOffset(months=3)
    pred_month_str = pred_month.strftime('%B %Y')
    
    print(f"Generating conflict risk predictions for {pred_month_str}...")
    
    # Create a risk category
    risk_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    risk_labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
    latest_data['risk_category'] = pd.cut(latest_data['conflict_probability'], 
                                         bins=risk_bins, labels=risk_labels)
    
    # Save predictions to CSV
    prediction_df = latest_data[['country', 'admin1', 'conflict_probability', 'risk_category']]
    prediction_df['prediction_for'] = pred_month_str
    prediction_df.to_csv(f'conflict_predictions_{pred_month.strftime("%Y_%m")}.csv', index=False)
    
    print(f"Saved predictions to 'conflict_predictions_{pred_month.strftime('%Y_%m')}.csv'")
    
    # Risk distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(latest_data['conflict_probability'], bins=20, kde=True)
    plt.title(f'Distribution of Conflict Risk - Prediction for {pred_month_str}')
    plt.xlabel('Conflict Probability')
    plt.ylabel('Count of Admin1 Regions')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'risk_distribution_{pred_month.strftime("%Y_%m")}.png')
    
    # Top highest risk regions
    top_risk = latest_data.sort_values('conflict_probability', ascending=False).head(20)
    plt.figure(figsize=(12, 8))
    g = sns.barplot(x='conflict_probability', y='admin1', hue='country', 
                   data=top_risk, dodge=False)
    plt.title(f'Top 20 Highest Risk Regions - Prediction for {pred_month_str}')
    plt.xlabel('Conflict Probability')
    plt.ylabel('Admin1 Region')
    plt.tight_layout()
    plt.savefig(f'top_risk_regions_{pred_month.strftime("%Y_%m")}.png')
    
    print("Top 10 highest risk regions:")
    for i, (_, row) in enumerate(top_risk.head(10).iterrows()):
        print(f"{i+1}. {row['admin1']}, {row['country']}: {row['conflict_probability']:.4f} ({row['risk_category']})")
    
    # If shapefile is provided, create a geographical visualization
    if admin1_shapefile:
        try:
            print("Creating geographical visualization...")
            # Load the shapefile
            gdf = gpd.read_file(admin1_shapefile)
            
            # Ensure shapefile has matching columns
            if 'COUNTRY' in gdf.columns and 'ADMIN1' in gdf.columns:
                gdf = gdf.rename(columns={'COUNTRY': 'country', 'ADMIN1': 'admin1'})
            
            # Merge predictions with geometry
            geo_risk = gdf.merge(latest_data[['country', 'admin1', 'conflict_probability', 'risk_category']], 
                                on=['country', 'admin1'], how='left')
            
            # Create a custom colormap for risk levels (yellow to red)
            colors = ['#ffffcc', '#ffeda0', '#feb24c', '#fc4e2a', '#bd0026']
            cmap = LinearSegmentedColormap.from_list('conflict_risk', colors, N=5)
            
            # Plot the map
            fig, ax = plt.subplots(1, 1, figsize=(15, 15))
            geo_risk.plot(column='conflict_probability', ax=ax, cmap=cmap, 
                         legend=True, legend_kwds={'label': 'Conflict Probability'},
                         missing_kwds={'color': 'lightgray'})
            
            # Add country borders with thicker lines
            gdf.dissolve(by='country').boundary.plot(ax=ax, color='black', linewidth=0.5)
            
            # Add title and annotations
            plt.title(f'Predicted Conflict Risk for {pred_month_str}', fontsize=16)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'conflict_risk_map_{pred_month.strftime("%Y_%m")}.png', dpi=300)
            print(f"Saved map visualization to 'conflict_risk_map_{pred_month.strftime('%Y_%m')}.png'")
            
        except Exception as e:
            print(f"Error creating geographical visualization: {e}")
    
    # Generate a country-level summary
    country_risk = latest_data.groupby('country').agg({
        'conflict_probability': ['mean', 'median', 'max', 'count']
    })
    country_risk.columns = ['mean_risk', 'median_risk', 'max_risk', 'regions_count']
    country_risk = country_risk.sort_values('mean_risk', ascending=False)
    
    # Save country summary
    country_risk.to_csv(f'country_risk_summary_{pred_month.strftime("%Y_%m")}.csv')
    
    # Plot country-level risk
    plt.figure(figsize=(12, 10))
    sns.barplot(x='mean_risk', y=country_risk.index, data=country_risk.reset_index(), 
               xerr=country_risk['mean_risk'].std())
    plt.title(f'Average Conflict Risk by Country - Prediction for {pred_month_str}')
    plt.xlabel('Mean Conflict Probability')
    plt.ylabel('Country')
    plt.tight_layout()
    plt.savefig(f'country_risk_{pred_month.strftime("%Y_%m")}.png')
    
    print("Top 5 highest risk countries (by mean risk):")
    for i, (country, row) in enumerate(country_risk.head(5).iterrows()):
        print(f"{i+1}. {country}: Mean Risk {row['mean_risk']:.4f}, Max Risk {row['max_risk']:.4f} ({row['regions_count']} regions)")
    
    print("\nAnalysis complete!")
    return prediction_df

def create_deployment_ready_prediction_function(model_path):
    """
    Create a standalone prediction function that can be deployed
    
    Parameters:
    -----------
    model_path : str
        Path to the trained model pickle file
        
    Returns:
    --------
    A function that can make predictions given new data
    """
    # Load the model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Get feature names from the model
    feature_names = model.feature_names_in_
    
    # Create the prediction function
    def predict_conflict_risk(new_data):
        """
        Predict conflict risk for new data
        
        Parameters:
        -----------
        new_data : DataFrame
            New data with the same features used during training
            
        Returns:
        --------
        DataFrame with the original data plus conflict probability predictions
        """
        # Ensure all required features are present
        missing_cols = set(feature_names) - set(new_data.columns)
        for col in missing_cols:
            new_data[col] = 0  # Add missing columns with default values
        
        # Ensure columns are in the right order
        X_new = new_data[feature_names]
        
        # Make predictions
        new_data['conflict_probability'] = model.predict_proba(X_new)[:, 1]
        
        # Add risk category
        risk_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        risk_labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
        new_data['risk_category'] = pd.cut(new_data['conflict_probability'], 
                                         bins=risk_bins, labels=risk_labels)
        
        return new_data
    
    print(f"Created deployment-ready prediction function using model: {model_path}")
    print(f"Required features: {', '.join(feature_names[:5])}... (total: {len(feature_names)})")
    
    return predict_conflict_risk

if __name__ == "__main__":
    # Adjust these paths as needed
    visualize_conflict_risk(
        model_path='spatial_conflict_model.pkl', 
        data_path='data\acled_modeling_data.csv',
        admin1_shapefile=None  # Add path to shapefile if available
    )
    
    # Create a deployable prediction function
    predict_fn = create_deployment_ready_prediction_function('spatial_conflict_model.pkl')
