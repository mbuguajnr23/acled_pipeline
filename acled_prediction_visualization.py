import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
# import pickle # No longer needed if model_object is passed
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
# import calendar # Not used
from datetime import datetime, timedelta
import logging
import os

# Configure logging
from acled_spatial_model import add_spatial_features

logger_viz = logging.getLogger("ACLED_Pipeline.Visualization")

def visualize_conflict_risk(model_object, data_path, admin1_shapefile=None, output_charts_dir='charts_dir', output_reports_dir='reports_dir', prediction_offset_months=1 ):
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
    model = model_object
    
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
    risk_bins = [0, 0.05, 0.2, 0.5, 0.8, 1.0] # Adjusted bins for more granularity at low end
    risk_labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
    df['risk_category'] = pd.cut(df['conflict_probability'],
                                 bins=risk_bins, labels=risk_labels, right=True, include_lowest=True)

    # Save all predictions to CSV (this will be a large file)
    full_predictions_path = os.path.join(output_reports_dir, f'full_conflict_predictions_{prediction_file_suffix}.csv')
    df[['date', 'country', 'admin1', 'conflict_probability', 'risk_category']].to_csv(full_predictions_path, index=False)
    logger_viz.info(f"Saved all predictions to: {full_predictions_path}")

    # --- Visualizations and Reports based on the LATEST available feature date ---
    # This means we are visualizing the predicted risk for (latest_date_in_data + prediction_offset_months)
    # using features from latest_date_in_data.
    data_for_latest_viz = df[df['date'] == latest_date_in_data].copy()
    if data_for_latest_viz.empty:
        logger_viz.warning(f"No data found for the latest date {latest_date_in_data} for visualization.")
    else:
        # Risk distribution for the latest prediction period
        plt.figure(figsize=(10, 6))
        sns.histplot(data_for_latest_viz['conflict_probability'], bins=20, kde=True)
        plt.title(f'Distribution of Conflict Risk - Prediction for {prediction_label_month_str}')
        plt.xlabel('Conflict Probability'); plt.ylabel('Count of Admin1 Regions')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_charts_dir, f'risk_distribution_{prediction_file_suffix}.png'))
        plt.close()
        logger_viz.info(f"Saved risk distribution plot for {prediction_label_month_str}.")

        # Top highest risk regions for the latest prediction period
        top_risk = data_for_latest_viz.sort_values('conflict_probability', ascending=False).head(20)
        plt.figure(figsize=(12, 8))
        try: # hue might cause issues if countries are not diverse in top N
            sns.barplot(x='conflict_probability', y='admin1', hue='country', data=top_risk, dodge=False)
        except:
            sns.barplot(x='conflict_probability', y='admin1', data=top_risk, dodge=False) # Fallback without hue
        plt.title(f'Top 20 Highest Risk Regions - Prediction for {prediction_label_month_str}')
        plt.xlabel('Conflict Probability'); plt.ylabel('Admin1 Region')
        plt.tight_layout()
        plt.savefig(os.path.join(output_charts_dir, f'top_risk_regions_{prediction_file_suffix}.png'))
        plt.close()
        logger_viz.info(f"Saved top risk regions plot for {prediction_label_month_str}.")

        logger_viz.info(f"Top 10 highest risk regions (prediction for {prediction_label_month_str}):")
        for i, row_tuple in enumerate(top_risk.head(10).itertuples()):
            logger_viz.info(f"  {i+1}. {getattr(row_tuple, 'admin1', 'N/A')}, {getattr(row_tuple, 'country', 'N/A')}: {getattr(row_tuple, 'conflict_probability', 0.0):.4f} ({getattr(row_tuple, 'risk_category', 'N/A')})")

        # Geographical visualization for the latest prediction period
        if admin1_shapefile_path:
            try:
                logger_viz.info(f"Creating geographical visualization using shapefile: {admin1_shapefile_path}")
                admin_gdf = gpd.read_file(admin1_shapefile_path)
                # Standardize columns in admin_gdf
                rename_map_geo = {}
                for col_n in ['CNTRY_NAME', 'COUNTRY', 'NAME_0', 'ADM0_NAME', 'admin0Name', 'COUNTRYNA']:
                    if col_n in admin_gdf.columns: rename_map_geo[col_n] = 'country'; break
                for col_n in ['ADMIN1', 'NAME_1', 'ADM1_NAME', 'admin1Name', 'province', 'state', 'ADM1_EN', 'shapeName']:
                    if col_n in admin_gdf.columns: rename_map_geo[col_n] = 'admin1'; break
                
                if 'country' not in rename_map_geo and 'country' in admin_gdf.columns: rename_map_geo['country'] = 'country'
                if 'admin1' not in rename_map_geo and 'admin1' in admin_gdf.columns: rename_map_geo['admin1'] = 'admin1'
                
                if 'country' not in rename_map_geo or 'admin1' not in rename_map_geo:
                     logger_viz.warning(f"Could not reliably find 'country'/'admin1' in shapefile {admin_gdf.columns.tolist()}. Map may be incorrect.")
                admin_gdf = admin_gdf.rename(columns=rename_map_geo)
                
                # Ensure CRS consistency
                if admin_gdf.crs and admin_gdf.crs.to_string() != "EPSG:4326":
                    admin_gdf = admin_gdf.to_crs("EPSG:4326")

                geo_risk = admin_gdf.merge(data_for_latest_viz[['country', 'admin1', 'conflict_probability', 'risk_category']],
                                           on=['country', 'admin1'], how='left')
                geo_risk['conflict_probability'] = geo_risk['conflict_probability'].fillna(-1) # Use -1 for missing for distinct color

                # Custom colormap: lightgrey for missing, then risk colors
                colors = ['lightgrey', '#ffffcc', '#ffeda0', '#feb24c', '#fc4e2a', '#bd0026'] # Added lightgrey for -1
                # Values for cmap bins should correspond to probabilities + missing category
                # If using bins for probability, map them to colors
                cmap = LinearSegmentedColormap.from_list('conflict_risk', colors, N=len(colors))

                fig, ax = plt.subplots(1, 1, figsize=(15, 15))
                geo_risk.plot(column='conflict_probability', ax=ax, cmap=cmap,
                              legend=True, legend_kwds={'label': 'Conflict Probability (Regions with -1 have no prediction data)'},
                              missing_kwds=None, # Already handled by fillna(-1) and cmap
                              vmin=-1, vmax=1) # Ensure -1 is at the bottom of cmap
                
                # Add country borders
                try:
                    # Create a dissolved layer for country boundaries if possible
                    country_boundaries = admin_gdf.dissolve(by='country', as_index=False)
                    country_boundaries.boundary.plot(ax=ax, color='black', linewidth=0.5)
                except Exception as e_bound:
                    logger_viz.warning(f"Could not plot dissolved country boundaries: {e_bound}. Plotting raw boundaries.")
                    admin_gdf.boundary.plot(ax=ax, color='grey', linewidth=0.3)


                plt.title(f'Predicted Conflict Risk for {prediction_label_month_str}', fontsize=16)
                plt.axis('off'); plt.tight_layout()
                map_save_path = os.path.join(output_charts_dir, f'conflict_risk_map_{prediction_file_suffix}.png')
                plt.savefig(map_save_path, dpi=300)
                plt.close(fig)
                logger_viz.info(f"Saved map visualization to {map_save_path}")
            except Exception as e:
                logger_viz.error(f"Error creating geographical visualization: {e}", exc_info=True)
        else:
            logger_viz.warning("No admin1 shapefile provided. Skipping map visualization.")

        # Country-level summary for the latest prediction period
        country_risk = data_for_latest_viz.groupby('country').agg(
            mean_risk=('conflict_probability', 'mean'),
            median_risk=('conflict_probability', 'median'),
            max_risk=('conflict_probability', 'max'),
            regions_with_data_count=('conflict_probability', 'count') # Count non-NaN probabilities
        ).sort_values('mean_risk', ascending=False)
        
        country_summary_path = os.path.join(output_reports_dir, f'country_risk_summary_{prediction_file_suffix}.csv')
        country_risk.to_csv(country_summary_path)
        logger_viz.info(f"Saved country risk summary to: {country_summary_path}")

        plt.figure(figsize=(12, 10))
        # Only plot countries with data
        country_risk_to_plot = country_risk[country_risk['regions_with_data_count'] > 0].head(20) # Top 20
        sns.barplot(x='mean_risk', y=country_risk_to_plot.index, data=country_risk_to_plot.reset_index())
        plt.title(f'Average Conflict Risk by Country - Prediction for {prediction_label_month_str}')
        plt.xlabel('Mean Conflict Probability'); plt.ylabel('Country')
        plt.tight_layout()
        plt.savefig(os.path.join(output_charts_dir, f'country_risk_barchart_{prediction_file_suffix}.png'))
        plt.close()
        logger_viz.info(f"Saved country risk bar chart for {prediction_label_month_str}.")

        logger_viz.info(f"Top 5 highest risk countries (prediction for {prediction_label_month_str}, by mean risk):")
        for i, (country_name, row_data) in enumerate(country_risk_to_plot.head(5).iterrows()):
            logger_viz.info(f"  {i+1}. {country_name}: Mean Risk {row_data['mean_risk']:.4f}, Max Risk {row_data['max_risk']:.4f} ({int(row_data['regions_with_data_count'])} regions)")

    logger_viz.info("Prediction visualization complete.")
    return df[['date', 'country', 'admin1', 'conflict_probability', 'risk_category']] # Return the full df with predictions

# create_deployment_ready_prediction_function remains mostly the same,
# but ensure it uses the same NaN filling strategy and feature list derivation.
def create_deployment_ready_prediction_function(model_object): # Changed to accept model_object
    """
    Create a standalone prediction function that can be deployed.
    """
    logger_viz.info("Creating deployment-ready prediction function...")
    model = model_object # Model is already loaded
    feature_names = model.feature_names_in_
    
    def predict_conflict_risk_deployed(new_data_df: pd.DataFrame):
        # Ensure new_data_df is a DataFrame
        if not isinstance(new_data_df, pd.DataFrame):
            try:
                new_data_df = pd.DataFrame(new_data_df) # Attempt to convert if list of dicts, etc.
            except Exception as e:
                raise ValueError(f"Input data must be a pandas DataFrame or convertible. Error: {e}")

        X_new = new_data_df.copy()
        missing_cols = set(feature_names) - set(X_new.columns)
        for col in missing_cols:
            X_new[col] = 0 # Or -999, consistent with training NaN strategy
        
        X_new = X_new[feature_names].fillna(-999) # Select in order and fill NaNs
        
        new_data_df['conflict_probability'] = model.predict_proba(X_new)[:, 1]
        
        risk_bins = [0, 0.05, 0.2, 0.5, 0.8, 1.0]
        risk_labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
        new_data_df['risk_category'] = pd.cut(new_data_df['conflict_probability'],
                                              bins=risk_bins, labels=risk_labels, right=True, include_lowest=True)
        return new_data_df[['conflict_probability', 'risk_category'] + [col for col in new_data_df.columns if col not in ['conflict_probability', 'risk_category']]]

    logger_viz.info(f"Deployment-ready function created. Model expects features like: {', '.join(list(feature_names)[:5])}... (total: {len(feature_names)})")
    return predict_conflict_risk_deployed


if __name__ == "__main__":
    # This __main__ block is for testing this script in isolation.
    # The acled_complete_pipeline.py will call visualize_conflict_risk directly.
    
    print("Running acled_prediction_visualization.py directly (example/test mode)")

    # Setup a dummy logger if running standalone
    if not logging.getLogger("ACLED_Pipeline").hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Create dummy model and data for testing
    class DummyModel:
        def __init__(self, features):
            self.feature_names_in_ = features
        def predict_proba(self, X):
            # Return probabilities for two classes, ensure it sums to 1 per row
            proba_class1 = np.random.rand(len(X), 1) * 0.7 # Max prob 0.7
            return np.hstack([1 - proba_class1, proba_class1])

    # Define some plausible feature names (should match what your model expects)
    test_feature_names = [
        'violent_events_count_lag1', 'fatalities_lag1', 'days_since_last_violent_event',
        'total_events_roll_mean3', 'protests_count_lag6', 'neighbor_conflict_density_lag1',
        'country_other_admin1_violent_events_lag1'
        # Add more features that your actual model uses
    ]
    if not test_feature_names: # Fallback if list is empty
        test_feature_names = [f'feature_{i}' for i in range(10)]


    dummy_model_obj = DummyModel(features=np.array(test_feature_names))
    
    sample_rows = 200
    dummy_pred_data = pd.DataFrame(np.random.rand(sample_rows, len(test_feature_names)) * 10, columns=test_feature_names)
    dummy_pred_data['date'] = pd.to_datetime('2023-01-01') + pd.to_timedelta(np.random.randint(0, 60, sample_rows), unit='D')
    dummy_pred_data['country'] = [f'Country_{i%2}' for i in range(sample_rows)]
    dummy_pred_data['admin1'] = [f'Admin1_{i%5}' for i in range(sample_rows)] # 5 admin1s per country
    
    # Create dummy data file
    os.makedirs('data_test_viz', exist_ok=True)
    dummy_data_file_path = 'data_test_viz/dummy_data_for_viz.csv'
    dummy_pred_data.to_csv(dummy_data_file_path, index=False)

    # Define output directories for the test
    test_charts_dir = 'charts_test_viz'
    test_reports_dir = 'reports_test_viz'

    # Call the visualization function
    output_df = visualize_conflict_risk(
        model_object=dummy_model_obj,
        data_for_prediction_path=dummy_data_file_path,
        admin1_shapefile_path=None, # Provide a path to a test shapefile if you have one
        output_charts_dir=test_charts_dir,
        output_reports_dir=test_reports_dir,
        prediction_offset_months=1
    )
    
    if not output_df.empty:
        print("\nSample of predictions from test run:")
        print(output_df.head())
    else:
        print("\nTest run of visualize_conflict_risk did not produce output DataFrame.")

    # Test the deployment function
    # deploy_fn = create_deployment_ready_prediction_function(dummy_model_obj)
    # sample_input_for_deploy = pd.DataFrame(np.random.rand(5, len(test_feature_names)) * 10, columns=test_feature_names)
    # deployed_predictions = deploy_fn(sample_input_for_deploy)
    # print("\nSample predictions from deployed function:")
    # print(deployed_predictions)