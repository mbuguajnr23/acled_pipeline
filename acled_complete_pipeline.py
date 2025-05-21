import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime
import logging

# Import our modules
from acled_feature_engineering import prepare_acled_data
from acled_spatial_model import train_spatial_model
from acled_prediction_visualization import visualize_conflict_risk

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("acled_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ACLED_Pipeline")

def run_complete_pipeline(
    acled_file_path,
    output_dir="output",
    prediction_window_months=3,
    event_threshold=5,
    run_baseline=True,
    shapefile_path=None
):
    """
    Run the complete ACLED conflict prediction pipeline
    
    Parameters:
    -----------
    acled_file_path : str
        Path to the ACLED CSV data file
    output_dir : str
        Directory to save outputs
    prediction_window_months : int
        How many months ahead to predict
    event_threshold : int
        Minimum number of violent events to classify as 'conflict'
    run_baseline : bool
        Whether to run the baseline model in addition to spatial model
    shapefile_path : str
        Path to admin1 boundary shapefile (optional)
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    # Set working directory to output directory
    original_dir = os.getcwd()
    os.chdir(output_dir)
    
    try:
        start_time = datetime.now()
        logger.info(f"Starting ACLED conflict prediction pipeline at {start_time}")
        logger.info(f"Using ACLED data from: {acled_file_path}")
        logger.info(f"Prediction window: {prediction_window_months} months")
        logger.info(f"Event threshold for conflict: {event_threshold} events")
        
        # Step 1: Data preparation and feature engineering
        logger.info("STEP 1: Data preparation and feature engineering")
        modeling_data = prepare_acled_data(
            acled_file_path, 
            pred_window_months=prediction_window_months,
            event_threshold=event_threshold
        )
        
        # Step 2: Train the spatial model
        logger.info("STEP 2: Training spatial model")
        model_results = train_spatial_model(
            data_path='data\acled_modeling_data.csv',
            shapefile_path=shapefile_path
        )
        
        # Step 3: Generate and visualize predictions
        logger.info("STEP 3: Generating and visualizing predictions")
        predictions = visualize_conflict_risk(
            model_path='spatial_conflict_model.pkl',
            data_path='data\acled_modeling_data.csv',
            admin1_shapefile=shapefile_path
        )
        
        # Step 4: Run baseline model for comparison if requested
        if run_baseline:
            logger.info("STEP 4: Training baseline model for comparison")
            from acled_baseline_model import train_baseline_model
            baseline_results = train_baseline_model(data_path='acled_modeling_data.csv')
            
            # Compare model performance
            if model_results and 'pr_auc' in model_results and baseline_results:
                baseline_pr_auc = max(model['pr_auc'] for model in baseline_results.values())
                spatial_pr_auc = model_results['pr_auc']
                
                logger.info(f"Model comparison:")
                logger.info(f"  Baseline model PR-AUC: {baseline_pr_auc:.4f}")
                logger.info(f"  Spatial model PR-AUC: {spatial_pr_auc:.4f}")
                logger.info(f"  Improvement: {(spatial_pr_auc - baseline_pr_auc) / baseline_pr_auc * 100:.2f}%")
        
        # Create a summary report
        end_time = datetime.now()
        runtime = end_time - start_time
        
        with open('pipeline_summary.txt', 'w') as f:
            f.write("ACLED Conflict Prediction Pipeline Summary\n")
            f.write("===========================================\n\n")
            f.write(f"Run date: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total runtime: {runtime.total_seconds() / 60:.2f} minutes\n\n")
            
            f.write("Pipeline Parameters:\n")
            f.write(f"  - ACLED data file: {acled_file_path}\n")
            f.write(f"  - Prediction window: {prediction_window_months} months\n")
            f.write(f"  - Event threshold for conflict: {event_threshold} events\n")
            f.write(f"  - Shapefile used: {'Yes' if shapefile_path else 'No'}\n\n")
            
            f.write("Model Performance:\n")
            f.write(f"  - Spatial model PR-AUC: {model_results['pr_auc']:.4f}\n")
            f.write(f"  - Spatial model ROC-AUC: {model_results['roc_auc']:.4f}\n")
            
            if run_baseline and baseline_results:
                best_baseline = max(baseline_results.items(), key=lambda x: x[1]['pr_auc'])
                f.write(f"  - Best baseline model: {best_baseline[0]}\n")
                f.write(f"  - Baseline PR-AUC: {best_baseline[1]['pr_auc']:.4f}\n")
                f.write(f"  - Baseline ROC-AUC: {best_baseline[1]['roc_auc']:.4f}\n\n")
            
            # Add top risk regions
            f.write("Top 10 Highest Risk Regions:\n")
            top_risk = predictions.sort_values('conflict_probability', ascending=False).head(10)
            for i, (_, row) in enumerate(top_risk.iterrows()):
                f.write(f"{i+1}. {row['admin1']}, {row['country']}: {row['conflict_probability']:.4f} ")
                f.write(f"({row['risk_category']})\n")
        
        logger.info(f"Pipeline completed successfully in {runtime.total_seconds() / 60:.2f} minutes")
        logger.info(f"All outputs saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}", exc_info=True)
    
    finally:
        # Return to original directory
        os.chdir(original_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ACLED conflict prediction pipeline')
    parser.add_argument('--acled_file', required=True, help='Path to ACLED CSV data file')
    parser.add_argument('--output_dir', default='output', help='Directory to save outputs')
    parser.add_argument('--pred_window', type=int, default=3, help='Prediction window in months')
    parser.add_argument('--event_threshold', type=int, default=5, help='Violent event threshold for conflict')
    parser.add_argument('--shapefile', default=None, help='Path to admin1 boundary shapefile (optional)')
    parser.add_argument('--skip_baseline', action='store_true', help='Skip running baseline model')
    
    args = parser.parse_args()
    
    run_complete_pipeline(
        acled_file_path=args.acled_file,
        output_dir=args.output_dir,
        prediction_window_months=args.pred_window,
        event_threshold=args.event_threshold,
        run_baseline=not args.skip_baseline,
        shapefile_path=args.shapefile
    )
