import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime
import logging

# Import our modules

from acled_feature_engineering import prepare_acled_data
from acled_spatial_model import train_spatial_model # Contains add_spatial_features too
from acled_prediction_visualization import visualize_conflict_risk


# --- Configuration ---
# Define standard subdirectories within the main output directory
PREPARED_DATA_SUBDIR = "prepared_data"
MODELS_SUBDIR = "models"
CHARTS_SUBDIR = "charts"
REPORTS_SUBDIR = "reports" # For text reports

# Set up logging
# Ensure log file is also in the output directory
# We'll set the log file handler path after creating output_dir

logger = logging.getLogger("ACLED_Pipeline")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Console handler (always active)
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)
# File handler will be added in run_complete_pipeline

def run_complete_pipeline(
    raw_acled_file_path, # Renamed for clarity
    main_output_dir="output_pipeline", # Renamed for clarity
    start_date_str='2012-01-01', # Added for prepare_acled_data
    end_date_str='2022-12-31',   # Added for prepare_acled_data
    prediction_window_months=1, # Defaulting to 1 for consistency
    event_threshold=1,          # Defaulting to 1 for consistency
    run_baseline=True,
    shapefile_path=None # This path is relative to original execution, not output_dir
):
    """
    Run the complete ACLED conflict prediction pipeline.
    All outputs (prepared data, models, charts, logs, reports) will be saved
    within subdirectories of `main_output_dir`.
    """
    original_cwd = os.getcwd() # Save original CWD

    try:
        # --- Setup Output Directories and Logging ---
        # Create main output directory if it doesn't exist
        os.makedirs(main_output_dir, exist_ok=True)
        logger.info(f"Main output directory: {os.path.abspath(main_output_dir)}")

        # Add file handler for logging now that main_output_dir is confirmed
        log_file_path = os.path.join(main_output_dir, "acled_pipeline.log")
        fh = logging.FileHandler(log_file_path)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Define paths for outputs within the main_output_dir
        prepared_data_dir = os.path.join(main_output_dir, PREPARED_DATA_SUBDIR)
        models_dir = os.path.join(main_output_dir, MODELS_SUBDIR)
        charts_dir = os.path.join(main_output_dir, CHARTS_SUBDIR)
        reports_dir = os.path.join(main_output_dir, REPORTS_SUBDIR)

        os.makedirs(prepared_data_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(charts_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)

        # Define specific file paths
        # Note: raw_acled_file_path and shapefile_path are absolute or relative to original CWD
        prepared_data_csv_path = os.path.join(prepared_data_dir, "acled_modeling_data_prepared.csv")
        spatial_model_path = os.path.join(models_dir, "spatial_conflict_model.pkl")
        # Baseline models might have different names, handle within its function

        start_time = datetime.now()
        logger.info(f"--- Starting ACLED Conflict Prediction Pipeline at {start_time} ---")
        logger.info(f"Using raw ACLED data from: {os.path.abspath(raw_acled_file_path)}")
        if shapefile_path:
            logger.info(f"Using shapefile from: {os.path.abspath(shapefile_path)}")
        logger.info(f"Analysis period: {start_date_str} to {end_date_str}")
        logger.info(f"Prediction window: {prediction_window_months} months")
        logger.info(f"Event threshold for conflict: {event_threshold} events")

        # --- Step 1: Data preparation and feature engineering ---
        logger.info("--- STEP 1: Data preparation and feature engineering ---")
        # prepare_acled_data should save its output to prepared_data_csv_path
       

        # Ensure raw_acled_file_path is absolute or resolvable from current context
        abs_raw_acled_file_path = os.path.abspath(raw_acled_file_path)

        modeling_df = prepare_acled_data(
            file_path=abs_raw_acled_file_path, # Pass absolute path
            start_date_str=start_date_str,
            end_date_str=end_date_str,
            pred_window_months=prediction_window_months,
            event_threshold=event_threshold
            # Consider adding output_csv_path to prepare_acled_data function
        )
        if modeling_df.empty:
            logger.error("Feature engineering resulted in an empty DataFrame. Exiting.")
            return
        modeling_df.to_csv(prepared_data_csv_path, index=False)
        logger.info(f"Prepared modeling data saved to: {prepared_data_csv_path}")


        # --- Step 2: Train the spatial model ---
        logger.info("--- STEP 2: Training spatial model ---")
        # train_spatial_model should be modified to accept output paths for model, charts
        abs_shapefile_path = os.path.abspath(shapefile_path) if shapefile_path else None
        model_results = train_spatial_model(
            data_path=prepared_data_csv_path, 
            shapefile_path=abs_shapefile_path,
            output_model_path=spatial_model_path, 
            output_charts_dir=charts_dir       
        if not model_results or 'model' not in model_results:
            logger.error("Spatial model training failed or did not return results. Exiting.")
            return


        # --- Step 3: Generate and visualize predictions ---
        logger.info("--- STEP 3: Generating and visualizing predictions ---")
        # visualize_conflict_risk needs to be adapted to save charts to charts_dir
        abs_admin1_shapefile_path = os.path.abspath(shapefile_path) if shapefile_path else None # Assuming admin1 shapefile is the same
        
     
        if model_results and 'model' in model_results: # Add a check
            actual_model_object = model_results['model']
            predictions_df = visualize_conflict_risk(
                model_object=actual_model_object, # <<< Pass the actual model object
                data_path=prepared_data_csv_path,
                admin1_shapefile=abs_admin1_shapefile_path,
                output_charts_dir=charts_dir,
                output_reports_dir=reports_dir,
                prediction_offset_months=prediction_window_months
            )
        else:
            logger.error("Spatial model object not found in model_results. Skipping visualization.")
            predictions_df = pd.DataFrame(columns=['admin1', 'country', 'conflict_probability', 'risk_category']) # Ensure it's defined
       

        # --- Step 4: Run baseline model for comparison if requested ---
        baseline_results_dict = None
        if run_baseline:
            logger.info("--- STEP 4: Training baseline model for comparison ---")
            # This import is here to avoid error if acled_baseline_model.py doesn't exist
            # and run_baseline is False.
            try:
                from acled_baseline_model import train_baseline_model
                # train_baseline_model should accept output_charts_dir
                baseline_results_dict = train_baseline_model(
                    data_path=prepared_data_csv_path,
                    output_charts_dir=charts_dir # New arg
                )
            except ImportError:
                logger.error("Could not import acled_baseline_model. Skipping baseline.")
            except Exception as e:
                logger.error(f"Error during baseline model training: {e}", exc_info=True)


        # --- Step 5: Create a summary report ---
        logger.info("--- STEP 5: Creating summary report ---")
        end_time = datetime.now()
        runtime = end_time - start_time
        report_path = os.path.join(reports_dir, 'pipeline_summary_report.txt')

        with open(report_path, 'w') as f:
            f.write("ACLED Conflict Prediction Pipeline Summary\n")
            f.write("===========================================\n\n")
            f.write(f"Pipeline Execution Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Runtime: {runtime.total_seconds() / 60:.2f} minutes\n\n")

            f.write("Pipeline Parameters:\n")
            f.write(f"  - Raw ACLED Data File: {os.path.abspath(raw_acled_file_path)}\n")
            f.write(f"  - Analysis Period: {start_date_str} to {end_date_str}\n")
            f.write(f"  - Prediction Window: {prediction_window_months} months\n")
            f.write(f"  - Conflict Event Threshold: {event_threshold} events\n")
            f.write(f"  - Shapefile Used: {'Yes, path: ' + os.path.abspath(shapefile_path) if shapefile_path else 'No'}\n\n")

            f.write("Output Locations:\n")
            f.write(f"  - Prepared Data: {prepared_data_csv_path}\n")
            f.write(f"  - Spatial Model: {spatial_model_path}\n")
            f.write(f"  - Charts: {os.path.abspath(charts_dir)}\n")
            f.write(f"  - Reports: {os.path.abspath(reports_dir)}\n")
            f.write(f"  - Log File: {log_file_path}\n\n")

            f.write("Model Performance (Spatial Model):\n")
            if model_results and 'pr_auc' in model_results:
                f.write(f"  - PR-AUC: {model_results['pr_auc']:.4f}\n")
                f.write(f"  - ROC-AUC: {model_results['roc_auc']:.4f}\n\n")
            else:
                f.write("  - Spatial model results not available.\n\n")

            if run_baseline and baseline_results_dict:
                f.write("Model Performance (Best Baseline):\n")
                # Find best baseline by PR-AUC
                if baseline_results_dict: # Check if not None and not empty
                    best_baseline_name = None
                    best_baseline_pr_auc = -1
                    best_baseline_roc_auc = -1
                    for name, metrics in baseline_results_dict.items():
                        if metrics.get('pr_auc', -1) > best_baseline_pr_auc:
                            best_baseline_pr_auc = metrics['pr_auc']
                            best_baseline_roc_auc = metrics.get('roc_auc', -1)
                            best_baseline_name = name
                    if best_baseline_name:
                        f.write(f"  - Model Type: {best_baseline_name}\n")
                        f.write(f"  - PR-AUC: {best_baseline_pr_auc:.4f}\n")
                        f.write(f"  - ROC-AUC: {best_baseline_roc_auc:.4f}\n\n")
                    else:
                        f.write("  - No valid baseline results found for comparison.\n\n")
                else:
                    f.write("  - Baseline models were run, but no results dictionary returned.\n\n")


            if not predictions_df.empty:
                f.write("Top 10 Highest Risk Regions (from Spatial Model Predictions):\n")
                # Ensure 'conflict_probability' column exists
                if 'conflict_probability' in predictions_df.columns:
                    top_risk = predictions_df.sort_values('conflict_probability', ascending=False).head(10)
                    for i, row_tuple in enumerate(top_risk.itertuples()): # Use itertuples for efficiency
                        f.write(f"  {i+1}. {getattr(row_tuple, 'admin1', 'N/A')}, {getattr(row_tuple, 'country', 'N/A')}: Probability={getattr(row_tuple, 'conflict_probability', 0.0):.4f} ")
                        f.write(f"(Risk Category: {getattr(row_tuple, 'risk_category', 'N/A')})\n")
                else:
                    f.write("  - 'conflict_probability' column not found in predictions_df.\n")
            else:
                f.write("  - No prediction data available to list top risk regions.\n")
            f.write("\n")

        logger.info(f"--- ACLED Pipeline completed successfully in {runtime.total_seconds() / 60:.2f} minutes ---")
        logger.info(f"All outputs saved to subdirectories within: {os.path.abspath(main_output_dir)}")

    except Exception as e:
        logger.error(f"--- ACLED Pipeline failed: {str(e)} ---", exc_info=True)
    finally:
        # Ensure we change back to the original directory even if errors occur
        # And remove the file handler to allow deletion of log file if pipeline is rerun quickly
        if 'fh' in locals() and fh in logger.handlers: # Check if fh was defined and added
            logger.removeHandler(fh)
            fh.close()
        os.chdir(original_cwd) # Change back to original directory
        logger.info(f"Switched back to original directory: {original_cwd}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ACLED conflict prediction pipeline.')
    parser.add_argument('--acled_file', required=True, help='Path to the raw ACLED CSV data file.')
    parser.add_argument('--output_dir', default='output_pipeline_run', help='Main directory to save all outputs.')
    parser.add_argument('--start_date', default='2012-01-01', help='Start date for analysis (YYYY-MM-DD).')
    parser.add_argument('--end_date', default='2022-12-31', help='End date for analysis (YYYY-MM-DD).')
    parser.add_argument('--pred_window', type=int, default=1, help='Prediction window in months (e.g., 1 for next month).')
    parser.add_argument('--event_threshold', type=int, default=1, help='Minimum future violent events for "conflict_occurs".')
    parser.add_argument('--shapefile', default=None, help='Path to Admin1 boundary shapefile (optional).')
    parser.add_argument('--skip_baseline', action='store_true', help='Skip running baseline models.')

    args = parser.parse_args()

    run_complete_pipeline(
        raw_acled_file_path=args.acled_file,
        main_output_dir=args.output_dir,
        start_date_str=args.start_date,
        end_date_str=args.end_date,
        prediction_window_months=args.pred_window,
        event_threshold=args.event_threshold,
        run_baseline=not args.skip_baseline,
        shapefile_path=args.shapefile
    )
    # Note: If shapefile_path is None, prepare_acled_data and train_spatial_model should handle it gracefully