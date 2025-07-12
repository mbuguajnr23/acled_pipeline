# ACLED Conflict Prediction Pipeline

This project implements an end-to-end machine learning pipeline to predict the likelihood of conflict events in African sub-national regions (Admin1 level). It uses historical data from the Armed Conflict Location & Event Data Project (ACLED) from 2012-2023 to train and evaluate models.

The primary model, an XGBoost classifier incorporating both temporal and spatial features, achieved a **PR-AUC of 0.8363** and an **ROC-AUC of 0.8809** on a temporal test set, demonstrating strong predictive capabilities for forecasting conflict three months in advance.

## Overview

The pipeline automates several key stages:
1.  **Data Ingestion & Preparation:** Loads and cleans ACLED event data.
2.  **Feature Engineering:** Creates a rich set of temporal features (lags, rolling statistics, trends) and spatial context features (based on neighboring region activity or country-level dynamics).
3.  **Model Training:** Trains an XGBoost model as the primary predictor and simpler baseline models (RandomForest, XGBoost without explicit spatial features) for comparison.
4.  **Model Evaluation:** Assesses performance using metrics like PR-AUC, ROC-AUC, precision, recall, and F1-score, with a temporal train-test split.
5.  **Prediction & Visualization:** Generates conflict risk scores, categorizes risk levels, and produces charts, reports, and (optionally) maps.

For a detailed explanation of the methodology, data processing, feature choices, and model interpretation, please see [EXPLANATION.md](EXPLANATION.md).


Getting Started
Prerequisites
Python 3.9+
Access to ACLED data
Installation
Clone the repository (if applicable):
# git clone 
# cd 

Bash
Create and activate a Python virtual environment (recommended):
python -m venv .venv
# Windows:
# .venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate


Bash
Install dependencies:
pip install -r requirements.txt

Bash
(Ensure requirements.txt includes pandas, numpy, matplotlib, seaborn, geopandas, scikit-learn, xgboost, shap).
Data Acquisition
Download Africa data from the ACLED Data Export Tool.
Date Range: It's recommended to download data from at least 2010 to allow for sufficient history for lag features if your analysis starts in 2012 (e.g., 2010-01-01 to 2023-12-31 or later).
Region: Africa.
Format: CSV.
Place the downloaded CSV file into the data/ directory (e.g., data/acled_data.csv).
Running the Pipeline
The main pipeline is executed via acled_complete_pipeline.py.
Basic command:
python acled_complete_pipeline.py --acled_file data/your_acled_download.csv
Use code with caution.
Bash

Example run:
python acled_complete_pipeline.py 
    --acled_file data/acled_data.csv
    --output_dir output_pipeline_run_YYYYMMDD_HHMM 
    --start_date 2012-01-01 
    --end_date 2023-12-31 
    --pred_window 3 
    --event_threshold 1 
    --shapefile data/Africa_Countries/Africa_Countries.shp 
    --skip_baseline
Use code with caution.

Bash

(Note: ^ is for line continuation in Windows cmd. Use \ for macOS/Linux. Consider naming output_dir with a timestamp for multiple runs).


Acknowledgements
Data provided by the Armed Conflict Location & Event Data Project (ACLED).
**Key Features of this README:**

*   **Concise Overview:** Quickly tells a visitor what the project is about and its main achievements.
*   **Clear "Getting Started":** Provides actionable steps for someone to set up and run the project.
*   **Focus on the Main Pipeline:** Emphasizes `acled_complete_pipeline.py` as the entry point.
*   **Highlights Key Results:** Includes the PR-AUC/ROC-AUC from your successful run to showcase capability.
*   **Points to `EXPLANATION.md`:** Directs users who want more technical depth to the detailed explanation file.
*   **Updated Parameters:** Reflects the parameters used in your successful run (like the `end_date` and `pred_window`).


