
---
## `EXPLANATION.md` 
---

```markdown
# Explanation: ACLED Conflict Prediction Pipeline

This document provides a more in-depth explanation of the methodologies, data processing steps, feature engineering choices, and modeling decisions made in the ACLED Conflict Prediction Pipeline project.

## 1. Project Objective and Scope

The core objective is to forecast the likelihood of conflict events (specifically violent events) at the sub-national administrative level 1 (Admin1) across Africa on a monthly basis. The project leverages historical data from ACLED (2012-2022) to build predictive models.

*   **Prediction Target:** Binary classification – will a conflict (defined as one or more violent events) occur in a given Admin1 region in a future month?
*   **Prediction Window:** Configurable, typically 1 to 3 months ahead.
*   **Spatial Unit:** Admin1 regions within African countries.
*   **Temporal Unit:** Month.

## 2. Data Source: ACLED

The Armed Conflict Location & Event Data Project (ACLED) provides disaggregated, real-time data on political violence and protest events.

*   **Key ACLED fields utilized:**
    *   `event_date`: Date of the event.
    *   `year`: Year of the event.
    *   `event_type`: Broad categorization (e.g., "Battles", "Protests", "Violence against civilians").
    *   `sub_event_type`: More granular event categorization.
    *   `actor1`, `actor2`: Names of primary actors involved.
    *   `assoc_actor_1`, `assoc_actor_2`: Associated actors.
    *   `inter1`, `inter2`: Categorization of actor types (e.g., State Forces, Rebel Groups, Political Militia, Civilians, etc.). This is crucial for creating actor-related features without dealing with thousands of unique actor names.
    *   `country`, `admin1`, `admin2`, `admin3`: Administrative divisions. `admin1` is the primary spatial unit for aggregation.
    *   `location`, `latitude`, `longitude`: Specific event location.
    *   `fatalities`: Estimated number of fatalities.

## 3. Pipeline Stages Detailed

### 3.1. Data Acquisition & Pre-processing (`acled_feature_engineering.py`)

1.  **Loading:** Raw ACLED data (CSV) for Africa is loaded.
2.  **Date Conversion:** `event_date` is converted to datetime objects.
3.  **Filtering:**
    *   The dataset is filtered to the specified analysis period (e.g., 2012-01-01 to 2022-12-31). This ensures that only relevant historical data is used for training and feature generation.
    *   Rows with missing `admin1` identifiers are dropped, as `admin1` is essential for regional aggregation.
4.  **Event Classification:**
    *   A binary column `is_violent` is created. Events like "Battles", "Violence against civilians", "Explosions/Remote violence", and "Riots" are typically classified as violent.
    *   Separate binary columns for specific event types (e.g., `is_battle`, `is_protest`) are also created for more granular feature generation.

### 3.2. Feature Engineering (`acled_feature_engineering.py`)

This is the core of transforming raw event data into a model-ready format. All features are generated at the `country-admin1-month` level.

1.  **Monthly Aggregation:**
    *   The event-level data is grouped by `country`, `admin1`, and `year_month`.
    *   For each group, the following base metrics are calculated:
        *   `total_events`: Count of all ACLED events.
        *   `fatalities`: Sum of fatalities.
        *   `violent_events_count`: Count of events flagged as `is_violent`.
        *   Counts for specific event types (e.g., `battles_count`, `protests_count`).
        *   `distinct_actors_count`: Number of unique `actor1` names active in the region-month.
        *   `distinct_actor_types_count`: Number of unique actor types (from `inter1`) active.
        *   `event_diversity`: Number of unique `event_type`s.
        *   `sub_event_diversity`: Number of unique `sub_event_type`s.

2.  **Time-Region Grid:**
    *   A complete grid is constructed containing every unique `(country, admin1)` pair for every month within the analysis period.
    *   The aggregated monthly data is merged onto this grid. Region-months with no recorded ACLED activity are filled with zeros for event counts and fatalities. This ensures a consistent dataset where "no events" is explicitly represented.

3.  **Temporal Feature Generation:**
    For each of the base metrics calculated in step 1, the following temporal features are derived for each region-month:
    *   **Lagged Features:** These capture the recent history. Values of a metric from 1, 2, 3, 6, and 12 months prior are included as features (e.g., `violent_events_count_lag1`, `fatalities_lag6`).
    *   **Rolling Window Features:** These smooth out short-term fluctuations and capture trends over different periods. For each base metric, 3-month, 6-month, and 12-month rolling *means* and *sums* are calculated (e.g., `violent_events_count_roll_mean12`, `total_events_roll_sum3`). `min_periods=1` is used to ensure values are generated even at the start of a series.
    *   **Trend Features:** Simple difference features are calculated to capture the rate of change:
        *   `metric_trend1 = metric_current_month - metric_lag1`
        *   `metric_trend3 = metric_current_month - metric_lag3`
    *   **`days_since_last_violent_event`:** This feature measures the duration of "peace" or absence of reported violence in a region. It's calculated by finding the number of days elapsed since the last recorded violent event for that admin1 region. For periods with no prior violent events, a large default value is used.

4.  **Target Variable (`conflict_occurs`):**
    *   The prediction task is formulated as binary classification.
    *   The target `conflict_occurs` for a given feature month `t` is determined by looking at the `violent_events_count` in that same region at a future month `t + prediction_window_months`.
    *   If `violent_events_count` at `t + prediction_window_months` is greater than or equal to `event_threshold` (e.g., 1), then `conflict_occurs` for month `t` is `1`, otherwise `0`.
    *   Data points at the end of the time series for which a future target cannot be constructed are dropped. Similarly, initial data points that have NaN values due to insufficient history for lags/rolling windows are also dropped.
    *   The distribution of this target variable is checked for class imbalance.

The output of this stage is a single CSV file (`acled_modeling_data_prepared.csv`) where each row represents a `country-admin1-month` and columns include all engineered features and the target variable.

### 3.3. Spatial Features (`acled_spatial_model.py`)

Spatial context can significantly influence conflict. This module adds features representing the conflict environment surrounding a focal region.

1.  **Shapefile-based (Primary Method, if shapefile provided):**
    *   Requires an Admin1 level shapefile for Africa. Column names in the shapefile corresponding to country and admin1 names need to be mapped.
    *   **Neighbor Identification:** For each Admin1 polygon, its contiguous neighbors (those sharing a boundary, i.e., `geom.touches(other_geom)`) are identified. Spatial indexing (`sindex`) is used to optimize this process.
    *   **Neighbor-based Features (calculated for month `t` using data from neighbors in month `t-1`):**
        *   `neighbor_violent_events_count_lag1_avg`: Average count of violent events in neighboring regions in the previous month.
        *   `neighbor_fatalities_lag1_avg`: Average sum of fatalities in neighboring regions in the previous month.
        *   `neighbor_conflict_density_lag1`: Proportion of neighboring regions that experienced at least one violent event in the previous month.
    *   These features are merged back into the main modeling DataFrame.

2.  **Country-based (Fallback Method):**
    *   Used if no shapefile is provided or if shapefile processing fails.
    *   Assumes that conflict dynamics within the same country, but outside the focal Admin1 region, provide relevant spatial context.
    *   **Features (calculated for month `t` using data from other Admin1s in the same country in month `t-1`):**
        *   `country_other_admin1_violent_events_lag1`: Sum of violent events in all *other* Admin1 regions of the same country in the previous month.
        *   `country_other_admin1_fatalities_lag1`: Sum of fatalities in all *other* Admin1 regions of the same country in the previous month.
        *   `country_conflict_density_lag1`: Proportion of *other* Admin1 regions in the same country that experienced at least one violent event in the previous month.

### 3.4. Model Training and Evaluation

1.  **Data Splitting:**
    *   A temporal train-test split is crucial for time-series forecasting. The data is typically split 80/20, where the older 80% of the time period is used for training and the most recent 20% for testing. This simulates a real-world scenario where the model predicts for future, unseen periods.

2.  **Feature Scaling (for some models):**
    *   For models like Logistic Regression (often used as a baseline), features are scaled using `StandardScaler` to have zero mean and unit variance. Tree-based models like XGBoost and RandomForest are generally insensitive to feature scaling. Scaled features are converted back to DataFrames with original column names to aid interpretability (e.g., with SHAP).

3.  **Model Selection:**
    *   **Primary Model (`acled_spatial_model.py`):** XGBoost Classifier (`xgb.XGBClassifier`). Chosen for its high performance, ability to handle complex interactions, and robustness. It's trained on data including temporal and spatial features.
    *   **Baseline Models (`acled_baseline_model.py`):** Simpler models are trained for comparison, typically using only the temporal features (excluding explicitly spatial ones). Examples include:
        *   RandomForest Classifier
        *   Logistic Regression (would require scaling)
        *   An XGBoost model without the spatial features could also serve as a strong baseline.

4.  **Evaluation Metrics:**
    *   **Accuracy:** Overall correctness.
    *   **Precision, Recall, F1-Score:** Calculated for both classes, but particular attention is paid to the "conflict" class (class 1).
    *   **Confusion Matrix:** Visualizes true positives, true negatives, false positives, and false negatives.
    *   **ROC AUC:** Measures discrimination across thresholds.
    *   **PR AUC (Precision-Recall Area Under Curve):** Especially important for imbalanced datasets as it focuses on the positive class performance. A higher PR AUC indicates better performance at identifying true conflicts while maintaining precision.

5.  **Feature Importance and Explainability:**
    *   **XGBoost/RandomForest:** Built-in `feature_importances_` attribute provides a measure of how much each feature contributes to the model's predictions (typically based on impurity reduction or split counts).
    *   **SHAP (SHapley Additive exPlanations):** Used for more nuanced model explainability. SHAP values explain the contribution of each feature to individual predictions and can be aggregated to understand global feature importance. `shap.summary_plot` is used to visualize this.

### 3.5. Prediction, Visualization, and Reporting (`acled_prediction_visualization.py`, `acled_complete_pipeline.py`)

1.  **Prediction Generation:** The trained primary model (XGBoost spatial) is used to generate conflict probabilities for all region-months in the dataset, or specifically for a future period (e.g., predicting for the month following the latest data available).
2.  **Risk Categorization:** Predicted probabilities are binned into qualitative risk categories (e.g., Very Low, Low, Moderate, High, Very High).
3.  **Outputs:**
    *   **CSVs:**
        *   Full predictions with probabilities and risk categories.
        *   Country-level risk summaries.
        *   Feature importance scores.
    *   **Charts:**
        *   Distribution of predicted conflict risk.
        *   Bar chart of top N highest-risk Admin1 regions.
        *   Bar chart of average risk by country.
        *   (If shapefile provided) Choropleth map visualizing predicted conflict risk geographically.
    *   **Pipeline Summary Report:** A text file summarizing the pipeline run parameters, key performance metrics for spatial and baseline models, and top predicted risk regions.

## 4. Key Considerations and Challenges

*   **Data Quality and Bias:** ACLED data relies on reported events, which can be subject to reporting biases (e.g., underreporting in remote or restricted areas).
*   **Definition of "Conflict":** The `event_threshold` for defining the binary target `conflict_occurs` is a critical parameter that can influence model performance and interpretation.
*   **Dynamic Nature of Conflict:** Conflict is highly dynamic. Features capturing recent changes (trends, lags) are important, but long-term patterns also play a role, as indicated by the importance of 12-month rolling features.
*   **Spatial Autocorrelation:** Conflict in one region often influences conflict in nearby regions. The spatial features aim to capture this, but more sophisticated methods (e.g., spatial econometrics, Graph Neural Networks) could be explored.
*   **Computational Resources:** Processing large datasets and training complex models, especially with SHAP value calculations, can be computationally intensive.
*   **Model Interpretability vs. Performance:** While complex models like XGBoost often yield higher performance, simpler models might be more interpretable. SHAP helps bridge this gap for tree-based ensembles.
*   **Ethical Implications:** Conflict prediction models must be used responsibly, acknowledging their limitations and potential for misuse. They are tools to aid analysis, not deterministic oracles.

## 5. Future Directions

*   Systematic hyperparameter tuning for all models.
*   More advanced spatial feature engineering and modeling techniques.
*   Integration of external data sources (socio-economic, political, environmental).
*   Ensemble modeling to combine strengths of different models.
*   Development of a more interactive dashboard for exploring predictions and risk factors.

This detailed explanation should provide a comprehensive understanding of the project's internal workings and rationale behind key decisions.