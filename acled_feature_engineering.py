import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def prepare_acled_data(file_path,
                       start_date_str='2012-01-01',
                       end_date_str='2022-12-31',
                       pred_window_months=1, # Defaulting to 1 month prediction
                       event_threshold=1):    # Defaulting to 1 violent event for 'conflict_occurs'
    """
    Prepare ACLED data for conflict prediction modeling.

    Parameters:
    -----------
    file_path : str
        Path to the ACLED CSV file.
    start_date_str : str
        The start date for filtering data (inclusive, YYYY-MM-DD).
    end_date_str : str
        The end date for filtering data (inclusive, YYYY-MM-DD).
    pred_window_months : int
        Number of months ahead to predict conflict (e.g., 1 means predict next month).
    event_threshold : int
        Minimum number of violent events in the future month to classify as 'conflict_occurs'.

    Returns:
    --------
    DataFrame with features and target variable.
    """
    print(f"Loading ACLED data from: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)

    # --- 1. Initial Cleaning and Filtering ---
    print("Performing initial cleaning and filtering...")
    df['event_date'] = pd.to_datetime(df['event_date'])

    # Filter by specified date range (CRUCIAL based on EDA)
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)
    df = df[(df['event_date'] >= start_date) & (df['event_date'] <= end_date)]
    print(f"Data filtered to range: {df['event_date'].min()} to {df['event_date'].max()}")

    if df.empty:
        print("No data available for the specified date range. Exiting.")
        return pd.DataFrame()

    # Handle missing admin1 (based on EDA)
    df.dropna(subset=['admin1'], inplace=True)
    print(f"Rows after dropping missing admin1: {len(df)}")

    # Create year_month for aggregation
    df['year_month'] = df['event_date'].dt.to_period('M')

    # Define violent events (consider adding/removing based on further analysis)
    # From your EDA: 'Protests', 'Battles', 'Violence against civilians', 'Riots', 'Strategic developments', 'Explosions/Remote violence'
    violent_event_types = ['Battles', 'Violence against civilians', 'Explosions/Remote violence', 'Riots']
    df['is_violent'] = df['event_type'].isin(violent_event_types).astype(int)
    df['is_battle'] = (df['event_type'] == 'Battles').astype(int)
    df['is_vac'] = (df['event_type'] == 'Violence against civilians').astype(int)
    df['is_explosion_remote'] = (df['event_type'] == 'Explosions/Remote violence').astype(int)
    df['is_riot'] = (df['event_type'] == 'Riots').astype(int)
    df['is_protest'] = (df['event_type'] == 'Protests').astype(int) # For potential precursor feature

    # --- 2. Monthly Aggregation by Region (Admin1) ---
    print("Aggregating data by country, admin1 region, and month...")

    # Define aggregation functions
    agg_funcs = {
        'event_id_cnty': 'count', # total_events
        'fatalities': 'sum',
        'is_violent': 'sum',     # violent_events
        'is_battle': 'sum',
        'is_vac': 'sum',
        'is_explosion_remote': 'sum',
        'is_riot': 'sum',
        'is_protest': 'sum',
        'actor1': pd.NamedAgg(column='actor1', aggfunc='nunique'), # actor1_count
        'inter1': pd.NamedAgg(column='inter1', aggfunc=lambda x: x.nunique()), # number of unique actor types involved as actor1
        'sub_event_type': pd.NamedAgg(column='sub_event_type', aggfunc='nunique'), # sub_event_diversity
        'event_type': pd.NamedAgg(column='event_type', aggfunc='nunique') # event_diversity
    }

    monthly_df = df.groupby(['country', 'admin1', 'year_month']).agg(agg_funcs).reset_index()
    monthly_df = monthly_df.rename(columns={'event_id_cnty': 'total_events',
                                            'is_violent': 'violent_events_count',
                                            'is_battle': 'battles_count',
                                            'is_vac': 'vac_count',
                                            'is_explosion_remote': 'explosion_remote_count',
                                            'is_riot': 'riots_count',
                                            'is_protest': 'protests_count',
                                            'actor1': 'distinct_actors_count',
                                            'inter1': 'distinct_actor_types_count',
                                            'sub_event_type': 'sub_event_diversity',
                                            'event_type': 'event_diversity'})


    # Convert year_month to a proper datetime for easier manipulation (start of the month)
    monthly_df['date'] = monthly_df['year_month'].dt.to_timestamp()

    # --- 3. Create Complete Time-Region Grid ---
    print("Creating complete time-region grid...")
    all_regions = monthly_df[['country', 'admin1']].drop_duplicates()
    # Use the filtered date range for the grid
    grid_date_range = pd.date_range(monthly_df['date'].min(), monthly_df['date'].max(), freq='MS') # MS for month start

    # Create cartesian product of regions and dates
    grid_index = pd.MultiIndex.from_product([all_regions.set_index(['country', 'admin1']).index, grid_date_range],
                                            names=['region_tuple', 'date'])
    grid_df = pd.DataFrame(index=grid_index).reset_index()
    grid_df[['country', 'admin1']] = pd.DataFrame(grid_df['region_tuple'].tolist(), index=grid_df.index)
    grid_df = grid_df.drop(columns=['region_tuple'])

    # Merge aggregated data with the complete grid
    # Use 'date' (datetime) for merging now
    full_df = pd.merge(grid_df, monthly_df.drop(columns=['year_month']),
                       on=['country', 'admin1', 'date'], how='left')

    # Fill NaNs for counts with 0 (regions with no events in some months)
    feature_cols_to_fill_zero = ['total_events', 'fatalities', 'violent_events_count', 'battles_count', 'vac_count',
                                 'explosion_remote_count', 'riots_count', 'protests_count',
                                 'distinct_actors_count', 'distinct_actor_types_count',
                                 'sub_event_diversity', 'event_diversity']
    fill_values = {col: 0 for col in feature_cols_to_fill_zero}
    full_df = full_df.fillna(value=fill_values)

    # Sort for lag/rolling creation
    full_df = full_df.sort_values(['country', 'admin1', 'date']).reset_index(drop=True)

    # --- 4. Feature Engineering (Lags, Rolling, Trends) ---
    print("Creating time-lagged, rolling window, and trend features...")
    # Columns to generate these features for:
    feature_columns_for_lags_rolls = feature_cols_to_fill_zero # Use the same list

    lag_periods = [1, 2, 3, 6, 12] # Added 12-month lag
    roll_windows = [3, 6, 12]    # Added 12-month rolling window

    grouped_by_region = full_df.groupby(['country', 'admin1'])

    for col in feature_columns_for_lags_rolls:
        # Lag features
        for lag in lag_periods:
            full_df[f'{col}_lag{lag}'] = grouped_by_region[col].shift(lag)

        # Rolling window features (mean and sum)
        for N in roll_windows:
            # Using min_periods=1 to avoid NaNs if window is not full at start
            full_df[f'{col}_roll_mean{N}'] = grouped_by_region[col].rolling(window=N, min_periods=1).mean().reset_index(level=[0,1], drop=True)
            full_df[f'{col}_roll_sum{N}'] = grouped_by_region[col].rolling(window=N, min_periods=1).sum().reset_index(level=[0,1], drop=True)

        # Trend features (difference from previous periods)
        if f'{col}_lag1' in full_df.columns: # Ensure lag1 exists
            full_df[f'{col}_trend1'] = full_df[col].fillna(0) - full_df[f'{col}_lag1'].fillna(0)
        if f'{col}_lag3' in full_df.columns: # Ensure lag3 exists
            full_df[f'{col}_trend3'] = full_df[col].fillna(0) - full_df[f'{col}_lag3'].fillna(0)

    # Additional Feature: Time since last violent event
    full_df['days_since_last_violent_event'] = grouped_by_region.apply(
        lambda g: g['date']. όπου(g['violent_events_count'] > 0).ffill().rsub(g['date']).dt.days
    ).reset_index(level=[0,1], drop=True)
    # Fill initial NaNs (before any event) with a large number or handle as special case
    full_df['days_since_last_violent_event'] = full_df['days_since_last_violent_event'].fillna(365*5) # e.g., 5 years

    # --- 5. Target Variable Creation ---
    print(f"Creating target variable for {pred_window_months}-month ahead prediction...")

    # Objective: For each (region, date_t), what will violent_events_count be at (region, date_t + pred_window_months)?
    target_info_df = monthly_df[['country', 'admin1', 'date', 'violent_events_count']].copy()
    target_info_df = target_info_df.rename(columns={'violent_events_count': 'future_violent_events'})

    # Create the 'feature_date' for which these future events will be the target
    target_info_df['feature_date_for_target'] = target_info_df['date'] - pd.DateOffset(months=pred_window_months)

    # Merge this future information back to full_df
    modeling_df = pd.merge(full_df,
                           target_info_df[['country', 'admin1', 'feature_date_for_target', 'future_violent_events']],
                           left_on=['country', 'admin1', 'date'],
                           right_on=['country', 'admin1', 'feature_date_for_target'],
                           how='left')

    modeling_df = modeling_df.drop(columns=['feature_date_for_target'])

    # Determine the latest possible feature date for which a target can exist
    max_original_data_date = monthly_df['date'].max() # From original aggregations
    latest_feature_date_for_valid_target = max_original_data_date - pd.DateOffset(months=pred_window_months)

    # Filter modeling_df to only include rows where a target *could* exist
    modeling_df = modeling_df[modeling_df['date'] <= latest_feature_date_for_valid_target]

    # For the remaining rows, if 'future_violent_events' is NaN, it means 0 events occurred in the target period for that region.
    modeling_df['future_violent_events'] = modeling_df['future_violent_events'].fillna(0)

    # Create binary target variable
    modeling_df['conflict_occurs'] = (modeling_df['future_violent_events'] >= event_threshold).astype(int)
    print(f"Target 'conflict_occurs' distribution:\n{modeling_df['conflict_occurs'].value_counts(normalize=True)}")

    # --- 6. Final Cleaning and Saving ---
    print("Final cleaning of modeling dataset...")
    # Drop rows with NaNs created by lag/rolling features (at the beginning of each group's series)
    first_lag_col_example = f'{feature_columns_for_lags_rolls[0]}_lag{lag_periods[0]}'
    if first_lag_col_example in modeling_df.columns:
        modeling_df = modeling_df.dropna(subset=[first_lag_col_example]) # This handles NaNs from earliest lags

    # Optionally, drop other columns not needed for modeling
    # e.g., 'year_month' from monthly_df if it was accidentally carried over.
    # columns_to_drop = ['year_month_x', 'year_month_y', 'date_y'] # Check for merge artifacts
    # modeling_df = modeling_df.drop(columns=columns_to_drop, errors='ignore')


    print(f"Final modeling dataset shape: {modeling_df.shape}")
    if modeling_df.empty:
        print("Modeling dataset is empty after processing. Check parameters and data.")
        return modeling_df

    output_filename = 'data/acled_modeling_data_prepared.csv'
    try:
        modeling_df.to_csv(output_filename, index=False)
        print(f"Saved prepared modeling dataset to '{output_filename}'")
    except Exception as e:
        print(f"Error saving prepared dataset: {e}")

    return modeling_df

if __name__ == "__main__":
    # Make sure the 'data' directory exists or adjust path
    prepared_data = prepare_acled_data('data/acled_data.csv',
                                       start_date_str='2012-01-01',
                                       end_date_str='2022-12-31',
                                       pred_window_months=1,
                                       event_threshold=1)
    if not prepared_data.empty:
        print("\nSample of prepared data:")
        print(prepared_data.head())
        print("\nMissing values in prepared data:")
        print(prepared_data.isnull().sum().sort_values(ascending=False).head(20)) # Show top 20 missing