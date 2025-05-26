import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def prepare_acled_data(file_path,
                       start_date_str='2012-01-01',
                       end_date_str='2022-12-31',
                       pred_window_months=1,
                       event_threshold=1):
    """
    Prepare ACLED data for conflict prediction modeling.

    Parameters:
    -----------
    file_path : str
        Path to the ACLED CSV file.
    start_date_str : str
        Start date for filtering data (inclusive, YYYY-MM-DD).
    end_date_str : str
        End date for filtering data (inclusive, YYYY-MM-DD).
    pred_window_months : int
        Number of months ahead to predict conflict.
    event_threshold : int
        Minimum violent events to classify as 'conflict_occurs'.

    Returns:
    --------
    DataFrame with features and target variable.
    """

    print(f"Loading ACLED data from: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)

    print("Cleaning and filtering data...")
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df[(df['event_date'] >= start_date_str) & (df['event_date'] <= end_date_str)]
    if df.empty:
        print("No data for the specified date range.")
        return pd.DataFrame()

    df.dropna(subset=['admin1'], inplace=True)
    df['year_month'] = df['event_date'].dt.to_period('M')

    # Define event type flags
    violent_event_types = [
        'Battles', 'Violence against civilians',
        'Explosions/Remote violence', 'Riots'
    ]
    df['is_violent'] = df['event_type'].isin(violent_event_types).astype(int)
    df['is_battle'] = (df['event_type'] == 'Battles').astype(int)
    df['is_vac'] = (df['event_type'] == 'Violence against civilians').astype(int)
    df['is_explosion_remote'] = (df['event_type'] == 'Explosions/Remote violence').astype(int)
    df['is_riot'] = (df['event_type'] == 'Riots').astype(int)
    df['is_protest'] = (df['event_type'] == 'Protests').astype(int)

    print("Aggregating monthly regional statistics...")
    agg_dict = {
        'total_events': ('event_id_cnty', 'count'),
        'fatalities': ('fatalities', 'sum'),
        'violent_events_count': ('is_violent', 'sum'),
        'battles_count': ('is_battle', 'sum'),
        'vac_count': ('is_vac', 'sum'),
        'explosion_remote_count': ('is_explosion_remote', 'sum'),
        'riots_count': ('is_riot', 'sum'),
        'protests_count': ('is_protest', 'sum'),
        'distinct_actors_count': ('actor1', 'nunique'),
        'distinct_actor_types_count': ('inter1', 'nunique'),
        'sub_event_diversity': ('sub_event_type', 'nunique'),
        'event_diversity': ('event_type', 'nunique'),
    }

    monthly_df = df.groupby(['country', 'admin1', 'year_month'], as_index=False).agg(**{
        k: pd.NamedAgg(column=v[0], aggfunc=v[1]) for k, v in agg_dict.items()
    })
    monthly_df['date'] = monthly_df['year_month'].dt.to_timestamp()

    print("Constructing complete region-month grid...")
    regions = monthly_df[['country', 'admin1']].drop_duplicates()
    date_range = pd.date_range(monthly_df['date'].min(), monthly_df['date'].max(), freq='MS')
        # Convert the MultiIndex of regions to a list of tuples
    region_tuples_list = list(regions.set_index(['country', 'admin1']).index)

    grid_index = pd.MultiIndex.from_product([region_tuples_list, date_range],
                                            names=['region_tuple', 'date'])

    grid_df = pd.DataFrame(index=grid_index).reset_index()
    grid_df[['country', 'admin1']] = pd.DataFrame(grid_df['region_tuple'].tolist(), index=grid_df.index)
    grid_df.drop(columns='region_tuple', inplace=True)

    full_df = pd.merge(grid_df, monthly_df.drop(columns='year_month'),
                       on=['country', 'admin1', 'date'], how='left')

    features_to_fill = list(agg_dict.keys())
    full_df[features_to_fill] = full_df[features_to_fill].fillna(0)
    full_df.sort_values(['country', 'admin1', 'date'], inplace=True)
    full_df.reset_index(drop=True, inplace=True)

    print("Generating lag, rolling, and trend features...")
    lag_periods = [1, 2, 3, 6, 12]
    roll_windows = [3, 6, 12]
    group = full_df.groupby(['country', 'admin1'])

    for col in features_to_fill:
        for lag in lag_periods:
            full_df[f'{col}_lag{lag}'] = group[col].shift(lag)
        for N in roll_windows:
            full_df[f'{col}_roll_mean{N}'] = group[col].rolling(N, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
            full_df[f'{col}_roll_sum{N}'] = group[col].rolling(N, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
        full_df[f'{col}_trend1'] = full_df[col].fillna(0) - full_df.get(f'{col}_lag1', 0).fillna(0)
        full_df[f'{col}_trend3'] = full_df[col].fillna(0) - full_df.get(f'{col}_lag3', 0).fillna(0)

    print("Calculating days since last violent event...")
    def days_since_last(series_dates, condition):
        last_date = series_dates.where(condition).ffill()
        return (series_dates - last_date).dt.days

    full_df['days_since_last_violent_event'] = group.apply(
        lambda g: days_since_last(g['date'], g['violent_events_count'] > 0)
    ).reset_index(level=[0,1], drop=True).fillna(365 * 5)

    print(f"Creating target variable for {pred_window_months}-month ahead prediction...")
    target_df = monthly_df[['country', 'admin1', 'date', 'violent_events_count']].copy()
    target_df.rename(columns={'violent_events_count': 'future_violent_events'}, inplace=True)
    target_df['feature_date_for_target'] = target_df['date'] - pd.DateOffset(months=pred_window_months)

    modeling_df = pd.merge(
        full_df,
        target_df[['country', 'admin1', 'feature_date_for_target', 'future_violent_events']],
        left_on=['country', 'admin1', 'date'],
        right_on=['country', 'admin1', 'feature_date_for_target'],
        how='left'
    )
    modeling_df.drop(columns='feature_date_for_target', inplace=True)
    modeling_df['future_violent_events'] = modeling_df['future_violent_events'].fillna(0)
    modeling_df['conflict_occurs'] = (modeling_df['future_violent_events'] >= event_threshold).astype(int)

    max_date = monthly_df['date'].max()
    modeling_df = modeling_df[modeling_df['date'] <= max_date - pd.DateOffset(months=pred_window_months)]

    print("Dropping rows with missing lag features...")
    first_lag_col = f'{features_to_fill[0]}_lag1'
    modeling_df.dropna(subset=[first_lag_col], inplace=True)

    print(f"Final dataset shape: {modeling_df.shape}")
    print(f"Conflict label distribution:\n{modeling_df['conflict_occurs'].value_counts(normalize=True)}")

    output_path = 'data/acled_modeling_data_prepared.csv'
    try:
        modeling_df.to_csv(output_path, index=False)
        print(f"Saved to: {output_path}")
    except Exception as e:
        print(f"Error saving file: {e}")

    return modeling_df


if __name__ == "__main__":
    prepared = prepare_acled_data(
        'data/acled_data.csv',
        start_date_str='2012-01-01',
        end_date_str='2022-12-31',
        pred_window_months=1,
        event_threshold=1
    )

    if not prepared.empty:
        print("\nSample of prepared data:")
        print(prepared.head())
        print("\nTop 20 columns with missing values:")
        print(prepared.isnull().sum().sort_values(ascending=False).head(20))
