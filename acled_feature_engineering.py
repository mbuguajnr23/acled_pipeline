import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import geopandas as gpd

def prepare_acled_data(file_path, pred_window_months=3, event_threshold=5):
    """
    Prepare ACLED data for conflict prediction modeling
    
    Parameters:
    -----------
    file_path : str
        Path to the ACLED CSV file
    pred_window_months : int
        Number of months ahead to predict conflict
    event_threshold : int
        Minimum number of violent events to classify as 'conflict'
        
    Returns:
    --------
    DataFrame with features and target variable
    """
    print("Loading ACLED data...")
    df = pd.read_csv(file_path)
    
    # Convert date to datetime
    df['event_date'] = pd.to_datetime(df['event_date'])
    
    # Filter only Africa data if needed
    # df = df[df['region'] == 'Africa']
    
    # Create year-month field for aggregation
    df['year_month'] = df['event_date'].dt.to_period('M')
    
    # Define violent events (adjust based on ACLED categories)
    violent_event_types = ['Battles', 'Violence against civilians', 'Explosions/Remote violence']
    df['is_violent'] = df['event_type'].isin(violent_event_types)
    
    print("Aggregating data by admin1 region and month...")
    # Aggregate by admin1 region and month
    monthly_region_data = []
    
    for (country, admin1, year_month), group in df.groupby(['country', 'admin1', 'year_month']):
        # Total events
        total_events = len(group)
        
        # Violent events
        violent_events = sum(group['is_violent'])
        
        # Fatalities
        fatalities = group['fatalities'].sum()
        
        # Event type diversity (number of unique event types)
        event_diversity = group['event_type'].nunique()
        
        # Actor counts
        actor1_count = group['actor1'].nunique()
        actor2_count = group['actor2'].nunique()
        
        # Store aggregated data
        monthly_region_data.append({
            'country': country,
            'admin1': admin1,
            'year_month': year_month,
            'total_events': total_events,
            'violent_events': violent_events,
            'fatalities': fatalities,
            'event_diversity': event_diversity,
            'actor1_count': actor1_count,
            'actor2_count': actor2_count
        })
    
    # Convert to DataFrame
    monthly_df = pd.DataFrame(monthly_region_data)
    
    # Convert year_month to datetime for easier manipulation
    monthly_df['date'] = monthly_df['year_month'].dt.to_timestamp()
    
    print("Creating time-lagged features...")
    # Create a complete time-region grid to ensure we have entries for all time periods
    all_regions = monthly_df[['country', 'admin1']].drop_duplicates()
    min_date = monthly_df['date'].min()
    max_date = monthly_df['date'].max() - pd.DateOffset(months=pred_window_months)  # Leave room for prediction window
    
    # Create date range
    date_range = pd.date_range(min_date, max_date, freq='M')
    
    # Create complete grid
    grid = pd.MultiIndex.from_product([all_regions.itertuples(index=False), date_range], 
                                      names=['region', 'date'])
    grid_df = pd.DataFrame(index=grid).reset_index()
    grid_df[['country', 'admin1']] = pd.DataFrame(grid_df['region'].tolist(), 
                                                index=grid_df.index)
    grid_df = grid_df.drop('region', axis=1)
    
    # Merge aggregated data with the complete grid
    full_df = pd.merge(grid_df, monthly_df, on=['country', 'admin1', 'date'], how='left')
    
    # Fill missing values (regions with no events in some months)
    full_df = full_df.fillna({
        'total_events': 0,
        'violent_events': 0,
        'fatalities': 0,
        'event_diversity': 0,+-
        'actor1_count': 0,
        'actor2_count': 0
    })
    
    # Sort by region and date for proper lag creation
    full_df = full_df.sort_values(['country', 'admin1', 'date'])
    
    # Create lag features (1, 2, 3, 6 months prior)
    lag_periods = [1, 2, 3, 6]
    feature_columns = ['total_events', 'violent_events', 'fatalities', 
                       'event_diversity', 'actor1_count', 'actor2_count']
    
    for col in feature_columns:
        for lag in lag_periods:
            # Create lag features by group
            full_df[f'{col}_lag{lag}'] = full_df.groupby(['country', 'admin1'])[col].shift(lag)
    
    # Create rolling window features
    for col in feature_columns:
        # 3-month rolling average
        full_df[f'{col}_roll3'] = full_df.groupby(['country', 'admin1'])[col].rolling(3).mean().reset_index(level=[0,1], drop=True)
        
        # 6-month rolling average
        full_df[f'{col}_roll6'] = full_df.groupby(['country', 'admin1'])[col].rolling(6).mean().reset_index(level=[0,1], drop=True)
    
    # Create trend features
    for col in feature_columns:
        # Trend compared to previous month
        full_df[f'{col}_trend1'] = full_df[col] - full_df[f'{col}_lag1']
        
        # Trend compared to 3 months ago
        full_df[f'{col}_trend3'] = full_df[col] - full_df[f'{col}_lag3']
    
    print("Creating target variable...")
    # Create target variable: conflict in the future prediction window
    target_df = monthly_df.copy()
    
    # Add prediction window to the date
    target_df['target_date'] = target_df['date'] + pd.DateOffset(months=pred_window_months)
    
    # Aggregate violent events in the target month for each region
    target_df = target_df[['country', 'admin1', 'target_date', 'violent_events']]
    target_df = target_df.rename(columns={'target_date': 'date', 'violent_events': 'future_violent_events'})
    
    # Merge with the feature dataframe
    modeling_df = pd.merge(full_df, target_df, on=['country', 'admin1', 'date'], how='left')
    
    # Create binary target variable based on threshold
    modeling_df['conflict_occurs'] = (modeling_df['future_violent_events'] >= event_threshold).astype(int)
    
    # Drop rows with NaN in target (happens at the end of the time series)
    modeling_df = modeling_df.dropna(subset=['conflict_occurs'])
    
    # Drop any remaining NaN values (happens at the beginning due to lag creation)
    modeling_df = modeling_df.dropna()
    
    print(f"Final dataset shape: {modeling_df.shape}")
    
    # Save the prepared dataset
    modeling_df.to_csv('data\acled_modeling_data.csv', index=False)
    print("Saved prepared dataset to 'acled_modeling_data.csv'")
    
    return modeling_df

if __name__ == "__main__":
    prepare_acled_data('data/acled_data.csv')
