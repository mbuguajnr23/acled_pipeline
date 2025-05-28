import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import geopandas as gpd # For map plotting
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_FILE_PATH = 'data/acled_data.csv' 
CHARTS_DIRECTORY = 'charts/'
# AFRICA_SHAPEFILE_PATH = 'path/to/your/africa_countries_shapefile.shp' # IMPORTANT: Update this path
AFRICA_SHAPEFILE_PATH = None

def save_and_show_plot(fig, filename_base, title=""):
    """Saves the current plot to the charts directory and shows it."""
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIRECTORY}{filename_base}.png")
    print(f"Generated '{CHARTS_DIRECTORY}{filename_base}.png'")
    plt.show()
    plt.close(fig) # Close the figure to free memory

# --- Main Script ---
print("Starting ACLED Data Exploration...")

# 1. Load the ACLED data
print(f"\nLoading ACLED data from {DATA_FILE_PATH}...")
try:
    acled_data = pd.read_csv(DATA_FILE_PATH, low_memory=False) # low_memory=False can help with mixed types
except FileNotFoundError:
    print(f"ERROR: Data file not found at {DATA_FILE_PATH}. Please check the path.")
    exit()

# 2. Basic data exploration
print(f"Dataset shape: {acled_data.shape}")
print("\nFirst few rows:")
print(acled_data.head())

print("\nColumns in the dataset:")
print(acled_data.columns.tolist())

print("\nData types:")
print(acled_data.dtypes)

print("\nMissing values (sum):")
print(acled_data.isnull().sum())
print("\nMissing values (percentage):")
print((acled_data.isnull().sum() / len(acled_data) * 100).sort_values(ascending=False))


# 3. Convert 'event_date' and check timeframe
print("\nProcessing 'event_date'...")
if 'event_date' in acled_data.columns:
    # ACLED dates are usually in 'YYYY-MM-DD' format.
    # If your date format is different, adjust pd.to_datetime accordingly
    # Example for 'DD-Month-YY': pd.to_datetime(acled_data['event_date'], format='%d-%b-%y')
    try:
        acled_data['event_date'] = pd.to_datetime(acled_data['event_date'])
        print(f"Event date range: From {acled_data['event_date'].min()} to {acled_data['event_date'].max()}")

        # Extract year and month for easier aggregation
        acled_data['year'] = acled_data['event_date'].dt.year
        acled_data['month_year'] = acled_data['event_date'].dt.to_period('M')

    except Exception as e:
        print(f"Error converting 'event_date': {e}. Please check date format.")
        # Attempt to infer format if standard conversion fails (can be slow)
        # acled_data['event_date'] = pd.to_datetime(acled_data['event_date'], infer_datetime_format=True)

else:
    print("Column 'event_date' not found. Skipping time-based analysis.")
    # Consider exiting if event_date is crucial and missing
    # exit()

# 4. Explore event types and their frequency
if 'event_type' in acled_data.columns:
    print("\nEvent types and frequencies:")
    print(acled_data['event_type'].value_counts())

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.countplot(y='event_type', data=acled_data, order=acled_data['event_type'].value_counts().index, ax=ax, palette="viridis")
    ax.set_xlabel('Number of Events')
    ax.set_ylabel('Event Type')
    save_and_show_plot(fig, "event_type_distribution", "Distribution of Event Types")

# 5. Explore countries represented
if 'country' in acled_data.columns:
    print("\nCountries in the dataset (Top 15 by event count):")
    print(acled_data['country'].value_counts().nlargest(15))

    fig, ax = plt.subplots(figsize=(12, 8))
    top_countries_events = acled_data['country'].value_counts().nlargest(10)
    sns.barplot(x=top_countries_events.values, y=top_countries_events.index, ax=ax, palette="mako")
    ax.set_xlabel('Number of Events')
    ax.set_ylabel('Country')
    save_and_show_plot(fig, "top_10_countries_events", "Top 10 Countries by Number of Events")

    if 'fatalities' in acled_data.columns:
        fig, ax = plt.subplots(figsize=(12, 8))
        country_fatalities = acled_data.groupby('country')['fatalities'].sum().nlargest(10)
        sns.barplot(x=country_fatalities.values, y=country_fatalities.index, ax=ax, palette="rocket")
        ax.set_xlabel('Total Fatalities')
        ax.set_ylabel('Country')
        save_and_show_plot(fig, "top_10_countries_fatalities", "Top 10 Countries by Total Fatalities")

# --- Time Series Visualizations (if 'event_date' was processed) ---
if 'month_year' in acled_data.columns:
    # 6. Visualize event distribution over time (Monthly)
    fig, ax = plt.subplots(figsize=(15, 6))
    monthly_events = acled_data.groupby('month_year').size()
    monthly_events.index = monthly_events.index.to_timestamp() # Convert PeriodIndex to Timestamp for plotting
    monthly_events.plot(kind='line', ax=ax)
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Events')
    save_and_show_plot(fig, "monthly_events_over_time", "Number of Events Over Time (Monthly)")

    # 7. Visualize fatalities over time (Monthly)
    if 'fatalities' in acled_data.columns:
        fig, ax = plt.subplots(figsize=(15, 6))
        monthly_fatalities_sum = acled_data.groupby('month_year')['fatalities'].sum()
        monthly_fatalities_sum.index = monthly_fatalities_sum.index.to_timestamp()
        monthly_fatalities_sum.plot(kind='line', ax=ax, color='red')
        ax.set_xlabel('Month')
        ax.set_ylabel('Total Fatalities')
        save_and_show_plot(fig, "monthly_fatalities_over_time", "Total Fatalities Over Time (Monthly)")

    # 8. Event Types Over Time (Monthly Line Plot)
    if 'event_type' in acled_data.columns:
        fig, ax = plt.subplots(figsize=(18, 8))
        events_by_type_monthly = acled_data.groupby(['month_year', 'event_type']).size().unstack(fill_value=0)
        events_by_type_monthly.index = events_by_type_monthly.index.to_timestamp()
        events_by_type_monthly.plot(kind='line', ax=ax)
        # For stacked area plot (can be good for proportions):
        # events_by_type_monthly.plot(kind='area', stacked=True, ax=ax, alpha=0.7)
        ax.set_xlabel('Month')
        ax.set_ylabel('Number of Events')
        ax.legend(title='Event Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        save_and_show_plot(fig, "event_types_over_time_monthly", "Event Types Over Time (Monthly)")


# --- Geographical Visualizations (if coordinates exist) ---
if 'latitude' in acled_data.columns and 'longitude' in acled_data.columns:
    print("\nGenerating geographical visualizations...")
    # Ensure no NaNs in coordinates for plotting
    plot_data_geo = acled_data.dropna(subset=['latitude', 'longitude'])

    # 9. Heatmap of Event Density
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.histplot(data=plot_data_geo, x="longitude", y="latitude", bins=100, pthresh=.05, cmap="inferno", ax=ax, cbar=True, cbar_kws={'label': 'Number of Events'})
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    # Overlay country borders if shapefile is available
    if AFRICA_SHAPEFILE_PATH:
        try:
            africa_map = gpd.read_file(AFRICA_SHAPEFILE_PATH)
            # Ensure CRS match or reproject if necessary. Assuming ACLED is WGS84 (EPSG:4326)
            if africa_map.crs != "EPSG:4326":
                 africa_map = africa_map.to_crs("EPSG:4326")
            africa_map.plot(ax=ax, facecolor='none', edgecolor='white', linewidth=0.5, alpha=0.7)
        except Exception as e:
            print(f"Could not load or plot shapefile from {AFRICA_SHAPEFILE_PATH}: {e}")
            print("Skipping map overlay. Heatmap will be generated without borders.")
            
    save_and_show_plot(fig, "event_density_heatmap_africa", "Heatmap of Event Density in Africa")


    # 10. Simple Scatter Plot of Event Locations (Optional - can be slow, consider sampling)
    # This creates a GeoDataFrame for potential further spatial analysis
    gdf_events = gpd.GeoDataFrame(
        plot_data_geo, geometry=gpd.points_from_xy(plot_data_geo.longitude, plot_data_geo.latitude), crs="EPSG:4326"
    )
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    if AFRICA_SHAPEFILE_PATH:
        try:
            africa_map_for_scatter = gpd.read_file(AFRICA_SHAPEFILE_PATH)
            if africa_map_for_scatter.crs != "EPSG:4326":
                africa_map_for_scatter = africa_map_for_scatter.to_crs("EPSG:4326")
            africa_map_for_scatter.plot(ax=ax, color='lightgray', edgecolor='black')
        except Exception as e:
            print(f"Could not load shapefile for scatter plot base: {e}")
    
    # Plot a sample to avoid overplotting and speed up rendering
    sample_size = min(50000, len(gdf_events)) # Plot up to 50k points or all if fewer
    gdf_events.sample(n=sample_size, random_state=42).plot(
        ax=ax, marker='o', color='darkred', markersize=1, alpha=0.3
    )
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    save_and_show_plot(fig, "event_locations_scatter_sample", "Sample of Event Locations in Africa")


print("\nExploratory Data Analysis script completed!")