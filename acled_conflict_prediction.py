import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import geopandas as gpd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# 1. Load the ACLED data
print("Loading ACLED data...")
# Adjust the path to where your data is stored
acled_data = pd.read_csv('data/acled_data.csv')

# 2. Basic data exploration
print(f"Dataset shape: {acled_data.shape}")
print("\nFirst few rows:")
print(acled_data.head())

print("\nColumns in the dataset:")
print(acled_data.columns.tolist())

print("\nData types:")
print(acled_data.dtypes)

print("\nMissing values:")
print(acled_data.isnull().sum())

# 3. Check the timeframe of the data
print("\nTime range of the data:")
if 'event_date' in acled_data.columns:
    acled_data['event_date'] = pd.to_datetime(acled_data['event_date'])
    print(f"From {acled_data['event_date'].min()} to {acled_data['event_date'].max()}")

# 4. Explore event types and their frequency
print("\nEvent types and frequencies:")
if 'event_type' in acled_data.columns:
    print(acled_data['event_type'].value_counts())

# 5. Explore countries represented
print("\nCountries in the dataset:")
if 'country' in acled_data.columns:
    print(acled_data['country'].value_counts())

# 6. Visualize event distribution over time
if 'event_date' in acled_data.columns:
    plt.figure(figsize=(15, 6))
    acled_data['event_date'].dt.date.value_counts().sort_index().plot()
    plt.title('Number of Events Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Events')
    plt.tight_layout()
    plt.savefig('charts/events_over_time.png')
    print("\nGenerated 'events_over_time.png'")

# 7. Visualize fatalities
if 'fatalities' in acled_data.columns:
    plt.figure(figsize=(15, 6))
    monthly_fatalities = acled_data.groupby(acled_data['event_date'].dt.to_period('M'))['fatalities'].sum()
    monthly_fatalities.plot(kind='bar')
    plt.title('Monthly Fatalities')
    plt.xlabel('Month')
    plt.ylabel('Total Fatalities')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('charts/monthly_fatalities.png')
    print("\nGenerated 'monthly_fatalities.png'")

print("\nExploration completed!")