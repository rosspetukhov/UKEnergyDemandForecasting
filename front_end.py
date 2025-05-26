import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from azure.storage.blob import BlobServiceClient, ContainerClient
from io import BytesIO
from datetime import datetime, timedelta
import numpy as np
import os

# Access secrets
connection_string = os.getenv("azure_connection_string")
if not connection_string:
    connection_string = st.secrets.get("azure_connection_string")

if not connection_string:
    st.error("Azure connection string not set!")
    st.stop()
container_name = "forecasting"

# Connect to container
container_client = ContainerClient.from_connection_string(conn_str=connection_string, container_name=container_name)

def forecast_data(container_client):
    # List blobs with the prefix
    blob_list = container_client.list_blobs(name_starts_with="next_day_forecast/")

    blobs_with_dates = []
    for blob in blob_list:
        filename = blob.name.split("/")[-1]
        
        if filename.startswith("forecast") and filename.endswith(".csv"):
            date_str = filename[len("forecast"): -len(".csv")]  
            try:
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                blobs_with_dates.append((blob.name, file_date))
            except ValueError:
                pass  

    # Sort by date descending and select latest 8
    blobs_sorted = sorted(blobs_with_dates, key=lambda x: x[1], reverse=True)
    latest_8_blobs = blobs_sorted[:8]

    # Read and combine CSVs
    dfs = []
    for blob_name, file_date in latest_8_blobs:
        blob_client = container_client.get_blob_client(blob_name)
        stream = blob_client.download_blob()
        df = pd.read_csv(BytesIO(stream.readall()))
        df['source_file_date'] = file_date.strftime('%Y-%m-%d')  # optional: track source
        dfs.append(df)

    # Combine into single DataFrame
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()

    # Convert 'Timestamp' column to datetime
    combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'])
    combined_df = combined_df.sort_values('Timestamp')

    return combined_df

def historical_data(container_client):
    blob_name = "nesodata/demanddataupdate.csv"

    # Get data from blob
    blob_client = container_client.get_blob_client(blob_name)
    stream = blob_client.download_blob()
    df = pd.read_csv(BytesIO(stream.readall()))

    # Filter relevant rows and columns
    df = df.loc[df['FORECAST_ACTUAL_INDICATOR'] == 'A']
    df = df[['SETTLEMENT_DATE', 'SETTLEMENT_PERIOD', 'ND']]

    # Convert dates and build timestamp
    df['SETTLEMENT_DATE'] = pd.to_datetime(df['SETTLEMENT_DATE'])
    df['time'] = pd.to_timedelta((df['SETTLEMENT_PERIOD'] - 1) * 0.5, unit='h')  # SP=1 means 00:00
    df['Timestamp'] = df['SETTLEMENT_DATE'] + df['time']

    # Get required columns
    df['Demand'] = df['ND']
    df = df[['Timestamp','Demand']]

    return df

def combine_historical_and_forecast():

    # Get data
    df_forecast = forecast_data(container_client)
    df_historical = historical_data(container_client)

    # Get today’s date (assuming UTC)
    now = datetime.utcnow()  # tz-naive (UTC time without tzinfo)
    seven_days_ago = now - timedelta(days=7)

    # Filter historical: last 7 days
    df_historical['Timestamp'] = pd.to_datetime(df_historical['Timestamp'])
    print(df_historical['Timestamp'])
    df_historical_filtered = df_historical[df_historical['Timestamp'] >= seven_days_ago].copy()
    df_historical_filtered = df_historical_filtered.rename(columns={'Demand': 'Historical_Demand'})

    # Filter forecast: only future timestamps
    df_forecast_filtered = df_forecast[df_forecast['Timestamp'] >= seven_days_ago].copy()
    df_forecast_filtered = df_forecast_filtered.rename(columns={'Demand': 'Forecast_Demand'})

    # Merge on Timestamp (outer join to keep all times)
    df_combined = pd.merge(df_historical_filtered, df_forecast_filtered, on='Timestamp', how='outer')

    # Sort by time
    df_combined = df_combined.sort_values('Timestamp').reset_index(drop=True)

    return df_combined

# Prepare chart data
fig, ax = plt.subplots(figsize=(10, 5))

# Set dark background color for figure and axes
fig.patch.set_facecolor('#121212')       # dark figure background
ax.set_facecolor('#121212')               # dark plot background

# Plot lines
df = combine_historical_and_forecast()
ax.plot(df['Timestamp'], df['Historical_Demand'], label='Historical Demand', color='cyan')
ax.plot(df['Timestamp'], df['Forecast_Demand'], label='Forecast Demand', color='orange')

# Set title and labels with light colors
ax.set_title("Historical (last 7 days) and Next Day Forecast", color='white')
ax.set_xlabel("Timestamp", color='white')
ax.set_ylabel("Demand", color='white')

# Customize tick params to light color
ax.tick_params(colors='white', which='both')

# Customize legend with facecolor and text color
legend = ax.legend(facecolor='#222222', edgecolor='white')
for text in legend.get_texts():
    text.set_color('white')

# Rotate x-axis labels and adjust layout
plt.xticks(rotation=45)
plt.tight_layout()

# Prepare Mape
df_overlap = df.dropna(subset=['Historical_Demand', 'Forecast_Demand'])

if not df_overlap.empty:
    mape = np.mean(
        np.abs(
            (df_overlap['Historical_Demand'] - df_overlap['Forecast_Demand']) / df_overlap['Historical_Demand']
        )
    ) * 100
    mape_text = f"Forecast MAPE: {mape:.2f}%"
else:
    mape_text = "Not enough overlapping data to calculate MAPE"


# Front-end
st.title("UK Energy Demand Forecast")
st.write("Last 7 days energy demand, and next day forecast")
st.write(mape_text)
st.pyplot(fig)

