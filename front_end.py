import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from azure.storage.blob import BlobServiceClient
from io import StringIO

# Access secrets
connection_string = st.secrets["azure_connection_string"]
container_name = "forecasting"
blob_name = "next_day_forecast/forecast2025-05-25.csv"

# Connect to Blob Storage
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
blob_client = blob_service_client.get_container_client(container_name).get_blob_client(blob_name)

# Download blob content
streamdownloader = blob_client.download_blob()
csv_content = streamdownloader.content_as_text()

# Read CSV into DataFrame
df = pd.read_csv(StringIO(csv_content))

# Convert 'Timestamp' column to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Sort by Timestamp (optional, but recommended)
df = df.sort_values('Timestamp')

# Streamlit frontend
st.title("Energy Demand")

st.write("UK energy demand forecast")

# Plot line chart
st.line_chart(df.set_index('Timestamp')['Demand'])

# Show raw data
if st.checkbox("Show raw data"):
    st.write(df)
