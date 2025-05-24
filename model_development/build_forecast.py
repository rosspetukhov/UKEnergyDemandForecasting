import matplotlib.pyplot as plt
import pandas as pd
import pickle
from datetime import date
from azure.storage.blob import BlobServiceClient
import os
import io
import logging 
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

# For local use
import json

with open('local.settings.json') as f:
    settings = json.load(f)
    os.environ.update(settings['Values'])


def load_data_and_model():
    logger.info("Starting loading data and model")

    try: 
        # Blob connection
        connect_str = os.environ['AzureWebJobsStorage']
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        container_name = 'forecasting'

        # Read CSV
        csv_blob = blob_service_client.get_blob_client(container=container_name, blob='nesodata/demanddataupdate.csv')
        csv_stream = io.BytesIO()
        csv_blob.download_blob().readinto(csv_stream)
        csv_stream.seek(0)
        df = pd.read_csv(csv_stream)

        # Filter relevant rows and columns
        df = df.loc[df['FORECAST_ACTUAL_INDICATOR'] == 'A']
        df = df[['SETTLEMENT_DATE', 'SETTLEMENT_PERIOD', 'ND']]

        # Convert dates and build datetime
        df['SETTLEMENT_DATE'] = pd.to_datetime(df['SETTLEMENT_DATE'])
        df['time'] = pd.to_timedelta((df['SETTLEMENT_PERIOD'] - 1) * 0.5, unit='h')  # SP=1 means 00:00
        df['datetime'] = df['SETTLEMENT_DATE'] + df['time']

        # Load the model from pickle
        pkl_blob = blob_service_client.get_blob_client(container=container_name, blob='models/ridge_model_24may.pkl')
        pkl_stream = io.BytesIO()
        pkl_blob.download_blob().readinto(pkl_stream)
        pkl_stream.seek(0)
        model = pickle.load(pkl_stream)

        logger.info("Data and model loaded succusefully")
        
        return df, model
    
    except Exception as e:
        logger.exception(f"An error occurred during loading data and model: {e}")
        sys.exit(1)

def forecast_next_day(df, model):
    logger.info("Starting forecasting for the next day")
    try:
        # Forecast next 48 half-hour periods 
        features = ['SETTLEMENT_PERIOD', 'month', 'dayofweek', 'lag_1', 'lag_48']
        last_datetime = df['datetime'].iloc[-1]
        future_timestamps = []
        future_preds = []
        recent_values = df.copy()

        for i in range(1, 49):  
            future_time = last_datetime + pd.Timedelta(minutes=30 * i)
            settlement_period = ((future_time.hour * 60 + future_time.minute) // 30) + 1

            month = future_time.month
            dayofweek = future_time.weekday()
            lag_1 = future_preds[-1] if len(future_preds) > 0 else recent_values.iloc[-1]['ND']
            lag_48 = recent_values.iloc[-48 + i]['ND'] if len(recent_values) >= 48 else lag_1

            X_future = pd.DataFrame([[settlement_period, month, dayofweek, lag_1, lag_48]], columns=features)
            pred = model.predict(X_future)[0]
            future_preds.append(pred)
            future_timestamps.append(future_time)

        # Save results to a df
        results_df = pd.DataFrame({
            'Timestamp': future_timestamps,
            'Demand': future_preds
        })

        # Blob connection
        connect_str = os.environ['AzureWebJobsStorage']
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        container_name = 'forecasting'

        # Save output
        today = date.today()
        output_csv = io.StringIO()
        results_df.to_csv(output_csv, index=False)
        output_blob = blob_service_client.get_blob_client(container=container_name, blob=f'next_day_forecast/forecast{today}.csv')
        output_blob.upload_blob(output_csv.getvalue(), overwrite=True)
        
        logger.info("Forecast saved succesfully")

    except Exception as e:
        logger.exception(f"An error occurred during forecasting for the next day: {e}")
        sys.exit(1)


def main():
    logger.info("Starting the script...")

    try: 
        df, model = load_data_and_model()
        forecast_next_day(df, model)
        
        logger.info("Script completed successfully.")

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
