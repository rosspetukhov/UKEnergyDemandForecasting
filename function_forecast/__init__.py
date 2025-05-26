import logging
import os
import io
import pandas as pd
import pickle
from datetime import date
from azure.storage.blob import BlobServiceClient
import azure.functions as func

def load_data_and_model(logger):
    """
    Load the latest NESO data and forecasting model from Azure Blob storage.

    Parameters:
        logger (logging.Logger): Logger instance for logging messages.

    Returns:
        tuple: A tuple containing:
            - df (pandas.DataFrame): DataFrame with NESO data filtered for actual demand.
            - model: The forecasting model loaded from a pickle file.

    Raises:
        Exception: If there is an error during data or model loading.
    """


    logger.info("Starting loading data and model")

    try:
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

        logger.info("Data and model loaded successfully")
        return df, model

    except Exception as e:
        logger.error(f"Error loading data and model: {e}", exc_info=True)
        raise

def forecast_next_day(df, model, logger):
    """
    Forecast for next day and save results in blob storage.

    Parameters:
        df (pandas.DataFrame): DataFrame with NESO data filtered for actual demand.
        model: The forecasting model loaded from a pickle file.
        logger (logging.Logger): Logger instance for logging messages.

    Returns:
        None

    Raises:
        Exception: If there is an error during forecast generation or saving to blob.
    """

    logger.info("Starting forecasting for the next day")
    try:
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

        results_df = pd.DataFrame({'Timestamp': future_timestamps, 'Demand': future_preds})

        connect_str = os.environ['AzureWebJobsStorage']
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        container_name = 'forecasting'

        today = date.today()
        output_csv = io.StringIO()
        results_df.to_csv(output_csv, index=False)
        output_blob = blob_service_client.get_blob_client(container=container_name, blob=f'next_day_forecast/forecast{today}.csv')
        output_blob.upload_blob(output_csv.getvalue(), overwrite=True)

        logger.info("Forecast saved successfully")

    except Exception as e:
        logger.error(f"Error during forecasting: {e}", exc_info=True)
        raise

# Azure Function entry point
def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    logging.info('Python HTTP trigger function started.')

    try:
        df, model = load_data_and_model(logging)
        forecast_next_day(df, model, logging)

        return func.HttpResponse("Forecast generated and saved successfully.", status_code=200)

    except Exception as e:
        logging.error(f"Function failed: {e}", exc_info=True)
        return func.HttpResponse(f"Function error: {str(e)}", status_code=500)
