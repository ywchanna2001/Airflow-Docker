import os
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import pickle
import pytz

# Ensure models directory exists
MODELS_DIR = '/opt/airflow/models'
os.makedirs(MODELS_DIR, exist_ok=True)

# Define file paths
DATA_FILE_PATH = os.path.join(MODELS_DIR, 'stock_data.csv')
MODEL_FILE_PATH = os.path.join(MODELS_DIR, 'arima_model.pkl')

def fetch_stock_data():
    """Fetch stock data from Yahoo Finance for the past 2 years and save it."""
    ticker = 'AAPL'
    period = '2y'
    stock_data = yf.download(ticker, period=period)
    stock_data.to_csv(DATA_FILE_PATH)
    print(f"Stock data saved to {DATA_FILE_PATH}")

def make_stationary(price_series):
    """Apply differencing until the series becomes stationary."""
    max_diffs = 2  # Limit the number of differences
    for d in range(max_diffs + 1):
        result = adfuller(price_series)
        if result[1] < 0.05:
            print(f"Series is stationary after {d} differencing(s).")
            return price_series, d  # Return transformed series and differencing count
        price_series = price_series.diff().dropna()
    
    print(f"Series is still not stationary after {max_diffs} differencing(s), proceeding with max differencing.")
    return price_series, max_diffs

def train_arima_model():
    """Train the ARIMA model on the fetched data, ensuring stationarity."""
    stock_data = pd.read_csv(DATA_FILE_PATH, index_col=0, parse_dates=True)

    if 'Close' not in stock_data.columns:
        raise ValueError("Missing 'Close' column in stock data")
    
    stock_data['Close'] = pd.to_numeric(stock_data['Close'], errors='coerce')
    price_series = stock_data['Close'].dropna()

    price_series, d = make_stationary(price_series)

    model = ARIMA(price_series, order=(5, d, 0))
    model_fit = model.fit()

    # Save model to a file so it can be accessed in the next task
    with open(MODEL_FILE_PATH, 'wb') as f:
        pickle.dump(model_fit, f)

    print(f"ARIMA model trained successfully with differencing order {d}")

def save_model():
    """Load the trained ARIMA model and save relevant details with a timestamped filename."""
    stock_data = pd.read_csv(DATA_FILE_PATH, index_col=0, parse_dates=True)

    # Load the model from file
    with open(MODELS_DIR + '/arima_model.pkl', 'rb') as f:
        model_fit = pickle.load(f)

    # Save the model with metadata
    model_metadata = {
        'model_fit': model_fit,
        'ticker': 'AAPL',
        'last_date': stock_data.index[-1],
        'last_price': stock_data['Close'].iloc[-1]
    }

    # Get current time in Colombo
    colombo_tz = pytz.timezone('Asia/Colombo')
    current_time_colombo = datetime.now(colombo_tz)
    timestamp = current_time_colombo.strftime("%Y%m%d_%H%M%S")

    # Define the new filename with the timestamp
    model_filename = f'arima_model_{timestamp}.pkl'
    model_file_path = os.path.join(MODELS_DIR, model_filename)

    # Save the model metadata to the new file
    with open(model_file_path, 'wb') as f:
        pickle.dump(model_metadata, f)

    print(f"Model saved successfully to {model_file_path}")

# Define default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    'stock_prediction_model_retraining',
    default_args=default_args,
    description='Weekly ARIMA Model Retraining for Apple Stock',
    schedule_interval='@weekly',  # Runs once a week
    catchup=False,
) as dag:

    fetch_data_task = PythonOperator(
        task_id='fetch_stock_data',
        python_callable=fetch_stock_data,
    )

    train_model_task = PythonOperator(
        task_id='train_arima_model',
        python_callable=train_arima_model,
    )

    save_model_task = PythonOperator(
        task_id='save_model',
        python_callable=save_model,
    )

    # Set task dependencies
    fetch_data_task >> train_model_task >> save_model_task

