# Imports and Region Definitions ###########################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import meteostat as ms
import sqlalchemy
import sqlite3
import datetime as dt
import os
import signal
import sys
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, MetaData, Table, ForeignKey, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from backend.regionweather import RegionWeather
from backend.autoarima import AutoARIMAForecast
from backend.prophet import ProphetForecast
from backend.lstm import LSTMForecast

# Signal handler for graceful shutdown
def signal_handler(signum, frame):
    print("\nReceived interrupt signal. Cleaning up...")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

region_data = {
    'US-FLA-FMPP': {'lat': 28.525581, 'lon': -81.536775, 'alt': 0},
    'US-FLA-FPC': {'lat': 28.996695, 'lon': -82.886613, 'alt': 0},
    'US-FLA-FPL': {'lat': 27.917488, 'lon': -81.450970, 'alt': 0},
    'US-FLA-GVL': {'lat': 29.619310, 'lon': -82.328732, 'alt': 0},
    'US-FLA-HST': {'lat': 25.456904, 'lon': -80.588092, 'alt': 0},
    'US-FLA-JEA': {'lat': 30.390902, 'lon': -83.679837, 'alt': 0},
    'US-FLA-SEC': {'lat': 28.805983, 'lon': -82.306291, 'alt': 0},
    'US-FLA-TAL': {'lat': 30.437174, 'lon': -84.248042, 'alt': 0},
    'US-FLA-TEC': {'lat': 27.959413, 'lon': -82.144821, 'alt': 0}
}

# ############################################################

# Database Configuration #####################################

def get_db_connection():
    """Create and return a PostgreSQL database connection with TimescaleDB"""
    
    # Get database connection parameters from environment variables or use defaults
    db_user = os.environ.get('DB_USER', 'postgres')
    db_password = os.environ.get('DB_PASSWORD', 'postgres')
    db_host = os.environ.get('DB_HOST', 'localhost')
    db_port = os.environ.get('DB_PORT', '5432')
    db_name = os.environ.get('DB_NAME', 'weather_forecasts')
    
    # Create the connection string
    connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    # Create the SQLAlchemy engine
    try:
        # Check if PostgreSQL driver is installed
        try:
            import psycopg2
            postgresql_available = True
        except ImportError:
            postgresql_available = False
            print("PostgreSQL driver (psycopg2) not installed. Install with: pip install psycopg2-binary")
            print("Falling back to SQLite database")
            
        if postgresql_available:
            engine = create_engine(connection_string)
            # Test connection
            with engine.connect() as conn:
                pass
            return engine
        else:
            # Fall back to SQLite
            sqlite_path = os.path.join(os.getcwd(), 'weather_forecasts.db')
            return create_engine(f"sqlite:///{sqlite_path}")
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        print("Falling back to SQLite database")
        sqlite_path = os.path.join(os.getcwd(), 'weather_forecasts.db')
        return create_engine(f"sqlite:///{sqlite_path}")

def create_database_tables(engine):
    """Create the necessary database tables if they don't exist"""
    
    # Create metadata object
    metadata = MetaData()
    
    # Define the regions table
    regions = Table(
        'regions', metadata,
        Column('id', String, primary_key=True),
        Column('lat', Float, nullable=False),
        Column('lon', Float, nullable=False),
        Column('alt', Float, nullable=False),
        Column('created_at', DateTime, default=dt.datetime.now)
    )
    
    # Define the historical weather data table
    historical_weather = Table(
        'historical_weather', metadata,
        Column('id', Integer, primary_key=True),
        Column('region_id', String, ForeignKey('regions.id'), nullable=False),
        Column('timestamp', DateTime, nullable=False),
        Column('temp', Float),
        Column('dwpt', Float),
        Column('rhum', Float),
        Column('prcp', Float),
        Column('snow', Float),
        Column('wdir', Float),
        Column('wspd', Float),
        Column('wpgt', Float),
        Column('pres', Float),
        Column('tsun', Float),
        Column('coco', Float)
    )
    
    # Define the forecasts table
    forecasts = Table(
        'forecasts', metadata,
        Column('id', Integer, primary_key=True),
        Column('region_id', String, ForeignKey('regions.id'), nullable=False),
        Column('model_type', String, nullable=False),  # 'aarima', 'prophet', or 'lstm'
        Column('timestamp', DateTime, nullable=False),
        Column('predicted_value', Float, nullable=False),
        Column('confidence_low', Float),
        Column('confidence_high', Float),
        Column('confidence_level', Integer),
        Column('created_at', DateTime, default=dt.datetime.now)
    )
    
    # Define the model metrics table
    model_metrics = Table(
        'model_metrics', metadata,
        Column('id', Integer, primary_key=True),
        Column('region_id', String, ForeignKey('regions.id'), nullable=False),
        Column('model_type', String, nullable=False),
        Column('metric_name', String, nullable=False),
        Column('metric_value', Float, nullable=False),
        Column('evaluated_at', DateTime, default=dt.datetime.now)
    )
    
    # Create the tables in the database
    metadata.create_all(engine)
    
    # Check if this is PostgreSQL and try to convert tables to TimescaleDB hypertables
    if 'postgresql' in engine.url.drivername:
        with engine.connect() as connection:
            # Check if TimescaleDB extension is installed
            try:
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))
                
                # Convert historical_weather to a hypertable
                connection.execute(text(
                    "SELECT create_hypertable('historical_weather', 'timestamp', if_not_exists => TRUE);"
                ))
                
                # Convert forecasts to a hypertable
                connection.execute(text(
                    "SELECT create_hypertable('forecasts', 'timestamp', if_not_exists => TRUE);"
                ))
                
                # Create indices for faster queries
                connection.execute(text(
                    "CREATE INDEX IF NOT EXISTS idx_historical_region_time ON historical_weather (region_id, timestamp DESC);"
                ))
                connection.execute(text(
                    "CREATE INDEX IF NOT EXISTS idx_forecasts_region_model_time ON forecasts (region_id, model_type, timestamp DESC);"
                ))
                
                print("TimescaleDB hypertables created successfully")
            except Exception as e:
                print(f"Warning: Could not create TimescaleDB hypertables: {e}")
                print("Continuing with regular PostgreSQL tables")
    else:
        print("Using SQLite database - TimescaleDB features not available")
    
    return metadata

def insert_region_data(engine, region_data):
    """Insert region data into the database"""
    
    with engine.connect() as connection:
        # Prepare data for insertion
        regions_to_insert = []
        for region_id, data in region_data.items():
            regions_to_insert.append({
                'id': region_id,
                'lat': data['lat'],
                'lon': data['lon'],
                'alt': data['alt'],
                'created_at': dt.datetime.now()
            })
        
        # Insert data if not exists
        for region in regions_to_insert:
            try:
                # Check if region already exists
                result = connection.execute(
                    text("SELECT id FROM regions WHERE id = :id"),
                    {"id": region['id']}
                ).fetchone()
                
                if not result:
                    # Insert if not exists
                    connection.execute(
                        text("INSERT INTO regions (id, lat, lon, alt, created_at) VALUES (:id, :lat, :lon, :alt, :created_at)"),
                        region
                    )
                
                # Commit the transaction
                connection.commit()
            except Exception as e:
                print(f"Error inserting region {region['id']}: {e}")
                connection.rollback()

def insert_historical_weather(engine, region_id, weather_df):
    """Insert historical weather data into the database"""
    
    try:
        # Create a copy to avoid modifying the original
        df = weather_df.copy()
        
        # Reset index to make datetime column available if it's in the index
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        
        # Rename time column if needed
        if 'time' in df.columns:
            df = df.rename(columns={'time': 'timestamp'})
        
        # Add region_id to the dataframe
        df['region_id'] = region_id
        
        # Select only the columns we need and rename them to match database schema
        columns_mapping = {
            'timestamp': 'timestamp', 
            'region_id': 'region_id',
            'temp': 'temp',
            'dwpt': 'dwpt', 
            'rhum': 'rhum',
            'prcp': 'prcp',
            'snow': 'snow',
            'wdir': 'wdir',
            'wspd': 'wspd',
            'wpgt': 'wpgt',
            'pres': 'pres',
            'tsun': 'tsun',
            'coco': 'coco'
        }
        
        # Filter and rename columns
        cols_to_use = [col for col in columns_mapping.keys() if col in df.columns]
        db_data = df[cols_to_use].rename(columns={k: v for k, v in columns_mapping.items() if k in cols_to_use})
        
        # Handle potential issues with the data
        if 'timestamp' not in db_data.columns:
            print(f"Error: No timestamp column found for region {region_id}. Available columns: {df.columns.tolist()}")
            return
        
        # Make sure timestamp is datetime
        db_data['timestamp'] = pd.to_datetime(db_data['timestamp'])
        
        # Insert into database
        db_data.to_sql('historical_weather', engine, if_exists='append', index=False)
        print(f"Successfully inserted {len(db_data)} weather records for {region_id}")
        
    except Exception as e:
        print(f"Error inserting historical weather data for {region_id}: {e}")

def insert_forecast_data(engine, region_id, model_type, forecast_df, confidence_level=None):
    """Insert forecast data into the database"""
    
    try:
        # Create a copy to avoid modifying the original
        df = forecast_df.copy()
        
        # Rename columns to match database schema
        if 'ds' in df.columns:
            df = df.rename(columns={'ds': 'timestamp'})
        
        # Determine the value column based on model type
        value_col = model_type.upper() if model_type.upper() in df.columns else df.columns[2]
        
        # Prepare data for database
        forecast_data = []
        
        for idx, row in df.iterrows():
            # Convert pandas Timestamp to Python datetime
            timestamp = row['timestamp']
            if isinstance(timestamp, pd.Timestamp):
                timestamp = timestamp.to_pydatetime()
                
            forecast_entry = {
                'region_id': region_id,
                'model_type': model_type,
                'timestamp': timestamp,
                'predicted_value': row[value_col],
                'confidence_low': None,
                'confidence_high': None,
                'confidence_level': None,
                'created_at': dt.datetime.now()
            }
            
            # Add confidence intervals if available
            if confidence_level is not None:
                lower_col = f"{model_type.upper()}-lo-{confidence_level}"
                upper_col = f"{model_type.upper()}-hi-{confidence_level}"
                
                if lower_col in df.columns and upper_col in df.columns:
                    forecast_entry['confidence_low'] = row[lower_col]
                    forecast_entry['confidence_high'] = row[upper_col]
                    forecast_entry['confidence_level'] = confidence_level
            
            forecast_data.append(forecast_entry)
        
        # Insert data into database
        with engine.connect() as connection:
            try:
                # First, delete any existing forecasts for this region and model
                connection.execute(
                    text("DELETE FROM forecasts WHERE region_id = :region_id AND model_type = :model_type"),
                    {"region_id": region_id, "model_type": model_type}
                )
                
                # Insert new forecasts
                for entry in forecast_data:
                    connection.execute(
                        text("""
                            INSERT INTO forecasts 
                            (region_id, model_type, timestamp, predicted_value, confidence_low, confidence_high, confidence_level, created_at)
                            VALUES (:region_id, :model_type, :timestamp, :predicted_value, :confidence_low, :confidence_high, :confidence_level, :created_at)
                        """),
                        entry
                    )
                
                # Commit the transaction
                connection.commit()
                print(f"Successfully inserted {len(forecast_data)} forecast points for {region_id} using {model_type}")
            
            except Exception as e:
                print(f"Error inserting forecast data: {e}")
                connection.rollback()
    
    except Exception as e:
        print(f"Error preparing forecast data for {region_id} with model {model_type}: {e}")

def insert_model_metrics(engine, region_id, model_type, metrics_df):
    """Insert model evaluation metrics into the database"""
    
    try:
        if metrics_df is None or len(metrics_df) == 0:
            print(f"No metrics data available for {region_id} with model {model_type}")
            return
            
        metrics_data = []
        
        for idx, row in metrics_df.iterrows():
            metric_entry = {
                'region_id': region_id,
                'model_type': model_type,
                'metric_name': row['metric'],
                'metric_value': row['value'],
                'evaluated_at': dt.datetime.now()
            }
            metrics_data.append(metric_entry)
        
        # Insert data into database
        with engine.connect() as connection:
            try:
                # First, delete any existing metrics for this region and model
                connection.execute(
                    text("DELETE FROM model_metrics WHERE region_id = :region_id AND model_type = :model_type"),
                    {"region_id": region_id, "model_type": model_type}
                )
                
                # Insert new metrics
                for entry in metrics_data:
                    connection.execute(
                        text("""
                            INSERT INTO model_metrics 
                            (region_id, model_type, metric_name, metric_value, evaluated_at)
                            VALUES (:region_id, :model_type, :metric_name, :metric_value, :evaluated_at)
                        """),
                        entry
                    )
                
                # Commit the transaction
                connection.commit()
                print(f"Successfully inserted {len(metrics_data)} metrics for {region_id} using {model_type}")
                
            except Exception as e:
                print(f"Error inserting metrics data: {e}")
                connection.rollback()
    
    except Exception as e:
        print(f"Error preparing metrics data for {region_id} with model {model_type}: {e}")

def get_latest_timestamp(engine, region_id):
    """Get the latest timestamp for a region from the historical_weather table"""
    try:
        with engine.connect() as connection:
            result = connection.execute(
                text("""
                    SELECT MAX(timestamp) as latest_timestamp 
                    FROM historical_weather 
                    WHERE region_id = :region_id
                """),
                {"region_id": region_id}
            ).fetchone()
            
            if result and result[0]:
                return pd.to_datetime(result[0])
            return None
    except Exception as e:
        print(f"Error getting latest timestamp for {region_id}: {e}")
        return None

# ############################################################

if __name__ == '__main__':
    # Create database connection first to check existing data
    print("Connecting to database...")
    engine = get_db_connection()
    
    # Create tables if they don't exist
    print("Creating database tables...")
    metadata = create_database_tables(engine)
    
    # Fetch Meteostat Data for each region #######################
    print("Fetching weather data for regions...")
    regionweather_list = []
    
    end = dt.datetime.now()
    start = end - dt.timedelta(days=365)  # Default to 1 year of historical data
    
    for region_name, coords in region_data.items():
        try:
            print(f"Checking existing data for {region_name}...")
            latest_timestamp = get_latest_timestamp(engine, region_name)
            
            if latest_timestamp:
                # If we have existing data, fetch from the latest timestamp to now
                start = latest_timestamp
                print(f"Found existing data for {region_name}, fetching from {start} to {end}")
            else:
                # If no existing data, fetch the last year of data
                start = end - dt.timedelta(days=365)
                print(f"No existing data found for {region_name}, fetching last 365 days")
            
            print(f"Fetching data for {region_name}...")
            rw = RegionWeather(region_name, coords['lat'], coords['lon'], coords['alt'], start, end)
            
            # Check if we have enough data points
            if hasattr(rw, 'to_dict'):
                data_dict = rw.to_dict()
                df = data_dict.get('df_hourly', None)
                if df is not None and len(df) > 1:
                    regionweather_list.append(rw)
                    print(f"Successfully fetched {len(df)} data points for {region_name}")
                else:
                    print(f"Warning: Not enough data points for {region_name}. Got {len(df) if df is not None else 0} points.")
            else:
                print(f"Warning: Could not extract data for {region_name}")
        except Exception as e:
            print(f"Error fetching data for {region_name}: {e}")
            
    # ############################################################
    
    # Forecast Weather with AARIMA ###############################
    
    print("\nGenerating AARIMA forecasts...")
    aarima_list = []
    
    for rw in regionweather_list:
        try:
            region_name = rw.region_name if hasattr(rw, 'region_name') else getattr(rw, '_region_name', None)
            if not region_name:
                print(f"Warning: Could not determine region name for {rw}")
                continue
                
            print(f"Creating AARIMA model for {region_name}...")
            
            # Get appropriate dataframe - try different methods to be flexible
            df = None
            if hasattr(rw, 'to_dict'):
                data_dict = rw.to_dict()
                df = data_dict.get('df_hourly', None)
            
            if df is None:
                print(f"Warning: Could not extract data for {region_name}")
                continue
            
            # Print the dataframe index and columns for debugging
            print(f"DataFrame index type: {type(df.index)}")
            print(f"DataFrame columns: {df.columns.tolist()}")
            
            # Transform the dataframe for AARIMA
            # Reset index if it's a DateTimeIndex
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                # Rename the index column to 'ds'
                if 'index' in df.columns:
                    df = df.rename(columns={'index': 'ds'})
            
            # Format the DataFrame properly for AARIMA
            # Check and create required columns
            if 'time' in df.columns and 'ds' not in df.columns:
                df = df.rename(columns={'time': 'ds'})
            
            if 'temp' in df.columns and 'y' not in df.columns:
                df = df.rename(columns={'temp': 'y'})
                
            if 'unique_id' not in df.columns:
                df['unique_id'] = region_name
                
            # Make sure ds is datetime type
            if 'ds' in df.columns:
                df['ds'] = pd.to_datetime(df['ds'])
            else:
                print(f"Warning: No timestamp column found for {region_name}. Available columns: {df.columns.tolist()}")
                continue
                
            # Create and fit AARIMA model
            print(f"AARIMA DataFrame columns: {df.columns.tolist()}")
            aarima = AutoARIMAForecast(region_name=region_name, initial_df=df)
            
            # Generate forecast
            print(f"Generating AARIMA forecast for {region_name}...")
            forecast = aarima.forecast(h=7, level=[80, 95])
            
            # Evaluate model
            print(f"Evaluating AARIMA model for {region_name}...")
            metrics = aarima.evaluate(h=7, step_size=7, n_windows=4)
            
            aarima_list.append(aarima)
            print(f"AARIMA processing complete for {region_name}")
            
        except Exception as e:
            print(f"Error processing AARIMA for region: {e}")
            import traceback
            traceback.print_exc()
    
    # ############################################################
    
    # Forecast Weather with Prophet ##############################
    
    print("\nGenerating Prophet forecasts...")
    prophet_list = []
    
    for rw in regionweather_list:
        try:
            region_name = rw.region_name if hasattr(rw, 'region_name') else getattr(rw, '_region_name', None)
            if not region_name:
                print(f"Warning: Could not determine region name for {rw}")
                continue
                
            print(f"Creating Prophet model for {region_name}...")
            
            # Get appropriate dataframe - try different methods to be flexible
            df = None
            if hasattr(rw, 'to_dict'):
                data_dict = rw.to_dict()
                df = data_dict.get('df_hourly', None)
            
            if df is None:
                print(f"Warning: Could not extract data for {region_name}")
                continue
                
            # Transform the dataframe for Prophet
            # Reset index if it's a DateTimeIndex
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                # Rename the index column to 'ds'
                if 'index' in df.columns:
                    df = df.rename(columns={'index': 'ds'})
            
            # Format the DataFrame properly for Prophet
            # Check and create required columns
            if 'time' in df.columns and 'ds' not in df.columns:
                df = df.rename(columns={'time': 'ds'})
            
            if 'temp' in df.columns and 'y' not in df.columns:
                df = df.rename(columns={'temp': 'y'})
                
            if 'unique_id' not in df.columns:
                df['unique_id'] = region_name
                
            # Make sure ds is datetime type
            if 'ds' in df.columns:
                df['ds'] = pd.to_datetime(df['ds'])
            else:
                print(f"Warning: No timestamp column found for {region_name}. Available columns: {df.columns.tolist()}")
                continue
                
            # Create and fit Prophet model
            print(f"Prophet DataFrame columns: {df.columns.tolist()}")
            prophet = ProphetForecast(region_name=region_name, data=df)
            
            # Generate forecast
            print(f"Generating Prophet forecast for {region_name}...")
            forecast = prophet.forecast(h=7, level=[80, 95])
            
            # Evaluate model
            print(f"Evaluating Prophet model for {region_name}...")
            metrics = prophet.evaluate(h=7, step_size=7, n_windows=4)
            
            prophet_list.append(prophet)
            print(f"Prophet processing complete for {region_name}")
            
        except Exception as e:
            print(f"Error processing Prophet for region: {e}")
            import traceback
            traceback.print_exc()
    
    # ############################################################
    
    # Forecast Weather with LSTM #################################
    
    print("\nGenerating LSTM forecasts...")
    lstm_list = []
    
    # Check if TensorFlow is available
    try:
        import tensorflow as tf
        tensorflow_available = True
        print("TensorFlow is available. LSTM forecasting will be enabled.")
    except ImportError:
        tensorflow_available = False
        print("TensorFlow is not installed. LSTM forecasting will be skipped.")
        print("To enable LSTM forecasting, install TensorFlow with: pip install tensorflow")
    
    if tensorflow_available:
        for rw in regionweather_list:
            try:
                region_name = rw.region_name if hasattr(rw, 'region_name') else getattr(rw, '_region_name', None)
                if not region_name:
                    print(f"Warning: Could not determine region name for {rw}")
                    continue
                    
                print(f"Creating LSTM model for {region_name}...")
                
                # Get appropriate dataframe - try different methods to be flexible
                df = None
                if hasattr(rw, 'to_dict'):
                    data_dict = rw.to_dict()
                    # LSTM uses daily data
                    df = data_dict.get('df_daily', None)
                
                if df is None:
                    print(f"Warning: Could not extract data for {region_name}")
                    continue
                    
                # Transform the dataframe for LSTM
                # Reset index if it's a DateTimeIndex
                if isinstance(df.index, pd.DatetimeIndex):
                    df = df.reset_index()
                    # Rename the index column to 'ds'
                    if 'index' in df.columns:
                        df = df.rename(columns={'index': 'ds'})
                
                # Format the DataFrame properly for LSTM
                # Check and create required columns
                if 'time' in df.columns and 'ds' not in df.columns:
                    df = df.rename(columns={'time': 'ds'})
                
                if 'temp' in df.columns and 'y' not in df.columns:
                    df = df.rename(columns={'temp': 'y'})
                    
                if 'unique_id' not in df.columns:
                    df['unique_id'] = region_name
                    
                # Make sure ds is datetime type
                if 'ds' in df.columns:
                    df['ds'] = pd.to_datetime(df['ds'])
                else:
                    print(f"Warning: No timestamp column found for {region_name}. Available columns: {df.columns.tolist()}")
                    continue
                    
                # Create and fit LSTM model
                print(f"LSTM DataFrame columns: {df.columns.tolist()}")
                lstm = LSTMForecast(region_name=region_name, initial_df=df)
                
                # Generate forecast
                print(f"Generating LSTM forecast for {region_name}...")
                forecast = lstm.forecast(h=7, level=[80, 95])
                
                # Evaluate model
                print(f"Evaluating LSTM model for {region_name}...")
                metrics = lstm.evaluate(h=7, step_size=7, n_windows=4)
                
                lstm_list.append(lstm)
                print(f"LSTM processing complete for {region_name}")
                
            except Exception as e:
                print(f"Error processing LSTM for region: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("Skipping LSTM forecasting due to missing TensorFlow dependency.")
    
    # ############################################################
    
    # Add data to database #######################################
    
    print("\nStoring data in database...")
    
    # Insert region data
    print("Inserting region data...")
    insert_region_data(engine, region_data)
    
    # Insert historical weather data and forecasts
    print("Inserting weather data and forecasts...")
    for rw in regionweather_list:
        try:
            region_name = rw.region_name if hasattr(rw, 'region_name') else getattr(rw, '_region_name', None)
            if not region_name:
                print(f"Warning: Could not determine region name for {rw}")
                continue
                    
            # Get weather data
            weather_df = None
            if hasattr(rw, 'to_dict'):
                data_dict = rw.to_dict()
                weather_df = data_dict.get('df_hourly', None)
            
            if weather_df is None:
                print(f"Warning: Could not extract weather data for {region_name}")
                continue
                    
            # Transform dataframe for storage
            # Reset index if it's a DateTimeIndex to make the timestamp a column
            if isinstance(weather_df.index, pd.DatetimeIndex):
                weather_df = weather_df.reset_index()
                    
            # Insert historical weather data
            print(f"Inserting historical weather data for {region_name}...")
            insert_historical_weather(engine, region_name, weather_df)
            
            # Find the corresponding forecast models for this region
            aarima_model = next((model for model in aarima_list if model.region_name == region_name), None)
            prophet_model = next((model for model in prophet_list if model.region_name == region_name), None)
            lstm_model = next((model for model in lstm_list if model.region_name == region_name), None)
            
            # Insert AARIMA forecasts
            if aarima_model and hasattr(aarima_model, 'forecast_df') and aarima_model.forecast_df is not None:
                print(f"Inserting AARIMA forecast for {region_name}...")
                insert_forecast_data(engine, region_name, 'aarima', aarima_model.forecast_df, confidence_level=80)
                
                # Insert metrics if available
                metrics = aarima_model.evaluate(h=7, step_size=7, n_windows=4)
                if metrics is not None and len(metrics) > 0:
                    print(f"Inserting AARIMA metrics for {region_name}...")
                    insert_model_metrics(engine, region_name, 'aarima', metrics)
            
            # Insert Prophet forecasts
            if prophet_model and hasattr(prophet_model, 'forecast_df') and prophet_model.forecast_df is not None:
                print(f"Inserting Prophet forecast for {region_name}...")
                insert_forecast_data(engine, region_name, 'prophet', prophet_model.forecast_df, confidence_level=80)
                
                # Insert metrics if available
                metrics = prophet_model.evaluate(h=7, step_size=7, n_windows=4)
                if metrics is not None and len(metrics) > 0:
                    print(f"Inserting Prophet metrics for {region_name}...")
                    insert_model_metrics(engine, region_name, 'prophet', metrics)
            
            # Insert LSTM forecasts
            if lstm_model and hasattr(lstm_model, 'forecast_df') and lstm_model.forecast_df is not None:
                print(f"Inserting LSTM forecast for {region_name}...")
                insert_forecast_data(engine, region_name, 'lstm', lstm_model.forecast_df, confidence_level=80)
                
                # Insert metrics if available
                metrics = lstm_model.evaluate(h=7, step_size=7, n_windows=4)
                if metrics is not None and len(metrics) > 0:
                    print(f"Inserting LSTM metrics for {region_name}...")
                    insert_model_metrics(engine, region_name, 'lstm', metrics)
        
        except Exception as e:
            print(f"Error processing database operations for region: {e}")
    
    print("All data has been successfully stored in the database.")
    
    # ############################################################
