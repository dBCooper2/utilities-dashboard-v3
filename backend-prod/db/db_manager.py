import psycopg2
import pandas as pd
from psycopg2 import sql
from psycopg2.extras import execute_values
import numpy as np

class DatabaseManager:
    def __init__(self, host='localhost', port=5432, dbname='weather_db', user='postgres', password='postgres'):
        """Initialize the database manager with connection parameters."""
        self.conn_params = {
            'host': host,
            'port': port,
            'dbname': dbname,
            'user': user,
            'password': password
        }
        self.conn = None
        self.cursor = None

    def connect(self):
        """Connect to the PostgreSQL database."""
        try:
            self.conn = psycopg2.connect(**self.conn_params)
            self.cursor = self.conn.cursor()
            print("Connected to the database successfully.")
        except Exception as e:
            print(f"Error connecting to the database: {e}")
            self.conn = None
            self.cursor = None
            raise

    def disconnect(self):
        """Close the database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            print("Database connection closed.")
        self.cursor = None
        self.conn = None

    def initialize_database(self):
        """Initialize the database with TimescaleDB extension and create tables."""
        try:
            self.connect()
            
            # Create TimescaleDB extension
            self.cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
            
            # Create regions table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS regions (
                    region_name TEXT PRIMARY KEY,
                    lat FLOAT NOT NULL,
                    lon FLOAT NOT NULL,
                    alt FLOAT,
                    start_date TIMESTAMP NOT NULL,
                    end_date TIMESTAMP NOT NULL
                );
            """)
            
            # Create hourly data table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS hourly_data (
                    time TIMESTAMP NOT NULL,
                    region_name TEXT NOT NULL,
                    temp FLOAT,
                    dwpt FLOAT,
                    rhum FLOAT,
                    prcp FLOAT,
                    snow FLOAT,
                    wdir FLOAT,
                    wspd FLOAT,
                    wpgt FLOAT,
                    pres FLOAT,
                    tsun FLOAT,
                    coco INT,
                    PRIMARY KEY (region_name, time),
                    FOREIGN KEY (region_name) REFERENCES regions(region_name)
                );
            """)
            
            # Create daily data table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_data (
                    time TIMESTAMP NOT NULL,
                    region_name TEXT NOT NULL,
                    temp FLOAT,
                    tmin FLOAT,
                    tmax FLOAT,
                    tavg FLOAT,
                    dwpt FLOAT,
                    rhum FLOAT,
                    prcp FLOAT,
                    snow FLOAT,
                    wdir FLOAT,
                    wspd FLOAT,
                    wpgt FLOAT,
                    pres FLOAT,
                    tsun FLOAT,
                    coco INT,
                    PRIMARY KEY (region_name, time),
                    FOREIGN KEY (region_name) REFERENCES regions(region_name)
                );
            """)
            
            # Convert tables to hypertables
            self.cursor.execute("SELECT create_hypertable('hourly_data', 'time', if_not_exists => TRUE);")
            self.cursor.execute("SELECT create_hypertable('daily_data', 'time', if_not_exists => TRUE);")
            
            self.conn.commit()
            print("Database initialized successfully.")
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            print(f"Error initializing database: {e}")
            raise
        finally:
            self.disconnect()

    def insert_region(self, region_weather):
        """Insert a region into the regions table."""
        try:
            self.connect()
            
            region_data = region_weather.to_dict()
            
            query = """
                INSERT INTO regions (region_name, lat, lon, alt, start_date, end_date)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (region_name) DO UPDATE SET
                    lat = EXCLUDED.lat,
                    lon = EXCLUDED.lon,
                    alt = EXCLUDED.alt,
                    start_date = EXCLUDED.start_date,
                    end_date = EXCLUDED.end_date;
            """
            
            self.cursor.execute(query, (
                region_data['region_name'],
                region_data['lat'],
                region_data['lon'],
                region_data['alt'],
                region_data['start_date'],
                region_data['end_date']
            ))
            
            self.conn.commit()
            print(f"Region {region_data['region_name']} inserted successfully.")
        except Exception as e:
            self.conn.rollback()
            print(f"Error inserting region: {e}")
            raise
        finally:
            self.disconnect()

    def insert_hourly_data(self, region_name, hourly_df):
        """Insert hourly data into the hourly_data table."""
        try:
            self.connect()
            
            # Ensure the DataFrame has a datetime index
            if not isinstance(hourly_df.index, pd.DatetimeIndex):
                raise ValueError("Hourly DataFrame must have a DatetimeIndex")
            
            # Prepare data for insertion
            df = hourly_df.reset_index()
            df.rename(columns={'index': 'time'}, inplace=True)
            df['region_name'] = region_name
            
            # Replace NaN values with None for proper SQL NULL handling
            df = df.replace({np.nan: None})
            
            # Get column names
            columns = df.columns.tolist()
            
            # Create SQL query
            table_name = "hourly_data"
            column_str = ", ".join(columns)
            placeholder_str = ", ".join(["%s"] * len(columns))
            
            query = f"""
                INSERT INTO {table_name} ({column_str})
                VALUES ({placeholder_str})
                ON CONFLICT (region_name, time) DO UPDATE SET
            """
            
            # Add SET clause for each column except primary key columns
            update_columns = [col for col in columns if col not in ['region_name', 'time']]
            update_str = ", ".join([f"{col} = EXCLUDED.{col}" for col in update_columns])
            query += update_str + ";"
            
            # Convert dataframe to list of tuples
            records = df.to_records(index=False)
            data = [tuple(record) for record in records]
            
            # Execute query
            self.cursor.executemany(query, data)
            
            self.conn.commit()
            print(f"Inserted {len(df)} hourly records for region {region_name}.")
        except Exception as e:
            self.conn.rollback()
            print(f"Error inserting hourly data: {e}")
            raise
        finally:
            self.disconnect()

    def insert_daily_data(self, region_name, daily_df):
        """Insert daily data into the daily_data table."""
        try:
            self.connect()
            
            # Ensure the DataFrame has a datetime index
            if not isinstance(daily_df.index, pd.DatetimeIndex):
                raise ValueError("Daily DataFrame must have a DatetimeIndex")
            
            # Prepare data for insertion
            df = daily_df.reset_index()
            df.rename(columns={'index': 'time'}, inplace=True)
            df['region_name'] = region_name
            
            # Replace NaN values with None for proper SQL NULL handling
            df = df.replace({np.nan: None})
            
            # Get column names
            columns = df.columns.tolist()
            
            # Create SQL query
            table_name = "daily_data"
            column_str = ", ".join(columns)
            placeholder_str = ", ".join(["%s"] * len(columns))
            
            query = f"""
                INSERT INTO {table_name} ({column_str})
                VALUES ({placeholder_str})
                ON CONFLICT (region_name, time) DO UPDATE SET
            """
            
            # Add SET clause for each column except primary key columns
            update_columns = [col for col in columns if col not in ['region_name', 'time']]
            update_str = ", ".join([f"{col} = EXCLUDED.{col}" for col in update_columns])
            query += update_str + ";"
            
            # Convert dataframe to list of tuples
            records = df.to_records(index=False)
            data = [tuple(record) for record in records]
            
            # Execute query
            self.cursor.executemany(query, data)
            
            self.conn.commit()
            print(f"Inserted {len(df)} daily records for region {region_name}.")
        except Exception as e:
            self.conn.rollback()
            print(f"Error inserting daily data: {e}")
            raise
        finally:
            self.disconnect()

    def get_regions(self):
        """Get all regions from the database."""
        try:
            self.connect()
            
            query = "SELECT * FROM regions;"
            self.cursor.execute(query)
            
            columns = [desc[0] for desc in self.cursor.description]
            regions = [dict(zip(columns, row)) for row in self.cursor.fetchall()]
            
            return regions
        except Exception as e:
            print(f"Error retrieving regions: {e}")
            raise
        finally:
            self.disconnect()

    def get_hourly_data(self, region_name, start_date=None, end_date=None):
        """Get hourly data for a specific region and time range."""
        try:
            self.connect()
            
            query = "SELECT * FROM hourly_data WHERE region_name = %s"
            params = [region_name]
            
            if start_date:
                query += " AND time >= %s"
                params.append(start_date)
            
            if end_date:
                query += " AND time <= %s"
                params.append(end_date)
            
            query += " ORDER BY time;"
            
            self.cursor.execute(query, params)
            
            columns = [desc[0] for desc in self.cursor.description]
            data = [dict(zip(columns, row)) for row in self.cursor.fetchall()]
            
            # Convert to DataFrame with time as index
            df = pd.DataFrame(data)
            if not df.empty:
                df.set_index('time', inplace=True)
                df.drop('region_name', axis=1, inplace=True)
            
            return df
        except Exception as e:
            print(f"Error retrieving hourly data: {e}")
            raise
        finally:
            self.disconnect()

    def get_daily_data(self, region_name, start_date=None, end_date=None):
        """Get daily data for a specific region and time range."""
        try:
            self.connect()
            
            query = "SELECT * FROM daily_data WHERE region_name = %s"
            params = [region_name]
            
            if start_date:
                query += " AND time >= %s"
                params.append(start_date)
            
            if end_date:
                query += " AND time <= %s"
                params.append(end_date)
            
            query += " ORDER BY time;"
            
            self.cursor.execute(query, params)
            
            columns = [desc[0] for desc in self.cursor.description]
            data = [dict(zip(columns, row)) for row in self.cursor.fetchall()]
            
            # Convert to DataFrame with time as index
            df = pd.DataFrame(data)
            if not df.empty:
                df.set_index('time', inplace=True)
                df.drop('region_name', axis=1, inplace=True)
            
            return df
        except Exception as e:
            print(f"Error retrieving daily data: {e}")
            raise
        finally:
            self.disconnect()
