import sys
import os
from datetime import datetime, timedelta
import socket

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from classes.regionweather import RegionWeather
from db.db_manager import DatabaseManager

def is_postgres_running(host, port=5432):
    """Check if PostgreSQL is running on the specified host and port."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        s.connect((host, port))
        s.close()
        return True
    except (socket.timeout, ConnectionRefusedError):
        return False

def test_without_db():
    """Test RegionWeather class functionality without database interaction."""
    # Define a time period (last 30 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Define Florida region
    region_name = 'US-FLA-FMPP'
    region_data = {
        'lat': 28.525581, 
        'lon': -81.536775, 
        'alt': 0
    }
    
    print(f"Processing {region_name} data without database...")
    
    # Create RegionWeather object
    region_weather = RegionWeather(
        region_name=region_name,
        lat=region_data['lat'],
        lon=region_data['lon'],
        alt=region_data['alt'],
        start_date=start_date,
        end_date=end_date
    )
    
    # Check hourly data
    hourly_data = region_weather._hourly
    if not hourly_data.empty:
        print(f"Retrieved {len(hourly_data)} hourly records")
        print("Hourly data columns:", hourly_data.columns.tolist())
        print("Hourly data sample:")
        print(hourly_data.head())
    else:
        print("No hourly data available")
    
    # Check daily data
    daily_data = region_weather._daily
    if not daily_data.empty:
        print(f"Retrieved {len(daily_data)} daily records")
        print("Daily data columns:", daily_data.columns.tolist())
        print("Daily data sample:")
        print(daily_data.head())
    else:
        print("No daily data available")

def test_with_db(db_host, db_port, db_name, db_user, db_password):
    """Test RegionWeather class with database interaction."""
    # Initialize database
    db_manager = DatabaseManager(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password
    )
    
    try:
        # Create database schema
        db_manager.initialize_database()
        
        # Define a time period (last 30 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Define Florida region
        region_name = 'US-FLA-FMPP'
        region_data = {
            'lat': 28.525581, 
            'lon': -81.536775, 
            'alt': 0
        }
        
        print(f"Processing {region_name} with database...")
        
        # Create RegionWeather object
        region_weather = RegionWeather(
            region_name=region_name,
            lat=region_data['lat'],
            lon=region_data['lon'],
            alt=region_data['alt'],
            start_date=start_date,
            end_date=end_date
        )
        
        # Insert region into database
        db_manager.insert_region(region_weather)
        
        # Insert hourly data
        hourly_data = region_weather._hourly
        if not hourly_data.empty:
            db_manager.insert_hourly_data(region_name, hourly_data)
            print(f"Inserted {len(hourly_data)} hourly records")
        else:
            print("No hourly data available")
        
        # Insert daily data
        daily_data = region_weather._daily
        if not daily_data.empty:
            db_manager.insert_daily_data(region_name, daily_data)
            print(f"Inserted {len(daily_data)} daily records")
        else:
            print("No daily data available")
        
        # Query the data back
        print("\nRetrieving region data:")
        regions = db_manager.get_regions()
        for region in regions:
            if region['region_name'] == region_name:
                print(f"Region: {region['region_name']}")
                print(f"Coordinates: ({region['lat']}, {region['lon']})")
                print(f"Time range: {region['start_date']} to {region['end_date']}")
        
        # Get daily data
        print(f"\nDaily data for {region_name}:")
        daily_data = db_manager.get_daily_data(region_name)
        if not daily_data.empty:
            print(f"Retrieved {len(daily_data)} daily records")
            print(daily_data.head())
        else:
            print("No daily data available")
        
        # Get hourly data (just the last day for brevity)
        print(f"\nHourly data for {region_name} (last day):")
        recent_hourly_date = end_date - timedelta(days=1)
        hourly_data = db_manager.get_hourly_data(
            region_name=region_name,
            start_date=recent_hourly_date,
            end_date=end_date
        )
        
        if not hourly_data.empty:
            print(f"Retrieved {len(hourly_data)} hourly records")
            print(hourly_data.head())
        else:
            print("No hourly data available for this period")
    
    except Exception as e:
        print(f"Database test failed with error: {e}")

def main():
    """Main function to run tests."""
    # Get database connection parameters from environment variables or use defaults
    db_host = os.environ.get('POSTGRES_HOST', 'localhost')
    db_port = int(os.environ.get('POSTGRES_PORT', 5432))
    db_name = os.environ.get('POSTGRES_DB', 'weather_db')
    db_user = os.environ.get('POSTGRES_USER', 'postgres')
    db_password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
    
    # Check if PostgreSQL is running
    postgres_running = is_postgres_running(db_host, db_port)
    print(f"PostgreSQL server status at {db_host}:{db_port}: {'Running' if postgres_running else 'Not running'}")
    
    # Test RegionWeather class without database
    test_without_db()
    
    # Test with database if PostgreSQL is running
    if postgres_running:
        print(f"\nTesting with database at {db_host}:{db_port}...")
        test_with_db(db_host, db_port, db_name, db_user, db_password)
    else:
        print("\nSkipping database tests because PostgreSQL is not running.")
        print("To run database tests, please ensure PostgreSQL and TimescaleDB are running.")

if __name__ == "__main__":
    main() 