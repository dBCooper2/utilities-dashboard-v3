import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from classes.regionweather import RegionWeather
from db.db_manager import DatabaseManager

def main():
    # Initialize database
    db_manager = DatabaseManager(
        host='localhost',
        port=5432,
        dbname='weather_db',
        user='postgres',
        password='postgres'
    )
    
    # Create database schema
    db_manager.initialize_database()
    
    # Define a time period (last 30 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Create sample region weather objects
    regions = [
        {
            'name': 'New York',
            'lat': 40.7128,
            'lon': -74.0060,
            'alt': 10
        },
        {
            'name': 'London',
            'lat': 51.5074,
            'lon': -0.1278,
            'alt': 25
        },
        {
            'name': 'Tokyo',
            'lat': 35.6762,
            'lon': 139.6503,
            'alt': 40
        }
    ]
    
    # Process each region
    for region in regions:
        print(f"Processing {region['name']}...")
        
        # Create RegionWeather object
        region_weather = RegionWeather(
            region_name=region['name'],
            lat=region['lat'],
            lon=region['lon'],
            alt=region['alt'],
            start_date=start_date,
            end_date=end_date
        )
        
        # Insert region into database
        db_manager.insert_region(region_weather)
        
        # Insert hourly data
        hourly_data = region_weather._hourly
        if not hourly_data.empty:
            db_manager.insert_hourly_data(region['name'], hourly_data)
            print(f"Inserted {len(hourly_data)} hourly records")
        else:
            print("No hourly data available")
        
        # Insert daily data
        daily_data = region_weather._daily
        if not daily_data.empty:
            db_manager.insert_daily_data(region['name'], daily_data)
            print(f"Inserted {len(daily_data)} daily records")
        else:
            print("No daily data available")
    
    # Query example: get regions
    print("\nRetrieving all regions:")
    regions = db_manager.get_regions()
    for region in regions:
        print(f"Region: {region['region_name']}, Coordinates: ({region['lat']}, {region['lon']})")
        
        # Get recent daily data for this region
        print(f"\nRecent daily data for {region['region_name']}:")
        recent_date = end_date - timedelta(days=7)
        daily_data = db_manager.get_daily_data(
            region_name=region['region_name'],
            start_date=recent_date,
            end_date=end_date
        )
        
        if not daily_data.empty:
            print(daily_data.head())
        else:
            print("No daily data available for this period")
            
        # Get recent hourly data for this region
        print(f"\nRecent hourly data for {region['region_name']}:")
        recent_hourly_date = end_date - timedelta(days=1)
        hourly_data = db_manager.get_hourly_data(
            region_name=region['region_name'],
            start_date=recent_hourly_date,
            end_date=end_date
        )
        
        if not hourly_data.empty:
            print(hourly_data.head())
        else:
            print("No hourly data available for this period")
    
    print("\nDatabase example completed.")

if __name__ == "__main__":
    main() 