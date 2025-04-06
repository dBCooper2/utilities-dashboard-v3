import pandas as pd
import meteostat as ms
import numpy as np
import math

class RegionWeather:
    def __init__(self, region_name, lat, lon, alt, start_date, end_date):
        self._region_name = region_name
        self._lat = lat
        self._lon = lon
        self._alt = alt
        self._start_date = start_date
        self._end_date = end_date

        # Create a Point object with lat, lon, and alt
        self._pt = ms.Point(self._lat, self._lon, self._alt)

        self._hourly = self._get_hourly_data()
        self._daily = self._fetch_daily_data()

    def _get_hourly_data(self):
        # Pass the time period to the Hourly constructor
        hourly = ms.Hourly(self._pt, self._start_date, self._end_date)
        data = hourly.fetch()
        print(f"Fetched {len(data)} hourly records")
        return data
    
    # Get circular mean of wind direction in degrees
    def _degree_mean(self, degrees):
        if degrees.isna().all():
            return np.nan
        
        # Convert to radians and calculate the averages of sin and cos components
        radians = np.radians(degrees.dropna())
        sin_mean = np.nanmean(np.sin(radians))
        cos_mean = np.nanmean(np.cos(radians))
        
        # Convert back to degrees
        degrees_mean = np.degrees(math.atan2(sin_mean, cos_mean))
        
        # Ensure the result is in [0, 360)
        return (degrees_mean + 360) % 360

    # Compute daily data instead of fetching from meteostat, Daily obj returns a lot of NaNs
    def _fetch_daily_data(self):
        print('Fetching Daily Data...')
        if self._hourly.empty:
            return pd.DataFrame()
        
        # Define aggregation methods for each column
        agg_methods = {
            'temp': 'mean',
            'dwpt': 'mean',
            'rhum': 'mean',
            'prcp': 'sum',
            'snow': 'max',
            'wdir': self._degree_mean,
            'wspd': 'mean',
            'wpgt': 'max',
            'pres': 'mean',
            'tsun': 'sum',
            'coco': 'max'
        }
        
        # Filter for columns that actually exist in the dataframe
        available_columns = self._hourly.columns.tolist()
        filtered_agg_methods = {col: method for col, method in agg_methods.items() 
                               if col in available_columns}
        
        # Create daily data with specific aggregation functions
        daily = self._hourly.resample('D').agg(filtered_agg_methods)
        
        # Calculate tmin, tmax, tavg from hourly temperature if temperature data exists
        if 'temp' in available_columns:
            temp_stats = self._hourly.resample('D').agg({
                'temp': ['min', 'max', 'mean']
            })
            temp_stats.columns = ['tmin', 'tmax', 'tavg']
            
            # Combine all daily data
            result = pd.concat([temp_stats, daily.drop('temp', axis=1, errors='ignore')], axis=1)
            return result
        
        return daily
    
    def to_dict(self):
        return {
            'region_name': self._region_name,
            'lat': self._lat,
            'lon': self._lon,
            'alt': self._alt,
            'start_date': self._start_date,
            'end_date': self._end_date,
            'hourly': self._hourly,
            'daily': self._daily
        }