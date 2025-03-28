# This script gets Weather Data from Meteostat or the NOAA API as a fallback
import numpy as np
import meteostat as ms
import requests as rq
import datetime as dt
import pandas as pd
import math

class RegionWeather:
    def __init__(self, region_name:str, lat:np.float64, lon:np.float64, alt:np.float64, start_date:dt.datetime, end_date:dt.datetime):
        self._region_name = region_name
        self._lat = lat
        self._lon = lon
        self._alt = alt
        self._start = start_date
        self._end = end_date
        self._pt = ms.Point(self._lat, self._lon, self._alt)

        self._hourly_obj = self.__get_hourly_obj()
        print('Hourly Object Fetched!')
        self._hourly = self.__get_hourly_data(self._hourly_obj)
        print('Hourly Object Fetched!')
        # self._hourly = self.__clean_data(self.hourly) Add Data Cleaning Later...
        print('Hourly Data Cleaned!')
        self._daily = self.__fetch_daily_data()
        print('Daily Data Fetched!')
        # Add Data Cleaning for Dailies
        self._weekly = self.__fetch_weekly_data()
        print('Weekly Data Aggregated and Fetched!')
        # Add Data Cleaning for Weeklies
        self._monthly = self.__fetch_monthly_data()
        print('Monthly Data Fetched!')
        # Add Data Cleaning for Monthlies
        self._fifteen_m = self.__interpolate_to_15m(self._hourly)
        print('15 Minute Data Interpolated!')

    # Fetches the Hourly Object from meteostat
    def __get_hourly_obj(self):
        print('Fetching Hourly Object...')
        return ms.Hourly(self._pt, self._start, self._end)

    # Fetches Hourly Data as a Time Series from meteostat
    def __get_hourly_data(self, hourly:ms.Hourly):
        print('Fetching Hourly Data from Object...')
        data = hourly.fetch()
        #print(f'Data:\n{data.head()}')
        #print(f'Columns: {data.columns.tolist()}')
        return data
    
    def __clean_data(self, data):
        '''
        Meteostat Hourly Data has fields:
            - station (id of weather station)
            - time (Datetime64 of the observation)
            - temp (Air temp in C)
            - dwpt (dew point in C)
            - rhum (relative humidity in pct)
            - prcp (1 hour precipitation total)
            - snow (snow depth in mm)
            - wdir (average wind direction in deg)
            - wpsd (avg wind speed in km/hr)
            - wpgt (peak wind gust in km/hr)
            - pres (avg sea-level air presure in hPa)
            - tsun (avg sunshine total in minutes)
            - coco (weather condition code)

        Source: https://dev.meteostat.net/python/hourly.html#api

        Dealing with Missing Data: https://www.e-education.psu.edu/meteo810/content/l5_p4.html
        '''
        return data
    
    # Function to interpolate hourly data to 15-minute increments
    def __interpolate_to_15m(self, df):
        df_15min = df.resample('15T').interpolate(method='linear')
        return df_15min

    # Circular mean for wind direction (degrees)
    def __degree_mean(self, degrees):
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

    # Function to aggregate hourly data to daily data
    def __fetch_daily_data(self):
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
            'wdir': self.__degree_mean,
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

    # Function to aggregate hourly data to weekly data
    def __fetch_weekly_data(self):
        print('Fetching Weekly Data...')
        if self._hourly.empty:
            return pd.DataFrame()
        
        # Define aggregation methods for each column
        agg_methods = {
            'temp': 'mean',
            'dwpt': 'mean',
            'rhum': 'mean',
            'prcp': 'sum',
            'snow': 'max',
            'wdir': self.__degree_mean,
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
        
        # Create weekly data with specific aggregation functions
        weekly = self._hourly.resample('W-SUN').agg(filtered_agg_methods)
        
        return weekly

    # Function to aggregate hourly data to monthly data
    def __fetch_monthly_data(self):
        print('Fetching Monthly Data...')
        if self._hourly.empty:
            return pd.DataFrame()
        
        # For monthly, we'll create it from daily data to get proper tmin/tmax
        daily_data = self.__fetch_daily_data()
        if daily_data.empty:
            return pd.DataFrame()
        
        # Define aggregation methods for monthly values
        agg_methods = {
            'tavg': 'mean',
            'tmin': 'min',
            'tmax': 'max',
            'prcp': 'sum',
            'wspd': 'mean',
            'pres': 'mean',
            'tsun': 'sum'
        }
        
        # Filter for columns that actually exist in the dataframe
        available_columns = daily_data.columns.tolist()
        filtered_agg_methods = {col: method for col, method in agg_methods.items() 
                               if col in available_columns}
        
        # Calculate monthly aggregations
        monthly = daily_data.resample('MS').agg(filtered_agg_methods)
        
        return monthly
    
    def to_dict(self):
        return {
            "region": self._region_name,
            "point": self._pt,
            "start_time": self._start,
            "end_time": self._end,
            "df_15m": self._fifteen_m,
            "df_hourly": self._hourly,
            "df_daily": self._daily,
            "df_weekly": self._weekly,
            "df_monthly": self._monthly
        }