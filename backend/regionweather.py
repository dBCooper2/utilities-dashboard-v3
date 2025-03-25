# This script gets Weather Data from Meteostat or the NOAA API as a fallback
import numpy as np
import meteostat as ms
import requests as rq
import datetime as dt

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
        self._daily = self.__fetch_daily_data(self._pt)
        print('Daily Data Fetched!')
        # Add Data Cleaning for Dailies
        self._weekly = self.__fetch_weekly_data(self._pt)
        print('Weekly Data Aggregated and Fetched!')
        # Add Data Cleaning for Weeklies
        self._monthly = self.__fetch_monthly_data(self._pt)
        print('Monthly Data Fetched!')
        # Add Data Cleaning for Monthlies
        self._fifteen_m = self.__interpolate_to_15m(self._hourly)
        print('15 Minute Data Interpolated!')

    # Fetches the Hourly Object from meteostat
    def __get_hourly_obj(self):
        print('fetching Hourly Object...')
        return ms.Hourly(self._pt, self._start, self._end)

    # Fetches Hourly Data as a Time Series from meteostat
    def __get_hourly_data(self, hourly:ms.Hourly):
        print('Fetching Hourly Data from Object...')
        data = hourly.fetch()
        print(f'Data:\n{data.head()}')
        print(f'Columns: {data.columns.tolist()}')
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


    # Function to aggregate to daily data
    def __fetch_daily_data(self, pt:ms.Point):
        daily_df = ms.Daily(pt)
        return daily_df.fetch()

    # Function to aggregate to weekly data
    def __fetch_weekly_data(self, pt:ms.Point):
        weekly_df = ms.Hourly(pt)
        data = weekly_df.aggregate('1W')
        return data.fetch()

    # Function to aggregate to monthly data
    def __fetch_monthly_data(self, pt:ms.Point):
        monthly_df = ms.Monthly(pt)
        return monthly_df.fetch()
    
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