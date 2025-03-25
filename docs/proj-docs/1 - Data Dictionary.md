## Table of Contents

- [[#Region Data]]
	- [[#Florida]]
	- [[#Georgia]]
	- [[#Alabama]]
	- [[#Mississippi]]
	- [[#North Carolina]]
	- [[#South Carolina]]
	- [[#Tennessee]]
	- [[#Kentucky]]
	- [[#Virginia]]
- [[#Weather Data]]
	- [[#meteostat.Point]]
	- [[#meteostat.Hourly]]
	- [[#meteostat.Normals]]
	- [[#Dealing with Timezones]]
- [[#Utilities Data]]
	- [[#Zonal Price Data]]
	- [[#Load and Demand Data]]
	- [[#Generation Mix]]
	- [[#Transmission Data]]
	- [[#Reserve Margins]]

## Region Data

The regions chosen are zonal data from the [`world.geojson`](<https://github.com/electricitymaps/electricitymaps-contrib/blob/master/web/geo/world.geojson>) file from [Electricity Maps](<https://github.com/electricitymaps>), a project dedicated to "mapping the carbon footprint of electricity worldwide". The latitude, longitude and altitude data was calculated in the script to get zones from the world `geojson` file.

This `world.geojson` file contains zonal data by state that we will break into separate state `geojson` files.
### Florida

- US-FLA-FMPP - Lat: 28.525581, Lon: -81.536775, Alt: 0
- US-FLA-FPC - Lat: 28.996695, Lon: -82.886613, Alt: 0
- US-FLA-FPL - Lat: 27.917488, Lon: -81.450970, Alt: 0
- US-FLA-GVL - Lat: 29.619310, Lon: -82.328732, Alt: 0
- US-FLA-HST - Lat: 25.456904, Lon: -80.588092, Alt: 0
- US-FLA-JEA - Lat: 30.390902, Lon: -83.679837, Alt: 0
- US-FLA-SEC - Lat: 28.805983, Lon: -82.306291, Alt: 0
- US-FLA-TAL - Lat: 30.437174, Lon: -84.248042, Alt: 0
- US-FLA-TEC - Lat: 27.959413, Lon: -82.144821, Alt: 0

### Georgia

- US-SE-SOCO - Lat: 32.154182, Lon: -85.283967, Alt: 0 (Southern Company, also covers Alabama and Mississippi)

### Alabama

- US-SE-SOCO - Lat: 32.154182, Lon: -85.283967, Alt: 0 (Southern Company, also covers Georgia and Mississippi)

### Mississippi

- US-SE-SOCO - Lat: 32.154182, Lon: -85.283967, Alt: 0 (Southern Company, also covers Alabama and Georgia)

### North Carolina

- US-CAR-CPLE - Lat: 35.299120, Lon: -77.893222, Alt: 0 (Carolina Power & Light East)
- US-CAR-CPLW - Lat: 35.602668, Lon: -82.602688, Alt: 0 (Carolina Power & Light West)
- US-CAR-DUK - Lat: 35.222090, Lon: -81.841604, Alt: 0 (Duke Energy)
- US-CAR-SC - Lat: 34.043817, Lon: -80.365381, Alt: 0
- US-CAR-SCEG - Lat: 33.514198, Lon: -81.052635, Alt: 0 (South Carolina Electric & Gas)

### South Carolina

- US-CAR-CPLE - Lat: 35.299120, Lon: -77.893222, Alt: 0 (Carolina Power & Light East)
- US-CAR-CPLW - Lat: 35.602668, Lon: -82.602688, Alt: 0 (Carolina Power & Light West)
- US-CAR-DUK - Lat: 35.222090, Lon: -81.841604, Alt: 0 (Duke Energy)
- US-CAR-SC - Lat: 34.043817, Lon: -80.365381, Alt: 0
- US-CAR-SCEG - Lat: 33.514198, Lon: -81.052635, Alt: 0 (South Carolina Electric & Gas)

### Tennessee

- US-MIDW-AECI - Lat: 37.574031, Lon: -93.321490, Alt: 0 (Associated Electric Cooperative Inc, part of Midwestern region)
- US-MIDW-MISO - Lat: 40.211376, Lon: -90.411633, Alt: 0 (Midcontinent Independent System Operator, part of Midwestern region)
- US-TEN-TVA - Lat: 35.166141, Lon: -86.702414, Alt: 0 (Tennessee Valley Authority)

### Kentucky

- US-MIDW-AECI - Lat: 37.574031, Lon: -93.321490, Alt: 0 (Associated Electric Cooperative Inc, part of Midwestern region)
- US-MIDW-LGEE - Lat: 37.589836, Lon: -85.663064, Alt: 0 (Louisville Gas & Electric/Kentucky Utilities)
- US-MIDW-MISO - Lat: 40.211376, Lon: -90.411633, Alt: 0 (Midcontinent Independent System Operator, part of Midwestern region)

### Virginia

- US-MIDW-AECI - Lat: 37.574031, Lon: -93.321490, Alt: 0 (Associated Electric Cooperative Inc, part of Midwestern region)
- US-MIDW-MISO - Lat: 40.211376, Lon: -90.411633, Alt: 0 (Midcontinent Independent System Operator, part of Midwestern region)

## Weather Data

All weather data is collected from the `meteostat` Python package. Creating the Weather Data will use the following Objects.

### meteostat.Point

The `Point` object is used to get weather data for any geographic location. It has parameters `lat`, `lon`, `alt`; for latitude, longitude, and altitude, respectively.

This will be passed into the `Hourly` and `Normals` constructors to get weather information.

Points can be constructed with the following command:

```python
from meteostat import Point
pt = Point(latitude, longitude, altitude)
```

[Documentation Link](<https://dev.meteostat.net/python/api/point/>)

### meteostat.Hourly

The `Hourly` object creates hourly weather data for a location over a specified period of time. It takes in a `Point` object, a `start` and `end` `datetime` object. 

Hourly objects can be constructed with the following command:

```python
import datetime as dt
from meteostat import Point, Hourly
pt = Point(lat, lon, alt)

start_dt = dt.datetime(YYYY, M, D, H, M, S)
end_dt = dt.datetime.now()

hourly = Hourly(pt, start_dt, end_dt)
```

The result of creating the `Hourly` object is a data source that you can use the `meteostat.TimeSeries` interfaces with. This is how you get the dataframe of time series data.

#### Example Script

 This example script gets data from the Tampa area in Florida, where the Florida Municipal Power Pool operates. It contains implementations of all of the `TimeSeries` interfaces. The data returned by the `fetch()` method returns the `pandas.Dataframe` object that we will store in the Database.

```python
from meteostat import Hourly
from datetime import datetime, timedelta
import pandas as pd

# Define location and time range
latitude, longitude, altitude = 27.9506, -82.4572, 6
start = datetime(2024, 3, 1)
end = datetime(2024, 3, 2)  # One day of data

# Fetch hourly data
data = Hourly(lat=latitude, lon=longitude, start=start, end=end, alt=altitude)

data = data.fetch()

# Normalize data
normalized_data = data.copy()
normalized_data = normalized_data.apply(lambda x: (x - x.mean()) / x.std() if x.dtype != 'O' else x)

# Aggregate (Example: Compute daily means from hourly data)
aggregated_data = data.resample('D').mean()

# Interpolate missing values
interpolated_data = data.interpolate()

# Convert temperature from Celsius to Fahrenheit
converted_data = data.copy()
if 'temp' in converted_data.columns:
    converted_data['temp'] = converted_data['temp'] * 9 / 5 + 32

# Check coverage (percentage of available data)
coverage = data.count() / len(data)

# Count non-null values
count_data = data.count()

# Find stations near the location
stations = Hourly.stations(lat=latitude, lon=longitude, limit=5)

# Clear cache (if needed)
Hourly.clear_cache()
```

One important thing to know is that this script normalizes the data. For the dashboard, we should display the raw data. For forecasting or other analysis techniques, normalized data should be used. For this reason, the database will store raw data and only normalize data for calculations. (Maybe store both?)

[Documentation Link](<https://dev.meteostat.net/python/api/hourly/#parameters>)

### meteostat.Normals

The `Normals` object queries for the normal climate of a region. This can be used as an aggregate of the weather in the region to be compared against the hourly weather data.

#### Example Script

This example uses the Tampa area again to get normal climate data. The end result, `data`, is a `pandas.Dataframe` object of the climate normals.

```python
from meteostat import Point, Normals
import pandas as pd

# Define location
pt = Point(27.9506, -82.4572, 6)

# Fetch climate normals
data = Normals(pt, start_date, end_date)
data = data.fetch()
```

[Documentation Link](<https://dev.meteostat.net/python/api/normals/>)

### Dealing with Timezones

`meteostat` uses Coordinated Universal Time (UTC) as its timezone. Because of this, data collection from each timezone must handle converting the timestamps generated in the time series data into the correct timezone for the region. 

## Utilities Data

The utilities data we want to fetch is based on the data presented in the [NYISO Realtime Dashboard](<https://www.nyiso.com/real-time-dashboard>). We will need time-series data of each category below for each region within each state.

The utilities data will be fetched from the [OpenEI API](<https://apps.openei.org/services/doc/rest/util_rates/?version=3>) and using the [NREL API](<https://developer.nrel.gov/docs/electricity/utility-rates-v3/>) as a fallback for pricing data. The rest of the data will be fetched from the [EIA API](<https://www.eia.gov/opendata/>)

### Zonal Price Data

- Locational Marginal Prices (LMPs) for each zone
- Day-ahead prices for comparison
- Historical price trends

### Load and Demand Data

- Current system load for each zone
- Forecasted load (day-ahead and week-ahead)
- Historical load for comparison
- Peak load records

### Generation Mix

- Current generation by fuel type for each zone
- Renewable vs conventional generation percentages
- Generation capacities

### Transmission Data

- Interface flows between zones
- Transmission constraints/congestion
- Available transmission capacity

### Reserve Margins

- Operating reserves (spinning and non-spinning)
- Available capacity by zone

