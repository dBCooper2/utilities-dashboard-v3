# Backend Structure

## Data Fetching

Data will be fetched from meteostat, NREL, OpenEI and EIA APIs. The following classes will fetch weather and energy data from these APIs.

- `regionweather.py`: Contains the RegionWeather class, which collects hourly and daily data for each region's weather.
- `energydata.py`: Contains the EnergyData class, which collects hourly data for each region's energy usage. This includes Zonal Price Data, Load and Demand Data, Fuel Generation Mix, Transmission Data, and Reserve Data. A more detailed outline of the data requirements can be found in the `docs/proj-docs/1 - Data Dictionary.md` file.

## Models

The following models forecast 24 hours ahead for weather data and energy usage data

- `autoarima.py`: A class containing an AutoARIMA model
- `prophet.py`: A class containing a Prophet model

The following creates a weekly weather forecast from RegionWeather data

- `lstm.py`: A class containing a Long Short-Term Memory model to forecast weekly weather data

## Database Manager

- `db.py`: A class that creates, connects to, and handles any database functions. This class has functions to take in dataframes of new weather and energy and their forecasts, and then saves it to a Postgres instance.

## Scheduler

- `scheduler.py`: A class to handle the automation of the ETL Pipeline. This class will spin up daily to get the latest weather and energy data , create forecasts for the upcoming day and week, and then add these new data for each region to the database.

## API Endpoints

- `api.py`: This class has endpoints that talk to the Postgres Database. This will set up endpoints that a frontend (that I will add at a later date) will access to get the weather and energy data and forecasts from the database.

## Docker Setup

This backend will be containerized in a Docker container and deployed to Railway. The deployed container should run the scheduled ETL pipeline once a day at 3AM, and handle serving database content to the frontend.
