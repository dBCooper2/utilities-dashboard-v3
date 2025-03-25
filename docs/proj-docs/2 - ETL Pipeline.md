
An ETL (Extract, Transform, Load) Pipeline is a system that extracts data from one system, transforms it, and then loads it into another system.

For this project, I will extract weather data from `meteostat` and utilities data from the `EIA`, `OpenEI`, and `NREL` APIs. That data will then be transformed in a python backend into optimized time-series data and related explanatory data, then loaded into a Postgres database with the TimescaleDB extension which a NextJS dashboard web application will display.

The pipeline will be dockerized and deployed to Railway for hosting.

## Table of Contents

- [[#Main Application]]
- [[#Data Extraction]]
- [[#Data Transformation]]
- [[#Data Loading]]

## Main Application

The main application will be a python script within the docker container that a cronjob from Railway will start once a day at 6AM (CHANGE THIS DEPENDING ON WHEN EACH API UPDATES). This application will first look in the Postgres database to determine what the oldest data is in the database. Once this is determined, the [[#Data Extraction]] functions will be called to get this data for each zone from the date of the oldest data to the time the application starts. 

The application will use the [[#Data Transformation]] functions to convert these JSONs into dataframes (or similar structure more optimized for time series stuff), create 15-minute frequency time series, and use forecasting methods in [[3 - Forecasting Methods]] to create forecasts for the day-ahead market.

Once transformations are completed, the function will use the [[#Data Loading]] scripts to load the transformed data into a Postgres database with TimescaleDB features.

## Data Extraction

We will break the data extraction process into separate python scripts for each set of time series data we wish to collect. Each script should be a series of functions used purely to fetch the data that will be called from the main application given a location or zone, a start date, and an end date from the API and return the raw JSON data.

### `weather_extraction.py`
### `zonal_price_extraction.py`
### `load_demand_extraction.py`
### `generation_mix_extraction.py`
### `transmission_extraction.py`
### `reserve_margin_extraction.py`


## Data Transformation

The data transformation process will be another set of python scripts that contain functions to be used in the main application. These functions will transform the JSON data into classes of `pandas.Dataframes` and another data structures, then return those classes.

### `weather_transformer.py`
### `zonal_price_transformer.py`
### `load_demand_transformer.py`
### `generation_mix_transformer.py`
### `transmission_transformer.py`
### `reserve_margin_transformer.py`

## Data Loading

The data loading process will again, be python scripts containing functions called in the main application. These functions should take the classes for each dataset and load them into a Postgres database with TimescaleDB features.

### `weather_db_loader.py`
### `zonal_price_db_loader.py`
### `load_demand_db_loader.py`
### `generation_mix_db_loader.py`
### `transmission_db_loader.py`
### `reserve_margin_db_loader.py`