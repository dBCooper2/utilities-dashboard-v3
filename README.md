# Southeastern Utilities ETL Pipeline and Dashboard

A dashboard of the day-ahead market for the Southeastern US, providing energy and weather forecasts. Built in Python, Postgres, and NextJS with Tailwind, and deployed with Vercel (and probably railway for the backend). Additionally, the repository contains notebooks exploring Time Series Analysis and testing various project functionalities.

## Backend Outline

The backend contains python classes to collect weather and energy data, forecasting methods for the extracted data, and database management for loading transformed data into the database for the dashboard.

### Current Implementation

- `setup.py`: installs backend classes as packages for notebooks, can be run using `pip install -e .`.
- `requirements.txt`: Installs packages, can be run with `pip install -r requirements.txt`.
- backend/
  - `regionweather.py`: creates an object to collect weather data from meteostat.
  - '`autoarima.py`: Uses statsforecast's AutoARIMA model to forecast day-ahead data for weather and energy data.