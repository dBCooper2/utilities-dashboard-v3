Now convert the backend folders into a production-ready docker container to be deployed on railway. 

Context:
- This backend should be stored in a folder named backend-prod
- The pipeline should run 1x a day at 4am, collecting hourly weather data for each region
- The pipeline should also have callable functions to perform the first fill
- Collected weather data should be added to a Postgres instance with TimescaleDB
- Collected weather data should be forecasted with AutoARIMA, Prophet, and XGBoost for the 24 hours ahead, and LSTM should be used to forecast the week ahead.
- The forecasts should be for general weather conditions to create a weather-app like dashboard where users can see the day and week-ahead forecast, along with the following dashboard elements:

Temperature & Load Correlation:
Current Temps on Map: Overlay the current temperature (temp) for each defined zone (US-FLA-FPL, US-SE-SOCO, etc.) directly onto your dashboard map. This visually explains why load might be high (hot temps in FL/GA driving AC use) or low in certain areas.
Heat Index / Wind Chill: Calculate and display "feels like" temperatures. Heat Index (using temp and rhum) is crucial for summer load in the Southeast, while Wind Chill (using temp and wspd) matters less frequently but can impact winter heating demand.
Temperature vs. Load Chart: Display a chart showing the recent trend (e.g., last 24 hours) of average temperature across the region (or selected zones) plotted against the actual or forecasted system load. This highlights the strong dependency.
Renewable Generation Context:
Wind Speed for Wind Power: Display current wind speed (wspd) and potentially wind gust (wpgt) on the map, especially near known wind generation areas (though less prevalent in the SE compared to other regions). This provides context for variable wind energy output.
Solar Potential (Proxy): Use tsun (sunshine duration) and/or coco (weather condition code indicating cloudiness) as proxies for solar irradiance. Overlaying an indicator of "sunny," "partly cloudy," or "cloudy" on the map gives context to solar generation levels during daylight hours.
Renewable Output vs. Weather Chart: Similar to the temp/load chart, plot recent wind speed trends against actual/estimated wind generation, and solar potential (proxy) against actual/estimated solar generation.
Forecast Integration:
Day-Ahead Weather Forecast: Since you're working on forecasting (autoarima.py, xgb.py), display the forecasted temperature (tavg, tmin, tmax), wind speed, and solar potential proxy for the next 24-48 hours alongside the energy forecasts (load/price). This helps users understand anticipated grid conditions.
Forecast vs. Actual Weather: Show a comparison of your recent weather forecast predictions against the actual observed weather data.
Precipitation & Hydro:
Current/Recent Precipitation: Display areas with significant recent precipitation (prcp). While less immediately impactful than temperature on an hourly basis, sustained rainfall affects hydro reservoir levels (relevant for TVA, parts of Carolinas/Georgia) over time. Extreme rainfall can also impact demand and operations.

- All of these data structures should be stored in the PostGres/Timescale instance in the docker container for outputting to the frontend

Lastly, the backend should have a FastAPI connection for the frontend to get all of this data from the backend. The frontend will be a dashboard with a dropdown to select a region, and get all of the relevant weather forecasts for that region as dashboard elements.

Build and test the backend, and let me know when everything is set up.

For any fine-tuning of the timescaledb setup or the models, write a markdown file in the backend-prod folder that explains to me how I can test the models.

Make sure robust logging exists for all code files.