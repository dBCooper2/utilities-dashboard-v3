# Testing and Comparing Forecasting Models

This guide will help you test and compare the performance of the three forecasting models: AutoARIMA, Prophet, and XGBoost.

## Setup

First, ensure you have all required packages installed:

```python
pip install pandas numpy matplotlib seaborn statsforecast prophet xgboost scikit-learn
```

## Data Preparation

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create sample data (replace with your actual data)
dates = pd.date_range(start='2023-01-01', end='2024-03-28', freq='15min')
np.random.seed(42)
data = pd.DataFrame({
    'value': np.random.normal(20, 5, len(dates)) + 
            10 * np.sin(2 * np.pi * np.arange(len(dates)) / (24 * 4)) +  # Daily seasonality
            5 * np.sin(2 * np.pi * np.arange(len(dates)) / (24 * 4 * 7))  # Weekly seasonality
}, index=dates)
```

## Testing AutoARIMA

```python
from backend.autoarima import AutoARIMA

# Initialize and fit model
arima_model = AutoARIMA(region_name='TestRegion', data=data['value'])

# Get forecasts
forecast_15min = arima_model.get_forecast_15min()
forecast_hourly = arima_model.get_forecast_hourly()
forecast_daily = arima_model.get_forecast_daily()

# Visualize results
arima_model.plot_forecast(horizon=96)
arima_model.plot_evaluation(test_size=0.2)

# Get evaluation metrics
arima_results = arima_model.evaluate_model(test_size=0.2)
print("\nAutoARIMA Metrics:")
for metric, value in arima_results['metrics'].items():
    print(f"{metric.upper()}: {value:.4f}")
```

## Testing Prophet

```python
from backend.prophet import ProphetForecast

# Initialize and fit model
prophet_model = ProphetForecast(region_name='TestRegion', data=data['value'])

# Get forecasts
forecast_15min = prophet_model.get_forecast_15min()
forecast_hourly = prophet_model.get_forecast_hourly()
forecast_daily = prophet_model.get_forecast_daily()

# Visualize results
prophet_model.plot_forecast(horizon=96)
prophet_model.plot_components()
prophet_model.plot_evaluation(test_size=0.2)

# Get evaluation metrics
prophet_results = prophet_model.evaluate_model(test_size=0.2)
print("\nProphet Metrics:")
for metric, value in prophet_results['metrics'].items():
    print(f"{metric.upper()}: {value:.4f}")
```

## Testing XGBoost

```python
from backend.xgboost import XGBoostForecast

# Initialize and fit model
xgboost_model = XGBoostForecast(region_name='TestRegion', data=data['value'])

# Get forecasts
forecast_15min = xgboost_model.get_forecast_15min()
forecast_hourly = xgboost_model.get_forecast_hourly()
forecast_daily = xgboost_model.get_forecast_daily()

# Visualize results
xgboost_model.plot_forecast(horizon=96)
xgboost_model.plot_feature_importance()
xgboost_model.plot_evaluation(test_size=0.2)

# Get evaluation metrics
xgboost_results = xgboost_model.evaluate_model(test_size=0.2)
print("\nXGBoost Metrics:")
for metric, value in xgboost_results['metrics'].items():
    print(f"{metric.upper()}: {value:.4f}")
```

## Comparing Models

```python
import matplotlib.pyplot as plt

# Create comparison plot
plt.figure(figsize=(15, 8))
plt.plot(arima_results['predictions'].index, arima_results['predictions']['actual'], 
         label='Actual', alpha=0.7)
plt.plot(arima_results['predictions'].index, arima_results['predictions']['predicted'], 
         label='AutoARIMA', alpha=0.7)
plt.plot(prophet_results['predictions'].index, prophet_results['predictions']['predicted'], 
         label='Prophet', alpha=0.7)
plt.plot(xgboost_results['predictions'].index, xgboost_results['predictions']['predicted'], 
         label='XGBoost', alpha=0.7)
plt.title('Model Comparison')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# Compare metrics
metrics_df = pd.DataFrame({
    'AutoARIMA': arima_results['metrics'],
    'Prophet': prophet_results['metrics'],
    'XGBoost': xgboost_results['metrics']
})
print("\nModel Comparison Metrics:")
print(metrics_df)
```

## Notes

1. **Data Requirements**:
   - All models require data with a datetime index
   - Data should be clean and free of missing values
   - Consider the frequency of your data (15min, hourly, daily)

2. **Model Selection**:
   - AutoARIMA: Good for linear patterns and seasonality
   - Prophet: Excellent for multiple seasonality patterns and trend changes
   - XGBoost: Powerful for non-linear patterns and complex relationships

3. **Performance Considerations**:
   - AutoARIMA: Fastest training, good for simple patterns
   - Prophet: Slower training, handles missing data well
   - XGBoost: Fast training, requires feature engineering

4. **Hyperparameter Tuning**:
   - Each model has parameters that can be tuned for better performance
   - Consider using cross-validation for parameter optimization
   - Monitor for overfitting when tuning parameters

5. **Evaluation Metrics**:
   - MAE: Mean Absolute Error
   - RMSE: Root Mean Square Error
   - MAPE: Mean Absolute Percentage Error
   - R²: R-squared score

Remember to adjust the test size and other parameters based on your specific needs and data characteristics. 