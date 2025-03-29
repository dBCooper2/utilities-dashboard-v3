'''
This module contains code for forecasting data using the AutoARIMA model.

This module forecasts day-ahead data for a specified column in a specific region.

Data must be passed in as a pandas DataFrame with a datetime index.
'''

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from typing import Dict, Union, Tuple

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose 

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsforecast.arima import arima_string

from backend.regionweather import RegionWeather

class AutoARIMA:
    def __init__(self, region_name:str, data:pd.DataFrame):
        """
        Initialize the AutoARIMA model.
        
        Args:
            region_name (str): Name of the region being forecasted
            data (pd.DataFrame): DataFrame with datetime index and values to forecast
        """
        self.region_name = region_name
        self.data = data
        self.model = None
        self.fitted = False
        
        # Determine the frequency of the data
        if isinstance(data.index, pd.DatetimeIndex):
            self.freq = pd.infer_freq(data.index)
            if self.freq is None:
                # If frequency cannot be inferred, try to determine from index
                time_diff = data.index[1] - data.index[0]
                if time_diff.total_seconds() <= 900:  # 15 minutes
                    self.freq = '15min'
                elif time_diff.total_seconds() <= 3600:  # 1 hour
                    self.freq = 'H'
                else:
                    self.freq = 'D'
        else:
            raise ValueError("Data index must be datetime")
            
        # Set seasonality based on frequency
        if self.freq == '15min':
            self.season_length = 96  # 24 hours * 4 (15-min intervals)
        elif self.freq == 'H':
            self.season_length = 24  # 24 hours
        else:
            self.season_length = 7  # 7 days
            
    def fit(self):
        """
        Fit the AutoARIMA model to the data.
        """
        # Prepare data for StatsForecast
        df = pd.DataFrame({
            'ds': self.data.index,
            'y': self.data.values,
            'unique_id': self.region_name
        })
        
        # Initialize and fit the model
        models = [AutoARIMA(season_length=self.season_length)]
        self.sf = StatsForecast(models=models, freq=self.freq)
        self.sf.fit(df=df)
        self.fitted = True
        self.model = self.sf.fitted_[0,0].model_
        
    def forecast(self, horizon:int=24) -> pd.DataFrame:
        """
        Generate forecasts for the specified horizon.
        
        Args:
            horizon (int): Number of periods to forecast
            
        Returns:
            pd.DataFrame: DataFrame containing the forecasts
        """
        if not self.fitted:
            self.fit()
            
        # Generate forecasts
        forecast_df = self.sf.forecast(horizon=horizon)
        
        # Convert to DataFrame with datetime index
        forecast_df = forecast_df.set_index('ds')
        forecast_df = forecast_df.rename(columns={'AutoARIMA': 'forecast'})
        
        return forecast_df
    
    def get_forecast_15min(self) -> pd.DataFrame:
        """
        Get 15-minute resolution forecasts for the next day.
        
        Returns:
            pd.DataFrame: DataFrame with 15-minute forecasts
        """
        if self.freq != '15min':
            raise ValueError("Data must be in 15-minute resolution")
        return self.forecast(horizon=96)  # 24 hours * 4 (15-min intervals)
    
    def get_forecast_hourly(self) -> pd.DataFrame:
        """
        Get hourly resolution forecasts for the next day.
        
        Returns:
            pd.DataFrame: DataFrame with hourly forecasts
        """
        if self.freq not in ['15min', 'H']:
            raise ValueError("Data must be in 15-minute or hourly resolution")
        return self.forecast(horizon=24)  # 24 hours
    
    def get_forecast_daily(self) -> pd.DataFrame:
        """
        Get daily resolution forecasts for the next week.
        
        Returns:
            pd.DataFrame: DataFrame with daily forecasts
        """
        return self.forecast(horizon=7)  # 7 days
    
    def plot_forecast(self, horizon:int=24, include_history:bool=True):
        """
        Plot the forecasts along with historical data.
        
        Args:
            horizon (int): Number of periods to forecast
            include_history (bool): Whether to include historical data in the plot
        """
        forecast_df = self.forecast(horizon=horizon)
        
        plt.figure(figsize=(12, 6))
        if include_history:
            plt.plot(self.data.index, self.data.values, label='Historical')
        plt.plot(forecast_df.index, forecast_df['forecast'], label='Forecast')
        plt.title(f'Forecast for {self.region_name}')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def get_model_summary(self) -> str:
        """
        Get a summary of the fitted ARIMA model.
        
        Returns:
            str: String containing the model summary
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        return arima_string(self.model)
        
    def evaluate_model(self, test_size:float=0.2) -> Dict[str, Union[float, pd.DataFrame]]:
        """
        Evaluate the model's performance using common time series metrics.
        
        Args:
            test_size (float): Proportion of data to use for testing (default: 0.2)
            
        Returns:
            Dict containing:
                - metrics: Dictionary of evaluation metrics
                - predictions: DataFrame with actual and predicted values
                - residuals: DataFrame with residuals and their statistics
        """
        if not self.fitted:
            self.fit()
            
        # Split data into train and test sets
        split_idx = int(len(self.data) * (1 - test_size))
        train_data = self.data[:split_idx]
        test_data = self.data[split_idx:]
        
        # Fit model on training data
        train_df = pd.DataFrame({
            'ds': train_data.index,
            'y': train_data.values,
            'unique_id': self.region_name
        })
        
        models = [AutoARIMA(season_length=self.season_length)]
        sf = StatsForecast(models=models, freq=self.freq)
        sf.fit(df=train_df)
        
        # Generate predictions for test period
        horizon = len(test_data)
        forecast_df = sf.forecast(horizon=horizon)
        forecast_df = forecast_df.set_index('ds')
        forecast_df = forecast_df.rename(columns={'AutoARIMA': 'forecast'})
        
        # Calculate residuals
        residuals = test_data - forecast_df['forecast']
        
        # Calculate metrics
        metrics = {
            'mae': np.mean(np.abs(residuals)),  # Mean Absolute Error
            'rmse': np.sqrt(np.mean(residuals**2)),  # Root Mean Square Error
            'mape': np.mean(np.abs(residuals / test_data)) * 100,  # Mean Absolute Percentage Error
            'r2': 1 - np.sum(residuals**2) / np.sum((test_data - np.mean(test_data))**2),  # R-squared
            'aic': self.model.aic,  # Akaike Information Criterion
            'bic': self.model.bic,  # Bayesian Information Criterion
        }
        
        # Create DataFrames for results
        predictions_df = pd.DataFrame({
            'actual': test_data,
            'predicted': forecast_df['forecast'],
            'residual': residuals
        })
        
        residuals_df = pd.DataFrame({
            'residual': residuals,
            'residual_abs': np.abs(residuals),
            'residual_pct': residuals / test_data * 100
        })
        
        return {
            'metrics': metrics,
            'predictions': predictions_df,
            'residuals': residuals_df
        }
        
    def plot_evaluation(self, test_size:float=0.2):
        """
        Plot evaluation results including actual vs predicted values and residuals.
        
        Args:
            test_size (float): Proportion of data to use for testing (default: 0.2)
        """
        results = self.evaluate_model(test_size)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot actual vs predicted
        ax1.plot(results['predictions'].index, results['predictions']['actual'], 
                label='Actual', alpha=0.7)
        ax1.plot(results['predictions'].index, results['predictions']['predicted'], 
                label='Predicted', alpha=0.7)
        ax1.set_title(f'Actual vs Predicted Values for {self.region_name}')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True)
        
        # Plot residuals
        ax2.plot(results['predictions'].index, results['predictions']['residual'], 
                label='Residuals', color='red', alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax2.set_title('Residuals Over Time')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Residual')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Print metrics
        print("\nModel Evaluation Metrics:")
        for metric, value in results['metrics'].items():
            print(f"{metric.upper()}: {value:.4f}")