'''
This module contains code for forecasting data using the AutoARIMA model.

This module forecasts day-ahead data for a specified column in a specific region.

Data must be passed in as a pandas DataFrame with columns:
- ds: datetime column
- y: observations column
- unique_id: identifier column
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

class AutoARIMAForecast:
    def __init__(self, region_name:str, data: pd.DataFrame):
        """
        Initialize the AutoARIMA model.
        
        Args:
            region_name (str): Name of the region
            data (pd.DataFrame): DataFrame with columns:
                - ds: datetime column
                - y: observations column
                - unique_id: identifier column
        """
        self.region_name = region_name
        # Validate input data format
        required_columns = ['ds', 'y', 'unique_id']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")
            
        self.data = data
        self.model = None
        self.fitted = False
        
        # Determine the frequency of the data
        if isinstance(data['ds'], pd.Series):
            self.freq = pd.infer_freq(data['ds'])
            if self.freq is None:
                # If frequency cannot be inferred, try to determine from index
                time_diff = data['ds'].iloc[1] - data['ds'].iloc[0]
                if time_diff.total_seconds() <= 900:  # 15 minutes
                    self.freq = '15min'
                elif time_diff.total_seconds() <= 3600:  # 1 hour
                    self.freq = 'H'
                else:
                    self.freq = 'D'
        else:
            raise ValueError("ds column must contain datetime values")
            
        # Set seasonality based on frequency
        if self.freq == '15min':
            self.season_length = 96  # 24 hours * 4 (15-min intervals)
        elif self.freq == 'H':
            self.season_length = 24  # 24 hours
        else:
            self.season_length = 7  # 7 days
            
    def _fit(self):
        """
        Fit the AutoARIMA model to the data.
        """
        print("Validating Data")
        # Validate data
        if self.data['y'].isnull().any():
            raise ValueError("Data contains NaN values")
        if not np.isfinite(self.data['y']).all():
            raise ValueError("Data contains infinite values")
        if not self.data['ds'].is_monotonic_increasing:
            self.data = self.data.sort_values('ds')

        print("Initializing Model")
        # Initialize and fit the model with controlled parameters
        models = [AutoARIMA(
            season_length=self.season_length,
            max_p=2,           # Limit maximum AR order
            max_q=2,           # Limit maximum MA order
            max_P=1,           # Limit maximum seasonal AR order
            max_Q=1,           # Limit maximum seasonal MA order
            max_order=4,       # Limit total order
            stepwise=True,     # Use stepwise search instead of exhaustive
            approximation=True # Use approximation for faster fitting
        )]
        print("Initializing StatsForecast")
        self.sf = StatsForecast(models=models, freq=self.freq)
        print("Fitting Model")
        self.sf.fit(df=self.data)
        print("Model Fitted")
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
            self._fit()
            
        # Generate forecasts
        forecast_df = self.sf.forecast(df=self.data, h=horizon)
        
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
            plt.plot(self.data['ds'], self.data['y'], label='Historical')
        plt.plot(forecast_df['ds'], forecast_df['AutoARIMA'], label='Forecast')
        plt.title(f'Forecast for {self.data["unique_id"].iloc[0]}')
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
        train_data = self.data.iloc[:split_idx]
        test_data = self.data.iloc[split_idx:]
        
        # Fit model on training data
        models = [AutoARIMA(season_length=self.season_length)]
        sf = StatsForecast(models=models, freq=self.freq)
        sf.fit(df=train_data)
        
        # Generate predictions for test period
        horizon = len(test_data)
        forecast_df = sf.forecast(horizon=horizon)
        
        # Calculate residuals
        residuals = test_data['y'].values - forecast_df['AutoARIMA'].values
        
        # Calculate metrics
        metrics = {
            'mae': np.mean(np.abs(residuals)),  # Mean Absolute Error
            'rmse': np.sqrt(np.mean(residuals**2)),  # Root Mean Square Error
            'mape': np.mean(np.abs(residuals / test_data['y'].values)) * 100,  # Mean Absolute Percentage Error
            'r2': 1 - np.sum(residuals**2) / np.sum((test_data['y'].values - np.mean(test_data['y'].values))**2),  # R-squared
            'aic': self.model.aic,  # Akaike Information Criterion
            'bic': self.model.bic,  # Bayesian Information Criterion
        }
        
        # Create DataFrames for results
        predictions_df = pd.DataFrame({
            'ds': test_data['ds'],
            'actual': test_data['y'],
            'predicted': forecast_df['AutoARIMA'],
            'residual': residuals
        })
        
        residuals_df = pd.DataFrame({
            'ds': test_data['ds'],
            'residual': residuals,
            'residual_abs': np.abs(residuals),
            'residual_pct': residuals / test_data['y'].values * 100
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
        ax1.plot(results['predictions']['ds'], results['predictions']['actual'], 
                label='Actual', alpha=0.7)
        ax1.plot(results['predictions']['ds'], results['predictions']['predicted'], 
                label='Predicted', alpha=0.7)
        ax1.set_title(f'Actual vs Predicted Values for {self.data["unique_id"].iloc[0]}')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True)
        
        # Plot residuals
        ax2.plot(results['predictions']['ds'], results['predictions']['residual'], 
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