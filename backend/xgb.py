'''
This class creates an XGBoost model for time series forecasting.

This class is capable of creating a forecast for data at 15 minute, hourly, and daily resolutions.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Union, Tuple
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb

class XGBoostForecast:
    def __init__(self, region_name: str, data: pd.DataFrame):
        """
        Initialize the XGBoost model.
        
        Args:
            region_name (str): Name of the region being forecasted
            data (pd.DataFrame): DataFrame with datetime index and values to forecast
        """
        self.region_name = region_name
        self.data = data
        self.model = None
        self.fitted = False
        self.scaler = StandardScaler()
        
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
            
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features for the model.
        
        Args:
            df (pd.DataFrame): DataFrame with datetime index
            
        Returns:
            pd.DataFrame: DataFrame with additional time-based features
        """
        df = df.copy()
        
        # Extract time-based features
        df['hour'] = df.index.hour
        df['day'] = df.index.day
        df['month'] = df.index.month
        df['day_of_week'] = df.index.dayofweek
        df['day_of_year'] = df.index.dayofyear
        df['quarter'] = df.index.quarter
        
        # Create cyclical features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Add lagged features
        for lag in [1, 2, 3, 4, 24, 96]:  # 15min, 1h, 4h, 1d, 4d
            if lag <= len(df):
                df[f'lag_{lag}'] = df['value'].shift(lag)
                
        # Add rolling statistics
        for window in [4, 24, 96]:  # 1h, 1d, 4d
            if window <= len(df):
                df[f'rolling_mean_{window}'] = df['value'].rolling(window=window).mean()
                df[f'rolling_std_{window}'] = df['value'].rolling(window=window).std()
                
        return df
        
    def fit(self):
        """
        Fit the XGBoost model to the data.
        """
        # Prepare data
        df = pd.DataFrame({'value': self.data.values}, index=self.data.index)
        df = self._create_features(df)
        
        # Remove rows with NaN values from lagged features
        df = df.dropna()
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col != 'value']
        X = df[feature_cols]
        y = df['value']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize and fit the model
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        self.model.fit(X_scaled, y)
        self.fitted = True
        
    def forecast(self, horizon: int = 24) -> pd.DataFrame:
        """
        Generate forecasts for the specified horizon.
        
        Args:
            horizon (int): Number of periods to forecast
            
        Returns:
            pd.DataFrame: DataFrame containing the forecasts
        """
        if not self.fitted:
            self.fit()
            
        # Create future dates
        last_date = self.data.index[-1]
        if self.freq == '15min':
            future_dates = pd.date_range(start=last_date + pd.Timedelta(minutes=15),
                                      periods=horizon,
                                      freq='15min')
        elif self.freq == 'H':
            future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1),
                                      periods=horizon,
                                      freq='H')
        else:
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                      periods=horizon,
                                      freq='D')
            
        # Create future features
        future_df = pd.DataFrame(index=future_dates)
        future_df = self._create_features(future_df)
        
        # Prepare features for prediction
        feature_cols = [col for col in future_df.columns if col != 'value']
        X_future = future_df[feature_cols]
        X_future_scaled = self.scaler.transform(X_future)
        
        # Generate predictions
        predictions = self.model.predict(X_future_scaled)
        
        # Calculate prediction intervals using standard deviation of residuals
        residuals = self.model.predict(self.scaler.transform(X)) - y
        std_residuals = np.std(residuals)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'forecast': predictions,
            'lower_bound': predictions - 1.96 * std_residuals,
            'upper_bound': predictions + 1.96 * std_residuals
        }, index=future_dates)
        
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
    
    def plot_forecast(self, horizon: int = 24, include_history: bool = True):
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
        plt.fill_between(forecast_df.index, 
                        forecast_df['lower_bound'],
                        forecast_df['upper_bound'],
                        color='red', alpha=0.1)
        plt.title(f'XGBoost Forecast for {self.region_name}')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_feature_importance(self):
        """
        Plot the importance of features used in the model.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
            
        # Get feature names
        feature_cols = [col for col in self._create_features(
            pd.DataFrame({'value': self.data.values}, index=self.data.index)
        ).columns if col != 'value']
        
        # Get feature importance
        importance = self.model.feature_importances_
        
        # Create DataFrame and sort by importance
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Plot
        plt.figure(figsize=(12, 6))
        sns.barplot(x='importance', y='feature', data=importance_df)
        plt.title(f'Feature Importance for {self.region_name}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()
        
    def evaluate_model(self, test_size: float = 0.2) -> Dict[str, Union[float, pd.DataFrame]]:
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
        
        # Prepare training data
        train_df = pd.DataFrame({'value': train_data.values}, index=train_data.index)
        train_df = self._create_features(train_df)
        train_df = train_df.dropna()
        
        # Prepare test data
        test_df = pd.DataFrame({'value': test_data.values}, index=test_data.index)
        test_df = self._create_features(test_df)
        test_df = test_df.dropna()
        
        # Prepare features and target
        feature_cols = [col for col in train_df.columns if col != 'value']
        X_train = train_df[feature_cols]
        y_train = train_df['value']
        X_test = test_df[feature_cols]
        y_test = test_df['value']
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Fit model on training data
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # Generate predictions
        predictions = model.predict(X_test_scaled)
        
        # Calculate residuals
        residuals = y_test - predictions
        
        # Calculate metrics
        metrics = {
            'mae': mean_absolute_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'mape': np.mean(np.abs(residuals / y_test)) * 100,
            'r2': r2_score(y_test, predictions)
        }
        
        # Create DataFrames for results
        predictions_df = pd.DataFrame({
            'actual': y_test,
            'predicted': predictions,
            'residual': residuals
        }, index=test_df.index)
        
        residuals_df = pd.DataFrame({
            'residual': residuals,
            'residual_abs': np.abs(residuals),
            'residual_pct': residuals / y_test * 100
        }, index=test_df.index)
        
        return {
            'metrics': metrics,
            'predictions': predictions_df,
            'residuals': residuals_df
        }
        
    def plot_evaluation(self, test_size: float = 0.2):
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