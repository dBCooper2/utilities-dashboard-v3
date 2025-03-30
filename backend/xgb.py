class XGBoostForecast:
    def __init__(self, region_name=None, initial_df=None, season_length=None, log_level='INFO'):
        """
        Initialize the XGBoost forecasting model.
        
        Parameters:
        -----------
        region_name : str, optional
            The name of the region being forecasted.
        initial_df : pandas.DataFrame, optional
            Initial dataframe to process during initialization.
        season_length : int, optional
            The seasonal period of the time series. If None, it will be inferred from the data.
        log_level : str, optional
            Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        """
        import logging
        import sys
        
        # Setup logging
        self.logger = logging.getLogger(f"XGBoostForecast_{region_name if region_name else 'default'}")
        self.logger.setLevel(getattr(logging, log_level))
        
        # Create handler if not already present
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.model = None
        self.season_length = season_length
        self.data = None
        self.freq = None
        self.forecast_df = None
        self.fitted = False
        self.region_name = region_name
        self.feature_names = None
        self.last_date = None
        self.target_column = 'y'
        
        print(f"Initializing XGBoost model for region: {region_name if region_name else 'default'}")
        self.logger.info(f"Initializing XGBoost model for region: {region_name if region_name else 'default'}")
        
        # Process initial dataframe if provided
        if initial_df is not None:
            print(f"Processing initial dataframe with {len(initial_df)} rows")
            self.logger.info(f"Processing initial dataframe with {len(initial_df)} rows")
            self.fit(initial_df)
    
    def _detect_frequency(self, df):
        """
        Detect the frequency of the time series data.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing the time series data with a 'ds' column.
            
        Returns:
        --------
        freq : str
            The detected frequency: '15min', 'H', or 'D'.
        """
        import pandas as pd
        
        print("Detecting time series frequency...")
        self.logger.info("Detecting time series frequency")
        
        # Convert 'ds' to datetime if it's not already
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Get the time differences
        diff = df['ds'].diff().dropna()
        
        # Calculate the most common time difference
        most_common_diff = diff.mode()[0]
        
        # Determine the frequency based on the most common difference
        if most_common_diff <= pd.Timedelta(minutes=15):
            freq = '15min'
        elif most_common_diff <= pd.Timedelta(hours=1):
            freq = 'H'
        else:
            freq = 'D'
            
        print(f"Detected frequency: {freq}")
        self.logger.info(f"Detected frequency: {freq}")
        return freq
    
    def _detect_season_length(self, freq):
        """
        Determine the appropriate season length based on frequency.
        
        Parameters:
        -----------
        freq : str
            The frequency of the time series.
            
        Returns:
        --------
        season_length : int
            The seasonal period for the given frequency.
        """
        print("Determining seasonal period...")
        self.logger.info("Determining seasonal period")
        
        if freq == '15min':
            season_length = 96  # 24 hours * 4 (15-min intervals)
        elif freq == 'H':
            season_length = 24  # 24 hours
        else:  # 'D'
            season_length = 7   # 7 days
            
        print(f"Determined season length: {season_length}")
        self.logger.info(f"Determined season length: {season_length}")
        return season_length
    
    def _create_features(self, df):
        """
        Create time series features for XGBoost.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing the time series data.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with additional features for XGBoost.
        """
        import pandas as pd
        import numpy as np
        
        print("Creating features for XGBoost...")
        self.logger.info("Creating features for XGBoost")
        
        df_features = df.copy()
        
        # Ensure data is sorted by date
        df_features = df_features.sort_values('ds')
        
        # Extract date features
        df_features['hour'] = df_features['ds'].dt.hour
        df_features['dayofweek'] = df_features['ds'].dt.dayofweek
        df_features['quarter'] = df_features['ds'].dt.quarter
        df_features['month'] = df_features['ds'].dt.month
        df_features['year'] = df_features['ds'].dt.year
        df_features['dayofyear'] = df_features['ds'].dt.dayofyear
        df_features['dayofmonth'] = df_features['ds'].dt.day
        df_features['weekofyear'] = df_features['ds'].dt.isocalendar().week
        
        # Add lag features based on the frequency
        lags = []
        if self.freq == '15min':
            lags = [1, 2, 3, 4, 96, 96*2, 96*7]  # 15min, 30min, 45min, 1h, 1d, 2d, 1w
        elif self.freq == 'H':
            lags = [1, 2, 3, 6, 12, 24, 24*7]    # 1h, 2h, 3h, 6h, 12h, 1d, 1w
        else:  # 'D'
            lags = [1, 2, 3, 7, 14, 30]          # 1d, 2d, 3d, 1w, 2w, 1m
        
        for lag in lags:
            df_features[f'lag_{lag}'] = df_features['y'].shift(lag)
        
        # Add rolling window statistics
        windows = []
        if self.freq == '15min':
            windows = [4, 24, 96]  # 1h, 6h, 1d
        elif self.freq == 'H':
            windows = [6, 12, 24]  # 6h, 12h, 1d
        else:  # 'D'
            windows = [7, 14, 30]  # 1w, 2w, 1m
            
        for window in windows:
            df_features[f'rolling_mean_{window}'] = df_features['y'].rolling(window=window, min_periods=1).mean()
            df_features[f'rolling_std_{window}'] = df_features['y'].rolling(window=window, min_periods=1).std()
            df_features[f'rolling_min_{window}'] = df_features['y'].rolling(window=window, min_periods=1).min()
            df_features[f'rolling_max_{window}'] = df_features['y'].rolling(window=window, min_periods=1).max()
            
        # Add date cyclical features
        if self.freq == '15min' or self.freq == 'H':
            # Hour of day
            df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24.0)
            df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24.0)
        
        # Day of week
        df_features['dayofweek_sin'] = np.sin(2 * np.pi * df_features['dayofweek'] / 7.0)
        df_features['dayofweek_cos'] = np.cos(2 * np.pi * df_features['dayofweek'] / 7.0)
        
        # Month of year
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12.0)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12.0)
        
        # Store the feature names for future use
        self.feature_names = [col for col in df_features.columns if col not in ['ds', 'y', 'unique_id']]
        
        print(f"Created {len(self.feature_names)} features for XGBoost model")
        self.logger.info(f"Created {len(self.feature_names)} features for XGBoost model")
        
        return df_features
    
    def fit(self, df):
        """
        Fit the XGBoost model to the provided dataframe.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing the time series data with columns 'ds', 'y', and 'unique_id'.
            
        Returns:
        --------
        self : XGBoostForecast
            The fitted model instance.
        """
        import pandas as pd
        import xgboost as xgb
        
        region_suffix = f" for region {self.region_name}" if self.region_name else ""
        print(f"Fitting XGBoost model{region_suffix} with {len(df)} data points...")
        self.logger.info(f"Fitting XGBoost model{region_suffix} with {len(df)} data points")
        
        # Store the original data
        self.data = df.copy()
        
        # Sort data by date to ensure proper time ordering
        self.data = self.data.sort_values('ds').reset_index(drop=True)
        
        # Detect frequency
        self.freq = self._detect_frequency(df)
        
        # Set season length if not provided
        if self.season_length is None:
            self.season_length = self._detect_season_length(self.freq)
        
        # For multiple time series (multiple unique_ids), we'll fit separate models
        # For now, we'll use the first unique_id if there are multiple
        unique_ids = self.data['unique_id'].unique()
        
        if len(unique_ids) > 1:
            print(f"Multiple unique_ids detected ({len(unique_ids)}). Using first id: {unique_ids[0]}")
            self.logger.warning(f"Multiple unique_ids detected ({len(unique_ids)}). Using first id: {unique_ids[0]}")
            
            # Filter data for the first unique_id
            self.data = self.data[self.data['unique_id'] == unique_ids[0]].copy()
        
        # Create features for XGBoost
        df_features = self._create_features(self.data)
        
        # Handle missing values created by lags and rolling windows
        df_features = df_features.dropna().reset_index(drop=True)
        
        if len(df_features) == 0:
            error_msg = "Not enough data points after creating lag features"
            print(f"Error: {error_msg}")
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        print(f"Training data shape after feature creation: {df_features.shape}")
        self.logger.info(f"Training data shape after feature creation: {df_features.shape}")
        
        # Keep track of the last date for forecasting
        self.last_date = df_features['ds'].max()
        
        # Prepare training data
        X_train = df_features[self.feature_names]
        y_train = df_features['y']
        
        # Configure the XGBoost model
        print("Configuring XGBoost model...")
        self.logger.info("Configuring XGBoost model")
        
        # Initialize model parameters
        params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.1,
            'max_depth': 6,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.01,
            'reg_lambda': 1,
            'random_state': 42
        }
        
        # Initialize the model
        self.model = xgb.XGBRegressor(**params)
        
        # Fit the model
        print("Fitting model to data...")
        self.logger.info("Fitting model to data")
        self.model.fit(X_train, y_train)
        self.fitted = True
        
        print("Model fitting complete!")
        self.logger.info("Model fitting complete")
        
        return self
    
    def forecast(self, h=96, level=None):
        """
        Generate forecasts for the next h periods.
        
        Parameters:
        -----------
        h : int, optional
            The forecast horizon. Default is 96 (24 hours in 15-min intervals).
        level : list, optional
            Confidence levels for prediction intervals.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the forecasts.
        """
        if not self.fitted:
            error_msg = "Model has not been fitted yet. Call fit() first."
            print(f"Error: {error_msg}")
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        import pandas as pd
        import numpy as np
        
        region_suffix = f" for region {self.region_name}" if self.region_name else ""
        # Adjust h based on frequency to always forecast 24 hours
        original_h = h
        if self.freq == 'H':
            h = 24
        elif self.freq == 'D':
            h = 1
            
        print(f"Generating forecast{region_suffix} for next {h} periods (requested: {original_h}, adjusted based on freq: {self.freq})...")
        self.logger.info(f"Generating forecast{region_suffix} for next {h} periods (freq: {self.freq})")
        
        # Create future dates
        print("Creating future dates for forecasting...")
        self.logger.info("Creating future dates for forecasting")
        
        if self.freq == '15min':
            future_dates = pd.date_range(start=self.last_date, periods=h+1, freq='15min')[1:]
        elif self.freq == 'H':
            future_dates = pd.date_range(start=self.last_date, periods=h+1, freq='H')[1:]
        else:  # 'D'
            future_dates = pd.date_range(start=self.last_date, periods=h+1, freq='D')[1:]
        
        # Use the last historical data to create initial forecast features
        last_data = self.data.copy()
        last_data = last_data.sort_values('ds')
        
        # Create a dataframe for forecasting with future dates
        forecast_df = pd.DataFrame({'ds': future_dates})
        forecast_df['unique_id'] = self.data['unique_id'].iloc[0]  # Use the same ID
        
        # Create empty predictions arrays
        predictions = np.zeros(len(future_dates))
        lower_bounds = np.zeros(len(future_dates))
        upper_bounds = np.zeros(len(future_dates))
        
        # Create a copy of the data with the latest values
        temp_df = pd.concat([last_data, forecast_df]).reset_index(drop=True)
        temp_df['y'] = temp_df['y'].fillna(0)  # Fill NaN for future dates temporarily
        
        # For each future step, make a prediction and update the DataFrame
        for i in range(len(future_dates)):
            # Create features for the forecast date
            future_features_df = self._create_features(temp_df.iloc[:(len(last_data) + i + 1)])
            
            # Get the last row (the one we want to forecast)
            future_features = future_features_df.iloc[-1:]
            X_future = future_features[self.feature_names]
            
            # Make prediction
            prediction = self.model.predict(X_future)[0]
            
            # Update the temp_df with the prediction for the next step
            temp_df.loc[len(last_data) + i, 'y'] = prediction
            predictions[i] = prediction
            
            # Create prediction intervals if requested (using a simple approach based on training error)
            if level is not None:
                # Simple approach: use constant error bounds based on training error
                error_margin = 0.1 * prediction  # 10% of the predicted value as a simple margin
                lower_bounds[i] = prediction - error_margin
                upper_bounds[i] = prediction + error_margin
        
        # Create the forecast dataframe
        forecast_result = pd.DataFrame({
            'unique_id': [self.data['unique_id'].iloc[0]] * len(future_dates),
            'ds': future_dates,
            'XGBoost': predictions
        })
        
        # Add prediction intervals if requested
        if level is not None:
            forecast_result['XGBoost-lo-80'] = lower_bounds
            forecast_result['XGBoost-hi-80'] = upper_bounds
        
        print(f"Forecast generated with {len(forecast_result)} future points")
        self.logger.info(f"Forecast generated with {len(forecast_result)} future points")
        
        # Store the forecast
        self.forecast_df = forecast_result
        
        return self.forecast_df
    
    def plot_acf(self, series=None, lags=40):
        """
        Plot the AutoCorrelation Function (ACF).
        
        Parameters:
        -----------
        series : pandas.Series, optional
            Time series to plot. If None, uses the 'y' column from the fitted data.
        lags : int, optional
            Number of lags to include in the plot.
            
        Returns:
        --------
        matplotlib.figure.Figure
            The ACF plot.
        """
        import matplotlib.pyplot as plt
        from statsmodels.graphics.tsaplots import plot_acf
        
        region_suffix = f" for region {self.region_name}" if self.region_name else ""
        print(f"Generating ACF plot{region_suffix} with {lags} lags...")
        self.logger.info(f"Generating ACF plot{region_suffix} with {lags} lags")
        
        if series is None:
            if self.data is None:
                error_msg = "No data available. Either provide a series or fit the model first."
                print(f"Error: {error_msg}")
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            series = self.data['y']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_acf(series, lags=lags, ax=ax)
        plt.title(f'Autocorrelation Function{region_suffix}')
        plt.tight_layout()
        
        print("ACF plot generated successfully")
        self.logger.info("ACF plot generated successfully")
        
        return fig
    
    def plot_pacf(self, series=None, lags=40):
        """
        Plot the Partial AutoCorrelation Function (PACF).
        
        Parameters:
        -----------
        series : pandas.Series, optional
            Time series to plot. If None, uses the 'y' column from the fitted data.
        lags : int, optional
            Number of lags to include in the plot.
            
        Returns:
        --------
        matplotlib.figure.Figure
            The PACF plot.
        """
        import matplotlib.pyplot as plt
        from statsmodels.graphics.tsaplots import plot_pacf
        
        region_suffix = f" for region {self.region_name}" if self.region_name else ""
        print(f"Generating PACF plot{region_suffix} with {lags} lags...")
        self.logger.info(f"Generating PACF plot{region_suffix} with {lags} lags")
        
        if series is None:
            if self.data is None:
                error_msg = "No data available. Either provide a series or fit the model first."
                print(f"Error: {error_msg}")
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            series = self.data['y']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_pacf(series, lags=lags, ax=ax)
        plt.title(f'Partial Autocorrelation Function{region_suffix}')
        plt.tight_layout()
        
        print("PACF plot generated successfully")
        self.logger.info("PACF plot generated successfully")
        
        return fig
    
    def adf_test(self, series=None):
        """
        Perform the Augmented Dickey-Fuller test for stationarity.
        
        Parameters:
        -----------
        series : pandas.Series, optional
            Time series to test. If None, uses the 'y' column from the fitted data.
            
        Returns:
        --------
        dict
            Dictionary containing the test statistic, p-value, and critical values.
        """
        from statsmodels.tsa.stattools import adfuller
        
        region_suffix = f" for region {self.region_name}" if self.region_name else ""
        print(f"Performing ADF test{region_suffix}...")
        self.logger.info(f"Performing ADF test{region_suffix}")
        
        if series is None:
            if self.data is None:
                error_msg = "No data available. Either provide a series or fit the model first."
                print(f"Error: {error_msg}")
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            series = self.data['y']
        
        result = adfuller(series)
        
        # Format the result
        adf_result = {
            'Test Statistic': result[0],
            'p-value': result[1],
            'Critical Values': result[4]
        }
        
        print(f"ADF test results: Test Statistic={adf_result['Test Statistic']:.4f}, p-value={adf_result['p-value']:.4f}")
        self.logger.info(f"ADF test results: Test Statistic={adf_result['Test Statistic']:.4f}, p-value={adf_result['p-value']:.4f}")
        
        return adf_result
    
    def evaluate(self, h=12, step_size=12, n_windows=5, metrics=None):
        """
        Evaluate the model using time series cross-validation.
        
        Parameters:
        -----------
        h : int, optional
            Forecast horizon (number of periods to forecast). Default is 12.
        step_size : int, optional
            Step size between each window. Default is 12.
        n_windows : int, optional
            Number of windows for cross-validation. Default is 5.
        metrics : list, optional
            List of metrics to calculate. If None, uses MAE, MAPE, RMSE, and SMAPE.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing evaluation metrics for the model.
        """
        if not self.fitted:
            error_msg = "Model has not been fitted yet. Call fit() first."
            print(f"Error: {error_msg}")
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        import pandas as pd
        import numpy as np
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        region_suffix = f" for region {self.region_name}" if self.region_name else ""
        print(f"Performing time series cross-validation{region_suffix} with h={h}, step_size={step_size}, n_windows={n_windows}...")
        self.logger.info(f"Performing time series cross-validation{region_suffix} with h={h}, step_size={step_size}, n_windows={n_windows}")
        
        # Initialize results dataframe
        result = pd.DataFrame(columns=['unique_id', 'metric', 'value'])
        
        # Sort data chronologically
        sorted_data = self.data.sort_values('ds').reset_index(drop=True)
        
        # Convert step_size and h to number of data points
        cutoffs = []
        for i in range(n_windows):
            cutoff_idx = len(sorted_data) - h - (n_windows - i - 1) * step_size
            if cutoff_idx <= 0:
                continue
            cutoffs.append(sorted_data.iloc[cutoff_idx - 1]['ds'])
        
        if len(cutoffs) == 0:
            error_msg = "No valid cutoff points for cross-validation"
            print(f"Error: {error_msg}")
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # For each cutoff, fit model and make predictions
        all_actuals = []
        all_predictions = []
        
        import xgboost as xgb
        
        for cutoff in cutoffs:
            # Split data at cutoff
            train_df = sorted_data[sorted_data['ds'] <= cutoff].copy()
            test_df = sorted_data[sorted_data['ds'] > cutoff].copy()
            test_df = test_df.iloc[:h].copy()  # Only take h steps for testing
            
            if len(test_df) == 0 or len(train_df) < 2:
                continue
            
            print(f"Cross-validation window: train until {cutoff}, test {len(test_df)} steps")
            self.logger.info(f"Cross-validation window: train until {cutoff}, test {len(test_df)} steps")
            
            # Create a temporary model for this fold
            model = XGBoostForecast(region_name=self.region_name, log_level='WARNING')
            model.fit(train_df)
            
            # Generate forecast for the test period
            forecast = model.forecast(h=len(test_df))
            
            # Merge forecast with actual values for comparison
            merged = pd.merge(test_df[['ds', 'y']], forecast[['ds', 'XGBoost']], on='ds', how='inner')
            
            if len(merged) > 0:
                actuals = merged['y'].values
                predictions = merged['XGBoost'].values
                
                all_actuals.extend(actuals)
                all_predictions.extend(predictions)
        
        # Calculate metrics
        if len(all_actuals) > 0 and len(all_predictions) > 0:
            # Convert to numpy arrays
            all_actuals = np.array(all_actuals)
            all_predictions = np.array(all_predictions)
            
            # MAE
            mae = mean_absolute_error(all_actuals, all_predictions)
            
            # RMSE
            rmse = np.sqrt(mean_squared_error(all_actuals, all_predictions))
            
            # MAPE
            non_zero = all_actuals != 0
            if np.any(non_zero):
                mape = np.mean(np.abs((all_actuals[non_zero] - all_predictions[non_zero]) / all_actuals[non_zero])) * 100
            else:
                mape = np.nan
            
            # SMAPE
            smape = np.mean(200.0 * np.abs(all_predictions - all_actuals) / (np.abs(all_predictions) + np.abs(all_actuals) + 1e-8))
            
            # Add to results
            unique_id = sorted_data['unique_id'].iloc[0]
            metrics_list = [
                {'metric': 'mae', 'value': mae},
                {'metric': 'rmse', 'value': rmse},
                {'metric': 'mape', 'value': mape},
                {'metric': 'smape', 'value': smape}
            ]
            
            for metric in metrics_list:
                result = pd.concat([
                    result,
                    pd.DataFrame({
                        'unique_id': [unique_id],
                        'metric': [metric['metric']],
                        'value': [metric['value']]
                    })
                ])
            
            print(f"Cross-validation complete with metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.4f}, SMAPE={smape:.4f}")
            self.logger.info(f"Cross-validation complete with metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.4f}, SMAPE={smape:.4f}")
        else:
            print("Warning: No valid cross-validation results generated")
            self.logger.warning("No valid cross-validation results generated")
        
        return result
    
    def plot_forecast(self, test_df=None, figsize=(12, 6)):
        """
        Plot the forecast along with the historical data.
        
        Parameters:
        -----------
        test_df : pandas.DataFrame, optional
            Test data to overlay on the plot.
        figsize : tuple, optional
            Figure size.
            
        Returns:
        --------
        matplotlib.figure.Figure
            The forecast plot.
        """
        if not self.fitted or self.forecast_df is None:
            error_msg = "Model has not been fitted or forecast has not been generated."
            print(f"Error: {error_msg}")
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        import matplotlib.pyplot as plt
        import pandas as pd
        from matplotlib.dates import DateFormatter
        
        region_suffix = f" for region {self.region_name}" if self.region_name else ""
        print(f"Creating forecast plot{region_suffix}...")
        self.logger.info(f"Creating forecast plot{region_suffix}")
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Ensure data is sorted by date
        sorted_data = self.data.sort_values('ds').reset_index(drop=True)
        forecast_df = self.forecast_df.sort_values('ds').reset_index(drop=True)
        
        # Get the last data point and first forecast point
        last_historical_date = sorted_data['ds'].max()
        last_historical_value = sorted_data[sorted_data['ds'] == last_historical_date]['y'].values[0]
        
        first_forecast_date = forecast_df['ds'].min()
        first_forecast_value = forecast_df[forecast_df['ds'] == first_forecast_date]['XGBoost'].values[0]
        
        print(f"Last historical date: {last_historical_date}, value: {last_historical_value}")
        print(f"First forecast date: {first_forecast_date}, value: {first_forecast_value}")
        print(f"Time difference: {first_forecast_date - last_historical_date}")
        self.logger.info(f"Last historical date: {last_historical_date}, value: {last_historical_value}")
        self.logger.info(f"First forecast date: {first_forecast_date}, value: {first_forecast_value}")
        self.logger.info(f"Time difference: {first_forecast_date - last_historical_date}")
        
        # Plot historical data
        ax.plot(sorted_data['ds'], sorted_data['y'], label='Historical Data')
        
        # Plot forecast data
        ax.plot(forecast_df['ds'], forecast_df['XGBoost'], label='XGBoost Forecast', color='red')
        
        # Add the bridge point to connect historical and forecast
        bridge_points = pd.DataFrame({
            'ds': [last_historical_date, first_forecast_date],
            'value': [last_historical_value, first_forecast_value]
        })
        ax.plot(bridge_points['ds'], bridge_points['value'], 'r--', alpha=0.5)
        
        # Plot prediction intervals if available
        if 'XGBoost-lo-80' in forecast_df.columns and 'XGBoost-hi-80' in forecast_df.columns:
            ax.fill_between(
                forecast_df['ds'],
                forecast_df['XGBoost-lo-80'],
                forecast_df['XGBoost-hi-80'],
                alpha=0.2,
                label='80% Confidence Interval'
            )
        
        # Plot test data if provided
        if test_df is not None:
            ax.plot(test_df['ds'], test_df['y'], label='Actual Values', linestyle='--', color='purple')
        
        # Format x-axis dates
        date_format = DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_formatter(date_format)
        plt.xticks(rotation=45)
        
        # Add labels and legend
        title = f'XGBoost Forecast{region_suffix}'
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend()
        plt.tight_layout()
        
        print("Forecast plot created successfully")
        self.logger.info("Forecast plot created successfully")
        
        return fig
    
    def plot_feature_importance(self, figsize=(12, 8)):
        """
        Plot feature importance from the XGBoost model.
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size.
            
        Returns:
        --------
        matplotlib.figure.Figure
            The feature importance plot.
        """
        if not self.fitted or self.model is None:
            error_msg = "Model has not been fitted yet. Call fit() first."
            print(f"Error: {error_msg}")
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        import matplotlib.pyplot as plt
        import pandas as pd
        
        region_suffix = f" for region {self.region_name}" if self.region_name else ""
        print(f"Creating feature importance plot{region_suffix}...")
        self.logger.info(f"Creating feature importance plot{region_suffix}")
        
        # Get feature importance
        importance = self.model.feature_importances_
        
        # Create a dataframe for plotting
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(importance_df['Feature'], importance_df['Importance'])
        ax.set_title(f'XGBoost Feature Importance{region_suffix}')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        plt.tight_layout()
        
        print("Feature importance plot created successfully")
        self.logger.info("Feature importance plot created successfully")
        
        return fig
    
    def get_results(self):
        """
        Get the forecast results.
        
        Returns:
        --------
        dict
            Dictionary containing the model results, including:
            - data: original data
            - forecast: forecast dataframe
            - freq: detected frequency
            - season_length: seasonal period
            - region_name: name of the region
        """
        if not self.fitted:
            error_msg = "Model has not been fitted yet. Call fit() first."
            print(f"Error: {error_msg}")
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        results = {
            'data': self.data,
            'forecast': self.forecast_df,
            'freq': self.freq,
            'season_length': self.season_length,
            'feature_names': self.feature_names,
            'region_name': self.region_name
        }
        
        print(f"Returning model results for {self.region_name if self.region_name else 'default'}")
        self.logger.info(f"Returning model results for {self.region_name if self.region_name else 'default'}")
        
        return results
    
    def get_model_summary(self):
        """
        Get a summary of the model configuration and status.
        
        Returns:
        --------
        dict
            Dictionary containing model configuration and status.
        """
        summary = {
            'region_name': self.region_name,
            'fitted': self.fitted,
            'frequency': self.freq,
            'season_length': self.season_length,
            'data_points': len(self.data) if self.data is not None else 0,
            'feature_count': len(self.feature_names) if self.feature_names is not None else 0,
            'forecast_generated': self.forecast_df is not None,
            'forecast_periods': len(self.forecast_df) if self.forecast_df is not None else 0
        }
        
        print(f"Model summary: {summary}")
        self.logger.info(f"Model summary: {summary}")
        
        return summary

