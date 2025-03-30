class AutoARIMAForecast:
    def __init__(self, region_name=None, initial_df=None, season_length=None, log_level='INFO'):
        """
        Initialize the AutoARIMA forecasting model.
        
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
        self.logger = logging.getLogger(f"AutoARIMAForecast_{region_name if region_name else 'default'}")
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
        
        print(f"Initializing AutoARIMA model for region: {region_name if region_name else 'default'}")
        self.logger.info(f"Initializing AutoARIMA model for region: {region_name if region_name else 'default'}")
        
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
    
    def fit(self, df):
        """
        Fit the AutoARIMA model to the provided dataframe.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing the time series data with columns 'ds', 'y', and 'unique_id'.
            
        Returns:
        --------
        self : AutoARIMAForecast
            The fitted model instance.
        """
        import pandas as pd
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA
        
        region_suffix = f" for region {self.region_name}" if self.region_name else ""
        print(f"Fitting AutoARIMA model{region_suffix} with {len(df)} data points...")
        self.logger.info(f"Fitting AutoARIMA model{region_suffix} with {len(df)} data points")
        
        # Store the original data
        self.data = df.copy()
        
        # Sort data by date to ensure proper time ordering
        self.data = self.data.sort_values('ds').reset_index(drop=True)
        
        # Detect frequency
        self.freq = self._detect_frequency(df)
        
        # Set season length if not provided
        if self.season_length is None:
            self.season_length = self._detect_season_length(self.freq)
        
        # Initialize the AutoARIMA model
        print(f"Creating AutoARIMA model with season_length={self.season_length}")
        self.logger.info(f"Creating AutoARIMA model with season_length={self.season_length}")
        models = [AutoARIMA(season_length=self.season_length)]
        
        # Create and fit the StatsForecast model
        print("Creating StatsForecast model...")
        self.logger.info("Creating StatsForecast model")
        self.model = StatsForecast(
            models=models,
            freq=self.freq,
            n_jobs=-1  # Use all available cores
        )
        
        # Fit the model
        print("Fitting model to data...")
        self.logger.info("Fitting model to data")
        self.model.fit(self.data)
        self.fitted = True
        
        print("Model fitting complete!")
        self.logger.info("Model fitting complete")
        
        return self
    
    def forecast(self, h=96, level=None):
        """
        Generate forecasts for the next h periods (default: 24 hours).
        
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
        
        region_suffix = f" for region {self.region_name}" if self.region_name else ""
        # Adjust h based on frequency to always forecast 24 hours
        original_h = h
        if self.freq == 'H':
            h = 24
        elif self.freq == 'D':
            h = 1
            
        print(f"Generating forecast{region_suffix} for next {h} periods (requested: {original_h}, adjusted based on freq: {self.freq})...")
        self.logger.info(f"Generating forecast{region_suffix} for next {h} periods (freq: {self.freq})")
        
        # Create a forecast df with just the unique_id and ds columns
        import pandas as pd
        import numpy as np
        
        # Get unique identifiers from the training data
        unique_ids = self.data['unique_id'].unique()
        
        # Generate future dates based on the last date in the training data
        # Sort data to ensure we get the actual last date
        sorted_data = self.data.sort_values('ds')
        last_date = pd.to_datetime(sorted_data['ds']).max()
        print(f"Last historical date: {last_date}")
        self.logger.info(f"Last historical date: {last_date}")
        
        # Determine the next timestamp directly based on the frequency using explicit timedelta
        if self.freq == '15min':
            next_timestamp = last_date + pd.Timedelta(minutes=15)
        elif self.freq == 'H':
            next_timestamp = last_date + pd.Timedelta(hours=1)
        else:  # 'D'
            next_timestamp = last_date + pd.Timedelta(days=1)
        
        print(f"Next forecast timestamp: {next_timestamp}")
        self.logger.info(f"Next forecast timestamp: {next_timestamp}")
        
        # Create a dataframe with future dates for forecasting
        # Use explicit parameters to avoid any ambiguity
        future_dates = pd.date_range(
            start=next_timestamp,
            periods=h,
            freq=self.freq
        )
        
        # Verify the generated dates - this is important for debugging
        print(f"Generated future dates from {future_dates[0]} to {future_dates[-1]}")
        print(f"Time difference between last historical and first forecast: {future_dates[0] - last_date}")
        self.logger.info(f"Generated future dates from {future_dates[0]} to {future_dates[-1]}")
        self.logger.info(f"Time difference between last historical and first forecast: {future_dates[0] - last_date}")
        
        # Store expected dates for later verification and correction
        expected_first_date = future_dates[0]
        expected_dates = future_dates
        
        # Create forecast dataframe with all combinations of unique_ids and future dates
        forecast_input_df = pd.DataFrame()
        for unique_id in unique_ids:
            temp_df = pd.DataFrame({
                'unique_id': [unique_id] * len(future_dates),
                'ds': future_dates,
                'y': [np.nan] * len(future_dates)  # Add y column with NaN values
            })
            forecast_input_df = pd.concat([forecast_input_df, temp_df])
        
        # Ensure proper sorting of the dataframe
        forecast_input_df = forecast_input_df.sort_values(['unique_id', 'ds']).reset_index(drop=True)
        
        print(f"Created forecast input dataframe with {len(forecast_input_df)} rows")
        self.logger.info(f"Created forecast input dataframe with {len(forecast_input_df)} rows")
        
        # Generate forecasts
        try:
            if level is not None:
                print(f"Including confidence intervals at levels: {level}")
                self.logger.info(f"Including confidence intervals at levels: {level}")
                self.forecast_df = self.model.forecast(df=forecast_input_df, h=h, level=level)
            else:
                self.forecast_df = self.model.forecast(df=forecast_input_df, h=h)
            
            print(f"Forecast generated with {len(self.forecast_df)} rows")
            self.logger.info(f"Forecast generated with {len(self.forecast_df)} rows")
            
            # Verify the forecast dates and correct them if needed
            forecast_dates = pd.to_datetime(self.forecast_df['ds']).sort_values().unique()
            actual_first_date = forecast_dates[0]
            
            print(f"Actual forecast dates: {actual_first_date} to {forecast_dates[-1]}")
            self.logger.info(f"Actual forecast dates: {actual_first_date} to {forecast_dates[-1]}")
            
            # Check if the actual first forecast date is significantly different from what we expected
            # This is likely due to StatsForecast possibly adding an extra day
            time_diff = actual_first_date - expected_first_date
            
            if abs(time_diff.total_seconds()) > 3600:  # More than an hour difference
                print(f"WARNING: First forecast date ({actual_first_date}) doesn't match expected date ({expected_first_date})")
                print(f"Time difference: {time_diff}")
                self.logger.warning(f"First forecast date ({actual_first_date}) doesn't match expected date ({expected_first_date})")
                self.logger.warning(f"Time difference: {time_diff}")
                
                # Correct the forecast dates to match what we expected
                print("Correcting forecast dates to match expected timeline...")
                self.logger.info("Correcting forecast dates to match expected timeline")
                
                # Create a mapping from the incorrect dates to the expected dates
                date_mapping = {}
                for i, incorrect_date in enumerate(forecast_dates):
                    if i < len(expected_dates):
                        date_mapping[incorrect_date] = expected_dates[i]
                    else:
                        # If we have more forecast dates than expected (unlikely), extend with the same interval
                        last_mapped_date = date_mapping[forecast_dates[i-1]]
                        if self.freq == '15min':
                            date_mapping[incorrect_date] = last_mapped_date + pd.Timedelta(minutes=15)
                        elif self.freq == 'H':
                            date_mapping[incorrect_date] = last_mapped_date + pd.Timedelta(hours=1)
                        else:  # 'D'
                            date_mapping[incorrect_date] = last_mapped_date + pd.Timedelta(days=1)
                
                # Apply the date correction
                corrected_forecast = self.forecast_df.copy()
                corrected_forecast['ds'] = corrected_forecast['ds'].apply(
                    lambda x: date_mapping.get(pd.to_datetime(x), pd.to_datetime(x))
                )
                
                self.forecast_df = corrected_forecast
                
                # Verify the correction
                corrected_dates = pd.to_datetime(self.forecast_df['ds']).sort_values().unique()
                print(f"Corrected forecast dates: {corrected_dates[0]} to {corrected_dates[-1]}")
                self.logger.info(f"Corrected forecast dates: {corrected_dates[0]} to {corrected_dates[-1]}")
                print(f"Time difference between last historical and first corrected forecast: {corrected_dates[0] - last_date}")
                self.logger.info(f"Time difference between last historical and first corrected forecast: {corrected_dates[0] - last_date}")
            
        except Exception as e:
            error_msg = f"Error generating forecast: {str(e)}"
            print(f"Error: {error_msg}")
            self.logger.error(error_msg)
            
            # Try a different approach - use predict instead of forecast
            print("Attempting alternative forecasting approach...")
            self.logger.info("Attempting alternative forecasting approach")
            
            try:
                # Some versions of statsforecast might use predict instead of forecast
                if level is not None:
                    self.forecast_df = self.model.predict(h=h, level=level)
                else:
                    self.forecast_df = self.model.predict(h=h)
                
                print(f"Forecast generated with {len(self.forecast_df)} rows")
                self.logger.info(f"Forecast generated with {len(self.forecast_df)} rows")
                
                # Apply the same date correction for the predict method
                forecast_dates = pd.to_datetime(self.forecast_df['ds']).sort_values().unique()
                actual_first_date = forecast_dates[0]
                
                # If there's a significant time difference, correct the dates
                if abs((actual_first_date - expected_first_date).total_seconds()) > 3600:
                    print(f"WARNING: First forecast date from predict ({actual_first_date}) doesn't match expected ({expected_first_date})")
                    self.logger.warning(f"First forecast date from predict ({actual_first_date}) doesn't match expected ({expected_first_date})")
                    
                    # Generate the correct series of dates
                    corrected_dates = pd.date_range(
                        start=expected_first_date,
                        periods=len(forecast_dates),
                        freq=self.freq
                    )
                    
                    # Map old dates to new dates
                    date_mapping = dict(zip(sorted(forecast_dates), corrected_dates))
                    
                    # Apply correction
                    corrected_forecast = self.forecast_df.copy()
                    corrected_forecast['ds'] = corrected_forecast['ds'].apply(
                        lambda x: date_mapping.get(pd.to_datetime(x), pd.to_datetime(x))
                    )
                    
                    self.forecast_df = corrected_forecast
                    print("Corrected predict dates to match expected timeline")
                    self.logger.info("Corrected predict dates to match expected timeline")
            except Exception as e2:
                error_msg = f"Error with alternative forecasting approach: {str(e2)}"
                print(f"Error: {error_msg}")
                self.logger.error(error_msg)
                raise ValueError(f"Unable to generate forecast: {str(e)}, then {str(e2)}")
        
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
        Evaluate the model using time series cross-validation as described in the Nixtla documentation.
        
        Parameters:
        -----------
        h : int, optional
            Forecast horizon (number of periods to forecast). Default is 12.
        step_size : int, optional
            Step size between each window. How often to run the forecasting process. Default is 12.
        n_windows : int, optional
            Number of windows for cross-validation. How many forecasting processes in the past to evaluate. Default is 5.
        metrics : list, optional
            List of metrics to calculate. If None, uses MAE, MAPE, MASE, RMSE, and SMAPE.
            
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
        
        if self.data is None:
            error_msg = "No data available for evaluation."
            print(f"Error: {error_msg}")
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        import pandas as pd
        import numpy as np
        from functools import partial
        import utilsforecast.losses as ufl
        from utilsforecast.evaluation import evaluate
        
        region_suffix = f" for region {self.region_name}" if self.region_name else ""
        print(f"Performing cross-validation{region_suffix} with h={h}, step_size={step_size}, n_windows={n_windows}...")
        self.logger.info(f"Performing cross-validation{region_suffix} with h={h}, step_size={step_size}, n_windows={n_windows}")
        
        # Store original model state to restore later
        original_data = self.data.copy()
        original_forecast_df = self.forecast_df.copy() if self.forecast_df is not None else None
        
        try:
            # Perform cross-validation using StatsForecast's cross_validation method
            print("Executing cross-validation...")
            self.logger.info("Executing cross-validation")
            
            # Use the model's cross_validation method if available
            if hasattr(self.model, 'cross_validation'):
                cv_df = self.model.cross_validation(
                    df=self.data,
                    h=h,
                    step_size=step_size,
                    n_windows=n_windows
                )
                print(f"Cross-validation generated {len(cv_df)} rows")
                self.logger.info(f"Cross-validation generated {len(cv_df)} rows")
            else:
                # If cross_validation is not available, fall back to manual approach
                print("StatsForecast cross_validation method not available. Implementing manual approach.")
                self.logger.warning("StatsForecast cross_validation method not available. Implementing manual approach.")
                
                # Sort data chronologically
                sorted_data = self.data.sort_values(['unique_id', 'ds']).reset_index(drop=True)
                
                # Create empty dataframe for cross-validation results
                cv_df = pd.DataFrame()
                
                # Identify unique time series
                unique_ids = sorted_data['unique_id'].unique()
                
                for uid in unique_ids:
                    uid_data = sorted_data[sorted_data['unique_id'] == uid].copy()
                    
                    # Calculate number of observations
                    n_obs = len(uid_data)
                    
                    # Check if we have enough data for cross-validation
                    if n_obs <= h + step_size * (n_windows - 1):
                        print(f"Warning: Not enough data for {uid} with {n_obs} observations. Need at least {h + step_size * (n_windows - 1) + 1}.")
                        self.logger.warning(f"Not enough data for {uid} with {n_obs} observations")
                        continue
                    
                    # Implement sliding window approach
                    for i in range(n_windows):
                        # Calculate cutoff point
                        cutoff_idx = n_obs - h - step_size * (n_windows - i - 1)
                        
                        # Split into train and test
                        train_data = uid_data.iloc[:cutoff_idx].copy()
                        test_data = uid_data.iloc[cutoff_idx:cutoff_idx + h].copy()
                        
                        if len(train_data) < 2:
                            print(f"Warning: Training data too small for window {i+1}. Skipping.")
                            self.logger.warning(f"Training data too small for window {i+1}. Skipping.")
                            continue
                        
                        # Get cutoff date
                        cutoff_date = train_data['ds'].max()
                        
                        # Create a temporary model and fit to training data
                        from statsforecast import StatsForecast
                        from statsforecast.models import AutoARIMA
                        
                        temp_model = StatsForecast(
                            models=[AutoARIMA(season_length=self.season_length)],
                            freq=self.freq,
                            n_jobs=-1
                        )
                        
                        # Fit the temporary model
                        temp_model.fit(train_data)
                        
                        # Generate forecast
                        forecast_df = temp_model.forecast(h=h)
                        
                        # Merge with actual values
                        forecast_df['cutoff'] = cutoff_date
                        merged = test_data.merge(
                            forecast_df[['unique_id', 'ds', 'AutoARIMA', 'cutoff']],
                            on=['unique_id', 'ds'],
                            how='left'
                        )
                        
                        # Add to cross-validation dataframe
                        cv_df = pd.concat([cv_df, merged])
            
            # If no cross-validation results, return empty metrics
            if len(cv_df) == 0:
                print("Warning: No cross-validation results generated.")
                self.logger.warning("No cross-validation results generated")
                return pd.DataFrame(columns=['unique_id', 'metric', 'value'])
            
            # Set up default metrics if not provided
            if metrics is None:
                metrics = [
                    ufl.mae, 
                    ufl.mape, 
                    partial(ufl.mase, seasonality=self.season_length), 
                    ufl.rmse, 
                    ufl.smape
                ]
            
            # Evaluate performance using the metrics
            print("Computing evaluation metrics...")
            self.logger.info("Computing evaluation metrics")
            
            # Prepare data for evaluation - ensure it has the columns: unique_id, ds, y, AutoARIMA
            # Check if 'y' and 'AutoARIMA' columns exist
            if 'y' not in cv_df.columns or 'AutoARIMA' not in cv_df.columns:
                print("Warning: Required columns missing in cross-validation results.")
                self.logger.warning("Required columns missing in cross-validation results")
                return pd.DataFrame(columns=['unique_id', 'metric', 'value'])
            
            # Check if we have the evaluate function from utilsforecast
            try:
                result = evaluate(
                    cv_df,
                    metrics=metrics,
                    train_df=self.data
                )
                print("Evaluation complete")
                self.logger.info("Evaluation complete")
            except Exception as e:
                print(f"Error during evaluation: {str(e)}")
                self.logger.error(f"Error during evaluation: {str(e)}")
                
                # Fall back to manual calculation of metrics
                print("Falling back to manual calculation of metrics")
                self.logger.info("Falling back to manual calculation of metrics")
                
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                
                result = pd.DataFrame(columns=['unique_id', 'metric', 'value'])
                
                # Group by unique_id to calculate metrics for each time series
                for uid in cv_df['unique_id'].unique():
                    uid_cv = cv_df[cv_df['unique_id'] == uid]
                    
                    # Extract actual and predicted values as numeric types
                    try:
                        y_true = pd.to_numeric(uid_cv['y'], errors='coerce').values
                        y_pred = pd.to_numeric(uid_cv['AutoARIMA'], errors='coerce').values
                        
                        # Filter out NaN values
                        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
                        if np.any(valid_mask):
                            y_true_valid = y_true[valid_mask]
                            y_pred_valid = y_pred[valid_mask]
                            
                            if len(y_true_valid) > 0:
                                # Calculate metrics
                                mae = mean_absolute_error(y_true_valid, y_pred_valid)
                                rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
                                
                                # MAPE (handle zeros)
                                non_zero = y_true_valid != 0
                                if np.any(non_zero):
                                    mape = np.mean(np.abs((y_true_valid[non_zero] - y_pred_valid[non_zero]) / y_true_valid[non_zero])) * 100
                                else:
                                    mape = np.nan
                                
                                # SMAPE
                                smape = np.mean(200.0 * np.abs(y_pred_valid - y_true_valid) / (np.abs(y_pred_valid) + np.abs(y_true_valid) + 1e-8))
                                
                                # MASE calculation (basic implementation)
                                mase = np.nan
                                try:
                                    uid_train = self.data[self.data['unique_id'] == uid].sort_values('ds')
                                    y_train = pd.to_numeric(uid_train['y'], errors='coerce').values
                                    
                                    # Calculate in-sample naive forecast errors (one-step)
                                    naive_errors = np.abs(y_train[1:] - y_train[:-1])
                                    
                                    # Calculate mean absolute in-sample naive error
                                    if len(naive_errors) > 0:
                                        mean_naive_error = np.mean(naive_errors)
                                        
                                        if mean_naive_error > 0:
                                            # MASE = MAE / mean_naive_error
                                            mase = mae / mean_naive_error
                                except Exception as mase_err:
                                    print(f"Error calculating MASE: {str(mase_err)}")
                                    self.logger.error(f"Error calculating MASE: {str(mase_err)}")
                                
                                # Add to results dataframe
                                metrics_df = pd.DataFrame({
                                    'unique_id': [uid] * 5,
                                    'metric': ['mae', 'mape', 'mase', 'rmse', 'smape'],
                                    'value': [mae, mape, mase, rmse, smape]
                                })
                                
                                result = pd.concat([result, metrics_df])
                            else:
                                print(f"Warning: No valid data points for {uid} after filtering NaNs")
                                self.logger.warning(f"No valid data points for {uid} after filtering NaNs")
                        else:
                            print(f"Warning: All data points for {uid} are NaN")
                            self.logger.warning(f"All data points for {uid} are NaN")
                    except Exception as calc_err:
                        print(f"Error calculating metrics for {uid}: {str(calc_err)}")
                        self.logger.error(f"Error calculating metrics for {uid}: {str(calc_err)}")
            
            return result
            
        finally:
            # Restore original model state
            self.data = original_data
            self.forecast_df = original_forecast_df
            
            # Ensure the model is in the same state as before
            print("Restoring model to original state...")
            self.logger.info("Restoring model to original state")
    
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
        import numpy as np
        from matplotlib.dates import DateFormatter
        
        region_suffix = f" for region {self.region_name}" if self.region_name else ""
        print(f"Creating forecast plot{region_suffix}...")
        self.logger.info(f"Creating forecast plot{region_suffix}")
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Ensure data is sorted by date
        sorted_data = self.data.sort_values('ds').reset_index(drop=True)
        sorted_forecast = self.forecast_df.sort_values('ds').reset_index(drop=True)
        
        # Get the last data point from historical data
        last_historical_date = sorted_data['ds'].max()
        last_historical_value = sorted_data[sorted_data['ds'] == last_historical_date]['y'].values[0]
        
        # Get the first forecast point
        first_forecast_date = sorted_forecast['ds'].min()
        first_forecast_value = sorted_forecast[sorted_forecast['ds'] == first_forecast_date][sorted_forecast.columns[2]].values[0]
        
        print(f"Last historical date: {last_historical_date}, value: {last_historical_value}")
        print(f"First forecast date: {first_forecast_date}, value: {first_forecast_value}")
        print(f"Time difference: {first_forecast_date - last_historical_date}")
        self.logger.info(f"Last historical date: {last_historical_date}, value: {last_historical_value}")
        self.logger.info(f"First forecast date: {first_forecast_date}, value: {first_forecast_value}")
        self.logger.info(f"Time difference: {first_forecast_date - last_historical_date}")
        
        # Plot historical data
        ax.plot(sorted_data['ds'], sorted_data['y'], label='Historical Data')
        
        # Create continuous data for plotting by adding interpolated points if needed
        continuous_forecast_df = sorted_forecast.copy()
        
        # Check if there's a large gap between the last historical point and first forecast
        # We'll define "large" as more than 2x the expected interval based on frequency
        expected_interval = None
        if self.freq == '15min':
            expected_interval = pd.Timedelta(minutes=15)
        elif self.freq == 'H':
            expected_interval = pd.Timedelta(hours=1)
        else:  # 'D'
            expected_interval = pd.Timedelta(days=1)
        
        actual_interval = first_forecast_date - last_historical_date
        
        # If the gap is significant, create intermediate points for a smoother transition
        if actual_interval > expected_interval * 2:
            print(f"Large gap detected ({actual_interval}), creating interpolation points for smoother transition")
            self.logger.info(f"Large gap detected ({actual_interval}), creating interpolation points for smoother transition")
            
            # Create interpolation points
            num_points = max(3, int(actual_interval / expected_interval) - 1)  # At least 3 points for smooth interpolation
            interp_dates = pd.date_range(
                start=last_historical_date + expected_interval,
                end=first_forecast_date - expected_interval,
                periods=num_points
            )
            
            # Linear interpolation between last historical and first forecast values
            interp_values = np.linspace(last_historical_value, first_forecast_value, num_points + 2)[1:-1]
            
            # Create DataFrame with interpolated points
            interp_df = pd.DataFrame({
                'unique_id': [continuous_forecast_df['unique_id'].iloc[0]] * len(interp_dates),
                'ds': interp_dates,
                sorted_forecast.columns[2]: interp_values
            })
            
            # Add confidence interval columns if they exist with interpolated values
            for level in [80, 95]:
                lower_col = f'AutoARIMA-lo-{level}'
                upper_col = f'AutoARIMA-hi-{level}'
                
                if lower_col in continuous_forecast_df.columns and upper_col in continuous_forecast_df.columns:
                    first_lower = continuous_forecast_df[lower_col].iloc[0]
                    first_upper = continuous_forecast_df[upper_col].iloc[0]
                    
                    # Create widening confidence intervals from historical point to forecast intervals
                    lower_values = np.linspace(last_historical_value, first_lower, num_points + 2)[1:-1]
                    upper_values = np.linspace(last_historical_value, first_upper, num_points + 2)[1:-1]
                    
                    interp_df[lower_col] = lower_values
                    interp_df[upper_col] = upper_values
            
            # Add interpolation points to forecast dataframe
            continuous_forecast_df = pd.concat([interp_df, continuous_forecast_df])
            continuous_forecast_df = continuous_forecast_df.sort_values('ds').reset_index(drop=True)
            
            print(f"Added {len(interp_dates)} interpolation points to create smooth transition")
            self.logger.info(f"Added {len(interp_dates)} interpolation points to create smooth transition")
        
        # Also add the last historical point to create a continuous line
        bridge_point = pd.DataFrame({
            'unique_id': [continuous_forecast_df['unique_id'].iloc[0]],
            'ds': [last_historical_date],
            sorted_forecast.columns[2]: [last_historical_value]
        })
        
        # Add any confidence interval columns if they exist
        for level in [80, 95]:
            lower_col = f'AutoARIMA-lo-{level}'
            upper_col = f'AutoARIMA-hi-{level}'
            
            if lower_col in continuous_forecast_df.columns and upper_col in continuous_forecast_df.columns:
                bridge_point[lower_col] = last_historical_value
                bridge_point[upper_col] = last_historical_value
        
        # Concatenate the bridge point with the forecast data
        continuous_forecast_df = pd.concat([bridge_point, continuous_forecast_df])
        continuous_forecast_df = continuous_forecast_df.sort_values('ds').reset_index(drop=True)
        
        print(f"Added bridge point to create continuous forecast line")
        self.logger.info(f"Added bridge point to create continuous forecast line")
        
        # Plot forecast with the bridge and interpolation points for a continuous line
        forecast_series = continuous_forecast_df[sorted_forecast.columns[2]]
        ax.plot(continuous_forecast_df['ds'], forecast_series, label='Forecast', color='red')
        
        # Plot confidence intervals if available
        for level in [80, 95]:
            lower_col = f'AutoARIMA-lo-{level}'
            upper_col = f'AutoARIMA-hi-{level}'
            
            if lower_col in continuous_forecast_df.columns and upper_col in continuous_forecast_df.columns:
                print(f"Adding {level}% confidence interval to plot...")
                self.logger.info(f"Adding {level}% confidence interval to plot")
                ax.fill_between(
                    continuous_forecast_df['ds'],
                    continuous_forecast_df[lower_col],
                    continuous_forecast_df[upper_col],
                    alpha=0.2,
                    label=f'{level}% Confidence Interval'
                )
        
        # Plot test data if provided
        if test_df is not None:
            print("Overlaying test data on plot...")
            self.logger.info("Overlaying test data on plot")
            ax.plot(test_df['ds'], test_df['y'], label='Actual Values', linestyle='--', color='green')
        
        # Format x-axis dates
        date_format = DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_formatter(date_format)
        plt.xticks(rotation=45)
        
        # Add labels and legend
        title = f'AutoARIMA Forecast{region_suffix}'
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend()
        plt.tight_layout()
        
        print("Forecast plot created successfully")
        self.logger.info("Forecast plot created successfully")
        
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
            'forecast_generated': self.forecast_df is not None,
            'forecast_periods': len(self.forecast_df) if self.forecast_df is not None else 0
        }
        
        print(f"Model summary: {summary}")
        self.logger.info(f"Model summary: {summary}")
        
        return summary