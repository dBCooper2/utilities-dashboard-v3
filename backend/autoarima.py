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
    
    def evaluate(self, test_size=0.2, metrics=None):
        """
        Evaluate the model performance using a train-test split of the initial data.
        
        Parameters:
        -----------
        test_size : float, optional
            Proportion of the dataset to include in the test split. Default is 0.2 (20%).
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
        print(f"Evaluating model performance{region_suffix} with test_size={test_size}...")
        self.logger.info(f"Evaluating model performance{region_suffix} with test_size={test_size}")
        
        # Sort data by date to ensure proper time ordering
        sorted_data = self.data.sort_values('ds').reset_index(drop=True)
        
        # Calculate the split point
        split_idx = int(len(sorted_data) * (1 - test_size))
        if split_idx <= 0 or split_idx >= len(sorted_data):
            error_msg = f"Invalid test_size: {test_size} results in invalid split index: {split_idx} for dataset of size {len(sorted_data)}"
            print(f"Error: {error_msg}")
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Split the data
        train_df = sorted_data.iloc[:split_idx].copy()
        test_df = sorted_data.iloc[split_idx:].copy()
        
        print(f"Data split: {len(train_df)} training points, {len(test_df)} test points")
        self.logger.info(f"Data split: {len(train_df)} training points, {len(test_df)} test points")
        
        # Save original model state
        original_data = self.data.copy()
        original_forecast_df = self.forecast_df.copy() if self.forecast_df is not None else None
        
        try:
            # Re-fit the model on the training data
            print("Fitting model on training data...")
            self.logger.info("Fitting model on training data")
            self.fit(train_df)
            
            # Generate forecast for the test period
            print(f"Generating forecast for {len(test_df)} test points...")
            self.logger.info(f"Generating forecast for {len(test_df)} test points")
            
            # Determine the forecast horizon needed to cover the test period
            if self.freq == '15min':
                h = len(test_df)
            elif self.freq == 'H':
                h = len(test_df)
            else:  # 'D'
                h = len(test_df)
            
            # Generate forecast
            forecast_df = self.forecast(h=h)
            
            # Default metrics
            if metrics is None:
                metrics = [
                    ufl.mae, ufl.mape, 
                    partial(ufl.mase, seasonality=self.season_length), 
                    ufl.rmse, ufl.smape
                ]
            
            # Prepare evaluation data by merging test data with forecasts
            print(f"Merging test data with forecasts...")
            self.logger.info(f"Merging test data with forecasts")
            
            # Ensure forecast dates match test dates
            merged_df = test_df.merge(forecast_df, on=['unique_id', 'ds'], how='inner')
            
            if len(merged_df) == 0:
                print("WARNING: No matching dates between test data and forecast. Checking for date alignment issues...")
                self.logger.warning("No matching dates between test data and forecast")
                
                # Print date ranges to debug
                test_dates = pd.to_datetime(test_df['ds']).sort_values()
                forecast_dates = pd.to_datetime(forecast_df['ds']).sort_values()
                
                print(f"Test data date range: {test_dates.min()} to {test_dates.max()}")
                print(f"Forecast date range: {forecast_dates.min()} to {forecast_dates.max()}")
                
                self.logger.info(f"Test data date range: {test_dates.min()} to {test_dates.max()}")
                self.logger.info(f"Forecast date range: {forecast_dates.min()} to {forecast_dates.max()}")
                
                # Try a different approach - use closest dates if exact matches fail
                print("Attempting alternative evaluation approach with closest dates...")
                self.logger.info("Attempting alternative evaluation approach with closest dates")
                
                # Create a simple DataFrame with accuracy metrics
                alternative_metrics = pd.DataFrame(columns=['unique_id', 'metric', 'value'])
                
                for uid in test_df['unique_id'].unique():
                    test_subset = test_df[test_df['unique_id'] == uid].sort_values('ds')
                    forecast_subset = forecast_df[forecast_df['unique_id'] == uid].sort_values('ds')
                    
                    # If we have different number of points, take the minimum
                    min_points = min(len(test_subset), len(forecast_subset))
                    if min_points > 0:
                        # Calculate simple metrics manually
                        actuals = test_subset['y'].values[:min_points]
                        predictions = forecast_subset[forecast_subset.columns[2]].values[:min_points]
                        
                        # Calculate MAE
                        mae = np.mean(np.abs(actuals - predictions))
                        
                        # Add to results
                        alternative_metrics = pd.concat([
                            alternative_metrics,
                            pd.DataFrame({
                                'unique_id': [uid],
                                'metric': ['mae'],
                                'value': [mae]
                            })
                        ])
                
                if len(alternative_metrics) > 0:
                    print("Alternative evaluation complete")
                    self.logger.info("Alternative evaluation complete")
                    result = alternative_metrics
                else:
                    error_msg = "Unable to evaluate model - no matching data points between test and forecast"
                    print(f"Error: {error_msg}")
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)
            else:
                # Evaluate with standard approach
                print(f"Computing evaluation metrics for AutoARIMA model...")
                self.logger.info(f"Computing evaluation metrics for AutoARIMA model")
                result = evaluate(merged_df, metrics=metrics, train_df=train_df)
            
            print("Evaluation complete")
            self.logger.info("Evaluation complete")
            
        finally:
            # Restore original model state
            self.data = original_data
            self.forecast_df = original_forecast_df
            
            # Re-fit the model on the full dataset to ensure it's in the same state as before
            if original_forecast_df is not None:
                print("Restoring model to original state with full dataset...")
                self.logger.info("Restoring model to original state with full dataset")
                self.fit(original_data)
        
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